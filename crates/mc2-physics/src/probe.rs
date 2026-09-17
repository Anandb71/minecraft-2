//! Sphere probes against voxel grids: the contact query shared by
//! collisions with the world and between bodies.

use crate::shape::VOXEL_M;
use glam::{DVec3, IVec3, Vec3};
use mc2_voxel::world::VoxelWorld;

/// Collision samples are spheres of half a voxel.
pub const SAMPLE_RADIUS: f32 = VOXEL_M * 0.5;

/// Contacts of a sphere of `r` voxels centred at `pv` (voxel units of some
/// grid) against the grid's solid cells: each axis direction whose
/// neighbour cell the sphere reaches, or, when the centre is already inside
/// a solid cell, the nearest face with empty space beyond. Yields the
/// outward normal (grid axes), the depth in voxels and the contact point.
pub fn sphere_contacts(
    pv: DVec3,
    r: f64,
    solid: impl Fn(IVec3) -> bool,
    out: &mut Vec<(IVec3, f64, DVec3)>,
) {
    let centre = pv.floor().as_ivec3();
    if solid(centre) {
        let mut best: Option<(f64, IVec3)> = None;
        for o in AXES {
            if solid(centre + o) {
                continue;
            }
            let axis = axis_of(o);
            let face = f64::from(centre[axis] + o[axis].max(0));
            let d = (face - pv[axis]).abs();
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, o));
            }
        }
        if let Some((d, o)) = best {
            out.push((o, d + r, pv - o.as_dvec3() * r));
        }
        return;
    }
    for o in AXES {
        let axis = axis_of(o);
        let q = pv + o.as_dvec3() * r;
        let v = q.floor().as_ivec3();
        if v == centre || !solid(v) {
            continue;
        }
        let face = f64::from(v[axis] + (-o[axis]).max(0));
        let depth = r - (face - pv[axis]).abs();
        if depth > 0.0 {
            out.push((-o, depth, q));
        }
    }
}

const AXES: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

fn axis_of(o: IVec3) -> usize {
    if o.x != 0 {
        0
    } else if o.y != 0 {
        1
    } else {
        2
    }
}

/// World contacts of a sample sphere at `p` (world metres): normal, depth
/// in metres, contact point.
pub fn sample_contacts(world: &VoxelWorld, p: DVec3, out: &mut Vec<(Vec3, f32, DVec3)>) {
    let mut raw = Vec::new();
    let r = f64::from(SAMPLE_RADIUS) * 16.0;
    sphere_contacts(p * 16.0, r, |v| world.voxel(v).is_solid(), &mut raw);
    for (o, d, q) in raw {
        out.push((o.as_vec3(), (d / 16.0) as f32, q / 16.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sphere_resting_on_a_face_reports_its_overlap() {
        let solid = |v: IVec3| v.y < 0;
        let mut out = Vec::new();
        // Centre 0.3 voxels above the floor, radius 0.5: 0.2 deep.
        sphere_contacts(DVec3::new(0.5, 0.3, 0.5), 0.5, solid, &mut out);
        assert_eq!(out.len(), 1);
        let (n, depth, point) = out[0];
        assert_eq!(n, IVec3::Y);
        assert!((depth - 0.2).abs() < 1e-9);
        assert!((point.y + 0.2).abs() < 1e-9);
        // Clear of the floor: nothing.
        out.clear();
        sphere_contacts(DVec3::new(0.5, 0.6, 0.5), 0.5, solid, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn a_buried_sphere_leaves_through_the_nearest_open_face() {
        // Solid below y = 4, open above: the centre at y = 3.8 is inside.
        let solid = |v: IVec3| v.y < 4;
        let mut out = Vec::new();
        sphere_contacts(DVec3::new(0.5, 3.8, 0.5), 0.5, solid, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, IVec3::Y);
        assert!((out[0].1 - 0.7).abs() < 1e-9);
    }
}
