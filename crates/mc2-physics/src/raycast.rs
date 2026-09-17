//! Rays against bodies: the CPU reference for the GPU body tracer, and
//! picking for gameplay.

use crate::body::{Body, BodyId};
use crate::shape::VOXEL_M;
use crate::world::PhysicsWorld;
use glam::{DVec3, IVec3, Vec3};
use mc2_voxel::material::MaterialId;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyHit {
    /// Distance along the ray, metres.
    pub t: f64,
    /// Grid voxel hit.
    pub voxel: IVec3,
    /// Grid axis of the face the ray entered.
    pub axis: usize,
    /// Outward normal of that face, world space.
    pub normal: Vec3,
    pub material: MaterialId,
}

/// First solid voxel of `body` along a world ray within `max_t` metres.
pub fn raycast_body(body: &Body, origin: DVec3, dir: DVec3, max_t: f64) -> Option<BodyHit> {
    let shape = &body.shape;
    let rot = body.grid_rotation().as_dquat();
    let inv_rot = rot.inverse();
    // Grid space in voxels.
    let p = inv_rot * (origin - body.grid_origin()) / f64::from(VOXEL_M);
    let d = inv_rot * dir;
    let d = DVec3::select(d.abs().cmplt(DVec3::splat(1e-12)), DVec3::splat(1e-12), d);
    let inv = d.recip();
    let size = shape.size.as_dvec3();
    let ta = -p * inv;
    let tb = (size - p) * inv;
    let (tn3, tf3) = (ta.min(tb), ta.max(tb));
    let tn = tn3.max_element().max(0.0);
    let tf = tf3.min_element().min(max_t / f64::from(VOXEL_M));
    if tn >= tf {
        return None;
    }
    let mut axis = if tn3.x >= tn3.y && tn3.x >= tn3.z {
        0
    } else if tn3.y >= tn3.z {
        1
    } else {
        2
    };
    let step = d.signum().as_ivec3();
    let mut v = (p + d * tn)
        .floor()
        .as_ivec3()
        .clamp(IVec3::ZERO, shape.size - 1);
    let upper = DVec3::select(d.cmpgt(DVec3::ZERO), DVec3::ONE, DVec3::ZERO);
    let mut tmax = (v.as_dvec3() + upper - p) * inv;
    let delta = inv.abs();
    let mut t = tn;
    loop {
        let m = shape.get(v);
        if m.is_solid() {
            let mut n = DVec3::ZERO;
            n[axis] = -d[axis].signum();
            return Some(BodyHit {
                t: t * f64::from(VOXEL_M),
                voxel: v,
                axis,
                normal: (rot * n).as_vec3(),
                material: m,
            });
        }
        let a = if tmax.x <= tmax.y && tmax.x <= tmax.z {
            0
        } else if tmax.y <= tmax.z {
            1
        } else {
            2
        };
        t = tmax[a];
        if t >= tf {
            return None;
        }
        v[a] += step[a];
        tmax[a] += delta[a];
        axis = a;
        if v.cmplt(IVec3::ZERO).any() || v.cmpge(shape.size).any() {
            return None;
        }
    }
}

impl PhysicsWorld {
    /// Nearest body along a world ray within `max_t` metres.
    pub fn raycast(&self, origin: DVec3, dir: DVec3, max_t: f64) -> Option<(BodyId, BodyHit)> {
        let mut best: Option<(BodyId, BodyHit)> = None;
        for b in &self.bodies {
            // Bounding sphere first.
            let to = b.pos - origin;
            let along = to.dot(dir);
            let r = f64::from(b.shape.radius);
            if along < -r || to.length_squared() - along * along > r * r {
                continue;
            }
            let limit = best.map_or(max_t, |(_, h)| h.t);
            if let Some(h) = raycast_body(b, origin, dir, limit) {
                best = Some((b.id, h));
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::BodyShape;
    use glam::Quat;
    use mc2_voxel::material::ids;
    use std::sync::Arc;

    fn slab() -> Arc<BodyShape> {
        // 8 x 2 x 8 voxels: a 0.5 m x 0.125 m x 0.5 m plate.
        Arc::new(BodyShape::from_voxels(IVec3::new(8, 2, 8), vec![ids::GRANITE; 128]).unwrap())
    }

    #[test]
    fn a_ray_from_above_hits_the_top_face() {
        let mut p = PhysicsWorld::new();
        let id = p.spawn(slab(), DVec3::new(1.0, 1.0, 1.0), Quat::IDENTITY);
        let (hit_id, h) = p
            .raycast(DVec3::new(1.03, 3.0, 0.97), DVec3::NEG_Y, 10.0)
            .expect("hit");
        assert_eq!(hit_id, id);
        // Top face at 1.0 + 1 voxel.
        assert!((h.t - (2.0 - 0.0625)).abs() < 1e-9, "{}", h.t);
        assert_eq!(h.axis, 1);
        assert!((h.normal - Vec3::Y).length() < 1e-6);
        assert_eq!(h.material, ids::GRANITE);
        assert!(
            p.raycast(DVec3::new(3.0, 3.0, 3.0), DVec3::NEG_Y, 10.0)
                .is_none()
        );
    }

    #[test]
    fn rotated_bodies_report_rotated_normals() {
        let mut p = PhysicsWorld::new();
        // Plate standing on edge: its thin axis now points along x.
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        p.spawn(slab(), DVec3::new(1.0, 1.0, 1.0), rot);
        let (_, h) = p
            .raycast(DVec3::new(-2.0, 1.02, 1.01), DVec3::X, 10.0)
            .expect("hit");
        assert!((h.t - (3.0 - 0.0625)).abs() < 1e-6, "{}", h.t);
        assert!((h.normal - Vec3::NEG_X).length() < 1e-5, "{}", h.normal);
    }
}
