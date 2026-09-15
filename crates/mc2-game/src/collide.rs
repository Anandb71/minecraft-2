//! Axis-aligned boxes against the voxel grid.
//!
//! Movement is swept one axis at a time in steps no longer than a voxel, so a
//! box can never tunnel through a 6.25 cm wall. When a step would overlap a
//! solid voxel the box stops flush against the voxel face on that axis.

use glam::{DVec3, IVec3};
use mc2_voxel::coords::{self, VOXELS_PER_BLOCK};
use mc2_voxel::material::{Kind, MaterialId};
use mc2_voxel::world::VoxelWorld;

pub const VOXEL_M: f64 = 1.0 / VOXELS_PER_BLOCK as f64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: DVec3,
    pub max: DVec3,
}

impl Aabb {
    /// Box standing on `feet` with the given width and height, metres.
    pub fn standing(feet: DVec3, width: f64, height: f64) -> Self {
        let h = width * 0.5;
        Self {
            min: DVec3::new(feet.x - h, feet.y, feet.z - h),
            max: DVec3::new(feet.x + h, feet.y + height, feet.z + h),
        }
    }

    pub fn translated(&self, d: DVec3) -> Self {
        Self {
            min: self.min + d,
            max: self.max + d,
        }
    }

    pub fn feet(&self) -> DVec3 {
        DVec3::new(
            (self.min.x + self.max.x) * 0.5,
            self.min.y,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    /// Voxel index range overlapped, inclusive, shrunk by a hair so a box
    /// resting exactly on a face does not overlap the voxel beyond it.
    pub fn voxel_range(&self) -> (IVec3, IVec3) {
        let eps = 1e-6;
        let lo = ((self.min + eps) / VOXEL_M).floor().as_ivec3();
        let hi = ((self.max - eps) / VOXEL_M).floor().as_ivec3();
        (lo, hi)
    }

    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.cmplt(other.max).all() && other.min.cmplt(self.max).all()
    }
}

/// Whether a material stops a body. Foliage, liquids and air do not.
pub fn blocks_movement(m: MaterialId) -> bool {
    matches!(m.get().kind, Kind::Solid | Kind::Transparent)
}

/// Something a box can collide with.
pub trait SolidField {
    fn solid(&self, v: IVec3) -> bool;
}

impl SolidField for VoxelWorld {
    fn solid(&self, v: IVec3) -> bool {
        if !coords::in_world_voxel(v) {
            // The world has walls and a floor, but open sky.
            return v.y < 0
                || (v.y >= 0 && (v.x < 0 || v.z < 0 || v.x >= 262_144 || v.z >= 262_144));
        }
        blocks_movement(self.voxel(v))
    }
}

pub fn overlaps_solid(field: &impl SolidField, b: &Aabb) -> bool {
    let (lo, hi) = b.voxel_range();
    for y in lo.y..=hi.y {
        for z in lo.z..=hi.z {
            for x in lo.x..=hi.x {
                if field.solid(IVec3::new(x, y, z)) {
                    return true;
                }
            }
        }
    }
    false
}

/// Result of a swept move.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MoveResult {
    pub moved: DVec3,
    /// Per axis: -1 hit moving negative, +1 hit moving positive, 0 free.
    pub hit: IVec3,
}

/// Sweeps `b` by `delta`, Y first, then X, then Z. Returns the new box.
pub fn sweep(field: &impl SolidField, b: Aabb, delta: DVec3) -> (Aabb, MoveResult) {
    let mut cur = b;
    let mut result = MoveResult::default();
    for axis in [1usize, 0, 2] {
        let d = delta[axis];
        if d == 0.0 {
            continue;
        }
        let steps = (d.abs() / (VOXEL_M * 0.9)).ceil().max(1.0) as i32;
        let step = d / f64::from(steps);
        for _ in 0..steps {
            let mut unit = DVec3::ZERO;
            unit[axis] = step;
            let next = cur.translated(unit);
            if overlaps_solid(field, &next) {
                // Snap flush to the voxel face the step ran into. The step is
                // shorter than a voxel, so that face lies between the two boxes.
                let shift = if step > 0.0 {
                    let face = ((next.max[axis] - 1e-9) / VOXEL_M).floor() * VOXEL_M;
                    (face - cur.max[axis]).max(0.0)
                } else {
                    let face = ((next.min[axis] + 1e-9) / VOXEL_M).floor() * VOXEL_M + VOXEL_M;
                    (face - cur.min[axis]).min(0.0)
                };
                let mut s = DVec3::ZERO;
                s[axis] = shift;
                let snapped = cur.translated(s);
                if !overlaps_solid(field, &snapped) {
                    cur = snapped;
                }
                result.hit[axis] = if step > 0.0 { 1 } else { -1 };
                break;
            }
            cur = next;
        }
    }
    result.moved = cur.min - b.min;
    (cur, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;

    fn floor_world() -> VoxelWorld {
        let mut w = VoxelWorld::new();
        // A 4 m square floor whose top face is at y = 10 m, and a wall.
        w.fill_box(IVec3::new(0, 0, 0), IVec3::new(63, 159, 63), ids::GRANITE);
        w.fill_box(
            IVec3::new(40, 160, 0),
            IVec3::new(41, 190, 63),
            ids::GRANITE,
        );
        w
    }

    #[test]
    fn falling_box_lands_flush_on_the_floor() {
        let w = floor_world();
        let b = Aabb::standing(DVec3::new(1.0, 12.0, 1.0), 0.6, 1.8);
        let (after, r) = sweep(&w, b, DVec3::new(0.0, -5.0, 0.0));
        assert_eq!(r.hit.y, -1);
        assert!((after.min.y - 10.0).abs() < 1e-9, "feet at {}", after.min.y);
        assert!(!overlaps_solid(&w, &after));
    }

    #[test]
    fn thin_wall_blocks_fast_motion() {
        let w = floor_world();
        let b = Aabb::standing(DVec3::new(1.0, 10.0, 1.0), 0.6, 1.8);
        // 3 m in one call, far more than the 12.5 cm wall is thick.
        let (after, r) = sweep(&w, b, DVec3::new(3.0, 0.0, 0.0));
        assert_eq!(r.hit.x, 1);
        assert!(
            (after.max.x - 2.5).abs() < 1e-9,
            "stopped at {}",
            after.max.x
        );
    }

    #[test]
    fn sliding_along_a_wall_keeps_the_free_axis() {
        let w = floor_world();
        let b = Aabb::standing(DVec3::new(2.2, 10.0, 1.0), 0.6, 1.8);
        let (after, r) = sweep(&w, b, DVec3::new(0.5, 0.0, 0.7));
        assert_eq!(r.hit.x, 1);
        assert_eq!(r.hit.z, 0);
        assert!((after.min.z - b.min.z - 0.7).abs() < 1e-9);
    }
}
