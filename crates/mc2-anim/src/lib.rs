//! Characters: a humanoid of eleven voxel parts, posed by procedural motion
//! and inverse kinematics.
//!
//! `skeleton` places the bones, `look` builds the parts a character is made
//! of, `gait` moves it, `ik` puts its feet on the ground and turns its head.

pub mod gait;
pub mod ik;
pub mod look;
pub mod skeleton;

pub use gait::{Gait, Motion};
pub use look::{Look, Part};
pub use skeleton::{BONES, Bone, Placed, Pose, Root, VOXEL_M, place};

use glam::{DVec3, Quat};

/// Where each part's voxel grid goes for drawing: its corner (metres) and
/// rotation, from the bones placed and the parts' pivots.
pub fn part_grids(placed: &[Placed; BONES], parts: &[Part; BONES]) -> [(DVec3, Quat); BONES] {
    std::array::from_fn(|i| {
        let p = placed[i];
        let corner = p.joint - (p.rot * (parts[i].pivot * VOXEL_M)).as_dvec3();
        (corner, p.rot)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_part_grid_puts_its_pivot_on_its_joint() {
        let look = Look::from_seed(3);
        let parts = look::parts(&look);
        let root = Root {
            feet: DVec3::new(4.0, 2.0, 1.0),
            yaw: 0.7,
        };
        let placed = place(root, &Pose::default());
        let grids = part_grids(&placed, &parts);
        for i in 0..BONES {
            let (corner, rot) = grids[i];
            let pivot = corner + (rot * (parts[i].pivot * VOXEL_M)).as_dvec3();
            assert!(pivot.distance(placed[i].joint) < 1e-6);
        }
    }
}
