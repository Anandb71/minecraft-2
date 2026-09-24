//! Characters: a humanoid of eleven voxel parts, posed by procedural motion
//! and inverse kinematics.

pub mod gait;
pub mod look;
pub mod skeleton;

pub use gait::{Gait, Motion};
pub use look::{Look, Part};
pub use skeleton::{BONES, Bone, Placed, Pose, Root, VOXEL_M, place};
