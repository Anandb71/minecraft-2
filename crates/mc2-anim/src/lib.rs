//! Characters: a humanoid of eleven voxel parts, posed by procedural motion
//! and inverse kinematics.

pub mod skeleton;

pub use skeleton::{BONES, Bone, Placed, Pose, Root, VOXEL_M, place};
