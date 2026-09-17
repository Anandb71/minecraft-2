//! Rigid body physics for voxel debris: extended position based dynamics
//! (Müller et al. 2020) with substeps, contacts against the voxel world and
//! between bodies, friction, restitution and sleeping; explosions that turn
//! terrain into bodies; and baking of settled bodies back into the world.

pub mod body;
pub mod eigen;
pub mod probe;
pub mod shape;

pub use body::{Body, BodyId};
pub use shape::BodyShape;
