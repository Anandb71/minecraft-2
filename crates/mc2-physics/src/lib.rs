//! Rigid body physics for voxel debris: extended position based dynamics
//! (Müller et al. 2020) with substeps, contacts against the voxel world and
//! between bodies, friction, restitution and sleeping; explosions that turn
//! terrain into bodies; and baking of settled bodies back into the world.

pub mod body;
pub mod collision;
pub mod eigen;
pub mod explode;
pub mod joint;
mod pair_contact;
pub mod probe;
pub mod raycast;
pub mod shape;
pub mod vehicle;
pub mod world;
mod world_contact;

pub use body::{Body, BodyId};
pub use joint::{Hinge, Joint};
pub use shape::BodyShape;
pub use vehicle::{Handling, Vehicle, Wheel};
pub use world::{Obstacle, PhysicsStats, PhysicsWorld};
