//! Traced audio: sounds made from nothing but maths, heard through the
//! world as it is.
//!
//! `dsp` holds the filters, delays and reverb. The sound of a space is
//! measured, not chosen: rays from the listener through the voxels give
//! how enclosed it is, how far sound travels between surfaces and how much
//! they swallow, and so the reverb's decay and the early echoes; a ray to
//! each source gives how much of the world stands between.

pub mod acoustics;
pub mod dsp;
pub mod synth;

pub use acoustics::{Acoustics, Occlusion, audible, measure, occlusion};
pub use synth::{Beds, Ground, Sound};
