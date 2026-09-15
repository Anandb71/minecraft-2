//! Engine-wide foundations shared by every other crate.
pub mod profiler;
pub mod stats;

pub use stats::RollingStats;
