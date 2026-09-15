//! Engine-wide foundations shared by every other crate.
pub mod hash;
pub mod profiler;
pub mod stats;

pub use hash::{FxHashMap, FxHashSet};
pub use stats::RollingStats;
