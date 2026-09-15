//! GPU foundations: device, frame graph, shaders, timestamp profiling.

pub mod device;
pub mod timing;

pub use device::{Gpu, GpuError, GpuOptions};
pub use timing::{GpuProfiler, GpuRow, TimestampScope};
