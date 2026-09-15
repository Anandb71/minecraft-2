//! GPU foundations: device, frame graph, shaders, timestamp profiling.

pub mod device;
pub mod shader;
pub mod timing;

pub use device::{Gpu, GpuError, GpuOptions};
pub use shader::{EmbeddedShaders, ShaderError, ShaderLibrary};
pub use timing::{GpuProfiler, GpuRow, TimestampScope};
