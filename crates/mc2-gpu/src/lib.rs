//! GPU foundations: device, frame graph, shaders, timestamp profiling.

pub mod device;
pub mod graph;
pub mod shader;
pub mod timing;

pub use device::{Gpu, GpuError, GpuOptions};
pub use graph::{
    FrameGraph, GraphError, GraphResources, Pass, PassBuilder, PassContext, SizePolicy, TexHandle,
    TextureDesc,
};
pub use shader::{EmbeddedShaders, ShaderError, ShaderLibrary};
pub use timing::{GpuProfiler, GpuRow, TimestampScope};
