//! Per-frame context threaded through every pass.

use crate::camera::FrameUniforms;
use crate::hud::HudCanvas;
use bytemuck::Zeroable;

pub struct FrameCtx {
    pub shaders: mc2_gpu::ShaderLibrary,
    pub time: f32,
    pub frame_index: u32,
    pub hud: HudCanvas,
    pub exposure: f32,
    pub tonemap: bool,
    pub uniforms: FrameUniforms,
    pub frame_buffer: wgpu::Buffer,
    pub frame_layout: wgpu::BindGroupLayout,
    pub frame_bind_group: wgpu::BindGroup,
    pub world_layout: wgpu::BindGroupLayout,
    pub world_bind_group: wgpu::BindGroup,
}

impl FrameCtx {
    pub fn new(
        device: &wgpu::Device,
        shaders: mc2_gpu::ShaderLibrary,
        world_layout: wgpu::BindGroupLayout,
        world_bind_group: wgpu::BindGroup,
    ) -> Self {
        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_layout = mc2_gpu::layout(
            device,
            "frame",
            wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT,
            &[mc2_gpu::bind::uniform()],
        );
        let frame_bind_group = mc2_gpu::bind_group(
            device,
            "frame",
            &frame_layout,
            &[frame_buffer.as_entire_binding()],
        );
        Self {
            shaders,
            time: 0.0,
            frame_index: 0,
            hud: HudCanvas::default(),
            exposure: 1.0,
            tonemap: true,
            uniforms: FrameUniforms::zeroed(),
            frame_buffer,
            frame_layout,
            frame_bind_group,
            world_layout,
            world_bind_group,
        }
    }
}
