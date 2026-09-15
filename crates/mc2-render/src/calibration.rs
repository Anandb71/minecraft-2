//! Calibration image pass: exposure ramp, saturated bars, moving bar.

use crate::frame::FrameCtx;
use bytemuck::{Pod, Zeroable};
use mc2_gpu::{HotCompute, Pass, PassBuilder, PassContext, TexHandle, bind, bind_group, layout};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CalibrationUniforms {
    time: f32,
    frame: u32,
    _pad: [f32; 2],
}

pub struct CalibrationPass {
    out: TexHandle,
    pipeline: HotCompute,
    bgl: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    bind_group: Option<(u64, wgpu::BindGroup)>,
}

impl CalibrationPass {
    pub fn new(device: &wgpu::Device, lib: &mc2_gpu::ShaderLibrary, out: TexHandle) -> Self {
        let bgl = layout(
            device,
            "calibration",
            wgpu::ShaderStages::COMPUTE,
            &[
                bind::uniform(),
                bind::write_2d(wgpu::TextureFormat::Rgba16Float),
            ],
        );
        let pipeline = HotCompute::new(device, lib, "calibration.wgsl", "main", &[&bgl]);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("calibration uniforms"),
            size: std::mem::size_of::<CalibrationUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            out,
            pipeline,
            bgl,
            uniforms,
            bind_group: None,
        }
    }
}

impl Pass<FrameCtx> for CalibrationPass {
    fn name(&self) -> &'static str {
        "calibration"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.write(self.out);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        ctx.queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&CalibrationUniforms {
                time: ctx.frame.time,
                frame: ctx.frame.frame_index,
                _pad: [0.0; 2],
            }),
        );
        let generation = ctx.graph.generation();
        if self
            .bind_group
            .as_ref()
            .is_none_or(|(g, _)| *g != generation)
        {
            let bg = bind_group(
                ctx.device,
                "calibration",
                &self.bgl,
                &[
                    self.uniforms.as_entire_binding(),
                    wgpu::BindingResource::TextureView(ctx.graph.view(self.out)),
                ],
            );
            self.bind_group = Some((generation, bg));
        }
        let extent = ctx.graph.extent(self.out);
        let pipeline = self.pipeline.get(ctx.device, &ctx.frame.shaders);
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("calibration"),
                timestamp_writes: ctx.timestamps.compute(),
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group.as_ref().expect("bind group").1, &[]);
        pass.dispatch_workgroups(
            mc2_gpu::groups(extent.width, 8),
            mc2_gpu::groups(extent.height, 8),
            1,
        );
    }
}
