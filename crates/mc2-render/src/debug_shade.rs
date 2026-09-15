//! Debug shading of the visibility buffer into the scene HDR target.

use crate::frame::FrameCtx;
use crate::vis::VisTargets;
use mc2_gpu::{HotCompute, Pass, PassBuilder, PassContext, TexHandle, bind, bind_group, layout};

pub struct DebugShadePass {
    vis: VisTargets,
    out: TexHandle,
    pipeline: HotCompute,
    bgl: wgpu::BindGroupLayout,
    bind_group: Option<(u64, wgpu::BindGroup)>,
}

impl DebugShadePass {
    pub fn new(device: &wgpu::Device, ctx: &FrameCtx, vis: VisTargets, out: TexHandle) -> Self {
        let bgl = layout(
            device,
            "debug shade",
            wgpu::ShaderStages::COMPUTE,
            &[
                bind::utexture_2d(),
                bind::texture(wgpu::TextureViewDimension::D2, false),
                bind::texture(wgpu::TextureViewDimension::D2, false),
                bind::write_2d(wgpu::TextureFormat::Rgba16Float),
            ],
        );
        let pipeline = HotCompute::new(
            device,
            &ctx.shaders,
            "debug_shade.wgsl",
            "main",
            &[&ctx.frame_layout, &ctx.world_layout, &bgl],
        );
        Self {
            vis,
            out,
            pipeline,
            bgl,
            bind_group: None,
        }
    }
}

impl Pass<FrameCtx> for DebugShadePass {
    fn name(&self) -> &'static str {
        "debug.shade"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.vis.id);
        b.read(self.vis.depth);
        b.read(self.vis.motion);
        b.write(self.out);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        if ctx.frame.uniforms.debug_mode == 0 {
            return;
        }
        let generation = ctx.graph.generation();
        if self
            .bind_group
            .as_ref()
            .is_none_or(|(g, _)| *g != generation)
        {
            let bg = bind_group(
                ctx.device,
                "debug shade",
                &self.bgl,
                &[
                    wgpu::BindingResource::TextureView(ctx.graph.view(self.vis.id)),
                    wgpu::BindingResource::TextureView(ctx.graph.view(self.vis.depth)),
                    wgpu::BindingResource::TextureView(ctx.graph.view(self.vis.motion)),
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
                label: Some("debug.shade"),
                timestamp_writes: ctx.timestamps.compute(),
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &ctx.frame.frame_bind_group, &[]);
        pass.set_bind_group(1, &ctx.frame.world_bind_group, &[]);
        pass.set_bind_group(2, &self.bind_group.as_ref().expect("bind group").1, &[]);
        pass.dispatch_workgroups(
            mc2_gpu::groups(extent.width, 8),
            mc2_gpu::groups(extent.height, 8),
            1,
        );
    }
}
