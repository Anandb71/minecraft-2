//! Visibility buffer targets and the primary ray march pass.

use crate::frame::FrameCtx;
use mc2_gpu::{
    FrameGraph, HotCompute, Pass, PassBuilder, PassContext, TexHandle, TextureDesc, bind,
    bind_group, layout,
};

pub const VIS_ID: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
pub const VIS_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
pub const VIS_MOTION: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Clone, Copy)]
pub struct VisTargets {
    pub id: TexHandle,
    pub depth: TexHandle,
    pub motion: TexHandle,
}

impl VisTargets {
    pub fn create(graph: &mut FrameGraph<FrameCtx>) -> Self {
        Self {
            id: graph.create_texture(TextureDesc::render_target("vis id", VIS_ID, 1.0)),
            depth: graph.create_texture(TextureDesc::render_target("vis depth", VIS_DEPTH, 1.0)),
            motion: graph.create_texture(TextureDesc::render_target("vis motion", VIS_MOTION, 1.0)),
        }
    }
}

pub struct PrimaryVisPass {
    targets: VisTargets,
    pipeline: HotCompute,
    bgl: wgpu::BindGroupLayout,
    bind_group: Option<(u64, wgpu::BindGroup)>,
}

impl PrimaryVisPass {
    pub fn new(device: &wgpu::Device, ctx: &FrameCtx, targets: VisTargets) -> Self {
        let bgl = layout(
            device,
            "vis primary targets",
            wgpu::ShaderStages::COMPUTE,
            &[
                bind::write_2d(VIS_ID),
                bind::write_2d(VIS_DEPTH),
                bind::write_2d(VIS_MOTION),
            ],
        );
        let pipeline = HotCompute::new(
            device,
            &ctx.shaders,
            "vis_primary.wgsl",
            "main",
            &[&ctx.frame_layout, &ctx.world_layout, &bgl],
        );
        Self {
            targets,
            pipeline,
            bgl,
            bind_group: None,
        }
    }
}

impl Pass<FrameCtx> for PrimaryVisPass {
    fn name(&self) -> &'static str {
        "vis.march"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.write(self.targets.id);
        b.write(self.targets.depth);
        b.write(self.targets.motion);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        if self
            .bind_group
            .as_ref()
            .is_none_or(|(g, _)| *g != generation)
        {
            let t = self.targets;
            let bg = bind_group(
                ctx.device,
                "vis primary targets",
                &self.bgl,
                &[
                    wgpu::BindingResource::TextureView(ctx.graph.view(t.id)),
                    wgpu::BindingResource::TextureView(ctx.graph.view(t.depth)),
                    wgpu::BindingResource::TextureView(ctx.graph.view(t.motion)),
                ],
            );
            self.bind_group = Some((generation, bg));
        }
        let extent = ctx.graph.extent(self.targets.id);
        let pipeline = self.pipeline.get(ctx.device, &ctx.frame.shaders);
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vis.march"),
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
