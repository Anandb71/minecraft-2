//! Temporal upsampling from render to display resolution.

use crate::frame::FrameCtx;
use crate::sky::{dispatch_all, linear_clamp};
use crate::vis::VisTargets;
use mc2_gpu::{
    FrameGraph, HotCompute, Pass, PassBuilder, PassContext, TexHandle, TextureDesc, bind,
    bind_group, layout,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Clone, Copy)]
pub struct UpsampleTargets {
    /// Display-resolution HDR result.
    pub color: TexHandle,
    /// Last frame's `color`.
    pub history: TexHandle,
}

impl UpsampleTargets {
    pub fn create(graph: &mut FrameGraph<FrameCtx>) -> Self {
        let mut history = TextureDesc::output_target("taa history", FORMAT);
        history.usage |= wgpu::TextureUsages::COPY_DST;
        Self {
            color: graph.create_texture(TextureDesc::output_target("taa color", FORMAT)),
            history: graph.create_history(history),
        }
    }
}

pub struct TemporalUpsamplePass {
    scene: TexHandle,
    vis: VisTargets,
    t: UpsampleTargets,
    exposure_prev: TexHandle,
    pipeline: HotCompute,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    group: Option<(u64, wgpu::BindGroup)>,
}

impl TemporalUpsamplePass {
    pub fn new(
        device: &wgpu::Device,
        ctx: &FrameCtx,
        scene: TexHandle,
        vis: VisTargets,
        t: UpsampleTargets,
        exposure_prev: TexHandle,
    ) -> Self {
        let unfilterable = || bind::texture(wgpu::TextureViewDimension::D2, false);
        let bgl = layout(
            device,
            "taa upsample",
            wgpu::ShaderStages::COMPUTE,
            &[
                unfilterable(),
                unfilterable(),
                unfilterable(),
                bind::texture_2d(),
                bind::sampler(true),
                bind::write_2d(FORMAT),
                unfilterable(),
            ],
        );
        let pipeline = HotCompute::new(
            device,
            &ctx.shaders,
            "taa_upsample.wgsl",
            "main",
            &[&ctx.frame_layout, &bgl],
        );
        Self {
            scene,
            vis,
            t,
            exposure_prev,
            pipeline,
            bgl,
            sampler: linear_clamp(device),
            group: None,
        }
    }
}

impl Pass<FrameCtx> for TemporalUpsamplePass {
    fn name(&self) -> &'static str {
        "denoise.upsample"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.scene);
        b.read(self.vis.depth);
        b.read(self.vis.motion);
        b.read(self.t.history);
        b.read(self.exposure_prev);
        b.write(self.t.color);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        if self.group.as_ref().is_none_or(|(g, _)| *g != generation) {
            let g = ctx.graph;
            let view = |h| wgpu::BindingResource::TextureView(g.view(h));
            let group = bind_group(
                ctx.device,
                "taa upsample",
                &self.bgl,
                &[
                    view(self.scene),
                    view(self.vis.depth),
                    view(self.vis.motion),
                    view(self.t.history),
                    wgpu::BindingResource::Sampler(&self.sampler),
                    view(self.t.color),
                    view(self.exposure_prev),
                ],
            );
            self.group = Some((generation, group));
        }
        let group = self.group.as_ref().expect("group").1.clone();
        let fg = ctx.frame.frame_bind_group.clone();
        let p = self.pipeline.get(ctx.device, &ctx.frame.shaders).clone();
        let e = ctx.graph.extent(self.t.color);
        dispatch_all(
            ctx,
            "denoise.upsample",
            &[(
                &p,
                &[&fg, &group],
                (e.width.div_ceil(8), e.height.div_ceil(8), 1),
            )],
        );
    }
}
