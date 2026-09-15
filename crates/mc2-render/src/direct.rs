//! Direct lighting passes: traced sun and moon visibility, ReSTIR over
//! emissive voxels, composition with sky and aerial perspective, and the
//! history copies that make next frame's temporal reuse possible.

use crate::frame::FrameCtx;
use crate::sky::dispatch_all;
use crate::vis::VisTargets;
use mc2_gpu::{
    FrameGraph, HotCompute, Pass, PassBuilder, PassContext, TexHandle, TextureDesc, bind,
    bind_group, layout,
};

const RGBA16F: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const RGBA32F: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

fn history_desc(label: &'static str, format: wgpu::TextureFormat) -> TextureDesc {
    let mut d = TextureDesc::render_target(label, format, 1.0);
    d.usage |= wgpu::TextureUsages::COPY_DST;
    d
}

fn unfilterable() -> wgpu::BindingType {
    bind::texture(wgpu::TextureViewDimension::D2, false)
}

#[derive(Clone, Copy)]
pub struct DirectTargets {
    pub vis_id_prev: TexHandle,
    pub light_vis: TexHandle,
    pub light_vis_prev: TexHandle,
    pub res_a0: TexHandle,
    pub res_b0: TexHandle,
    pub res_a1: TexHandle,
    pub res_b1: TexHandle,
    pub res_a2: TexHandle,
    pub res_b2: TexHandle,
    pub res_a_prev: TexHandle,
    pub res_b_prev: TexHandle,
    pub direct_lights: TexHandle,
}

impl DirectTargets {
    pub fn create(graph: &mut FrameGraph<FrameCtx>) -> Self {
        let t = |g: &mut FrameGraph<FrameCtx>, l, f| {
            g.create_texture(TextureDesc::render_target(l, f, 1.0))
        };
        Self {
            vis_id_prev: graph.create_history(history_desc("vis id prev", crate::vis::VIS_ID)),
            light_vis: t(graph, "light visibility", RGBA16F),
            light_vis_prev: graph.create_history(history_desc("light visibility prev", RGBA16F)),
            res_a0: t(graph, "reservoir a0", RGBA32F),
            res_b0: t(graph, "reservoir b0", RGBA32F),
            res_a1: t(graph, "reservoir a1", RGBA32F),
            res_b1: t(graph, "reservoir b1", RGBA32F),
            res_a2: t(graph, "reservoir a2", RGBA32F),
            res_b2: t(graph, "reservoir b2", RGBA32F),
            res_a_prev: graph.create_history(history_desc("reservoir a prev", RGBA32F)),
            res_b_prev: graph.create_history(history_desc("reservoir b prev", RGBA32F)),
            direct_lights: t(graph, "direct lights", RGBA16F),
        }
    }
}

fn views<'a>(
    ctx: &'a PassContext<'_, FrameCtx>,
    handles: &[TexHandle],
) -> Vec<wgpu::BindingResource<'a>> {
    handles
        .iter()
        .map(|&h| wgpu::BindingResource::TextureView(ctx.graph.view(h)))
        .collect()
}

pub struct SunPass {
    vis: VisTargets,
    t: DirectTargets,
    pipeline: HotCompute,
    bgl: wgpu::BindGroupLayout,
    group: Option<(u64, wgpu::BindGroup)>,
}

impl SunPass {
    pub fn new(device: &wgpu::Device, ctx: &FrameCtx, vis: VisTargets, t: DirectTargets) -> Self {
        let bgl = layout(
            device,
            "direct sun",
            wgpu::ShaderStages::COMPUTE,
            &[
                bind::utexture_2d(),
                unfilterable(),
                unfilterable(),
                bind::utexture_2d(),
                unfilterable(),
                bind::write_2d(RGBA16F),
            ],
        );
        let pipeline = HotCompute::new(
            device,
            &ctx.shaders,
            "direct_sun.wgsl",
            "main",
            &[&ctx.frame_layout, &ctx.world_layout, &bgl],
        );
        Self {
            vis,
            t,
            pipeline,
            bgl,
            group: None,
        }
    }
}

impl Pass<FrameCtx> for SunPass {
    fn name(&self) -> &'static str {
        "direct.sun"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.vis.id);
        b.read(self.vis.depth);
        b.read(self.vis.motion);
        b.read(self.t.vis_id_prev);
        b.read(self.t.light_vis_prev);
        b.write(self.t.light_vis);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        if self.group.as_ref().is_none_or(|(g, _)| *g != generation) {
            let v = self.vis;
            let res = views(
                ctx,
                &[
                    v.id,
                    v.depth,
                    v.motion,
                    self.t.vis_id_prev,
                    self.t.light_vis_prev,
                    self.t.light_vis,
                ],
            );
            self.group = Some((
                generation,
                bind_group(ctx.device, "direct sun", &self.bgl, &res),
            ));
        }
        let group = self.group.as_ref().expect("group").1.clone();
        let (fg, wg) = (
            ctx.frame.frame_bind_group.clone(),
            ctx.frame.world_bind_group.clone(),
        );
        let p = self.pipeline.get(ctx.device, &ctx.frame.shaders).clone();
        let e = ctx.graph.extent(self.t.light_vis);
        dispatch_all(
            ctx,
            "direct.sun",
            &[(
                &p,
                &[&fg, &wg, &group],
                (e.width.div_ceil(8), e.height.div_ceil(8), 1),
            )],
        );
    }
}
