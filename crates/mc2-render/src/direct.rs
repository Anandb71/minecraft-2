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

pub struct RestirPass {
    vis: VisTargets,
    t: DirectTargets,
    candidates: HotCompute,
    temporal: HotCompute,
    spatial: HotCompute,
    shade: HotCompute,
    layouts: [wgpu::BindGroupLayout; 4],
    params: [wgpu::Buffer; 2],
    groups: Option<(u64, Vec<wgpu::BindGroup>)>,
}

impl RestirPass {
    pub fn new(
        device: &wgpu::Device,
        ctx: &FrameCtx,
        lights: &wgpu::BindGroupLayout,
        vis: VisTargets,
        t: DirectTargets,
    ) -> Self {
        let vis_bindings = [bind::utexture_2d(), unfilterable(), unfilterable()];
        let mk = |label: &str, extra: &[wgpu::BindingType]| {
            let mut all = vis_bindings.to_vec();
            all.extend_from_slice(extra);
            layout(device, label, wgpu::ShaderStages::COMPUTE, &all)
        };
        let candidates_l = mk(
            "restir candidates",
            &[bind::write_2d(RGBA32F), bind::write_2d(RGBA32F)],
        );
        let temporal_l = mk(
            "restir temporal",
            &[
                unfilterable(),
                unfilterable(),
                bind::utexture_2d(),
                unfilterable(),
                unfilterable(),
                bind::write_2d(RGBA32F),
                bind::write_2d(RGBA32F),
            ],
        );
        let spatial_l = mk(
            "restir spatial",
            &[
                unfilterable(),
                unfilterable(),
                bind::write_2d(RGBA32F),
                bind::write_2d(RGBA32F),
                bind::uniform(),
            ],
        );
        let shade_l = mk(
            "restir shade",
            &[unfilterable(), unfilterable(), bind::write_2d(RGBA16F)],
        );
        let hc = |file, l: &wgpu::BindGroupLayout| {
            HotCompute::new(
                device,
                &ctx.shaders,
                file,
                "main",
                &[&ctx.frame_layout, &ctx.world_layout, l, lights],
            )
        };
        let params = [0u32, 1].map(|i| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("restir spatial params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            buf.get_mapped_range_mut(..)
                .expect("mapped at creation")
                .copy_from_slice(bytemuck::cast_slice(&[i, 0, 0, 0]));
            buf.unmap();
            buf
        });
        Self {
            vis,
            t,
            candidates: hc("restir_candidates.wgsl", &candidates_l),
            temporal: hc("restir_temporal.wgsl", &temporal_l),
            spatial: hc("restir_spatial.wgsl", &spatial_l),
            shade: hc("restir_shade.wgsl", &shade_l),
            layouts: [candidates_l, temporal_l, spatial_l, shade_l],
            params,
            groups: None,
        }
    }
}

impl Pass<FrameCtx> for RestirPass {
    fn name(&self) -> &'static str {
        "direct.restir"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        let t = self.t;
        for h in [
            self.vis.id,
            self.vis.depth,
            self.vis.motion,
            t.vis_id_prev,
            t.res_a_prev,
            t.res_b_prev,
        ] {
            b.read(h);
        }
        for h in [
            t.res_a0,
            t.res_b0,
            t.res_a1,
            t.res_b1,
            t.res_a2,
            t.res_b2,
            t.direct_lights,
        ] {
            b.write(h);
        }
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        if self.groups.as_ref().is_none_or(|(g, _)| *g != generation) {
            let v = self.vis;
            let t = self.t;
            let base = [v.id, v.depth, v.motion];
            let with = |extra: &[TexHandle]| {
                let mut h = base.to_vec();
                h.extend_from_slice(extra);
                h
            };
            let dev = ctx.device;
            let candidates = bind_group(
                dev,
                "restir candidates",
                &self.layouts[0],
                &views(ctx, &with(&[t.res_a0, t.res_b0])),
            );
            let temporal = bind_group(
                dev,
                "restir temporal",
                &self.layouts[1],
                &views(
                    ctx,
                    &with(&[
                        t.res_a0,
                        t.res_b0,
                        t.vis_id_prev,
                        t.res_a_prev,
                        t.res_b_prev,
                        t.res_a1,
                        t.res_b1,
                    ]),
                ),
            );
            let spatial = |input: (TexHandle, TexHandle),
                           output: (TexHandle, TexHandle),
                           params: &wgpu::Buffer| {
                let mut r = views(ctx, &with(&[input.0, input.1, output.0, output.1]));
                r.push(params.as_entire_binding());
                bind_group(dev, "restir spatial", &self.layouts[2], &r)
            };
            let spatial1 = spatial((t.res_a1, t.res_b1), (t.res_a2, t.res_b2), &self.params[0]);
            let spatial2 = spatial((t.res_a2, t.res_b2), (t.res_a1, t.res_b1), &self.params[1]);
            let shade = bind_group(
                dev,
                "restir shade",
                &self.layouts[3],
                &views(ctx, &with(&[t.res_a1, t.res_b1, t.direct_lights])),
            );
            self.groups = Some((
                generation,
                vec![candidates, temporal, spatial1, spatial2, shade],
            ));
        }
        let groups = self.groups.as_ref().expect("groups").1.clone();
        let fg = ctx.frame.frame_bind_group.clone();
        let wg = ctx.frame.world_bind_group.clone();
        let lg = ctx.frame.lights_bind_group.clone();
        let e = ctx.graph.extent(self.t.direct_lights);
        let work = (e.width.div_ceil(8), e.height.div_ceil(8), 1);
        let dev = ctx.device;
        let lib = &ctx.frame.shaders;
        let pc = self.candidates.get(dev, lib).clone();
        let pt = self.temporal.get(dev, lib).clone();
        let ps = self.spatial.get(dev, lib).clone();
        let pd = self.shade.get(dev, lib).clone();
        dispatch_all(
            ctx,
            "direct.restir",
            &[
                (&pc, &[&fg, &wg, &groups[0], &lg], work),
                (&pt, &[&fg, &wg, &groups[1], &lg], work),
                (&ps, &[&fg, &wg, &groups[2], &lg], work),
                (&ps, &[&fg, &wg, &groups[3], &lg], work),
                (&pd, &[&fg, &wg, &groups[4], &lg], work),
            ],
        );
    }
}
