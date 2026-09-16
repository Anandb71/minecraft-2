//! Direct lighting passes: traced sun and moon visibility, ReSTIR over
//! emissive voxels, composition with sky and aerial perspective, and the
//! history copies that make next frame's temporal reuse possible.

use crate::frame::FrameCtx;
use crate::sky::{SkyTargets, dispatch_all, linear_clamp};
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
    pub light_trace: TexHandle,
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
    /// Outgoing diffuse radiance per pixel, and last frame's.
    pub surface: TexHandle,
    pub surface_prev: TexHandle,
}

impl DirectTargets {
    pub fn create(graph: &mut FrameGraph<FrameCtx>) -> Self {
        let t = |g: &mut FrameGraph<FrameCtx>, l, f| {
            g.create_texture(TextureDesc::render_target(l, f, 1.0))
        };
        Self {
            vis_id_prev: graph.create_history(history_desc("vis id prev", crate::vis::VIS_ID)),
            light_trace: t(graph, "light trace", RGBA16F),
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
            surface: t(graph, "surface radiance", RGBA16F),
            surface_prev: graph.create_history(history_desc("surface radiance prev", RGBA16F)),
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
    trace: HotCompute,
    resolve: HotCompute,
    trace_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    groups: Option<(u64, wgpu::BindGroup, wgpu::BindGroup)>,
}

impl SunPass {
    pub fn new(device: &wgpu::Device, ctx: &FrameCtx, vis: VisTargets, t: DirectTargets) -> Self {
        let trace_layout = layout(
            device,
            "visibility trace",
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
        let resolve_layout = layout(
            device,
            "visibility resolve",
            wgpu::ShaderStages::COMPUTE,
            &[
                bind::utexture_2d(),
                unfilterable(),
                bind::utexture_2d(),
                unfilterable(),
                unfilterable(),
                bind::write_2d(RGBA16F),
                unfilterable(),
            ],
        );
        let trace = HotCompute::new(
            device,
            &ctx.shaders,
            "visibility_trace.wgsl",
            "main",
            &[&ctx.frame_layout, &ctx.world_layout, &trace_layout],
        );
        let resolve = HotCompute::new(
            device,
            &ctx.shaders,
            "visibility_resolve.wgsl",
            "main",
            &[&ctx.frame_layout, &ctx.world_layout, &resolve_layout],
        );
        Self {
            vis,
            t,
            trace,
            resolve,
            trace_layout,
            resolve_layout,
            groups: None,
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
        b.write(self.t.light_trace);
        b.write(self.t.light_vis);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        if self
            .groups
            .as_ref()
            .is_none_or(|(g, _, _)| *g != generation)
        {
            let (v, t) = (self.vis, self.t);
            let trace = views(
                ctx,
                &[
                    v.id,
                    v.depth,
                    v.motion,
                    t.vis_id_prev,
                    t.light_vis_prev,
                    t.light_trace,
                ],
            );
            let resolve = views(
                ctx,
                &[
                    v.id,
                    v.motion,
                    t.vis_id_prev,
                    t.light_vis_prev,
                    t.light_trace,
                    t.light_vis,
                    v.depth,
                ],
            );
            self.groups = Some((
                generation,
                bind_group(ctx.device, "visibility trace", &self.trace_layout, &trace),
                bind_group(
                    ctx.device,
                    "visibility resolve",
                    &self.resolve_layout,
                    &resolve,
                ),
            ));
        }
        let (_, tg, rg) = self.groups.as_ref().expect("groups");
        let (tg, rg) = (tg.clone(), rg.clone());
        let (fg, wg) = (
            ctx.frame.frame_bind_group.clone(),
            ctx.frame.world_bind_group.clone(),
        );
        let pt = self.trace.get(ctx.device, &ctx.frame.shaders).clone();
        let pr = self.resolve.get(ctx.device, &ctx.frame.shaders).clone();
        let e = ctx.graph.extent(self.t.light_vis);
        let stride = ctx.frame.uniforms.trace_stride.max(1);
        let blocks = (e.width.div_ceil(stride), e.height.div_ceil(stride));
        dispatch_all(
            ctx,
            "direct.sun",
            &[
                (
                    &pt,
                    &[&fg, &wg, &tg],
                    (blocks.0.div_ceil(8), blocks.1.div_ceil(8), 1),
                ),
                (
                    &pr,
                    &[&fg, &wg, &rg],
                    (e.width.div_ceil(8), e.height.div_ceil(8), 1),
                ),
            ],
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
        // No emitter in range: composition ignores the output, skip it all.
        if ctx.frame.uniforms.light_count == 0 {
            return;
        }
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

/// Denoised lighting signals composition reads, and where it writes
/// surface radiance for next frame's indirect rays.
#[derive(Clone)]
pub struct ComposeInputs {
    /// Emitter irradiance (denoised).
    pub emitters: TexHandle,
    /// Indirect irradiance (denoised, GI resolution).
    pub gi: TexHandle,
    pub surface: TexHandle,
    pub clouds: TexHandle,
    pub cloud_shadow: TexHandle,
    pub cloud_uniforms: wgpu::Buffer,
    /// Integrated froxel fog.
    pub fog: TexHandle,
    /// Denoised glossy reflections (GI resolution).
    pub reflections: TexHandle,
}

pub struct ComposePass {
    vis: VisTargets,
    t: DirectTargets,
    sky: SkyTargets,
    inputs: ComposeInputs,
    out: TexHandle,
    pipeline: HotCompute,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    group: Option<(u64, wgpu::BindGroup)>,
}

impl ComposePass {
    pub fn new(
        device: &wgpu::Device,
        ctx: &FrameCtx,
        vis: VisTargets,
        t: DirectTargets,
        sky: SkyTargets,
        inputs: ComposeInputs,
        out: TexHandle,
    ) -> Self {
        let bgl = layout(
            device,
            "shade compose",
            wgpu::ShaderStages::COMPUTE,
            &[
                bind::utexture_2d(),
                unfilterable(),
                unfilterable(),
                unfilterable(),
                unfilterable(),
                bind::texture_2d(),
                bind::texture_2d(),
                bind::texture(wgpu::TextureViewDimension::D3, true),
                bind::sampler(true),
                bind::write_2d(RGBA16F),
                bind::texture_2d(),
                bind::texture(wgpu::TextureViewDimension::D3, true),
                unfilterable(),
                unfilterable(),
                bind::write_2d(RGBA16F),
                bind::texture_2d(),
                unfilterable(),
                bind::uniform(),
                bind::texture(wgpu::TextureViewDimension::D3, true),
                unfilterable(),
            ],
        );
        let pipeline = HotCompute::new(
            device,
            &ctx.shaders,
            "shade_compose.wgsl",
            "main",
            &[&ctx.frame_layout, &ctx.world_layout, &bgl],
        );
        Self {
            vis,
            t,
            sky,
            inputs,
            out,
            pipeline,
            bgl,
            sampler: linear_clamp(device),
            group: None,
        }
    }
}

impl Pass<FrameCtx> for ComposePass {
    fn name(&self) -> &'static str {
        "direct.compose"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        for h in [
            self.vis.id,
            self.vis.depth,
            self.vis.motion,
            self.t.light_vis,
            self.inputs.emitters,
            self.inputs.gi,
        ] {
            b.read(h);
        }
        b.write(self.inputs.surface);
        b.read(self.inputs.clouds);
        b.read(self.inputs.cloud_shadow);
        b.read(self.inputs.fog);
        b.read(self.inputs.reflections);
        b.read(self.sky.transmittance);
        b.read(self.sky.sky_view);
        b.read(self.sky.aerial);
        b.read(self.sky.sky_view_moon);
        b.read(self.sky.aerial_moon);
        b.read(self.sky.ambient);
        b.write(self.out);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        if self.group.as_ref().is_none_or(|(g, _)| *g != generation) {
            let v = self.vis;
            let mut r = views(
                ctx,
                &[
                    v.id,
                    v.depth,
                    v.motion,
                    self.t.light_vis,
                    self.inputs.emitters,
                    self.sky.transmittance,
                    self.sky.sky_view,
                    self.sky.aerial,
                ],
            );
            r.push(wgpu::BindingResource::Sampler(&self.sampler));
            r.push(wgpu::BindingResource::TextureView(ctx.graph.view(self.out)));
            r.push(wgpu::BindingResource::TextureView(
                ctx.graph.view(self.sky.sky_view_moon),
            ));
            r.push(wgpu::BindingResource::TextureView(
                ctx.graph.view(self.sky.aerial_moon),
            ));
            r.push(wgpu::BindingResource::TextureView(
                ctx.graph.view(self.sky.ambient),
            ));
            r.push(wgpu::BindingResource::TextureView(
                ctx.graph.view(self.inputs.gi),
            ));
            r.push(wgpu::BindingResource::TextureView(
                ctx.graph.view(self.inputs.surface),
            ));
            r.push(wgpu::BindingResource::TextureView(
                ctx.graph.view(self.inputs.clouds),
            ));
            r.push(wgpu::BindingResource::TextureView(
                ctx.graph.view(self.inputs.cloud_shadow),
            ));
            r.push(self.inputs.cloud_uniforms.as_entire_binding());
            r.push(wgpu::BindingResource::TextureView(
                ctx.graph.view(self.inputs.fog),
            ));
            r.push(wgpu::BindingResource::TextureView(
                ctx.graph.view(self.inputs.reflections),
            ));
            self.group = Some((
                generation,
                bind_group(ctx.device, "shade compose", &self.bgl, &r),
            ));
        }
        let group = self.group.as_ref().expect("group").1.clone();
        let (fg, wg) = (
            ctx.frame.frame_bind_group.clone(),
            ctx.frame.world_bind_group.clone(),
        );
        let p = self.pipeline.get(ctx.device, &ctx.frame.shaders).clone();
        let e = ctx.graph.extent(self.out);
        dispatch_all(
            ctx,
            "direct.compose",
            &[(
                &p,
                &[&fg, &wg, &group],
                (e.width.div_ceil(8), e.height.div_ceil(8), 1),
            )],
        );
    }
}

/// Copies this frame's results into history textures for the next frame.
pub struct HistoryPass {
    copies: Vec<(TexHandle, TexHandle)>,
}

impl HistoryPass {
    pub fn new(copies: Vec<(TexHandle, TexHandle)>) -> Self {
        Self { copies }
    }
}

impl Pass<FrameCtx> for HistoryPass {
    fn name(&self) -> &'static str {
        "post.history"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        for &(src, dst) in &self.copies {
            b.read(src);
            b.write(dst);
        }
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        for &(src, dst) in &self.copies {
            let s = ctx.graph.texture(src);
            let d = ctx.graph.texture(dst);
            ctx.encoder
                .copy_texture_to_texture(s.as_image_copy(), d.as_image_copy(), s.size());
        }
    }
}

/// Metered exposure with temporal adaptation.
pub struct ExposurePass {
    scene: TexHandle,
    previous: TexHandle,
    out: TexHandle,
    pipeline: HotCompute,
    bgl: wgpu::BindGroupLayout,
    group: Option<(u64, wgpu::BindGroup)>,
}

impl ExposurePass {
    pub fn targets(graph: &mut FrameGraph<FrameCtx>) -> (TexHandle, TexHandle) {
        let desc = |label| TextureDesc {
            size: mc2_gpu::SizePolicy::Fixed(1, 1, 1),
            ..history_desc(label, RGBA16F)
        };
        (
            graph.create_texture(desc("exposure")),
            graph.create_history(desc("exposure prev")),
        )
    }

    pub fn new(
        device: &wgpu::Device,
        ctx: &FrameCtx,
        scene: TexHandle,
        (out, previous): (TexHandle, TexHandle),
    ) -> Self {
        let bgl = layout(
            device,
            "exposure",
            wgpu::ShaderStages::COMPUTE,
            &[unfilterable(), unfilterable(), bind::write_2d(RGBA16F)],
        );
        let pipeline = HotCompute::new(
            device,
            &ctx.shaders,
            "exposure.wgsl",
            "main",
            &[&ctx.frame_layout, &bgl],
        );
        Self {
            scene,
            previous,
            out,
            pipeline,
            bgl,
            group: None,
        }
    }
}

impl Pass<FrameCtx> for ExposurePass {
    fn name(&self) -> &'static str {
        "post.exposure"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.scene);
        b.read(self.previous);
        b.write(self.out);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        if self.group.as_ref().is_none_or(|(g, _)| *g != generation) {
            let r = views(ctx, &[self.scene, self.previous, self.out]);
            self.group = Some((
                generation,
                bind_group(ctx.device, "exposure", &self.bgl, &r),
            ));
        }
        let group = self.group.as_ref().expect("group").1.clone();
        let fg = ctx.frame.frame_bind_group.clone();
        let p = self.pipeline.get(ctx.device, &ctx.frame.shaders).clone();
        dispatch_all(ctx, "post.exposure", &[(&p, &[&fg, &group], (1, 1, 1))]);
    }
}
