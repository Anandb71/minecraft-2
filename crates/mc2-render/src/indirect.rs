//! Indirect diffuse light: ReSTIR GI and radiance cascades behind one output,
//! selected by quality settings, both shading their hits through
//! `gi_common.wgsl` (last frame's lit surfaces on screen, the sky map off it).

use crate::frame::FrameCtx;
use crate::sky::{Job, SkyTargets, dispatch_all, linear_clamp};
use crate::vis::VisTargets;
use mc2_gpu::{
    FrameGraph, HotCompute, Pass, PassBuilder, PassContext, SizePolicy, TexHandle, TextureDesc,
    bind, bind_group, layout,
};

const RGBA16F: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const RGBA32F: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GiMethod {
    RestirGi,
    RadianceCascades,
}

impl GiMethod {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "restir" | "restir-gi" => Some(Self::RestirGi),
            "cascades" | "rc" => Some(Self::RadianceCascades),
            _ => None,
        }
    }
}

/// Radiance cascade layout: probe spacing and directions double per cascade,
/// interval ends grow fourfold (D19).
pub const RC_CASCADES: u32 = 6;
pub const RC_SPACING0: u32 = 16;
pub const RC_DIRS0: u32 = 4;
/// Length of cascade 0's interval; cascade i covers
/// [T0 (4^i - 1) / 3, T0 (4^(i+1) - 1) / 3], so six cascades reach 341 m.
pub const RC_T0_M: f32 = 0.25;

fn unf() -> wgpu::BindingType {
    bind::texture(wgpu::TextureViewDimension::D2, false)
}

fn uniform_buffer(device: &wgpu::Device, label: &str, words: &[u32]) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (words.len() * 4) as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    buf.get_mapped_range_mut(..)
        .expect("mapped at creation")
        .copy_from_slice(bytemuck::cast_slice(words));
    buf.unmap();
    buf
}

/// Inputs every indirect pass shades hits with.
#[derive(Clone, Copy)]
pub struct GiInputs {
    pub vis: VisTargets,
    pub prev_id: TexHandle,
    pub surface_prev: TexHandle,
    pub sky: SkyTargets,
}

impl GiInputs {
    fn read(&self, b: &mut PassBuilder<'_>) {
        for h in [
            self.vis.id,
            self.vis.depth,
            self.vis.motion,
            self.prev_id,
            self.surface_prev,
            self.sky.ambient,
            self.sky.transmittance,
            self.sky.sky_view,
            self.sky.sky_view_moon,
        ] {
            b.read(h);
        }
    }

    /// Layout entries for bindings 0..=10 of gi_common's expectations.
    fn layout_entries() -> Vec<wgpu::BindingType> {
        vec![
            bind::utexture_2d(),
            unf(),
            unf(),
            bind::utexture_2d(),
            unf(),
            unf(),
            unf(),
            bind::texture_2d(),
            bind::texture_2d(),
            bind::texture_2d(),
            bind::sampler(true),
        ]
    }

    fn resources<'a>(
        &self,
        ctx: &'a PassContext<'_, FrameCtx>,
        sky_map: &'a wgpu::TextureView,
        sampler: &'a wgpu::Sampler,
    ) -> Vec<wgpu::BindingResource<'a>> {
        let g = ctx.graph;
        let view = |h| wgpu::BindingResource::TextureView(g.view(h));
        vec![
            view(self.vis.id),
            view(self.vis.depth),
            view(self.vis.motion),
            view(self.prev_id),
            view(self.surface_prev),
            wgpu::BindingResource::TextureView(sky_map),
            view(self.sky.ambient),
            view(self.sky.transmittance),
            view(self.sky.sky_view),
            view(self.sky.sky_view_moon),
            wgpu::BindingResource::Sampler(sampler),
        ]
    }
}

fn gi_desc(
    label: &'static str,
    format: wgpu::TextureFormat,
    shift: u32,
    history: bool,
) -> TextureDesc {
    let mut d = TextureDesc {
        size: SizePolicy::RenderBlocks {
            block: 1 << shift,
            extra: 0,
        },
        ..TextureDesc::render_target(label, format, 1.0)
    };
    if history {
        d.usage |= wgpu::TextureUsages::COPY_DST;
    }
    d
}

/// The noisy demodulated irradiance both methods write.
pub fn gi_output(graph: &mut FrameGraph<FrameCtx>, shift: u32) -> TexHandle {
    graph.create_texture(gi_desc("gi noisy", RGBA16F, shift, false))
}

#[derive(Clone, Copy)]
pub struct RestirGiTargets {
    init_pos: TexHandle,
    init_rad: TexHandle,
    t_pos: TexHandle,
    t_rad: TexHandle,
    t_res: TexHandle,
    prev_pos: TexHandle,
    prev_rad: TexHandle,
    prev_res: TexHandle,
    s_pos: TexHandle,
    s_rad: TexHandle,
    s_res: TexHandle,
}

impl RestirGiTargets {
    pub fn create(graph: &mut FrameGraph<FrameCtx>, shift: u32) -> Self {
        let mut t = |label, history| {
            let d = gi_desc(label, RGBA32F, shift, history);
            if history {
                graph.create_history(d)
            } else {
                graph.create_texture(d)
            }
        };
        Self {
            init_pos: t("gi initial position", false),
            init_rad: t("gi initial radiance", false),
            t_pos: t("gi temporal position", false),
            t_rad: t("gi temporal radiance", false),
            t_res: t("gi temporal reservoir", false),
            prev_pos: t("gi temporal position prev", true),
            prev_rad: t("gi temporal radiance prev", true),
            prev_res: t("gi temporal reservoir prev", true),
            s_pos: t("gi spatial position", false),
            s_rad: t("gi spatial radiance", false),
            s_res: t("gi spatial reservoir", false),
        }
    }

    pub fn history_copies(&self) -> [(TexHandle, TexHandle); 3] {
        [
            (self.t_pos, self.prev_pos),
            (self.t_rad, self.prev_rad),
            (self.t_res, self.prev_res),
        ]
    }
}

pub struct RestirGiPass {
    inputs: GiInputs,
    t: RestirGiTargets,
    out: TexHandle,
    sky_map: wgpu::TextureView,
    sampler: wgpu::Sampler,
    pipelines: [HotCompute; 4],
    layouts: [wgpu::BindGroupLayout; 4],
    params: wgpu::Buffer,
    groups: Option<(u64, Vec<wgpu::BindGroup>)>,
}

impl RestirGiPass {
    pub fn new(
        device: &wgpu::Device,
        ctx: &FrameCtx,
        inputs: GiInputs,
        t: RestirGiTargets,
        out: TexHandle,
        sky_map: wgpu::TextureView,
        shift: u32,
    ) -> Self {
        let vis3 = [bind::utexture_2d(), unf(), unf()];
        let mut initial = GiInputs::layout_entries();
        initial.extend([
            bind::write_2d(RGBA32F),
            bind::write_2d(RGBA32F),
            bind::uniform(),
        ]);
        let mut temporal = vis3.to_vec();
        temporal.extend([
            bind::utexture_2d(),
            unf(),
            unf(),
            unf(),
            unf(),
            unf(),
            bind::write_2d(RGBA32F),
            bind::write_2d(RGBA32F),
            bind::write_2d(RGBA32F),
            bind::uniform(),
        ]);
        let mut spatial = vis3.to_vec();
        spatial.extend([
            unf(),
            unf(),
            unf(),
            bind::write_2d(RGBA32F),
            bind::write_2d(RGBA32F),
            bind::write_2d(RGBA32F),
            bind::uniform(),
        ]);
        let mut shade = vis3.to_vec();
        shade.extend([
            unf(),
            unf(),
            unf(),
            bind::write_2d(RGBA16F),
            bind::uniform(),
        ]);
        let stages = wgpu::ShaderStages::COMPUTE;
        let layouts = [
            layout(device, "restir gi initial", stages, &initial),
            layout(device, "restir gi temporal", stages, &temporal),
            layout(device, "restir gi spatial", stages, &spatial),
            layout(device, "restir gi shade", stages, &shade),
        ];
        let f = &ctx.frame_layout;
        let w = &ctx.world_layout;
        let hc = |file, l: &[&wgpu::BindGroupLayout]| {
            HotCompute::new(device, &ctx.shaders, file, "main", l)
        };
        let pipelines = [
            hc("restir_gi_initial.wgsl", &[f, w, &layouts[0]]),
            hc("restir_gi_temporal.wgsl", &[f, &layouts[1]]),
            hc("restir_gi_spatial.wgsl", &[f, w, &layouts[2]]),
            hc("restir_gi_shade.wgsl", &[f, &layouts[3]]),
        ];
        Self {
            inputs,
            t,
            out,
            sky_map,
            sampler: linear_clamp(device),
            pipelines,
            layouts,
            params: uniform_buffer(device, "restir gi params", &[shift, 0, 0, 0]),
            groups: None,
        }
    }
}

impl Pass<FrameCtx> for RestirGiPass {
    fn name(&self) -> &'static str {
        "indirect.restir_gi"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        self.inputs.read(b);
        let t = self.t;
        for h in [t.prev_pos, t.prev_rad, t.prev_res] {
            b.read(h);
        }
        for h in [
            t.init_pos, t.init_rad, t.t_pos, t.t_rad, t.t_res, t.s_pos, t.s_rad, t.s_res, self.out,
        ] {
            b.write(h);
        }
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        if ctx.frame.gi != GiMethod::RestirGi {
            return;
        }
        let generation = ctx.graph.generation();
        if self.groups.as_ref().is_none_or(|(g, _)| *g != generation) {
            let t = self.t;
            let v = self.inputs.vis;
            let g = ctx.graph;
            let view = |h| wgpu::BindingResource::TextureView(g.view(h));
            let params = || self.params.as_entire_binding();
            let mut initial = self.inputs.resources(ctx, &self.sky_map, &self.sampler);
            initial.extend([view(t.init_pos), view(t.init_rad), params()]);
            let temporal = vec![
                view(v.id),
                view(v.depth),
                view(v.motion),
                view(self.inputs.prev_id),
                view(t.init_pos),
                view(t.init_rad),
                view(t.prev_pos),
                view(t.prev_rad),
                view(t.prev_res),
                view(t.t_pos),
                view(t.t_rad),
                view(t.t_res),
                params(),
            ];
            let spatial = vec![
                view(v.id),
                view(v.depth),
                view(v.motion),
                view(t.t_pos),
                view(t.t_rad),
                view(t.t_res),
                view(t.s_pos),
                view(t.s_rad),
                view(t.s_res),
                params(),
            ];
            let shade = vec![
                view(v.id),
                view(v.depth),
                view(v.motion),
                view(t.s_pos),
                view(t.s_rad),
                view(t.s_res),
                view(self.out),
                params(),
            ];
            let groups = [initial, temporal, spatial, shade]
                .iter()
                .zip(&self.layouts)
                .map(|(r, l)| bind_group(ctx.device, "restir gi", l, r))
                .collect();
            self.groups = Some((generation, groups));
        }
        let groups = self.groups.as_ref().expect("groups").1.clone();
        let fg = ctx.frame.frame_bind_group.clone();
        let wg = ctx.frame.world_bind_group.clone();
        let p: Vec<wgpu::ComputePipeline> = self
            .pipelines
            .iter_mut()
            .map(|h| h.get(ctx.device, &ctx.frame.shaders).clone())
            .collect();
        let e = ctx.graph.extent(self.out);
        let work = (e.width.div_ceil(8), e.height.div_ceil(8), 1);
        let s0 = [&fg, &wg, &groups[0]];
        let s1 = [&fg, &groups[1]];
        let s2 = [&fg, &wg, &groups[2]];
        let s3 = [&fg, &groups[3]];
        let jobs: [Job<'_>; 4] = [
            (&p[0], &s0, work),
            (&p[1], &s1, work),
            (&p[2], &s2, work),
            (&p[3], &s3, work),
        ];
        dispatch_all(ctx, "indirect.restir_gi", &jobs);
    }
}

#[derive(Clone, Copy)]
pub struct CascadeTargets {
    raw: [TexHandle; RC_CASCADES as usize],
    merged: [TexHandle; RC_CASCADES as usize],
}

impl CascadeTargets {
    pub fn create(graph: &mut FrameGraph<FrameCtx>) -> Self {
        // Every cascade's atlas is (render / spacing0 * dirs0) plus room for
        // the last partial probe of the coarsest cascade.
        let extra = RC_DIRS0 << (RC_CASCADES - 1);
        let desc = |label| TextureDesc {
            size: SizePolicy::RenderBlocks {
                block: RC_SPACING0 / RC_DIRS0,
                extra,
            },
            ..TextureDesc::render_target(label, RGBA16F, 1.0)
        };
        let raw_labels = [
            "rc raw 0", "rc raw 1", "rc raw 2", "rc raw 3", "rc raw 4", "rc raw 5",
        ];
        let merged_labels = [
            "rc merged 0",
            "rc merged 1",
            "rc merged 2",
            "rc merged 3",
            "rc merged 4",
            "rc merged 5",
        ];
        Self {
            raw: raw_labels.map(|l| graph.create_texture(desc(l))),
            merged: merged_labels.map(|l| graph.create_texture(desc(l))),
        }
    }
}

pub struct CascadesPass {
    inputs: GiInputs,
    t: CascadeTargets,
    out: TexHandle,
    sky_map: wgpu::TextureView,
    sampler: wgpu::Sampler,
    trace: HotCompute,
    merge: HotCompute,
    gather: HotCompute,
    layouts: [wgpu::BindGroupLayout; 3],
    /// One per cascade, plus the gather params.
    params: Vec<wgpu::Buffer>,
    groups: Option<(u64, Vec<wgpu::BindGroup>)>,
}

impl CascadesPass {
    pub fn new(
        device: &wgpu::Device,
        ctx: &FrameCtx,
        inputs: GiInputs,
        t: CascadeTargets,
        out: TexHandle,
        sky_map: wgpu::TextureView,
        shift: u32,
    ) -> Self {
        let mut trace = GiInputs::layout_entries();
        trace.extend([bind::write_2d(RGBA16F), bind::uniform()]);
        let merge = [
            bind::utexture_2d(),
            unf(),
            unf(),
            unf(),
            unf(),
            bind::write_2d(RGBA16F),
            bind::uniform(),
        ];
        let gather = [
            bind::utexture_2d(),
            unf(),
            unf(),
            unf(),
            bind::write_2d(RGBA16F),
            bind::uniform(),
        ];
        let stages = wgpu::ShaderStages::COMPUTE;
        let layouts = [
            layout(device, "rc trace", stages, &trace),
            layout(device, "rc merge", stages, &merge),
            layout(device, "rc gather", stages, &gather),
        ];
        let params = (0..=RC_CASCADES)
            .map(|i| {
                let c = i.min(RC_CASCADES - 1);
                // Interval ends grow 4x per cascade: T0 (4^i - 1) / 3.
                let t_min = RC_T0_M * ((1u32 << (2 * c)) - 1) as f32 / 3.0;
                let t_max = RC_T0_M * ((1u32 << (2 * c + 2)) - 1) as f32 / 3.0;
                let words = [
                    c,
                    RC_CASCADES,
                    RC_SPACING0 << c,
                    RC_DIRS0 << c,
                    t_min.to_bits(),
                    t_max.to_bits(),
                    shift,
                    0,
                ];
                // The last buffer is cascade 0's layout for the gather pass.
                let words = if i == RC_CASCADES {
                    [0, RC_CASCADES, RC_SPACING0, RC_DIRS0, 0, 0, shift, 0]
                } else {
                    words
                };
                uniform_buffer(device, "rc params", &words)
            })
            .collect();
        let f = &ctx.frame_layout;
        let w = &ctx.world_layout;
        Self {
            inputs,
            t,
            out,
            sky_map,
            sampler: linear_clamp(device),
            trace: HotCompute::new(
                device,
                &ctx.shaders,
                "rc_trace.wgsl",
                "main",
                &[f, w, &layouts[0]],
            ),
            merge: HotCompute::new(
                device,
                &ctx.shaders,
                "rc_merge.wgsl",
                "main",
                &[f, &layouts[1]],
            ),
            gather: HotCompute::new(
                device,
                &ctx.shaders,
                "rc_gather.wgsl",
                "main",
                &[f, &layouts[2]],
            ),
            layouts,
            params,
            groups: None,
        }
    }
}

impl Pass<FrameCtx> for CascadesPass {
    fn name(&self) -> &'static str {
        "indirect.cascades"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        self.inputs.read(b);
        for i in 0..RC_CASCADES as usize {
            b.write(self.t.raw[i]);
            b.write(self.t.merged[i]);
        }
        b.write(self.out);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        if ctx.frame.gi != GiMethod::RadianceCascades {
            return;
        }
        let n = RC_CASCADES as usize;
        let generation = ctx.graph.generation();
        if self.groups.as_ref().is_none_or(|(g, _)| *g != generation) {
            let g = ctx.graph;
            let view = |h| wgpu::BindingResource::TextureView(g.view(h));
            let v = self.inputs.vis;
            let mut groups = Vec::new();
            for i in 0..n {
                let mut r = self.inputs.resources(ctx, &self.sky_map, &self.sampler);
                r.extend([view(self.t.raw[i]), self.params[i].as_entire_binding()]);
                groups.push(bind_group(ctx.device, "rc trace", &self.layouts[0], &r));
            }
            // merge i reads raw i and merged i+1; the coarsest copies raw
            // through with a merge against itself that its opaque last
            // interval makes a no-op.
            for i in 0..n {
                let coarser = if i + 1 < n {
                    self.t.merged[i + 1]
                } else {
                    self.t.raw[i]
                };
                let r = vec![
                    view(v.id),
                    view(v.depth),
                    view(v.motion),
                    view(self.t.raw[i]),
                    view(coarser),
                    view(self.t.merged[i]),
                    self.params[i].as_entire_binding(),
                ];
                groups.push(bind_group(ctx.device, "rc merge", &self.layouts[1], &r));
            }
            let r = vec![
                view(v.id),
                view(v.depth),
                view(v.motion),
                view(self.t.merged[0]),
                view(self.out),
                self.params[n].as_entire_binding(),
            ];
            groups.push(bind_group(ctx.device, "rc gather", &self.layouts[2], &r));
            self.groups = Some((generation, groups));
        }
        let groups = self.groups.as_ref().expect("groups").1.clone();
        let fg = ctx.frame.frame_bind_group.clone();
        let wg = ctx.frame.world_bind_group.clone();
        let pt = self.trace.get(ctx.device, &ctx.frame.shaders).clone();
        let pm = self.merge.get(ctx.device, &ctx.frame.shaders).clone();
        let pg = self.gather.get(ctx.device, &ctx.frame.shaders).clone();
        let atlas = ctx.graph.extent(self.t.raw[0]);
        let atlas_work = (atlas.width.div_ceil(8), atlas.height.div_ceil(8), 1);
        let e = ctx.graph.extent(self.out);
        let gather_work = (e.width.div_ceil(8), e.height.div_ceil(8), 1);
        let trace_sets: Vec<[&wgpu::BindGroup; 3]> =
            groups[..n].iter().map(|g| [&fg, &wg, g]).collect();
        let merge_sets: Vec<[&wgpu::BindGroup; 2]> =
            groups[n..2 * n].iter().map(|g| [&fg, g]).collect();
        let gather_set = [&fg, &groups[2 * n]];
        let mut jobs: Vec<Job<'_>> = Vec::new();
        for set in &trace_sets {
            jobs.push((&pt, set, atlas_work));
        }
        // Coarsest first so each merge reads a finished coarser cascade.
        for set in merge_sets.iter().rev() {
            jobs.push((&pm, set, atlas_work));
        }
        jobs.push((&pg, &gather_set, gather_work));
        dispatch_all(ctx, "indirect.cascades", &jobs);
    }
}

/// Traced glossy reflections at GI resolution, shaded like indirect hits.
pub struct ReflectionPass {
    inputs: GiInputs,
    out: TexHandle,
    sky_map: wgpu::TextureView,
    sampler: wgpu::Sampler,
    pipeline: HotCompute,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    group: Option<(u64, wgpu::BindGroup)>,
}

impl ReflectionPass {
    pub fn new(
        device: &wgpu::Device,
        ctx: &FrameCtx,
        inputs: GiInputs,
        out: TexHandle,
        sky_map: wgpu::TextureView,
        shift: u32,
    ) -> Self {
        let mut entries = GiInputs::layout_entries();
        entries.extend([bind::write_2d(RGBA16F), bind::uniform()]);
        let layout = layout(device, "reflections", wgpu::ShaderStages::COMPUTE, &entries);
        let pipeline = HotCompute::new(
            device,
            &ctx.shaders,
            "reflect_trace.wgsl",
            "main",
            &[&ctx.frame_layout, &ctx.world_layout, &layout],
        );
        Self {
            inputs,
            out,
            sky_map,
            sampler: linear_clamp(device),
            pipeline,
            layout,
            params: uniform_buffer(device, "reflection params", &[shift, 0, 0, 0]),
            group: None,
        }
    }
}

impl Pass<FrameCtx> for ReflectionPass {
    fn name(&self) -> &'static str {
        "refl.trace"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        self.inputs.read(b);
        b.write(self.out);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        if self.group.as_ref().is_none_or(|(g, _)| *g != generation) {
            let mut r = self.inputs.resources(ctx, &self.sky_map, &self.sampler);
            r.push(wgpu::BindingResource::TextureView(ctx.graph.view(self.out)));
            r.push(self.params.as_entire_binding());
            self.group = Some((
                generation,
                bind_group(ctx.device, "reflections", &self.layout, &r),
            ));
        }
        let group = self.group.as_ref().expect("group").1.clone();
        let fg = ctx.frame.frame_bind_group.clone();
        let wg = ctx.frame.world_bind_group.clone();
        let p = self.pipeline.get(ctx.device, &ctx.frame.shaders).clone();
        let e = ctx.graph.extent(self.out);
        let set = [&fg, &wg, &group];
        dispatch_all(
            ctx,
            "refl.trace",
            &[(&p, &set, (e.width.div_ceil(8), e.height.div_ceil(8), 1))],
        );
    }
}
