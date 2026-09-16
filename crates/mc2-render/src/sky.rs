//! Atmosphere passes: LUTs built once, sky-view and aerial perspective per frame.

use crate::frame::FrameCtx;
use mc2_gpu::{
    FrameGraph, HotCompute, Pass, PassBuilder, PassContext, SizePolicy, TexHandle, TextureDesc,
    bind, bind_group, layout,
};

pub const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
pub const MULTISCATTER_SIZE: (u32, u32) = (32, 32);
pub const SKY_VIEW_SIZE: (u32, u32) = (200, 100);
pub const AERIAL_SIZE: (u32, u32, u32) = (32, 32, 32);
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Clone, Copy)]
pub struct SkyTargets {
    pub transmittance: TexHandle,
    pub multiscatter: TexHandle,
    pub sky_view: TexHandle,
    pub aerial: TexHandle,
    /// Moon-lit counterparts, rendered only while moonlight matters.
    pub sky_view_moon: TexHandle,
    pub aerial_moon: TexHandle,
    /// Sky irradiance for the six axis normals (+X -X +Y -Y +Z -Z).
    pub ambient: TexHandle,
}

fn fixed(label: &'static str, w: u32, h: u32, d: u32) -> TextureDesc {
    TextureDesc {
        size: SizePolicy::Fixed(w, h, d),
        dimension: if d > 1 {
            wgpu::TextureDimension::D3
        } else {
            wgpu::TextureDimension::D2
        },
        ..TextureDesc::render_target(label, FORMAT, 1.0)
    }
}

impl SkyTargets {
    pub fn create(graph: &mut FrameGraph<FrameCtx>) -> Self {
        let (tw, th) = TRANSMITTANCE_SIZE;
        let (mw, mh) = MULTISCATTER_SIZE;
        let (sw, sh) = SKY_VIEW_SIZE;
        let (aw, ah, ad) = AERIAL_SIZE;
        Self {
            // LUTs persist across frames and are rebuilt only on demand.
            transmittance: graph.create_history(fixed("sky transmittance", tw, th, 1)),
            multiscatter: graph.create_history(fixed("sky multiscatter", mw, mh, 1)),
            sky_view: graph.create_texture(fixed("sky view", sw, sh, 1)),
            aerial: graph.create_texture(fixed("sky aerial", aw, ah, ad)),
            // History textures persist, so a skipped moon frame keeps valid
            // (if stale) contents rather than whatever the graph reallocated.
            sky_view_moon: graph.create_history(fixed("sky view moon", sw, sh, 1)),
            aerial_moon: graph.create_history(fixed("sky aerial moon", aw, ah, ad)),
            ambient: graph.create_texture(fixed("sky ambient", 6, 1, 1)),
        }
    }
}

pub fn linear_clamp(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("linear clamp"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// A pipeline, its bind groups by index, and a workgroup count.
pub type Job<'a> = (
    &'a wgpu::ComputePipeline,
    &'a [&'a wgpu::BindGroup],
    (u32, u32, u32),
);

/// One compute pass (one timestamp scope) running several dispatches.
pub fn dispatch_all(ctx: &mut PassContext<'_, FrameCtx>, label: &str, jobs: &[Job<'_>]) {
    let mut pass = ctx
        .encoder
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: ctx.timestamps.compute(),
        });
    for (pipeline, groups, work) in jobs {
        pass.set_pipeline(pipeline);
        for (i, g) in groups.iter().enumerate() {
            pass.set_bind_group(i as u32, *g, &[]);
        }
        pass.dispatch_workgroups(work.0, work.1, work.2);
    }
}

/// Transmittance then multiple scattering, once.
pub struct SkyLutPass {
    t: SkyTargets,
    transmittance: HotCompute,
    multiscatter: HotCompute,
    t_layout: wgpu::BindGroupLayout,
    m_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    built_generation: Option<u64>,
}

impl SkyLutPass {
    pub fn new(device: &wgpu::Device, ctx: &FrameCtx, t: SkyTargets) -> Self {
        let t_layout = layout(
            device,
            "sky transmittance",
            wgpu::ShaderStages::COMPUTE,
            &[bind::write_2d(FORMAT)],
        );
        let m_layout = layout(
            device,
            "sky multiscatter",
            wgpu::ShaderStages::COMPUTE,
            &[
                bind::texture_2d(),
                bind::sampler(true),
                bind::write_2d(FORMAT),
            ],
        );
        Self {
            t,
            transmittance: HotCompute::new(
                device,
                &ctx.shaders,
                "sky_transmittance.wgsl",
                "main",
                &[&t_layout],
            ),
            multiscatter: HotCompute::new(
                device,
                &ctx.shaders,
                "sky_multiscatter.wgsl",
                "main",
                &[&m_layout],
            ),
            t_layout,
            m_layout,
            sampler: linear_clamp(device),
            built_generation: None,
        }
    }
}

impl Pass<FrameCtx> for SkyLutPass {
    fn name(&self) -> &'static str {
        "sky.luts"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.write(self.t.transmittance);
        b.write(self.t.multiscatter);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        // Rebuild after allocation or a shader reload.
        let generation = ctx.graph.generation() ^ (ctx.frame.shaders.generation() << 32);
        if self.built_generation == Some(generation) {
            return;
        }
        self.built_generation = Some(generation);
        let tv = ctx.graph.view(self.t.transmittance);
        let mv = ctx.graph.view(self.t.multiscatter);
        let tg = bind_group(
            ctx.device,
            "sky transmittance",
            &self.t_layout,
            &[wgpu::BindingResource::TextureView(tv)],
        );
        let mg = bind_group(
            ctx.device,
            "sky multiscatter",
            &self.m_layout,
            &[
                wgpu::BindingResource::TextureView(tv),
                wgpu::BindingResource::Sampler(&self.sampler),
                wgpu::BindingResource::TextureView(mv),
            ],
        );
        let (tw, th) = TRANSMITTANCE_SIZE;
        let (mw, mh) = MULTISCATTER_SIZE;
        let pt = self
            .transmittance
            .get(ctx.device, &ctx.frame.shaders)
            .clone();
        let pm = self
            .multiscatter
            .get(ctx.device, &ctx.frame.shaders)
            .clone();
        dispatch_all(
            ctx,
            "sky.luts",
            &[
                (&pt, &[&tg], (tw.div_ceil(8), th.div_ceil(8), 1)),
                (&pm, &[&mg], (mw.div_ceil(8), mh.div_ceil(8), 1)),
            ],
        );
    }
}

/// Per-frame sky-view LUT and aerial perspective volume, for the sun and,
/// at night, the moon.
pub struct SkyFramePass {
    t: SkyTargets,
    view: HotCompute,
    aerial: HotCompute,
    view_layout: wgpu::BindGroupLayout,
    aerial_layout: wgpu::BindGroupLayout,
    ambient: HotCompute,
    ambient_layout: wgpu::BindGroupLayout,
    ambient_group: Option<(u64, wgpu::BindGroup)>,
    sampler: wgpu::Sampler,
    /// Light index uniforms: sun, moon.
    params: [wgpu::Buffer; 2],
    /// (generation, [sun view, sun aerial, moon view, moon aerial]).
    groups: Option<(u64, [wgpu::BindGroup; 4])>,
}

impl SkyFramePass {
    pub fn new(device: &wgpu::Device, ctx: &FrameCtx, t: SkyTargets) -> Self {
        let luts = [bind::texture_2d(), bind::texture_2d(), bind::sampler(true)];
        let view_layout = layout(
            device,
            "sky view",
            wgpu::ShaderStages::COMPUTE,
            &[
                luts[0],
                luts[1],
                luts[2],
                bind::write_2d(FORMAT),
                bind::uniform(),
            ],
        );
        let aerial_layout = layout(
            device,
            "sky aerial",
            wgpu::ShaderStages::COMPUTE,
            &[
                luts[0],
                luts[1],
                luts[2],
                bind::storage_texture(
                    FORMAT,
                    wgpu::TextureViewDimension::D3,
                    wgpu::StorageTextureAccess::WriteOnly,
                ),
                bind::uniform(),
            ],
        );
        let ambient_layout = layout(
            device,
            "sky ambient",
            wgpu::ShaderStages::COMPUTE,
            &[
                bind::texture_2d(),
                bind::texture_2d(),
                bind::texture_2d(),
                bind::sampler(true),
                bind::write_2d(FORMAT),
            ],
        );
        let params = [0u32, 1].map(|light| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sky light params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            });
            buf.get_mapped_range_mut(..)
                .expect("mapped at creation")
                .copy_from_slice(bytemuck::cast_slice(&[light, 0, 0, 0]));
            buf.unmap();
            buf
        });
        Self {
            t,
            view: HotCompute::new(
                device,
                &ctx.shaders,
                "sky_view.wgsl",
                "main",
                &[&ctx.frame_layout, &view_layout],
            ),
            aerial: HotCompute::new(
                device,
                &ctx.shaders,
                "sky_aerial.wgsl",
                "main",
                &[&ctx.frame_layout, &aerial_layout],
            ),
            view_layout,
            aerial_layout,
            ambient: HotCompute::new(
                device,
                &ctx.shaders,
                "sky_ambient.wgsl",
                "main",
                &[&ctx.frame_layout, &ambient_layout],
            ),
            ambient_layout,
            ambient_group: None,
            sampler: linear_clamp(device),
            params,
            groups: None,
        }
    }
}

impl Pass<FrameCtx> for SkyFramePass {
    fn name(&self) -> &'static str {
        "sky.frame"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.t.transmittance);
        b.read(self.t.multiscatter);
        b.write(self.t.sky_view);
        b.write(self.t.aerial);
        b.write(self.t.sky_view_moon);
        b.write(self.t.aerial_moon);
        b.write(self.t.ambient);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        if self.groups.as_ref().is_none_or(|(g, _)| *g != generation) {
            let g = ctx.graph;
            let group = |layout: &wgpu::BindGroupLayout, out: TexHandle, light: usize| {
                bind_group(
                    ctx.device,
                    "sky frame",
                    layout,
                    &[
                        wgpu::BindingResource::TextureView(g.view(self.t.transmittance)),
                        wgpu::BindingResource::TextureView(g.view(self.t.multiscatter)),
                        wgpu::BindingResource::Sampler(&self.sampler),
                        wgpu::BindingResource::TextureView(g.view(out)),
                        self.params[light].as_entire_binding(),
                    ],
                )
            };
            let groups = [
                group(&self.view_layout, self.t.sky_view, 0),
                group(&self.aerial_layout, self.t.aerial, 0),
                group(&self.view_layout, self.t.sky_view_moon, 1),
                group(&self.aerial_layout, self.t.aerial_moon, 1),
            ];
            self.groups = Some((generation, groups));
        }
        let groups = self.groups.as_ref().expect("groups").1.clone();
        if self
            .ambient_group
            .as_ref()
            .is_none_or(|(g, _)| *g != generation)
        {
            let g = ctx.graph;
            let view = |h| wgpu::BindingResource::TextureView(g.view(h));
            let group = bind_group(
                ctx.device,
                "sky ambient",
                &self.ambient_layout,
                &[
                    view(self.t.transmittance),
                    view(self.t.sky_view),
                    view(self.t.sky_view_moon),
                    wgpu::BindingResource::Sampler(&self.sampler),
                    view(self.t.ambient),
                ],
            );
            self.ambient_group = Some((generation, group));
        }
        let ambient_group = self.ambient_group.as_ref().expect("group").1.clone();
        let frame_group = ctx.frame.frame_bind_group.clone();
        let (sw, sh) = SKY_VIEW_SIZE;
        let (aw, ah, ad) = AERIAL_SIZE;
        let view_work = (sw.div_ceil(8), sh.div_ceil(8), 1);
        let aerial_work = (aw.div_ceil(4), ah.div_ceil(4), ad.div_ceil(4));
        let pv = self.view.get(ctx.device, &ctx.frame.shaders).clone();
        let pa = self.aerial.get(ctx.device, &ctx.frame.shaders).clone();
        let pamb = self.ambient.get(ctx.device, &ctx.frame.shaders).clone();
        let moon = ctx.frame.uniforms.sky_flags & crate::camera::SKY_MOON != 0;
        let sets = [
            [&frame_group, &groups[0]],
            [&frame_group, &groups[1]],
            [&frame_group, &groups[2]],
            [&frame_group, &groups[3]],
            [&frame_group, &ambient_group],
        ];
        let mut jobs: Vec<Job<'_>> = vec![(&pv, &sets[0], view_work), (&pa, &sets[1], aerial_work)];
        if moon {
            jobs.push((&pv, &sets[2], view_work));
            jobs.push((&pa, &sets[3], aerial_work));
        }
        // The ambient cube reads the sky-view LUTs written above.
        jobs.push((&pamb, &sets[4], (1, 1, 1)));
        dispatch_all(ctx, "sky.frame", &jobs);
    }
}
