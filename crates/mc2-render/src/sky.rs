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
