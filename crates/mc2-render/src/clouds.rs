//! Volumetric clouds: noise volumes, the amortised ray march, its temporal
//! resolve and the cloud shadow map.

use crate::frame::FrameCtx;
use crate::sky::{Job, SkyTargets, dispatch_all, linear_clamp};
use bytemuck::{Pod, Zeroable};
use mc2_gpu::{
    FrameGraph, HotCompute, Pass, PassBuilder, PassContext, SizePolicy, TexHandle, TextureDesc,
    bind, bind_group, layout,
};

const RGBA16F: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const NOISE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const SHADOW_MAP_SIZE: u32 = 256;

/// Weather as the cloud layer sees it. The weather system (step 12) drives
/// these; until then they hold a fair-weather default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudSettings {
    pub coverage: f32,
    /// 0 stratus to 1 cumulus.
    pub cloud_type: f32,
    pub precipitation: f32,
    /// Wind velocity at cloud height, metres per second (x, z).
    pub wind: [f32; 2],
    pub base_m: f32,
    pub top_m: f32,
}

impl Default for CloudSettings {
    fn default() -> Self {
        Self {
            coverage: 0.38,
            cloud_type: 0.8,
            precipitation: 0.0,
            wind: [9.0, 4.0],
            base_m: 1500.0,
            top_m: 4000.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct CloudUniforms {
    pub coverage: f32,
    pub cloud_type: f32,
    pub wind: [f32; 2],
    pub base_m: f32,
    pub top_m: f32,
    pub extinction: f32,
    pub steps: u32,
    pub light_steps: u32,
    pub precipitation: f32,
    pub shadow_texel_m: f32,
    pub history_valid: u32,
    pub camera_world: [f32; 3],
    pub _pad0: f32,
}

#[derive(Clone, Copy)]
pub struct CloudTargets {
    trace: TexHandle,
    pub clouds: TexHandle,
    clouds_prev: TexHandle,
    /// Transmittance toward the sun per world cell of the cloud base plane.
    pub shadow: TexHandle,
}

impl CloudTargets {
    pub fn create(graph: &mut FrameGraph<FrameCtx>) -> Self {
        let blocks = |label, block| TextureDesc {
            size: SizePolicy::RenderBlocks { block, extra: 0 },
            ..TextureDesc::render_target(label, RGBA16F, 1.0)
        };
        let mut prev = blocks("clouds prev", 2);
        prev.usage |= wgpu::TextureUsages::COPY_DST;
        Self {
            trace: graph.create_texture(blocks("clouds trace", 4)),
            clouds: graph.create_texture(blocks("clouds", 2)),
            clouds_prev: graph.create_history(prev),
            // Persistent: each frame refreshes a quarter of it.
            shadow: graph.create_history(TextureDesc {
                size: SizePolicy::Fixed(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1),
                ..TextureDesc::render_target("cloud shadow", RGBA16F, 1.0)
            }),
        }
    }

    pub fn history_copies(&self) -> [(TexHandle, TexHandle); 1] {
        [(self.clouds, self.clouds_prev)]
    }
}

pub struct CloudsPass {
    t: CloudTargets,
    sky: SkyTargets,
    shape: wgpu::TextureView,
    detail: wgpu::TextureView,
    /// Shape job, detail job.
    noise_groups: [wgpu::BindGroup; 2],
    noise_built: Option<u64>,
    noise: HotCompute,
    trace: HotCompute,
    resolve: HotCompute,
    shadow: HotCompute,
    layouts: [wgpu::BindGroupLayout; 3],
    repeat: wgpu::Sampler,
    clamp: wgpu::Sampler,
    pub uniforms: wgpu::Buffer,
    groups: Option<(u64, [wgpu::BindGroup; 3])>,
    last_generation: Option<u64>,
}

fn noise_volume(device: &wgpu::Device, label: &str, size: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: NOISE,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    })
}

fn uniform_words(device: &wgpu::Device, words: [u32; 4]) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloud noise job"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    buf.get_mapped_range_mut(..)
        .expect("mapped at creation")
        .copy_from_slice(bytemuck::cast_slice(&words));
    buf.unmap();
    buf
}

impl CloudsPass {
    pub fn new(device: &wgpu::Device, ctx: &FrameCtx, t: CloudTargets, sky: SkyTargets) -> Self {
        let shape_tex = noise_volume(device, "cloud shape noise", 128);
        let detail_tex = noise_volume(device, "cloud detail noise", 32);
        let shape = shape_tex.create_view(&Default::default());
        let detail = detail_tex.create_view(&Default::default());
        let write3d = bind::storage_texture(
            NOISE,
            wgpu::TextureViewDimension::D3,
            wgpu::StorageTextureAccess::WriteOnly,
        );
        let stages = wgpu::ShaderStages::COMPUTE;
        let noise_layout = layout(
            device,
            "cloud noise",
            stages,
            &[write3d, write3d, bind::uniform()],
        );
        let noise_groups = [0u32, 1].map(|volume| {
            let job = uniform_words(device, [volume, 0, 0, 0]);
            bind_group(
                device,
                "cloud noise",
                &noise_layout,
                &[
                    wgpu::BindingResource::TextureView(&shape),
                    wgpu::BindingResource::TextureView(&detail),
                    job.as_entire_binding(),
                ],
            )
        });
        let vol = bind::texture(wgpu::TextureViewDimension::D3, true);
        let trace_layout = layout(
            device,
            "clouds trace",
            stages,
            &[
                vol,
                vol,
                bind::sampler(true),
                bind::texture_2d(),
                bind::sampler(true),
                bind::texture(wgpu::TextureViewDimension::D2, false),
                bind::write_2d(RGBA16F),
                bind::uniform(),
            ],
        );
        let resolve_layout = layout(
            device,
            "clouds resolve",
            stages,
            &[
                bind::texture_2d(),
                bind::texture_2d(),
                bind::sampler(true),
                bind::write_2d(RGBA16F),
                bind::uniform(),
            ],
        );
        let shadow_layout = layout(
            device,
            "clouds shadow",
            stages,
            &[
                vol,
                vol,
                bind::sampler(true),
                bind::write_2d(RGBA16F),
                bind::uniform(),
            ],
        );
        let f = &ctx.frame_layout;
        let hc = |file, l: &wgpu::BindGroupLayout| {
            HotCompute::new(device, &ctx.shaders, file, "main", &[f, l])
        };
        let noise = HotCompute::new(
            device,
            &ctx.shaders,
            "cloud_noise.wgsl",
            "main",
            &[&noise_layout],
        );
        let repeat = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cloud noise repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud uniforms"),
            size: std::mem::size_of::<CloudUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            t,
            sky,
            shape,
            detail,
            noise_groups,
            noise_built: None,
            noise,
            trace: hc("clouds_trace.wgsl", &trace_layout),
            resolve: hc("clouds_resolve.wgsl", &resolve_layout),
            shadow: hc("clouds_shadow_update.wgsl", &shadow_layout),
            layouts: [trace_layout, resolve_layout, shadow_layout],
            repeat,
            clamp: linear_clamp(device),
            uniforms,
            groups: None,
            last_generation: None,
        }
    }

    fn write_uniforms(&self, ctx: &PassContext<'_, FrameCtx>, history_valid: bool) {
        let s = ctx.frame.clouds;
        let q = ctx.frame.cloud_quality;
        let time = f64::from(ctx.frame.time);
        let u = CloudUniforms {
            coverage: s.coverage,
            cloud_type: s.cloud_type,
            wind: [
                (f64::from(s.wind[0]) * time) as f32,
                (f64::from(s.wind[1]) * time) as f32,
            ],
            base_m: s.base_m,
            top_m: s.top_m,
            extinction: 0.012,
            steps: q.0,
            light_steps: q.1,
            precipitation: s.precipitation,
            shadow_texel_m: 50.0,
            history_valid: u32::from(history_valid),
            camera_world: ctx.frame.camera_world,
            _pad0: 0.0,
        };
        ctx.queue
            .write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
    }
}

impl Pass<FrameCtx> for CloudsPass {
    fn name(&self) -> &'static str {
        "vol.clouds"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.sky.transmittance);
        b.read(self.sky.ambient);
        b.read(self.t.clouds_prev);
        b.write(self.t.trace);
        b.write(self.t.clouds);
        b.write(self.t.shadow);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        let history_valid = self.last_generation == Some(generation);
        self.last_generation = Some(generation);
        self.write_uniforms(ctx, history_valid);
        if self.groups.as_ref().is_none_or(|(g, _)| *g != generation) {
            let g = ctx.graph;
            let view = |h| wgpu::BindingResource::TextureView(g.view(h));
            let trace = bind_group(
                ctx.device,
                "clouds trace",
                &self.layouts[0],
                &[
                    wgpu::BindingResource::TextureView(&self.shape),
                    wgpu::BindingResource::TextureView(&self.detail),
                    wgpu::BindingResource::Sampler(&self.repeat),
                    view(self.sky.transmittance),
                    wgpu::BindingResource::Sampler(&self.clamp),
                    view(self.sky.ambient),
                    view(self.t.trace),
                    self.uniforms.as_entire_binding(),
                ],
            );
            let resolve = bind_group(
                ctx.device,
                "clouds resolve",
                &self.layouts[1],
                &[
                    view(self.t.trace),
                    view(self.t.clouds_prev),
                    wgpu::BindingResource::Sampler(&self.clamp),
                    view(self.t.clouds),
                    self.uniforms.as_entire_binding(),
                ],
            );
            let shadow = bind_group(
                ctx.device,
                "clouds shadow",
                &self.layouts[2],
                &[
                    wgpu::BindingResource::TextureView(&self.shape),
                    wgpu::BindingResource::TextureView(&self.detail),
                    wgpu::BindingResource::Sampler(&self.repeat),
                    view(self.t.shadow),
                    self.uniforms.as_entire_binding(),
                ],
            );
            self.groups = Some((generation, [trace, resolve, shadow]));
        }
        let groups = self.groups.as_ref().expect("groups").1.clone();
        let fg = ctx.frame.frame_bind_group.clone();
        let lib_generation = ctx.frame.shaders.generation();
        let build_noise = self.noise_built != Some(lib_generation);
        self.noise_built = Some(lib_generation);
        let pn = self.noise.get(ctx.device, &ctx.frame.shaders).clone();
        let pt = self.trace.get(ctx.device, &ctx.frame.shaders).clone();
        let pr = self.resolve.get(ctx.device, &ctx.frame.shaders).clone();
        let ps = self.shadow.get(ctx.device, &ctx.frame.shaders).clone();
        let noise_shape = [&self.noise_groups[0]];
        let noise_detail = [&self.noise_groups[1]];
        let trace_set = [&fg, &groups[0]];
        let resolve_set = [&fg, &groups[1]];
        let shadow_set = [&fg, &groups[2]];
        let trace_extent = ctx.graph.extent(self.t.trace);
        let clouds_extent = ctx.graph.extent(self.t.clouds);
        let mut jobs: Vec<Job<'_>> = Vec::new();
        if build_noise {
            jobs.push((&pn, &noise_shape, (32, 32, 32)));
            jobs.push((&pn, &noise_detail, (8, 8, 8)));
        }
        jobs.push((&ps, &shadow_set, (16, 16, 1)));
        jobs.push((
            &pt,
            &trace_set,
            (
                trace_extent.width.div_ceil(8),
                trace_extent.height.div_ceil(8),
                1,
            ),
        ));
        jobs.push((
            &pr,
            &resolve_set,
            (
                clouds_extent.width.div_ceil(8),
                clouds_extent.height.div_ceil(8),
                1,
            ),
        ));
        dispatch_all(ctx, "vol.clouds", &jobs);
    }
}
