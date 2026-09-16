//! Local height fog in a froxel volume: light injection with temporal
//! reprojection, then front-to-back integration.

use crate::frame::FrameCtx;
use crate::sky::{Job, SkyTargets, dispatch_all, linear_clamp};
use bytemuck::{Pod, Zeroable};
use mc2_gpu::{
    FrameGraph, HotCompute, Pass, PassBuilder, PassContext, SizePolicy, TexHandle, TextureDesc,
    bind, bind_group, layout,
};

const RGBA16F: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const FROXEL_BLOCK: u32 = 8;
pub const FROXEL_SLICES: u32 = 64;

/// Fog weather. Step 12's weather system drives these.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FogSettings {
    /// Extinction per metre at and below `height_m`.
    pub density: f32,
    pub height_m: f32,
    pub falloff_m: f32,
    pub anisotropy: f32,
    pub albedo: f32,
}

impl Default for FogSettings {
    fn default() -> Self {
        // Light haze: about 3 km visibility at the fog height, halving every
        // 60 m above it.
        Self {
            density: 0.0012,
            height_m: 120.0,
            falloff_m: 60.0,
            anisotropy: 0.6,
            albedo: 0.9,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct FogUniforms {
    pub density: f32,
    pub height_m: f32,
    pub falloff_m: f32,
    pub anisotropy: f32,
    pub albedo: f32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub history_valid: u32,
    pub _pad: [f32; 3],
}

#[derive(Clone, Copy)]
pub struct FogTargets {
    injected: TexHandle,
    injected_prev: TexHandle,
    /// Integrated scattering (rgb) and transmittance (a) per froxel.
    pub integrated: TexHandle,
}

impl FogTargets {
    pub fn create(graph: &mut FrameGraph<FrameCtx>) -> Self {
        let volume = |label| TextureDesc {
            size: SizePolicy::RenderVolume {
                block: FROXEL_BLOCK,
                depth: FROXEL_SLICES,
            },
            dimension: wgpu::TextureDimension::D3,
            ..TextureDesc::render_target(label, RGBA16F, 1.0)
        };
        let mut prev = volume("fog injected prev");
        prev.usage |= wgpu::TextureUsages::COPY_DST;
        Self {
            injected: graph.create_texture(volume("fog injected")),
            injected_prev: graph.create_history(prev),
            integrated: graph.create_texture(volume("fog integrated")),
        }
    }

    pub fn history_copies(&self) -> [(TexHandle, TexHandle); 1] {
        [(self.injected, self.injected_prev)]
    }
}

pub struct FogPass {
    t: FogTargets,
    sky: SkyTargets,
    cloud_shadow: TexHandle,
    cloud_uniforms: wgpu::Buffer,
    sky_map: wgpu::TextureView,
    inject: HotCompute,
    integrate: HotCompute,
    layouts: [wgpu::BindGroupLayout; 2],
    sampler: wgpu::Sampler,
    pub uniforms: wgpu::Buffer,
    groups: Option<(u64, [wgpu::BindGroup; 2])>,
    last_generation: Option<u64>,
}

impl FogPass {
    pub fn new(
        device: &wgpu::Device,
        ctx: &FrameCtx,
        t: FogTargets,
        sky: SkyTargets,
        (cloud_shadow, cloud_uniforms): (TexHandle, wgpu::Buffer),
        sky_map: wgpu::TextureView,
    ) -> Self {
        let stages = wgpu::ShaderStages::COMPUTE;
        let write3d = bind::storage_texture(
            RGBA16F,
            wgpu::TextureViewDimension::D3,
            wgpu::StorageTextureAccess::WriteOnly,
        );
        let unf = bind::texture(wgpu::TextureViewDimension::D2, false);
        let inject_layout = layout(
            device,
            "fog inject",
            stages,
            &[
                bind::texture(wgpu::TextureViewDimension::D3, true),
                write3d,
                bind::sampler(true),
                bind::texture_2d(),
                unf,
                unf,
                unf,
                bind::uniform(),
                bind::uniform(),
            ],
        );
        let integrate_layout = layout(
            device,
            "fog integrate",
            stages,
            &[
                bind::texture(wgpu::TextureViewDimension::D3, false),
                write3d,
                bind::uniform(),
            ],
        );
        let f = &ctx.frame_layout;
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fog uniforms"),
            size: std::mem::size_of::<FogUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            t,
            sky,
            cloud_shadow,
            cloud_uniforms,
            sky_map,
            inject: HotCompute::new(
                device,
                &ctx.shaders,
                "fog_inject.wgsl",
                "main",
                &[f, &inject_layout],
            ),
            integrate: HotCompute::new(
                device,
                &ctx.shaders,
                "fog_integrate.wgsl",
                "main",
                &[f, &integrate_layout],
            ),
            layouts: [inject_layout, integrate_layout],
            sampler: linear_clamp(device),
            uniforms,
            groups: None,
            last_generation: None,
        }
    }
}

impl Pass<FrameCtx> for FogPass {
    fn name(&self) -> &'static str {
        "vol.fog"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.t.injected_prev);
        b.read(self.sky.transmittance);
        b.read(self.sky.ambient);
        b.read(self.cloud_shadow);
        b.write(self.t.injected);
        b.write(self.t.integrated);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        let history_valid = self.last_generation == Some(generation);
        self.last_generation = Some(generation);
        let e = ctx.graph.extent(self.t.injected);
        let s = ctx.frame.fog;
        let u = FogUniforms {
            density: s.density,
            height_m: s.height_m,
            falloff_m: s.falloff_m,
            anisotropy: s.anisotropy,
            albedo: s.albedo,
            width: e.width,
            height: e.height,
            depth: e.depth_or_array_layers,
            history_valid: u32::from(history_valid),
            _pad: [0.0; 3],
        };
        ctx.queue
            .write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        if self.groups.as_ref().is_none_or(|(g, _)| *g != generation) {
            let g = ctx.graph;
            let view = |h| wgpu::BindingResource::TextureView(g.view(h));
            let inject = bind_group(
                ctx.device,
                "fog inject",
                &self.layouts[0],
                &[
                    view(self.t.injected_prev),
                    view(self.t.injected),
                    wgpu::BindingResource::Sampler(&self.sampler),
                    view(self.sky.transmittance),
                    view(self.sky.ambient),
                    wgpu::BindingResource::TextureView(&self.sky_map),
                    view(self.cloud_shadow),
                    self.cloud_uniforms.as_entire_binding(),
                    self.uniforms.as_entire_binding(),
                ],
            );
            let integrate = bind_group(
                ctx.device,
                "fog integrate",
                &self.layouts[1],
                &[
                    view(self.t.injected),
                    view(self.t.integrated),
                    self.uniforms.as_entire_binding(),
                ],
            );
            self.groups = Some((generation, [inject, integrate]));
        }
        let groups = self.groups.as_ref().expect("groups").1.clone();
        let fg = ctx.frame.frame_bind_group.clone();
        let pi = self.inject.get(ctx.device, &ctx.frame.shaders).clone();
        let pg = self.integrate.get(ctx.device, &ctx.frame.shaders).clone();
        let inject_set = [&fg, &groups[0]];
        let integrate_set = [&fg, &groups[1]];
        let jobs: [Job<'_>; 2] = [
            (
                &pi,
                &inject_set,
                (
                    e.width.div_ceil(4),
                    e.height.div_ceil(4),
                    e.depth_or_array_layers.div_ceil(4),
                ),
            ),
            (
                &pg,
                &integrate_set,
                (e.width.div_ceil(8), e.height.div_ceil(8), 1),
            ),
        ];
        dispatch_all(ctx, "vol.fog", &jobs);
    }
}
