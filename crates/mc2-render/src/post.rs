//! Display-resolution post processing: bloom, camera motion blur and depth
//! of field. Exposure, night vision, tonemapping, sharpening, vignette and
//! grain happen in the present pass.

use crate::frame::FrameCtx;
use crate::sky::{Job, dispatch_all, linear_clamp};
use crate::vis::VisTargets;
use bytemuck::{Pod, Zeroable};
use mc2_gpu::{
    FrameGraph, HotCompute, Pass, PassBuilder, PassContext, SizePolicy, TexHandle, TextureDesc,
    bind, bind_group, layout,
};

const RGBA16F: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const BLOOM_LEVELS: usize = 6;

/// Thin-lens depth of field, used by photo mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthOfField {
    pub focus_m: f32,
    pub focal_mm: f32,
    pub f_number: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostSettings {
    pub bloom_intensity: f32,
    /// Fraction of a frame the virtual shutter is open; 0 disables blur.
    pub shutter: f32,
    pub sharpen: f32,
    pub vignette: f32,
    pub grain: f32,
    pub purkinje: bool,
    pub dof: Option<DepthOfField>,
}

impl Default for PostSettings {
    fn default() -> Self {
        Self {
            bloom_intensity: 0.04,
            shutter: 0.5,
            sharpen: 0.3,
            vignette: 0.25,
            grain: 0.012,
            purkinje: true,
            dof: None,
        }
    }
}

impl PostSettings {
    /// Everything off: the display transform alone.
    pub fn neutral() -> Self {
        Self {
            bloom_intensity: 0.0,
            shutter: 0.0,
            sharpen: 0.0,
            vignette: 0.0,
            grain: 0.0,
            purkinje: false,
            dof: None,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostUniforms {
    bloom_intensity: f32,
    shutter: f32,
    focus_m: f32,
    aperture_mm: f32,
    focal_mm: f32,
    sensor_mm: f32,
    max_coc_px: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomUniforms {
    mode: u32,
    karis: u32,
    radius: f32,
    _pad: f32,
}

#[derive(Clone, Copy)]
pub struct PostTargets {
    down: [TexHandle; BLOOM_LEVELS],
    up: [TexHandle; BLOOM_LEVELS - 1],
    /// Post-processed HDR at display resolution.
    pub color: TexHandle,
}

impl PostTargets {
    pub fn create(graph: &mut FrameGraph<FrameCtx>) -> Self {
        let level = |label, i: usize| TextureDesc {
            size: SizePolicy::Output(0.5f32.powi(i as i32 + 1)),
            ..TextureDesc::render_target(label, RGBA16F, 1.0)
        };
        let down_labels = [
            "bloom down 0",
            "bloom down 1",
            "bloom down 2",
            "bloom down 3",
            "bloom down 4",
            "bloom down 5",
        ];
        let up_labels = [
            "bloom up 0",
            "bloom up 1",
            "bloom up 2",
            "bloom up 3",
            "bloom up 4",
        ];
        Self {
            down: std::array::from_fn(|i| graph.create_texture(level(down_labels[i], i))),
            up: std::array::from_fn(|i| graph.create_texture(level(up_labels[i], i))),
            color: graph.create_texture(TextureDesc::output_target("post color", RGBA16F)),
        }
    }
}

pub struct PostPass {
    input: TexHandle,
    vis: VisTargets,
    t: PostTargets,
    bloom: HotCompute,
    combine: HotCompute,
    bloom_layout: wgpu::BindGroupLayout,
    combine_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    down_params: [wgpu::Buffer; 2],
    up_params: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    /// One zero texture for downsample jobs, whose `base` is unused.
    unused: wgpu::TextureView,
    groups: Option<(u64, Vec<wgpu::BindGroup>)>,
}

fn params_buffer(device: &wgpu::Device, u: BloomUniforms) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bloom params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    buf.get_mapped_range_mut(..)
        .expect("mapped at creation")
        .copy_from_slice(bytemuck::bytes_of(&u));
    buf.unmap();
    buf
}

impl PostPass {
    pub fn new(
        device: &wgpu::Device,
        ctx: &FrameCtx,
        input: TexHandle,
        vis: VisTargets,
        t: PostTargets,
    ) -> Self {
        let stages = wgpu::ShaderStages::COMPUTE;
        let unf = bind::texture(wgpu::TextureViewDimension::D2, false);
        let bloom_layout = layout(
            device,
            "bloom",
            stages,
            &[
                bind::texture_2d(),
                unf,
                bind::sampler(true),
                bind::write_2d(RGBA16F),
                bind::uniform(),
            ],
        );
        let combine_layout = layout(
            device,
            "post combine",
            stages,
            &[
                bind::texture_2d(),
                bind::texture_2d(),
                unf,
                unf,
                bind::sampler(true),
                bind::write_2d(RGBA16F),
                bind::uniform(),
            ],
        );
        let unused = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("bloom unused base"),
                size: wgpu::Extent3d::default(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: RGBA16F,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post uniforms"),
            size: std::mem::size_of::<PostUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bloom_params = |mode, karis| BloomUniforms {
            mode,
            karis,
            radius: 1.0,
            _pad: 0.0,
        };
        Self {
            input,
            vis,
            t,
            bloom: HotCompute::new(
                device,
                &ctx.shaders,
                "post_bloom.wgsl",
                "main",
                &[&bloom_layout],
            ),
            combine: HotCompute::new(
                device,
                &ctx.shaders,
                "post_combine.wgsl",
                "main",
                &[&ctx.frame_layout, &combine_layout],
            ),
            bloom_layout,
            combine_layout,
            sampler: linear_clamp(device),
            down_params: [
                params_buffer(device, bloom_params(0, 1)),
                params_buffer(device, bloom_params(0, 0)),
            ],
            up_params: params_buffer(device, bloom_params(1, 0)),
            uniforms,
            unused,
            groups: None,
        }
    }
}

impl Pass<FrameCtx> for PostPass {
    fn name(&self) -> &'static str {
        "post.combine"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.input);
        b.read(self.vis.depth);
        b.read(self.vis.motion);
        for h in self.t.down {
            b.write(h);
        }
        for h in self.t.up {
            b.write(h);
        }
        b.write(self.t.color);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let s = ctx.frame.post;
        let dof = s.dof.unwrap_or(DepthOfField {
            focus_m: 10.0,
            focal_mm: 35.0,
            f_number: 0.0,
        });
        let u = PostUniforms {
            bloom_intensity: s.bloom_intensity,
            shutter: s.shutter,
            focus_m: dof.focus_m,
            aperture_mm: if s.dof.is_some() && dof.f_number > 0.0 {
                dof.focal_mm / dof.f_number
            } else {
                0.0
            },
            focal_mm: dof.focal_mm,
            sensor_mm: 24.0,
            max_coc_px: 24.0,
            _pad: 0.0,
        };
        ctx.queue
            .write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        let generation = ctx.graph.generation();
        if self.groups.as_ref().is_none_or(|(g, _)| *g != generation) {
            let g = ctx.graph;
            let view = |h| wgpu::BindingResource::TextureView(g.view(h));
            let t = self.t;
            let mut groups = Vec::new();
            for i in 0..BLOOM_LEVELS {
                let src = if i == 0 { self.input } else { t.down[i - 1] };
                groups.push(bind_group(
                    ctx.device,
                    "bloom down",
                    &self.bloom_layout,
                    &[
                        view(src),
                        wgpu::BindingResource::TextureView(&self.unused),
                        wgpu::BindingResource::Sampler(&self.sampler),
                        view(t.down[i]),
                        self.down_params[usize::from(i != 0)].as_entire_binding(),
                    ],
                ));
            }
            // up[i] = down[i] + tent(coarser), coarsest first.
            for i in (0..BLOOM_LEVELS - 1).rev() {
                let coarser = if i + 1 == BLOOM_LEVELS - 1 {
                    t.down[i + 1]
                } else {
                    t.up[i + 1]
                };
                groups.push(bind_group(
                    ctx.device,
                    "bloom up",
                    &self.bloom_layout,
                    &[
                        view(coarser),
                        view(t.down[i]),
                        wgpu::BindingResource::Sampler(&self.sampler),
                        view(t.up[i]),
                        self.up_params.as_entire_binding(),
                    ],
                ));
            }
            groups.push(bind_group(
                ctx.device,
                "post combine",
                &self.combine_layout,
                &[
                    view(self.input),
                    view(t.up[0]),
                    view(self.vis.depth),
                    view(self.vis.motion),
                    wgpu::BindingResource::Sampler(&self.sampler),
                    view(t.color),
                    self.uniforms.as_entire_binding(),
                ],
            ));
            self.groups = Some((generation, groups));
        }
        let groups = self.groups.as_ref().expect("groups").1.clone();
        let fg = ctx.frame.frame_bind_group.clone();
        let pb = self.bloom.get(ctx.device, &ctx.frame.shaders).clone();
        let pc = self.combine.get(ctx.device, &ctx.frame.shaders).clone();
        let work = |h: TexHandle| {
            let e = ctx.graph.extent(h);
            (e.width.div_ceil(8), e.height.div_ceil(8), 1)
        };
        let t = self.t;
        let mut sizes = Vec::new();
        for h in t.down {
            sizes.push(work(h));
        }
        for i in (0..BLOOM_LEVELS - 1).rev() {
            sizes.push(work(t.up[i]));
        }
        let color_work = work(t.color);
        let sets: Vec<[&wgpu::BindGroup; 1]> =
            groups[..groups.len() - 1].iter().map(|g| [g]).collect();
        let combine_set = [&fg, &groups[groups.len() - 1]];
        let mut jobs: Vec<Job<'_>> = sets
            .iter()
            .zip(&sizes)
            .map(|(set, w)| (&pb, &set[..], *w))
            .collect();
        jobs.push((&pc, &combine_set, color_work));
        dispatch_all(ctx, "post.combine", &jobs);
    }
}
