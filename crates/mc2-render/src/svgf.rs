//! Spatiotemporal variance-guided filtering (Schied et al. 2017) for one
//! demodulated lighting signal at render or half resolution.

use crate::frame::FrameCtx;
use crate::sky::{Job, dispatch_all};
use crate::vis::VisTargets;
use mc2_gpu::{
    FrameGraph, HotCompute, Pass, PassBuilder, PassContext, SizePolicy, TexHandle, TextureDesc,
    bind, bind_group, layout,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Five levels give the paper's 65x65 pixel footprint.
pub const ITERATIONS: u32 = 5;

#[derive(Clone, Copy)]
pub struct SvgfTargets {
    /// Temporally integrated colour (rgb) and variance (a).
    pub integrated: TexHandle,
    pub moments: TexHandle,
    pub moments_prev: TexHandle,
    /// Output of the first a-trous level: next frame's colour history.
    pub color_prev: TexHandle,
    ping: TexHandle,
    pong: TexHandle,
    /// Final filtered colour.
    pub output: TexHandle,
}

impl SvgfTargets {
    /// `shift` 0 filters at render resolution, 1 at half resolution.
    /// `names`: integrated, moments, moments prev, colour prev, ping, pong.
    pub fn create(graph: &mut FrameGraph<FrameCtx>, names: [&'static str; 6], shift: u32) -> Self {
        let desc = |label: &'static str| {
            let mut d = TextureDesc {
                size: SizePolicy::RenderBlocks {
                    block: 1 << shift,
                    extra: 0,
                },
                ..TextureDesc::render_target(label, FORMAT, 1.0)
            };
            d.usage |= wgpu::TextureUsages::COPY_DST;
            d
        };
        let ping = graph.create_texture(desc(names[4]));
        let pong = graph.create_texture(desc(names[5]));
        Self {
            integrated: graph.create_texture(desc(names[0])),
            moments: graph.create_texture(desc(names[1])),
            moments_prev: graph.create_history(desc(names[2])),
            color_prev: graph.create_history(desc(names[3])),
            ping,
            pong,
            // Levels 1..5 alternate ping, pong, ping, pong: pong is last.
            output: pong,
        }
    }
}

pub struct SvgfPass {
    name: &'static str,
    noisy: TexHandle,
    vis: VisTargets,
    prev_id: TexHandle,
    t: SvgfTargets,
    temporal: HotCompute,
    atrous: HotCompute,
    temporal_layout: wgpu::BindGroupLayout,
    atrous_layout: wgpu::BindGroupLayout,
    /// Temporal params, then one per a-trous level.
    params: Vec<wgpu::Buffer>,
    groups: Option<(u64, Vec<wgpu::BindGroup>)>,
}

impl SvgfPass {
    #[expect(
        clippy::too_many_arguments,
        reason = "a pass wires a signal to its visibility buffer and history"
    )]
    pub fn new(
        device: &wgpu::Device,
        ctx: &FrameCtx,
        name: &'static str,
        noisy: TexHandle,
        vis: VisTargets,
        prev_id: TexHandle,
        t: SvgfTargets,
        shift: u32,
    ) -> Self {
        let unf = || bind::texture(wgpu::TextureViewDimension::D2, false);
        let temporal_layout = layout(
            device,
            "svgf temporal",
            wgpu::ShaderStages::COMPUTE,
            &[
                bind::utexture_2d(),
                unf(),
                unf(),
                bind::utexture_2d(),
                unf(),
                unf(),
                unf(),
                bind::write_2d(FORMAT),
                bind::write_2d(FORMAT),
                bind::uniform(),
            ],
        );
        let atrous_layout = layout(
            device,
            "svgf atrous",
            wgpu::ShaderStages::COMPUTE,
            &[
                bind::utexture_2d(),
                unf(),
                unf(),
                unf(),
                bind::write_2d(FORMAT),
                bind::uniform(),
            ],
        );
        let params = (0..=ITERATIONS)
            .map(|i| {
                let step = if i == 0 { 0 } else { 1u32 << (i - 1) };
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("svgf params"),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: true,
                });
                buf.get_mapped_range_mut(..)
                    .expect("mapped at creation")
                    .copy_from_slice(bytemuck::cast_slice(&[shift, step, 0, 0]));
                buf.unmap();
                buf
            })
            .collect();
        Self {
            name,
            noisy,
            vis,
            prev_id,
            t,
            temporal: HotCompute::new(
                device,
                &ctx.shaders,
                "svgf_temporal.wgsl",
                "main",
                &[&ctx.frame_layout, &temporal_layout],
            ),
            atrous: HotCompute::new(
                device,
                &ctx.shaders,
                "svgf_atrous.wgsl",
                "main",
                &[&ctx.frame_layout, &atrous_layout],
            ),
            temporal_layout,
            atrous_layout,
            params,
            groups: None,
        }
    }

    /// Pairs for the history pass: this frame's moments become next frame's.
    pub fn history_copies(t: &SvgfTargets) -> [(TexHandle, TexHandle); 1] {
        [(t.moments, t.moments_prev)]
    }
}

impl Pass<FrameCtx> for SvgfPass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.noisy);
        b.read(self.vis.id);
        b.read(self.vis.depth);
        b.read(self.vis.motion);
        b.read(self.prev_id);
        b.read(self.t.moments_prev);
        b.write(self.t.integrated);
        b.write(self.t.moments);
        b.write(self.t.color_prev);
        b.write(self.t.ping);
        b.write(self.t.pong);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let generation = ctx.graph.generation();
        if self.groups.as_ref().is_none_or(|(g, _)| *g != generation) {
            let g = ctx.graph;
            let view = |h| wgpu::BindingResource::TextureView(g.view(h));
            let t = self.t;
            let v = self.vis;
            let mut groups = vec![bind_group(
                ctx.device,
                "svgf temporal",
                &self.temporal_layout,
                &[
                    view(v.id),
                    view(v.depth),
                    view(v.motion),
                    view(self.prev_id),
                    view(self.noisy),
                    view(t.color_prev),
                    view(t.moments_prev),
                    view(t.integrated),
                    view(t.moments),
                    self.params[0].as_entire_binding(),
                ],
            )];
            // Level 0 writes the colour history; later levels ping-pong.
            let chain = [
                (t.integrated, t.color_prev),
                (t.color_prev, t.ping),
                (t.ping, t.pong),
                (t.pong, t.ping),
                (t.ping, t.pong),
            ];
            for (level, (src, dst)) in chain.into_iter().enumerate() {
                groups.push(bind_group(
                    ctx.device,
                    "svgf atrous",
                    &self.atrous_layout,
                    &[
                        view(v.id),
                        view(v.depth),
                        view(v.motion),
                        view(src),
                        view(dst),
                        self.params[level + 1].as_entire_binding(),
                    ],
                ));
            }
            self.groups = Some((generation, groups));
        }
        let groups = self.groups.as_ref().expect("groups").1.clone();
        let fg = ctx.frame.frame_bind_group.clone();
        let pt = self.temporal.get(ctx.device, &ctx.frame.shaders).clone();
        let pa = self.atrous.get(ctx.device, &ctx.frame.shaders).clone();
        let e = ctx.graph.extent(self.t.integrated);
        let work = (e.width.div_ceil(8), e.height.div_ceil(8), 1);
        let sets: Vec<[&wgpu::BindGroup; 2]> = groups.iter().map(|g| [&fg, g]).collect();
        let mut jobs: Vec<Job<'_>> = vec![(&pt, &sets[0], work)];
        for set in &sets[1..] {
            jobs.push((&pa, set, work));
        }
        dispatch_all(ctx, self.name, &jobs);
    }
}
