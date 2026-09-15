//! Frame graph: declared passes over declared textures.
//!
//! Each pass states which graph textures it reads and writes during `setup`.
//! `compile` checks every read has an earlier writer (or is an import or a
//! history texture), culls passes whose outputs nobody consumes, and
//! (re)allocates textures whose size policy changed. `execute` runs the
//! surviving passes in order, each wrapped in a CPU scope and a GPU
//! timestamp scope. Texture reallocation bumps `generation` so passes can
//! rebuild bind groups lazily.

use crate::timing::{GpuProfiler, TimestampScope};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SizePolicy {
    /// Fraction of the internal render resolution.
    Render(f32),
    /// Fraction of the display resolution.
    Output(f32),
    Fixed(u32, u32, u32),
}

#[derive(Clone, Debug)]
pub struct TextureDesc {
    pub label: &'static str,
    pub format: wgpu::TextureFormat,
    pub size: SizePolicy,
    pub usage: wgpu::TextureUsages,
    pub dimension: wgpu::TextureDimension,
    pub mips: u32,
}

impl TextureDesc {
    /// Storage-writable, sampleable 2D target at a fraction of render size.
    pub fn render_target(label: &'static str, format: wgpu::TextureFormat, scale: f32) -> Self {
        Self {
            label,
            format,
            size: SizePolicy::Render(scale),
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            dimension: wgpu::TextureDimension::D2,
            mips: 1,
        }
    }

    pub fn output_target(label: &'static str, format: wgpu::TextureFormat) -> Self {
        Self {
            size: SizePolicy::Output(1.0),
            ..Self::render_target(label, format, 1.0)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TexHandle(pub u32);

struct GraphTexture {
    desc: TextureDesc,
    texture: Option<wgpu::Texture>,
    view: Option<wgpu::TextureView>,
    extent: wgpu::Extent3d,
    kind: TexKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TexKind {
    Transient,
    /// Valid to read before the first write of a frame: it holds last
    /// frame's contents (temporal accumulation).
    History,
    /// Owned outside the graph; supplied with `import`.
    Imported,
}

/// Registration-time view a pass uses to declare its resources.
pub struct PassBuilder<'a> {
    reads: &'a mut Vec<TexHandle>,
    writes: &'a mut Vec<TexHandle>,
    output: &'a mut bool,
}

impl PassBuilder<'_> {
    pub fn read(&mut self, t: TexHandle) {
        self.reads.push(t);
    }

    pub fn write(&mut self, t: TexHandle) {
        self.writes.push(t);
    }

    /// Marks the pass as a root that is never culled (present, readback).
    pub fn output(&mut self) {
        *self.output = true;
    }
}

pub struct PassContext<'a, C> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub graph: &'a GraphResources,
    pub timestamps: TimestampScope,
    pub frame: &'a mut C,
}

pub trait Pass<C> {
    fn name(&self) -> &'static str;
    fn setup(&mut self, builder: &mut PassBuilder<'_>);
    fn execute(&mut self, ctx: &mut PassContext<'_, C>);
}

struct PassEntry<C> {
    pass: Box<dyn Pass<C>>,
    reads: Vec<TexHandle>,
    writes: Vec<TexHandle>,
    output: bool,
    live: bool,
}

/// Texture storage handed to executing passes.
pub struct GraphResources {
    textures: Vec<GraphTexture>,
    render_size: (u32, u32),
    output_size: (u32, u32),
    generation: u64,
}

impl GraphResources {
    pub fn view(&self, t: TexHandle) -> &wgpu::TextureView {
        self.textures[t.0 as usize]
            .view
            .as_ref()
            .unwrap_or_else(|| panic!("graph texture `{}` not allocated", self.label(t)))
    }

    pub fn texture(&self, t: TexHandle) -> &wgpu::Texture {
        self.textures[t.0 as usize]
            .texture
            .as_ref()
            .unwrap_or_else(|| panic!("graph texture `{}` not allocated", self.label(t)))
    }

    pub fn extent(&self, t: TexHandle) -> wgpu::Extent3d {
        self.textures[t.0 as usize].extent
    }

    /// Looks a texture up by label, for tests and debug readback.
    pub fn find(&self, label: &str) -> Option<&wgpu::Texture> {
        self.textures
            .iter()
            .find(|t| t.desc.label == label)
            .and_then(|t| t.texture.as_ref())
    }

    pub fn label(&self, t: TexHandle) -> &'static str {
        self.textures[t.0 as usize].desc.label
    }

    pub fn render_size(&self) -> (u32, u32) {
        self.render_size
    }

    pub fn output_size(&self) -> (u32, u32) {
        self.output_size
    }

    /// Increments whenever any texture is reallocated.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn extent_for(&self, size: SizePolicy) -> wgpu::Extent3d {
        let scaled = |(w, h): (u32, u32), s: f32| wgpu::Extent3d {
            width: ((w as f32 * s).round() as u32).max(1),
            height: ((h as f32 * s).round() as u32).max(1),
            depth_or_array_layers: 1,
        };
        match size {
            SizePolicy::Render(s) => scaled(self.render_size, s),
            SizePolicy::Output(s) => scaled(self.output_size, s),
            SizePolicy::Fixed(w, h, d) => wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: d,
            },
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum GraphError {
    ReadBeforeWrite {
        pass: &'static str,
        texture: &'static str,
    },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::ReadBeforeWrite { pass, texture } => {
                write!(
                    f,
                    "pass `{pass}` reads `{texture}` before any pass writes it"
                )
            }
        }
    }
}

impl std::error::Error for GraphError {}

pub struct FrameGraph<C> {
    res: GraphResources,
    passes: Vec<PassEntry<C>>,
    dirty: bool,
}

impl<C> FrameGraph<C> {
    pub fn new(render_size: (u32, u32), output_size: (u32, u32)) -> Self {
        Self {
            res: GraphResources {
                textures: Vec::new(),
                render_size,
                output_size,
                generation: 0,
            },
            passes: Vec::new(),
            dirty: true,
        }
    }

    fn add_texture(&mut self, desc: TextureDesc, kind: TexKind) -> TexHandle {
        self.res.textures.push(GraphTexture {
            desc,
            texture: None,
            view: None,
            extent: wgpu::Extent3d::default(),
            kind,
        });
        self.dirty = true;
        TexHandle(self.res.textures.len() as u32 - 1)
    }

    pub fn create_texture(&mut self, desc: TextureDesc) -> TexHandle {
        self.add_texture(desc, TexKind::Transient)
    }

    /// A texture whose previous-frame contents are meaningful to read.
    pub fn create_history(&mut self, desc: TextureDesc) -> TexHandle {
        self.add_texture(desc, TexKind::History)
    }

    /// Declares an externally owned texture, e.g. the swapchain image.
    pub fn declare_import(&mut self, desc: TextureDesc) -> TexHandle {
        self.add_texture(desc, TexKind::Imported)
    }

    pub fn import(&mut self, t: TexHandle, texture: wgpu::Texture) {
        let entry = &mut self.res.textures[t.0 as usize];
        debug_assert!(entry.kind == TexKind::Imported);
        entry.extent = texture.size();
        entry.view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        entry.texture = Some(texture);
    }

    /// Drops the graph's reference to an imported texture, e.g. before the
    /// swapchain image is presented and the surface possibly reconfigured.
    pub fn release_import(&mut self, t: TexHandle) {
        let entry = &mut self.res.textures[t.0 as usize];
        entry.view = None;
        entry.texture = None;
    }

    pub fn add_pass(&mut self, mut pass: Box<dyn Pass<C>>) {
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        let mut output = false;
        pass.setup(&mut PassBuilder {
            reads: &mut reads,
            writes: &mut writes,
            output: &mut output,
        });
        self.passes.push(PassEntry {
            pass,
            reads,
            writes,
            output,
            live: true,
        });
        self.dirty = true;
    }

    pub fn resize(&mut self, render_size: (u32, u32), output_size: (u32, u32)) {
        if self.res.render_size != render_size || self.res.output_size != output_size {
            self.res.render_size = render_size;
            self.res.output_size = output_size;
            self.dirty = true;
        }
    }

    pub fn resources(&self) -> &GraphResources {
        &self.res
    }

    /// Names of passes that survive culling, in execution order.
    pub fn live_passes(&self) -> Vec<&'static str> {
        self.passes
            .iter()
            .filter(|p| p.live)
            .map(|p| p.pass.name())
            .collect()
    }

    /// Validates, culls and allocates. Cheap when nothing changed.
    pub fn compile(&mut self, device: &wgpu::Device) -> Result<(), GraphError> {
        if !self.dirty {
            return Ok(());
        }
        let n = self.res.textures.len();
        let mut written = vec![false; n];
        for p in &self.passes {
            for r in &p.reads {
                let tex = &self.res.textures[r.0 as usize];
                if tex.kind == TexKind::Transient && !written[r.0 as usize] {
                    return Err(GraphError::ReadBeforeWrite {
                        pass: p.pass.name(),
                        texture: tex.desc.label,
                    });
                }
            }
            for w in &p.writes {
                written[w.0 as usize] = true;
            }
        }

        // Cull backwards from outputs. History textures are consumed by the
        // next frame, so their writers always stay live.
        let mut needed = vec![false; n];
        for p in self.passes.iter_mut().rev() {
            let textures = &self.res.textures;
            p.live = p.output
                || p.writes.iter().any(|w| {
                    needed[w.0 as usize] || textures[w.0 as usize].kind == TexKind::History
                });
            if p.live {
                for r in &p.reads {
                    needed[r.0 as usize] = true;
                }
            }
        }

        let mut used = vec![false; n];
        for p in self.passes.iter().filter(|p| p.live) {
            for t in p.reads.iter().chain(&p.writes) {
                used[t.0 as usize] = true;
            }
        }
        let mut reallocated = false;
        for (i, is_used) in used.iter().enumerate() {
            let entry = &self.res.textures[i];
            if !is_used || entry.kind == TexKind::Imported {
                continue;
            }
            let extent = self.res.extent_for(entry.desc.size);
            if entry.texture.is_some() && entry.extent == extent {
                continue;
            }
            let desc = entry.desc.clone();
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(desc.label),
                size: extent,
                mip_level_count: desc.mips,
                sample_count: 1,
                dimension: desc.dimension,
                format: desc.format,
                usage: desc.usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(desc.label),
                mip_level_count: Some(1),
                ..Default::default()
            });
            let entry = &mut self.res.textures[i];
            entry.texture = Some(texture);
            entry.view = Some(view);
            entry.extent = extent;
            reallocated = true;
        }
        if reallocated {
            self.res.generation += 1;
        }
        self.dirty = false;
        Ok(())
    }

    /// Runs live passes. `compile` must have succeeded since the last change.
    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut GpuProfiler,
        frame: &mut C,
    ) {
        debug_assert!(!self.dirty, "frame graph executed without compile");
        for p in self.passes.iter_mut().filter(|p| p.live) {
            let name = p.pass.name();
            mc2_core::scope!(name);
            let timestamps = profiler.scope(name);
            let mut ctx = PassContext {
                device,
                queue,
                encoder,
                graph: &self.res,
                timestamps,
                frame,
            };
            p.pass.execute(&mut ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Probe {
        name: &'static str,
        reads: Vec<TexHandle>,
        writes: Vec<TexHandle>,
        output: bool,
    }

    impl Pass<Vec<&'static str>> for Probe {
        fn name(&self) -> &'static str {
            self.name
        }

        fn setup(&mut self, b: &mut PassBuilder<'_>) {
            self.reads.iter().for_each(|&t| b.read(t));
            self.writes.iter().for_each(|&t| b.write(t));
            if self.output {
                b.output();
            }
        }

        fn execute(&mut self, ctx: &mut PassContext<'_, Vec<&'static str>>) {
            ctx.frame.push(self.name);
        }
    }

    fn probe(name: &'static str, r: &[TexHandle], w: &[TexHandle], output: bool) -> Box<Probe> {
        Box::new(Probe {
            name,
            reads: r.to_vec(),
            writes: w.to_vec(),
            output,
        })
    }

    fn desc(label: &'static str) -> TextureDesc {
        TextureDesc::render_target(label, wgpu::TextureFormat::Rgba8Unorm, 1.0)
    }

    #[test]
    fn culls_unconsumed_passes_and_runs_in_order() {
        let Some(gpu) = crate::device::test_gpu() else {
            return;
        };
        let mut g: FrameGraph<Vec<&'static str>> = FrameGraph::new((64, 32), (128, 64));
        let a = g.create_texture(desc("a"));
        let unused = g.create_texture(desc("unused"));
        let hist = g.create_history(desc("hist"));
        g.add_pass(probe("write_a", &[], &[a], false));
        g.add_pass(probe("dead", &[], &[unused], false));
        g.add_pass(probe("temporal", &[hist, a], &[hist], false));
        g.add_pass(probe("present", &[a], &[], true));
        g.compile(&gpu.device).unwrap();
        assert_eq!(g.live_passes(), vec!["write_a", "temporal", "present"]);
        assert_eq!(g.resources().extent(a).width, 64);

        let mut prof = GpuProfiler::new(&gpu.device, &gpu.queue, false);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut log = Vec::new();
        g.execute(&gpu.device, &gpu.queue, &mut enc, &mut prof, &mut log);
        assert_eq!(log, vec!["write_a", "temporal", "present"]);

        let gen_before = g.resources().generation();
        g.resize((32, 16), (64, 32));
        g.compile(&gpu.device).unwrap();
        assert!(g.resources().generation() > gen_before);
        assert_eq!(g.resources().extent(a).width, 32);
    }

    #[test]
    fn read_before_write_is_rejected() {
        let Some(gpu) = crate::device::test_gpu() else {
            return;
        };
        let mut g: FrameGraph<Vec<&'static str>> = FrameGraph::new((8, 8), (8, 8));
        let a = g.create_texture(desc("a"));
        g.add_pass(probe("reader", &[a], &[], true));
        g.add_pass(probe("writer", &[], &[a], false));
        assert_eq!(
            g.compile(&gpu.device),
            Err(GraphError::ReadBeforeWrite {
                pass: "reader",
                texture: "a"
            })
        );
    }
}
