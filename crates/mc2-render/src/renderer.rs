//! Owns the frame graph and drives one frame end to end.

use crate::calibration::CalibrationPass;
use crate::camera::{Camera, FrameInputs, FrameUniforms};
use crate::debug_shade::DebugShadePass;
use crate::frame::FrameCtx;
use crate::hud::HudPass;
use crate::present::PresentPass;
use crate::vis::{PrimaryVisPass, VisTargets};
use crate::voxel_gpu::{GpuWorld, GpuWorldConfig};
use glam::Vec3;
use mc2_gpu::{FrameGraph, Gpu, GpuProfiler, TexHandle, TextureDesc};
use mc2_voxel::world::VoxelWorld;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    /// Exposure ramp and colour bars; validates the display pipeline.
    Calibration,
    World,
}

#[derive(Clone, Copy, Debug)]
pub struct RendererOptions {
    pub mode: RenderMode,
    pub world: GpuWorldConfig,
    pub render_scale: f32,
}

impl Default for RendererOptions {
    fn default() -> Self {
        Self {
            mode: RenderMode::World,
            world: GpuWorldConfig::default(),
            render_scale: 1.0,
        }
    }
}

pub struct Renderer {
    graph: FrameGraph<FrameCtx>,
    pub frame: FrameCtx,
    pub profiler: GpuProfiler,
    pub world: GpuWorld,
    output: TexHandle,
    pub output_format: wgpu::TextureFormat,
    output_size: (u32, u32),
    /// Internal render resolution as a fraction of the display.
    pub render_scale: f32,
    prev_camera: Option<Camera>,
    pub sun_dir: Vec3,
    pub debug_mode: u32,
    pub lod_pixels: f32,
}

pub fn render_size(output: (u32, u32), scale: f32) -> (u32, u32) {
    (
        ((output.0 as f32 * scale).round() as u32).max(1),
        ((output.1 as f32 * scale).round() as u32).max(1),
    )
}

impl Renderer {
    pub fn new(
        gpu: &Gpu,
        output_format: wgpu::TextureFormat,
        output_size: (u32, u32),
        opts: RendererOptions,
    ) -> Self {
        let dev = &gpu.device;
        let shaders = crate::shaders::library();
        let world = GpuWorld::new(dev, opts.world);
        let frame = FrameCtx::new(dev, shaders, world.layout.clone(), world.bind_group.clone());
        let mut graph = FrameGraph::new(render_size(output_size, opts.render_scale), output_size);
        let output = graph.declare_import(TextureDesc {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            ..TextureDesc::output_target("display", output_format)
        });
        let scene = graph.create_texture(TextureDesc::render_target(
            "scene hdr",
            wgpu::TextureFormat::Rgba16Float,
            1.0,
        ));
        match opts.mode {
            RenderMode::Calibration => {
                graph.add_pass(Box::new(CalibrationPass::new(dev, &frame.shaders, scene)));
            }
            RenderMode::World => {
                let vis = VisTargets::create(&mut graph);
                graph.add_pass(Box::new(PrimaryVisPass::new(dev, &frame, vis)));
                graph.add_pass(Box::new(DebugShadePass::new(dev, &frame, vis, scene)));
            }
        }
        graph.add_pass(Box::new(PresentPass::new(
            dev,
            &frame.shaders,
            scene,
            output,
            output_format,
        )));
        graph.add_pass(Box::new(HudPass::new(
            dev,
            &gpu.queue,
            &frame.shaders,
            output,
            output_format,
        )));
        Self {
            graph,
            frame,
            profiler: GpuProfiler::new(dev, &gpu.queue, gpu.timestamps),
            world,
            output,
            output_format,
            output_size,
            render_scale: opts.render_scale,
            prev_camera: None,
            sun_dir: Vec3::new(0.4, 0.8, 0.3),
            debug_mode: 0,
            lod_pixels: 1.0,
        }
    }

    pub fn resize(&mut self, output_size: (u32, u32)) {
        self.output_size = output_size;
        self.graph
            .resize(render_size(output_size, self.render_scale), output_size);
    }

    pub fn output_size(&self) -> (u32, u32) {
        self.output_size
    }

    pub fn render_size(&self) -> (u32, u32) {
        render_size(self.output_size, self.render_scale)
    }

    /// A frame graph texture by label, e.g. "vis id", for tests and tools.
    pub fn graph_texture(&self, label: &str) -> Option<&wgpu::Texture> {
        self.graph.resources().find(label)
    }

    pub fn live_passes(&self) -> Vec<&'static str> {
        self.graph.live_passes()
    }

    /// Streams world changes and sets the camera for the next frame.
    pub fn prepare(&mut self, gpu: &Gpu, world: &mut VoxelWorld, camera: &Camera, dt: f32) {
        self.world.update(&gpu.queue, world, camera.position);
        let prev = self.prev_camera.unwrap_or(*camera);
        self.frame.uniforms = FrameUniforms::build(&FrameInputs {
            camera: *camera,
            prev_camera: prev,
            render_size: self.render_size(),
            output_size: self.output_size,
            frame_index: self.frame.frame_index,
            time: self.frame.time,
            dt,
            jitter: false,
            sun_dir: self.sun_dir,
            exposure: self.frame.exposure,
            debug_mode: self.debug_mode,
            quality: 0,
            lod_pixels: self.lod_pixels,
        });
        self.prev_camera = Some(*camera);
    }

    /// Renders into the target texture, which must match the output format.
    pub fn render(&mut self, gpu: &Gpu, target: wgpu::Texture) {
        mc2_core::scope!("renderer.frame");
        self.frame.shaders.poll();
        self.profiler.begin_frame();
        self.graph.import(self.output, target);
        if let Err(e) = self.graph.compile(&gpu.device) {
            panic!("frame graph invalid: {e}");
        }
        gpu.queue.write_buffer(
            &self.frame.frame_buffer,
            0,
            bytemuck::bytes_of(&self.frame.uniforms),
        );
        self.world.begin_frame(&gpu.queue);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        self.graph.execute(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &mut self.profiler,
            &mut self.frame,
        );
        self.world.end_frame(&mut encoder);
        self.profiler.end_frame(&mut encoder);
        {
            mc2_core::scope!("renderer.submit");
            gpu.queue.submit([encoder.finish()]);
        }
        self.world.after_submit();
        self.profiler.after_submit();
        self.graph.release_import(self.output);
        self.frame.frame_index = self.frame.frame_index.wrapping_add(1);
    }
}
