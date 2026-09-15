//! Owns the frame graph and drives one frame end to end.

use crate::calibration::CalibrationPass;
use crate::frame::FrameCtx;
use crate::hud::HudPass;
use crate::present::PresentPass;
use mc2_gpu::{FrameGraph, Gpu, GpuProfiler, TexHandle, TextureDesc};

pub struct Renderer {
    graph: FrameGraph<FrameCtx>,
    pub frame: FrameCtx,
    pub profiler: GpuProfiler,
    output: TexHandle,
    pub output_format: wgpu::TextureFormat,
    output_size: (u32, u32),
    /// Internal render resolution as a fraction of the display.
    pub render_scale: f32,
}

pub fn render_size(output: (u32, u32), scale: f32) -> (u32, u32) {
    (
        ((output.0 as f32 * scale).round() as u32).max(1),
        ((output.1 as f32 * scale).round() as u32).max(1),
    )
}

impl Renderer {
    pub fn new(gpu: &Gpu, output_format: wgpu::TextureFormat, output_size: (u32, u32)) -> Self {
        let shaders = crate::shaders::library();
        let render_scale = 1.0;
        let mut graph = FrameGraph::new(render_size(output_size, render_scale), output_size);
        let output = graph.declare_import(TextureDesc {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            ..TextureDesc::output_target("display", output_format)
        });
        let scene = graph.create_texture(TextureDesc::render_target(
            "scene hdr",
            wgpu::TextureFormat::Rgba16Float,
            1.0,
        ));
        let dev = &gpu.device;
        graph.add_pass(Box::new(CalibrationPass::new(dev, &shaders, scene)));
        graph.add_pass(Box::new(PresentPass::new(
            dev,
            &shaders,
            scene,
            output,
            output_format,
        )));
        graph.add_pass(Box::new(HudPass::new(
            dev,
            &gpu.queue,
            &shaders,
            output,
            output_format,
        )));
        Self {
            graph,
            frame: FrameCtx::new(shaders),
            profiler: GpuProfiler::new(dev, &gpu.queue, gpu.timestamps),
            output,
            output_format,
            output_size,
            render_scale,
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

    pub fn live_passes(&self) -> Vec<&'static str> {
        self.graph.live_passes()
    }

    /// Renders into `target`, which must match the output format and size.
    pub fn render(&mut self, gpu: &Gpu, target: wgpu::Texture) {
        mc2_core::scope!("renderer.frame");
        self.frame.shaders.poll();
        self.profiler.begin_frame();
        self.graph.import(self.output, target);
        if let Err(e) = self.graph.compile(&gpu.device) {
            panic!("frame graph invalid: {e}");
        }
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
        self.profiler.end_frame(&mut encoder);
        {
            mc2_core::scope!("renderer.submit");
            gpu.queue.submit([encoder.finish()]);
        }
        self.profiler.after_submit();
        self.frame.frame_index = self.frame.frame_index.wrapping_add(1);
    }
}
