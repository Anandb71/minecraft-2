//! Owns the frame graph and drives one frame end to end.

use crate::calibration::CalibrationPass;
use crate::camera::{Camera, Celestial, FrameInputs, FrameUniforms};
use crate::debug_shade::DebugShadePass;
use crate::frame::FrameCtx;
use crate::hud::HudPass;
use crate::present::PresentPass;
use crate::quality::Quality;
use crate::vis::{BeamPass, PrimaryVisPass, VisTargets};
use crate::voxel_gpu::{GpuWorld, GpuWorldConfig};
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
    pub quality: Quality,
}

impl Default for RendererOptions {
    fn default() -> Self {
        Self {
            mode: RenderMode::World,
            world: GpuWorldConfig::default(),
            quality: Quality::default(),
        }
    }
}

pub struct Renderer {
    graph: FrameGraph<FrameCtx>,
    pub frame: FrameCtx,
    pub profiler: GpuProfiler,
    pub world: GpuWorld,
    pub lights: crate::lights::LightRegistry,
    pub skymap: crate::skymap::SkyMap,
    output: TexHandle,
    pub output_format: wgpu::TextureFormat,
    output_size: (u32, u32),
    quality: Quality,
    prev_camera: Option<Camera>,
    pub celestial: Celestial,
    /// Cloud layer weather.
    pub clouds: crate::clouds::CloudSettings,
    pub fog: crate::fog::FogSettings,
    /// Rigid bodies to draw next frame, posed for this frame.
    pub bodies: Vec<crate::bodies::BodyInstance>,
    pub debug_mode: u32,
    /// Start primary rays from the beam prepass distances.
    pub beam: bool,
    /// Sub-pixel camera jitter for temporal upsampling; on in world mode.
    pub jitter: bool,
}

/// Indirect light and its denoiser run at half the render resolution.
pub const GI_SHIFT: u32 = 1;

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
        let lights = crate::lights::LightRegistry::new(dev);
        let frame = FrameCtx::new(
            dev,
            shaders,
            world.layout.clone(),
            world.bind_group.clone(),
            lights.bind_group.clone(),
        );
        let mut frame = frame;
        if opts.mode == RenderMode::Calibration {
            // The calibration chart shows the display transform alone.
            frame.post = crate::post::PostSettings::neutral();
        }
        let mut graph = FrameGraph::new(
            render_size(output_size, opts.quality.render_scale),
            output_size,
        );
        let output = graph.declare_import(TextureDesc {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            ..TextureDesc::output_target("display", output_format)
        });
        let scene = graph.create_texture(TextureDesc::render_target(
            "scene hdr",
            wgpu::TextureFormat::Rgba16Float,
            1.0,
        ));
        let mut exposure = None;
        let skymap = crate::skymap::SkyMap::new(dev);
        let mut history = Vec::new();
        // What the present pass shows: the render-resolution scene, or its
        // temporally upsampled display-resolution version.
        let mut presented = scene;
        let vis = match opts.mode {
            RenderMode::Calibration => {
                graph.add_pass(Box::new(CalibrationPass::new(dev, &frame.shaders, scene)));
                None
            }
            RenderMode::World => {
                use crate::direct::{ComposePass, DirectTargets, HistoryPass, RestirPass, SunPass};
                use crate::sky::{SkyFramePass, SkyLutPass, SkyTargets};
                let vis = VisTargets::create(&mut graph);
                let sky = SkyTargets::create(&mut graph);
                let direct = DirectTargets::create(&mut graph);
                graph.add_pass(Box::new(BeamPass::new(dev, &frame, vis.beam)));
                graph.add_pass(Box::new(PrimaryVisPass::new(dev, &frame, vis)));
                graph.add_pass(Box::new(SkyLutPass::new(dev, &frame, sky)));
                graph.add_pass(Box::new(SkyFramePass::new(dev, &frame, sky)));
                graph.add_pass(Box::new(SunPass::new(dev, &frame, vis, direct)));
                graph.add_pass(Box::new(RestirPass::new(
                    dev,
                    &frame,
                    &lights.layout,
                    vis,
                    direct,
                )));
                // Emitter light, denoised at render resolution.
                let emitters = crate::svgf::SvgfTargets::create(
                    &mut graph,
                    [
                        "emitters integrated",
                        "emitters moments",
                        "emitters moments prev",
                        "emitters color prev",
                        "emitters ping",
                        "emitters pong",
                    ],
                    0,
                );
                graph.add_pass(Box::new(
                    crate::svgf::SvgfPass::new(
                        dev,
                        &frame,
                        "denoise.emitters",
                        direct.direct_lights,
                        vis,
                        direct.vis_id_prev,
                        emitters,
                        0,
                    )
                    .only_with_lights(),
                ));
                // Indirect light at GI resolution, both methods writing one
                // signal, then denoised.
                use crate::indirect::{
                    CascadeTargets, CascadesPass, GiInputs, RestirGiPass, RestirGiTargets,
                };
                let cloud_targets = crate::clouds::CloudTargets::create(&mut graph);
                let clouds_pass = crate::clouds::CloudsPass::new(dev, &frame, cloud_targets, sky);
                let cloud_uniforms = clouds_pass.uniforms.clone();
                graph.add_pass(Box::new(clouds_pass));
                let fog_targets = crate::fog::FogTargets::create(&mut graph);
                graph.add_pass(Box::new(crate::fog::FogPass::new(
                    dev,
                    &frame,
                    fog_targets,
                    sky,
                    (cloud_targets.shadow, cloud_uniforms.clone()),
                    skymap.view.clone(),
                )));
                let gi_inputs = GiInputs {
                    vis,
                    prev_id: direct.vis_id_prev,
                    surface_prev: direct.surface_prev,
                    sky,
                };
                let gi_noisy = crate::indirect::gi_output(&mut graph, GI_SHIFT);
                let restir_gi = RestirGiTargets::create(&mut graph, GI_SHIFT);
                graph.add_pass(Box::new(RestirGiPass::new(
                    dev,
                    &frame,
                    gi_inputs,
                    restir_gi,
                    gi_noisy,
                    skymap.view.clone(),
                    GI_SHIFT,
                )));
                let cascades = CascadeTargets::create(&mut graph);
                graph.add_pass(Box::new(CascadesPass::new(
                    dev,
                    &frame,
                    gi_inputs,
                    cascades,
                    gi_noisy,
                    skymap.view.clone(),
                    GI_SHIFT,
                )));
                let gi = crate::svgf::SvgfTargets::create(
                    &mut graph,
                    [
                        "gi integrated",
                        "gi moments",
                        "gi moments prev",
                        "gi color prev",
                        "gi ping",
                        "gi pong",
                    ],
                    GI_SHIFT,
                );
                graph.add_pass(Box::new(crate::svgf::SvgfPass::new(
                    dev,
                    &frame,
                    "denoise.gi",
                    gi_noisy,
                    vis,
                    direct.vis_id_prev,
                    gi,
                    GI_SHIFT,
                )));
                // Glossy reflections, denoised like indirect light.
                let reflections_noisy = graph.create_texture(TextureDesc {
                    size: mc2_gpu::SizePolicy::RenderBlocks {
                        block: 1 << GI_SHIFT,
                        extra: 0,
                    },
                    ..TextureDesc::render_target(
                        "reflections noisy",
                        wgpu::TextureFormat::Rgba16Float,
                        1.0,
                    )
                });
                graph.add_pass(Box::new(crate::indirect::ReflectionPass::new(
                    dev,
                    &frame,
                    gi_inputs,
                    reflections_noisy,
                    skymap.view.clone(),
                    GI_SHIFT,
                )));
                let reflections = crate::svgf::SvgfTargets::create(
                    &mut graph,
                    [
                        "reflections integrated",
                        "reflections moments",
                        "reflections moments prev",
                        "reflections color prev",
                        "reflections ping",
                        "reflections pong",
                    ],
                    GI_SHIFT,
                );
                graph.add_pass(Box::new(crate::svgf::SvgfPass::new(
                    dev,
                    &frame,
                    "refl.denoise",
                    reflections_noisy,
                    vis,
                    direct.vis_id_prev,
                    reflections,
                    GI_SHIFT,
                )));
                // What is seen through glass and water, full resolution.
                let refraction = graph.create_texture(TextureDesc::render_target(
                    "refraction",
                    wgpu::TextureFormat::Rgba16Float,
                    1.0,
                ));
                graph.add_pass(Box::new(crate::indirect::RefractionPass::new(
                    dev,
                    &frame,
                    gi_inputs,
                    refraction,
                    skymap.view.clone(),
                )));
                graph.add_pass(Box::new(ComposePass::new(
                    dev,
                    &frame,
                    vis,
                    direct,
                    sky,
                    crate::direct::ComposeInputs {
                        emitters: emitters.output,
                        gi: gi.output,
                        surface: direct.surface,
                        clouds: cloud_targets.clouds,
                        cloud_shadow: cloud_targets.shadow,
                        cloud_uniforms,
                        fog: fog_targets.integrated,
                        reflections: reflections.output,
                        refraction,
                    },
                    scene,
                )));
                history.extend(restir_gi.history_copies());
                history.extend(cloud_targets.history_copies());
                history.extend(fog_targets.history_copies());
                history.extend(crate::svgf::SvgfPass::history_copies(&emitters));
                history.extend(crate::svgf::SvgfPass::history_copies(&gi));
                history.extend(crate::svgf::SvgfPass::history_copies(&reflections));
                history.push((direct.surface, direct.surface_prev));
                graph.add_pass(Box::new(DebugShadePass::new(dev, &frame, vis, scene)));
                let exposure_targets = crate::direct::ExposurePass::targets(&mut graph);
                graph.add_pass(Box::new(crate::direct::ExposurePass::new(
                    dev,
                    &frame,
                    scene,
                    exposure_targets,
                )));
                exposure = Some(exposure_targets.0);
                let taa = crate::upsample::UpsampleTargets::create(&mut graph);
                graph.add_pass(Box::new(crate::upsample::TemporalUpsamplePass::new(
                    dev,
                    &frame,
                    scene,
                    vis,
                    taa,
                    exposure_targets.1,
                )));
                let post = crate::post::PostTargets::create(&mut graph);
                graph.add_pass(Box::new(crate::post::PostPass::new(
                    dev, &frame, taa.color, vis, post,
                )));
                presented = post.color;
                history.extend([
                    (taa.color, taa.history),
                    (exposure_targets.0, exposure_targets.1),
                    (vis.id, direct.vis_id_prev),
                    (direct.light_vis, direct.light_vis_prev),
                    (direct.res_a1, direct.res_a_prev),
                    (direct.res_b1, direct.res_b_prev),
                ]);
                graph.add_pass(Box::new(HistoryPass::new(std::mem::take(&mut history))));
                Some(vis)
            }
        };
        graph.add_pass(Box::new(PresentPass::new(
            dev,
            &frame.shaders,
            presented,
            output,
            exposure,
            output_format,
        )));
        if let Some(vis) = vis {
            graph.add_pass(Box::new(crate::gizmo::GizmoPass::new(
                dev,
                &frame,
                vis.depth,
                output,
                output_format,
            )));
        }
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
            lights,
            skymap,
            output,
            output_format,
            output_size,
            quality: opts.quality,
            prev_camera: None,
            celestial: Celestial::default(),
            clouds: crate::clouds::CloudSettings::default(),
            fog: crate::fog::FogSettings::default(),
            bodies: Vec::new(),
            debug_mode: 0,
            beam: true,
            jitter: opts.mode == RenderMode::World,
        }
    }

    pub fn resize(&mut self, output_size: (u32, u32)) {
        self.output_size = output_size;
        self.graph.resize(
            render_size(output_size, self.quality.render_scale),
            output_size,
        );
    }

    pub fn quality(&self) -> Quality {
        self.quality
    }

    /// Applies quality settings, resizing internal targets if the render
    /// scale changed.
    pub fn set_quality(&mut self, quality: Quality) {
        let rescale = quality.render_scale != self.quality.render_scale;
        self.quality = quality;
        if rescale {
            self.resize(self.output_size);
        }
    }

    pub fn output_size(&self) -> (u32, u32) {
        self.output_size
    }

    pub fn render_size(&self) -> (u32, u32) {
        render_size(self.output_size, self.quality.render_scale)
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
        self.world
            .bodies
            .update(&gpu.queue, &self.bodies, camera.position);
        {
            mc2_core::scope!("lights.update");
            for pos in self.world.take_changed() {
                self.lights.update_chunk(pos, world.chunk(pos));
                self.skymap.update_chunk(pos, world.chunk(pos));
            }
            self.lights.upload(&gpu.device, &gpu.queue, camera.position);
            self.frame.lights_bind_group = self.lights.bind_group.clone();
            self.skymap.upload(&gpu.queue, camera.position);
        }
        self.frame.gi = self.quality.gi;
        self.frame.cloud_quality = (self.quality.cloud_steps, self.quality.cloud_light_steps);
        self.frame.camera_world = camera.position.as_vec3().to_array();
        self.frame.clouds = self.clouds;
        self.frame.fog = self.fog;
        let prev = self.prev_camera.unwrap_or(*camera);
        self.frame.uniforms = FrameUniforms::build(&FrameInputs {
            camera: *camera,
            prev_camera: prev,
            render_size: self.render_size(),
            output_size: self.output_size,
            frame_index: self.frame.frame_index,
            time: self.frame.time,
            dt,
            // Jittered sub-pixel samples feed temporal upsampling.
            jitter: self.jitter,
            celestial: self.celestial,
            exposure: self.frame.exposure,
            debug_mode: self.debug_mode,
            restir_candidates: self.quality.restir_candidates,
            lod_pixels: self.quality.lod_pixels,
            beam: self.beam,
            trace_stride: self.quality.trace_stride,
            light_count: self.lights.sampled_len(),
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
