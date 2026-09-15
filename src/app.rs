//! Windowed application: event loop, surface management, input, frame pacing.

use crate::cli::Args;
use crate::flycam::{FlyCam, Input};
use crate::world::{GameWorld, spawn_camera};
use glam::Vec2;
use mc2_core::RollingStats;
use mc2_gpu::{Gpu, GpuOptions};
use mc2_render::Renderer;
use mc2_render::camera::Camera;
use mc2_render::overlay::{OverlayInput, draw_profiler};
use mc2_render::renderer::RendererOptions;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    renderer: Renderer,
}

pub struct App {
    running: Option<Running>,
    world: GameWorld,
    camera: Camera,
    flycam: FlyCam,
    input: Input,
    grabbed: bool,
    start: Instant,
    last_frame: Instant,
    frame_ms: RollingStats,
    show_profiler: bool,
    vsync: bool,
    debug_mode: u32,
    error: Option<String>,
    exit_after: Option<u32>,
    frames: u32,
}

impl App {
    fn new(args: &Args) -> Self {
        Self {
            running: None,
            world: GameWorld::start(args.seed, args.world_dir.clone(), Default::default()),
            camera: Camera::default(),
            flycam: FlyCam::default(),
            input: Input::default(),
            grabbed: false,
            start: Instant::now(),
            last_frame: Instant::now(),
            frame_ms: RollingStats::default(),
            show_profiler: true,
            vsync: true,
            debug_mode: 0,
            error: None,
            exit_after: args.exit_after,
            frames: 0,
        }
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let attrs = Window::default_attributes()
            .with_title("MINECRAFT 2")
            .with_inner_size(winit::dpi::PhysicalSize::new(1600, 900));
        let window = Arc::new(event_loop.create_window(attrs).map_err(|e| e.to_string())?);
        let instance = Gpu::instance();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| e.to_string())?;
        let gpu = Gpu::new(instance, Some(&surface), &GpuOptions::default())
            .map_err(|e| e.to_string())?;
        log::info!("adapter: {}", gpu.describe());
        let caps = surface.get_capabilities(&gpu.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&gpu.device, &config);
        let renderer = Renderer::new(
            &gpu,
            format,
            (config.width, config.height),
            RendererOptions::default(),
        );
        self.running = Some(Running {
            window,
            surface,
            config,
            gpu,
            renderer,
        });
        Ok(())
    }

    fn reconfigure(&mut self) {
        if let Some(r) = &mut self.running {
            r.config.present_mode = if self.vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            };
            r.surface.configure(&r.gpu.device, &r.config);
            r.renderer.resize((r.config.width, r.config.height));
        }
    }

    fn set_grab(&mut self, grab: bool) {
        let Some(r) = &self.running else {
            return;
        };
        if grab {
            let ok = r
                .window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| r.window.set_cursor_grab(CursorGrabMode::Confined))
                .is_ok();
            r.window.set_cursor_visible(!ok);
            self.grabbed = ok;
        } else {
            let _ = r.window.set_cursor_grab(CursorGrabMode::None);
            r.window.set_cursor_visible(true);
            self.grabbed = false;
            self.input.release_all();
        }
    }

    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        self.frames += 1;
        if self.exit_after.is_some_and(|n| self.frames > n) {
            event_loop.exit();
            return;
        }
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.frame_ms.push(dt * 1000.0);
        self.last_frame = now;

        match self.world.poll() {
            Ok(Some(terrain)) => self.camera = spawn_camera(&terrain),
            Ok(None) => {}
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
                return;
            }
        }
        {
            mc2_core::scope!("game.update");
            self.flycam
                .update(&mut self.camera, &self.input, dt.min(0.1), self.grabbed);
            self.input.clear_frame();
            self.world.update(self.camera.position);
        }

        let Some(r) = &mut self.running else {
            return;
        };
        let surface_tex = match r.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                r.surface.configure(&r.gpu.device, &r.config);
                return;
            }
            _ => return,
        };

        let renderer = &mut r.renderer;
        renderer.frame.time = self.start.elapsed().as_secs_f32();
        renderer.debug_mode = self.debug_mode;
        renderer.prepare(&r.gpu, &mut self.world.voxels, &self.camera, dt);
        renderer.frame.hud.clear();
        if self.show_profiler {
            let cpu_rows = mc2_core::profiler::rows();
            let s = renderer.world.stats;
            let p = self.camera.position;
            let stream = self
                .world
                .streamer
                .as_ref()
                .map(|st| st.stats)
                .unwrap_or_default();
            let extra = [
                self.world.loading_text().unwrap_or_else(|| {
                    format!(
                        "streaming: {} chunks ({} full), {} queued, {} generating, +{} this frame",
                        stream.loaded,
                        stream.full,
                        stream.queued,
                        stream.inflight,
                        stream.inserted_last_frame
                    )
                }),
                format!(
                    "chunks {}  bricks {}  +{} ({:.1} MB)  feedback {}  tree {:.1} MB  voxels {:.1} MB",
                    s.chunks,
                    s.bricks_resident,
                    s.bricks_uploaded_last_frame,
                    s.bytes_uploaded_last_frame as f32 / 1e6,
                    s.feedback_requests_last_frame,
                    s.tree_mb,
                    s.voxel_mb
                ),
                format!(
                    "pos {:.1} {:.1} {:.1}  speed {:.1} m/s",
                    p.x, p.y, p.z, self.flycam.speed
                ),
                format!(
                    "click look  WASD fly  wheel speed  F3 HUD  F4 view {}  V vsync {}  Esc release/quit",
                    self.debug_mode,
                    if self.vsync { "on" } else { "off" }
                ),
            ];
            let input = OverlayInput {
                gpu: &renderer.profiler,
                cpu: &cpu_rows,
                frame_ms: &self.frame_ms,
                adapter: &r.gpu.info.name,
                resolution: renderer.output_size(),
                render_resolution: renderer.render_size(),
                extra: &extra,
            };
            let mut hud = std::mem::take(&mut renderer.frame.hud);
            draw_profiler(&mut hud, &input, 1.0);
            renderer.frame.hud = hud;
        }
        renderer.render(&r.gpu, surface_tex.texture.clone());
        {
            mc2_core::scope!("present");
            r.window.pre_present_notify();
            r.gpu.queue.present(surface_tex);
        }
        mc2_core::profiler::end_frame();
    }

    fn key(&mut self, event_loop: &ActiveEventLoop, event: &KeyEvent) {
        let PhysicalKey::Code(code) = event.physical_key else {
            return;
        };
        if event.state == ElementState::Released {
            self.input.release(code);
            return;
        }
        if event.repeat {
            return;
        }
        self.input.press(code);
        match code {
            KeyCode::Escape => {
                if self.grabbed {
                    self.set_grab(false);
                } else {
                    event_loop.exit();
                }
            }
            KeyCode::F3 => self.show_profiler = !self.show_profiler,
            KeyCode::F4 => self.debug_mode = (self.debug_mode + 1) % 3,
            KeyCode::KeyV => {
                self.vsync = !self.vsync;
                self.reconfigure();
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_none()
            && let Err(e) = self.init(event_loop)
        {
            self.error = Some(e);
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.running
                    && size.width > 0
                    && size.height > 0
                {
                    r.config.width = size.width;
                    r.config.height = size.height;
                    self.reconfigure();
                }
            }
            WindowEvent::Focused(false) => self.set_grab(false),
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !self.grabbed => self.set_grab(true),
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
            }
            WindowEvent::KeyboardInput { event, .. } => self.key(event_loop, &event),
            WindowEvent::RedrawRequested => self.frame(event_loop),
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.grabbed
        {
            self.input.mouse_delta += Vec2::new(delta.0 as f32, delta.1 as f32);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(r) = &self.running {
            r.window.request_redraw();
        }
    }
}

pub fn run(args: &Args) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| e.to_string())?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(args);
    event_loop.run_app(&mut app).map_err(|e| e.to_string())?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
