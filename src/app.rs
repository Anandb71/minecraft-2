//! Windowed application: event loop, surface management, frame pacing.

use mc2_core::RollingStats;
use mc2_gpu::{Gpu, GpuOptions};
use mc2_render::Renderer;
use mc2_render::overlay::{OverlayInput, draw_profiler};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    renderer: Renderer,
}

pub struct App {
    running: Option<Running>,
    start: Instant,
    last_frame: Instant,
    frame_ms: RollingStats,
    show_profiler: bool,
    vsync: bool,
    error: Option<String>,
    exit_after: Option<u32>,
    frames: u32,
}

impl App {
    fn new(exit_after: Option<u32>) -> Self {
        Self {
            running: None,
            start: Instant::now(),
            last_frame: Instant::now(),
            frame_ms: RollingStats::default(),
            show_profiler: true,
            vsync: true,
            error: None,
            exit_after,
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
        let renderer = Renderer::new(&gpu, format, (config.width, config.height));
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

    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        self.frames += 1;
        if self.exit_after.is_some_and(|n| self.frames > n) {
            event_loop.exit();
            return;
        }
        let Some(r) = &mut self.running else {
            return;
        };
        let now = Instant::now();
        self.frame_ms
            .push((now - self.last_frame).as_secs_f32() * 1000.0);
        self.last_frame = now;

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
        renderer.frame.hud.clear();
        if self.show_profiler {
            let cpu_rows = mc2_core::profiler::rows();
            let extra = [format!(
                "F3 profiler  V vsync ({})  Esc quit",
                if self.vsync { "on" } else { "off" }
            )];
            let input = OverlayInput {
                gpu: &renderer.profiler,
                cpu: &cpu_rows,
                frame_ms: &self.frame_ms,
                adapter: &r.gpu.info.name,
                resolution: renderer.output_size(),
                render_resolution: mc2_render::renderer::render_size(
                    renderer.output_size(),
                    renderer.render_scale,
                ),
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
        if event.state != ElementState::Pressed || event.repeat {
            return;
        }
        match event.physical_key {
            PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
            PhysicalKey::Code(KeyCode::F3) => self.show_profiler = !self.show_profiler,
            PhysicalKey::Code(KeyCode::KeyV) => {
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
            WindowEvent::KeyboardInput { event, .. } => self.key(event_loop, &event),
            WindowEvent::RedrawRequested => self.frame(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(r) = &self.running {
            r.window.request_redraw();
        }
    }
}

pub fn run(exit_after: Option<u32>) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| e.to_string())?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(exit_after);
    event_loop.run_app(&mut app).map_err(|e| e.to_string())?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
