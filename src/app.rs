//! Windowed application: event loop, surface management, input, frame pacing.

use crate::cli::Args;
use crate::gamepad::{Gamepads, Pad};
use crate::world::{TerrainLoader, spawn_camera, stream, stream_stats};
use glam::Vec2;
use mc2_core::RollingStats;
use mc2_game::Game;
use mc2_game::input::{Button, Key};
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
    game: Game,
    loader: TerrainLoader,
    camera: Camera,
    grabbed: bool,
    gamepads: Gamepads,
    pad_connected: bool,
    last_frame: Instant,
    start: Instant,
    frame_ms: RollingStats,
    show_profiler: bool,
    vsync: bool,
    debug_mode: u32,
    quality: mc2_render::quality::Preset,
    gi: Option<mc2_render::indirect::GiMethod>,
    photo: crate::photo::PhotoMode,
    inventory: crate::inventory_ui::Screen,
    audio: crate::audio::Audio,
    water: Option<crate::water_sim::WaterSim>,
    modifiers: winit::keyboard::ModifiersState,
    screenshot_requested: bool,
    error: Option<String>,
    exit_after: Option<u32>,
    frames: u32,
}

fn map_key(code: KeyCode) -> Option<Key> {
    Some(match code {
        KeyCode::KeyW => Key::Forward,
        KeyCode::KeyS => Key::Back,
        KeyCode::KeyA => Key::Left,
        KeyCode::KeyD => Key::Right,
        KeyCode::Space => Key::Jump,
        KeyCode::ControlLeft | KeyCode::KeyC => Key::Crouch,
        KeyCode::ShiftLeft => Key::Sprint,
        KeyCode::KeyF => Key::ToggleFly,
        KeyCode::F5 => Key::ToggleView,
        KeyCode::Tab => Key::ToggleMode,
        KeyCode::KeyE => Key::Interact,
        KeyCode::KeyP => Key::Photo,
        KeyCode::Digit1 => Key::Slot(0),
        KeyCode::Digit2 => Key::Slot(1),
        KeyCode::Digit3 => Key::Slot(2),
        KeyCode::Digit4 => Key::Slot(3),
        KeyCode::Digit5 => Key::Slot(4),
        KeyCode::Digit6 => Key::Slot(5),
        KeyCode::Digit7 => Key::Slot(6),
        KeyCode::Digit8 => Key::Slot(7),
        KeyCode::Digit9 => Key::Slot(8),
        _ => return None,
    })
}

impl App {
    fn new(args: &Args) -> Self {
        let mut game = Game::new();
        game.world
            .insert_resource(mc2_game::interact::Interaction::new(args.creative));
        if let Some(sky) = args.weather {
            game.world
                .resource_mut::<mc2_game::weather::Weather>()
                .set(sky);
        }
        // Physics runs on its own thread in the window; tests and headless
        // captures step it inline.
        game.world.resource_mut::<mc2_game::physics::Physics>().host =
            mc2_game::physics_host::PhysicsHost::threaded();
        Self {
            running: None,
            game,
            loader: TerrainLoader::start(args.seed, args.world_dir.clone(), Default::default()),
            camera: Camera::default(),
            grabbed: false,
            gamepads: Gamepads::new(),
            pad_connected: false,
            last_frame: Instant::now(),
            start: Instant::now(),
            frame_ms: RollingStats::default(),
            show_profiler: true,
            vsync: true,
            debug_mode: 0,
            quality: args.quality,
            gi: args.gi,
            photo: crate::photo::PhotoMode::default(),
            inventory: Default::default(),
            audio: crate::audio::Audio::start(),
            water: None,
            modifiers: Default::default(),
            screenshot_requested: false,
            error: None,
            exit_after: args.exit_after,
            frames: 0,
        }
    }

    fn quality_settings(&self) -> mc2_render::quality::Quality {
        let mut q = self.quality.settings();
        if let Some(gi) = self.gi {
            q.gi = gi;
        }
        q
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
            RendererOptions {
                quality: self.quality_settings(),
                ..Default::default()
            },
        );
        let mut renderer = renderer;
        renderer.frame.hud.set_icons(crate::icons::icons().atlas());
        self.water = Some(crate::water_sim::WaterSim::new(
            &gpu,
            &renderer.frame.shaders,
        ));
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

    fn set_play(&mut self, play: bool) {
        let Some(r) = &self.running else {
            return;
        };
        if play {
            r.window.focus_window();
            let grabbed = r
                .window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| r.window.set_cursor_grab(CursorGrabMode::Confined))
                .is_ok();
            r.window.set_cursor_visible(false);
            self.grabbed = grabbed;
            self.game.input().captured = true;
        } else {
            let _ = r.window.set_cursor_grab(CursorGrabMode::None);
            r.window.set_cursor_visible(true);
            self.grabbed = false;
            self.game.input().release_all();
            self.game.input().captured = false;
        }
    }

    /// Opens (at a station, if one was used) or closes the inventory
    /// screen, freeing or recapturing the cursor.
    fn toggle_inventory(&mut self, station: Option<mc2_game::crafting::Station>) {
        if self.inventory.open {
            self.inventory.close(&mut self.game);
            self.set_play(true);
        } else {
            self.set_play(false);
            self.inventory.open_at(station);
            if let Some(r) = &self.running {
                self.inventory.center((r.config.width, r.config.height));
            }
        }
    }

    fn playing(&self) -> bool {
        self.game
            .world
            .resource::<mc2_game::input::Input>()
            .captured
    }

    fn toggle_photo(&mut self) {
        if self.photo.active {
            let paused = self.photo.exit();
            self.game.clock().paused = paused;
        } else {
            let paused = self.game.clock().paused;
            self.game.input().release_all();
            self.photo.enter(self.camera, paused);
            self.game.clock().paused = true;
        }
    }

    fn apply_gamepad(&mut self, dt: f32) {
        let pad = self.gamepads.poll();
        self.pad_connected = pad.connected;
        self.game.input().clear_pad();
        if !pad.connected {
            self.photo.set_analog(Vec2::ZERO, 0.0, false);
            return;
        }
        if self.inventory.open {
            if pad.select || pad.start || pad.crouch_pressed {
                self.toggle_inventory(None);
                return;
            }
            let size = self
                .running
                .as_ref()
                .map(|r| (r.config.width, r.config.height))
                .unwrap_or((1280, 720));
            self.inventory.stick_move(pad.move_axis, dt, size);
            if pad.jump_pressed {
                self.inventory.click(&mut self.game, false, false);
            }
            if pad.interact {
                self.inventory.click(&mut self.game, true, false);
            }
            if pad.dpad_up {
                self.inventory.wheel(1.0);
            }
            if pad.dpad_down {
                self.inventory.wheel(-1.0);
            }
            return;
        }
        if self.photo.active {
            if pad.select {
                self.toggle_photo();
                self.photo.set_analog(Vec2::ZERO, 0.0, false);
                return;
            }
            if pad.start || pad.interact {
                self.photo.request_capture();
            }
            apply_photo_pad(&mut self.photo, &pad, dt);
            return;
        }
        if pad.start {
            self.set_play(!self.playing());
            return;
        }
        if !self.playing() {
            if pad.stick_active || pad.any_edge {
                self.set_play(true);
            } else {
                return;
            }
        }
        if pad.select {
            self.toggle_inventory(None);
            return;
        }
        let mut input = self.game.input();
        input.move_axis = pad.move_axis;
        input.look_axis = pad.look_axis;
        if pad.jump {
            input.pad_hold_key(Key::Jump);
        }
        if pad.crouch {
            input.pad_hold_key(Key::Crouch);
        }
        if pad.sprint {
            input.pad_hold_key(Key::Sprint);
        }
        if pad.fly {
            input.pad_press_key(Key::ToggleFly);
        }
        if pad.view {
            input.pad_press_key(Key::ToggleView);
        }
        if pad.mode {
            input.pad_press_key(Key::ToggleMode);
        }
        if pad.interact {
            input.pad_press_key(Key::Interact);
        }
        if pad.break_block {
            input.pad_hold_button(Button::Primary);
        }
        if pad.break_pressed {
            input.pad_press_button(Button::Primary);
        }
        if pad.place {
            input.pad_hold_button(Button::Secondary);
        }
        if pad.place_pressed {
            input.pad_press_button(Button::Secondary);
        }
        if pad.preview {
            input.pad_hold_button(Button::Middle);
        }
        if pad.slot_next {
            input.scroll -= 1.0;
        }
        if pad.slot_prev {
            input.scroll += 1.0;
        }
    }

    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        self.frames += 1;
        if self.exit_after.is_some_and(|n| self.frames > n) {
            event_loop.exit();
            return;
        }
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.frame_ms
            .push((now - self.last_frame).as_secs_f32() * 1000.0);
        self.last_frame = now;
        self.apply_gamepad(dt);

        match self.loader.poll(&mut self.game) {
            Ok(Some(terrain)) => {
                let spawn = spawn_camera(&terrain);
                let feet = spawn.position - glam::DVec3::Y * mc2_game::player::EYE;
                self.game.spawn_player(feet, spawn.yaw, spawn.pitch);
            }
            Ok(None) => {}
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
                return;
            }
        }

        if self.photo.active {
            // The world holds still; only the photographer moves.
            self.photo.update(dt);
            self.camera = self.photo.camera;
        } else {
            self.game.update(dt);
            // Using a crafting table or furnace opens its recipes.
            let opened = self
                .game
                .world
                .resource_mut::<mc2_game::interact::Interaction>()
                .opened
                .take();
            if opened.is_some() && !self.inventory.open {
                self.toggle_inventory(opened);
            }
            let view = self.game.view();
            self.camera.position = view.position;
            self.camera.yaw = view.yaw;
            self.camera.pitch = view.pitch;
            self.camera.fov_y = Camera::default().fov_y;
            self.audio
                .update(&mut self.game, view.position, view.yaw, None);
        }
        stream(&mut self.game, self.camera.position);
        if let (Some(w), Some(r)) = (&mut self.water, &self.running)
            && !self.photo.active
        {
            w.frame(&r.gpu, &r.renderer.frame.shaders, &mut self.game, dt);
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
        renderer.celestial = crate::world::celestial(&self.game);
        crate::world::weather(&self.game, renderer, self.camera.position);
        if self.photo.active {
            self.photo.apply(renderer);
        } else {
            renderer.frame.post.dof = None;
            renderer.frame.exposure = 1.0;
        }
        if self.photo.take_capture_request() {
            let mut voxels = self.game.world.resource_mut::<mc2_game::Voxels>();
            match crate::photo::capture(
                &r.gpu,
                renderer,
                &mut voxels.0,
                &self.camera,
                std::path::Path::new("captures"),
            ) {
                Ok(path) => log::info!("photo saved to {}", path.display()),
                Err(e) => log::error!("photo failed: {e}"),
            }
        }
        if std::mem::take(&mut self.screenshot_requested) {
            let mut voxels = self.game.world.resource_mut::<mc2_game::Voxels>();
            match crate::photo::screenshot(
                &r.gpu,
                renderer,
                &mut voxels.0,
                &self.camera,
                std::path::Path::new("captures"),
            ) {
                Ok(path) => log::info!("screenshot saved to {}", path.display()),
                Err(e) => log::error!("screenshot failed: {e}"),
            }
        }
        crate::world::pose_bodies(&self.game, renderer);
        {
            let mut voxels = self.game.world.resource_mut::<mc2_game::Voxels>();
            renderer.prepare(&r.gpu, &mut voxels.0, &self.camera, dt);
        }
        renderer.frame.hud.clear();
        let screen = renderer.output_size();
        if self.photo.active {
            let status = self.photo.status(self.pad_connected);
            renderer.frame.hud.text(8.0, 8.0, 1.0, 0xffff_ffff, &status);
        } else {
            crate::game_hud::draw(
                &mut self.game,
                &mut renderer.frame.hud,
                &mut renderer.frame.gizmos,
                screen,
                self.pad_connected,
            );
        }
        if !self.photo.active {
            self.inventory.draw(
                &mut self.game,
                &mut renderer.frame.hud,
                screen,
                self.pad_connected,
            );
        }
        if self.show_profiler && !self.photo.active {
            let cpu_rows = mc2_core::profiler::rows();
            let s = renderer.world.stats;
            let st = stream_stats(&self.game).unwrap_or_default();
            let p = self.camera.position;
            let extra = [
                self.loader.loading_text().unwrap_or_else(|| {
                    format!(
                        "streaming: {} chunks ({} full), {} queued, {} generating",
                        st.loaded, st.full, st.queued, st.inflight
                    )
                }),
                format!(
                    "gpu: {} chunks ({} pending), {} bricks, +{} ({:.1} MB), tree {:.0} MB, voxels {:.0} MB",
                    s.chunks,
                    s.pending_chunks,
                    s.bricks_resident,
                    s.bricks_uploaded_last_frame,
                    s.bytes_uploaded_last_frame as f32 / 1e6,
                    s.tree_mb,
                    s.voxel_mb
                ),
                crate::world::physics_line(&self.game, renderer),
                crate::world::structure_line(&self.game),
                crate::world::weather_line(&self.game),
                self.water
                    .as_ref()
                    .map_or_else(String::new, |w| w.line(&self.game)),
                {
                    let clock = self.game.world.resource::<mc2_game::clock::WorldClock>();
                    let hour = clock.hour();
                    format!(
                        "pos {:.1} {:.1} {:.1}  quality {} (F6)  time {:02}:{:02}{} ([ ] T)",
                        p.x,
                        p.y,
                        p.z,
                        self.quality.name(),
                        hour.floor() as u32,
                        (hour.fract() * 60.0).floor() as u32,
                        if clock.paused { " paused" } else { "" }
                    )
                },
                if self.pad_connected {
                    format!(
                        "LS move  RS look  A jump  B crouch  RT break  LT place  LB/RB hotbar  Y fly  X TNT  L3 sprint  R3 view  D-pad mode  Start play  Back pack  P photo  F4 view {}  vsync {}",
                        self.debug_mode,
                        if self.vsync { "on" } else { "off" }
                    )
                } else {
                    format!(
                        "click capture  WASD move  space jump  ctrl crouch  shift sprint  F fly  F5 view  Tab mode  1-9 slot  E light TNT  F2 shot  F3 HUD  F4 view {}  V vsync {}  P photo",
                        self.debug_mode,
                        if self.vsync { "on" } else { "off" }
                    )
                },
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
        let pressed = event.state == ElementState::Pressed;
        if self.photo.active && code != KeyCode::KeyP && code != KeyCode::Escape {
            if !event.repeat {
                self.photo.key(code, pressed);
            }
            if code != KeyCode::F3 {
                return;
            }
        }
        if event.state == ElementState::Released {
            if let Some(k) = map_key(code) {
                self.game.input().key_up(k);
            }
            return;
        }
        if event.repeat {
            return;
        }
        // I opens and closes the inventory; while it is open, Esc closes
        // it and number keys swap the slot under the cursor to the hotbar.
        if code == KeyCode::KeyI || (self.inventory.open && code == KeyCode::Escape) {
            self.toggle_inventory(None);
            return;
        }
        if self.inventory.open
            && let Some(Key::Slot(n)) = map_key(code)
        {
            self.inventory.number(&mut self.game, usize::from(n));
            return;
        }
        if let Some(k) = map_key(code)
            && self.playing()
        {
            self.game.input().key_down(k);
        }
        match code {
            KeyCode::Escape => {
                if self.photo.active {
                    self.toggle_photo();
                } else if self.playing() {
                    self.set_play(false);
                } else {
                    event_loop.exit();
                }
            }
            KeyCode::F2 => self.screenshot_requested = true,
            KeyCode::F3 => self.show_profiler = !self.show_profiler,
            KeyCode::KeyP => self.toggle_photo(),
            KeyCode::F4 => self.debug_mode = (self.debug_mode + 1) % 4,
            KeyCode::F6 => {
                self.quality = self.quality.next();
                let settings = self.quality_settings();
                if let Some(r) = &mut self.running {
                    r.renderer.set_quality(settings);
                }
            }
            KeyCode::KeyV => {
                self.vsync = !self.vsync;
                self.reconfigure();
            }
            KeyCode::KeyT => {
                let mut clock = self.game.clock();
                clock.paused = !clock.paused;
            }
            KeyCode::BracketLeft | KeyCode::BracketRight => {
                let step = if code == KeyCode::BracketLeft {
                    -1.0
                } else {
                    1.0
                };
                // Step through midnight into the next or previous day.
                self.game.clock().days += step / 24.0;
            }
            _ => {}
        }
    }
}

fn apply_photo_pad(photo: &mut crate::photo::PhotoMode, pad: &Pad, dt: f32) {
    let up = f32::from(u8::from(pad.jump)) - f32::from(u8::from(pad.crouch));
    photo.set_analog(pad.move_axis, up, pad.sprint);
    photo.stick_look(pad.look_axis, dt);
    if pad.zoom.abs() > 0.05 {
        photo.zoom(pad.zoom * dt * 4.0);
    }
    if pad.dpad_up {
        photo.bump_focus(-1);
    }
    if pad.dpad_down {
        photo.bump_focus(1);
    }
    if pad.dpad_left {
        photo.bump_exposure(-1);
    }
    if pad.dpad_right {
        photo.bump_exposure(1);
    }
    if pad.lb {
        photo.bump_aperture(-1);
    }
    if pad.rb {
        photo.bump_aperture(1);
    }
    if pad.fly {
        photo.depth_of_field = !photo.depth_of_field;
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
            WindowEvent::Focused(false) => self.set_play(false),
            WindowEvent::MouseInput { state, button, .. } => {
                let mapped = match button {
                    MouseButton::Left => Some(Button::Primary),
                    MouseButton::Right => Some(Button::Secondary),
                    MouseButton::Middle => Some(Button::Middle),
                    _ => None,
                };
                if self.inventory.open {
                    if state == ElementState::Pressed && button != MouseButton::Middle {
                        let secondary = button == MouseButton::Right;
                        let shift = self.modifiers.shift_key();
                        self.inventory.click(&mut self.game, secondary, shift);
                    }
                    return;
                }
                match (state, mapped) {
                    (ElementState::Pressed, _) if !self.playing() => self.set_play(true),
                    (ElementState::Pressed, Some(b)) => self.game.input().button_down(b),
                    (ElementState::Released, Some(b)) => self.game.input().button_up(b),
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                if self.inventory.open {
                    self.inventory.wheel(lines);
                } else if self.photo.active {
                    self.photo.zoom(lines);
                } else {
                    self.game.input().scroll += lines;
                }
            }
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::CursorMoved { position, .. } => {
                self.inventory
                    .mouse_move(position.x as f32, position.y as f32);
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
            if self.photo.active {
                self.photo.look(delta.0 as f32, delta.1 as f32);
            } else {
                self.game.input().mouse_delta += Vec2::new(delta.0 as f32, delta.1 as f32);
            }
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
