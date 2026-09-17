//! Photo mode: time stops, the camera detaches and flies freely, and the
//! lens (focal length, aperture, focus, exposure) is adjustable. A capture
//! renders the frozen scene at the highest quality tier until temporal
//! accumulation settles and writes a PNG.

use mc2_gpu::Gpu;
use mc2_render::Renderer;
use mc2_render::camera::Camera;
use mc2_render::post::DepthOfField;
use mc2_render::quality::Preset;
use std::path::{Path, PathBuf};
use winit::keyboard::KeyCode;

/// Full-frame sensor height, mm.
const SENSOR_MM: f32 = 24.0;
/// Frames rendered for a capture: enough for every temporal history
/// (visibility, SVGF, clouds, fog, upsampling) to converge.
pub const CAPTURE_FRAMES: u32 = 96;

pub struct PhotoMode {
    pub active: bool,
    pub camera: Camera,
    pub focal_mm: f32,
    pub f_number: f32,
    pub focus_m: f32,
    pub exposure_ev: f32,
    pub depth_of_field: bool,
    /// Held movement keys: forward, back, left, right, up, down.
    held: [bool; 6],
    clock_was_paused: bool,
    capture_requested: bool,
}

impl Default for PhotoMode {
    fn default() -> Self {
        Self {
            active: false,
            camera: Camera::default(),
            focal_mm: 35.0,
            f_number: 2.8,
            focus_m: 10.0,
            exposure_ev: 0.0,
            depth_of_field: true,
            held: [false; 6],
            clock_was_paused: false,
            capture_requested: false,
        }
    }
}

/// Vertical field of view of a lens on the full-frame sensor.
pub fn fov_for_focal(focal_mm: f32) -> f32 {
    2.0 * (SENSOR_MM * 0.5 / focal_mm).atan()
}

impl PhotoMode {
    /// Enters photo mode at `camera`; returns the clock pause state to set.
    pub fn enter(&mut self, camera: Camera, clock_paused: bool) {
        self.active = true;
        self.camera = camera;
        self.clock_was_paused = clock_paused;
        self.held = [false; 6];
    }

    /// Leaves photo mode; returns the clock pause state to restore.
    pub fn exit(&mut self) -> bool {
        self.active = false;
        self.clock_was_paused
    }

    /// Handles a key while active; returns whether it was consumed.
    pub fn key(&mut self, code: KeyCode, pressed: bool) -> bool {
        let movement = match code {
            KeyCode::KeyW => Some(0),
            KeyCode::KeyS => Some(1),
            KeyCode::KeyA => Some(2),
            KeyCode::KeyD => Some(3),
            KeyCode::Space => Some(4),
            KeyCode::ControlLeft | KeyCode::KeyC => Some(5),
            _ => None,
        };
        if let Some(i) = movement {
            self.held[i] = pressed;
            return true;
        }
        if !pressed {
            return false;
        }
        match code {
            KeyCode::KeyQ => self.focus_m = (self.focus_m / 1.15).max(0.3),
            KeyCode::KeyE => self.focus_m = (self.focus_m * 1.15).min(2000.0),
            KeyCode::KeyZ => self.f_number = next_stop(self.f_number, -1),
            KeyCode::KeyX => self.f_number = next_stop(self.f_number, 1),
            KeyCode::Comma => self.exposure_ev -= 1.0 / 3.0,
            KeyCode::Period => self.exposure_ev += 1.0 / 3.0,
            KeyCode::KeyG => self.depth_of_field = !self.depth_of_field,
            KeyCode::Enter | KeyCode::F2 => self.capture_requested = true,
            _ => return false,
        }
        true
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        // Slower with longer lenses, so framing stays precise when zoomed.
        let scale = 0.0025 * 35.0 / self.focal_mm;
        self.camera.yaw -= dx * scale;
        self.camera.pitch = (self.camera.pitch - dy * scale).clamp(-1.55, 1.55);
    }

    /// Mouse wheel zooms: focal length steps of about 10%.
    pub fn zoom(&mut self, steps: f32) {
        self.focal_mm = (self.focal_mm * 1.1f32.powf(steps)).clamp(12.0, 400.0);
    }

    pub fn update(&mut self, dt: f32) {
        let f = self.camera.forward();
        let flat = glam::Vec3::new(f.x, 0.0, f.z).normalize_or_zero();
        let right = self.camera.right();
        let mut v = glam::Vec3::ZERO;
        let axes = [flat, -flat, -right, right, glam::Vec3::Y, -glam::Vec3::Y];
        for (held, axis) in self.held.iter().zip(axes) {
            if *held {
                v += axis;
            }
        }
        self.camera.position += (v.normalize_or_zero() * 4.0 * dt).as_dvec3();
        self.camera.fov_y = fov_for_focal(self.focal_mm);
    }

    /// Lens settings for the renderer this frame.
    pub fn apply(&self, renderer: &mut Renderer) {
        renderer.frame.post.dof = self.depth_of_field.then_some(DepthOfField {
            focus_m: self.focus_m,
            focal_mm: self.focal_mm,
            f_number: self.f_number,
        });
        renderer.frame.exposure = 2f32.powf(self.exposure_ev);
    }

    pub fn take_capture_request(&mut self) -> bool {
        std::mem::take(&mut self.capture_requested)
    }

    pub fn status(&self) -> String {
        format!(
            "PHOTO  {:.0} mm  f/{:.1}  focus {:.1} m  {:+.1} EV  DoF {}  |  WASD/space/ctrl fly  wheel zoom  Q/E focus  Z/X aperture  , . exposure  G DoF  Enter capture  P exit",
            self.focal_mm,
            self.f_number,
            self.focus_m,
            self.exposure_ev,
            if self.depth_of_field { "on" } else { "off" }
        )
    }
}

/// Full stops between f/1.4 and f/22.
fn next_stop(f: f32, dir: i32) -> f32 {
    const STOPS: [f32; 9] = [1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0];
    let i = STOPS.iter().position(|s| (s - f).abs() < 0.05).unwrap_or(2) as i32;
    STOPS[(i + dir).clamp(0, STOPS.len() as i32 - 1) as usize]
}

/// Renders the current scene offscreen at the top quality tier and saves it.
pub fn capture(
    gpu: &Gpu,
    renderer: &mut Renderer,
    world: &mut mc2_voxel::world::VoxelWorld,
    camera: &Camera,
    dir: &Path,
) -> Result<PathBuf, String> {
    let quality = renderer.quality();
    let mut top = Preset::SuperUltraCrazyDuperRealistic.settings();
    top.gi = quality.gi;
    renderer.set_quality(top);
    let shutter = renderer.frame.post.shutter;
    // A still frame: no motion blur.
    renderer.frame.post.shutter = 0.0;
    let result = render_to_png(gpu, renderer, world, camera, dir, CAPTURE_FRAMES, "photo");
    renderer.frame.post.shutter = shutter;
    renderer.set_quality(quality);
    result
}

/// Saves the scene as the player sees it, at the current quality; the
/// temporal histories are already settled, so one frame is enough.
pub fn screenshot(
    gpu: &Gpu,
    renderer: &mut Renderer,
    world: &mut mc2_voxel::world::VoxelWorld,
    camera: &Camera,
    dir: &Path,
) -> Result<PathBuf, String> {
    render_to_png(gpu, renderer, world, camera, dir, 1, "screenshot")
}

/// Renders `frames` frames of the scene into an offscreen target without the
/// HUD and writes the last one to `dir/<prefix>_<unix time>.png`.
fn render_to_png(
    gpu: &Gpu,
    renderer: &mut Renderer,
    world: &mut mc2_voxel::world::VoxelWorld,
    camera: &Camera,
    dir: &Path,
    frames: u32,
    prefix: &str,
) -> Result<PathBuf, String> {
    let size = renderer.output_size();
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("capture target"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: renderer.output_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let hud = std::mem::take(&mut renderer.frame.hud);
    let mut rendered = Ok(());
    for _ in 0..frames {
        renderer.prepare(gpu, world, camera, 1.0 / 60.0);
        renderer.render(gpu, target.clone());
        if let Err(e) = gpu.device.poll(wgpu::PollType::wait_indefinitely()) {
            rendered = Err(e.to_string());
            break;
        }
    }
    renderer.frame.hud = hud;
    rendered?;
    let rgba = mc2_gpu::capture::read_rgba8(&gpu.device, &gpu.queue, &target);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("{prefix}_{stamp}.png"));
    mc2_gpu::capture::save_png(&path, size.0, size.1, &rgba).map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lens_controls_stay_in_range() {
        assert!((fov_for_focal(35.0).to_degrees() - 37.8).abs() < 0.1);
        assert_eq!(next_stop(2.8, 1), 4.0);
        assert_eq!(next_stop(22.0, 1), 22.0);
        assert_eq!(next_stop(1.4, -1), 1.4);
        let mut p = PhotoMode::default();
        p.zoom(100.0);
        assert_eq!(p.focal_mm, 400.0);
        assert!(p.key(KeyCode::KeyQ, true));
        assert!(!p.key(KeyCode::KeyL, true));
    }
}
