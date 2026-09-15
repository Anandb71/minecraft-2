//! Free-flying camera used before the player controller exists, and later
//! by photo mode.

use glam::{DVec3, Vec2};
use mc2_render::camera::Camera;
use winit::keyboard::KeyCode;

#[derive(Default)]
pub struct Input {
    held: Vec<KeyCode>,
    pub mouse_delta: Vec2,
    pub scroll: f32,
}

impl Input {
    pub fn press(&mut self, key: KeyCode) {
        if !self.held.contains(&key) {
            self.held.push(key);
        }
    }

    pub fn release(&mut self, key: KeyCode) {
        self.held.retain(|k| *k != key);
    }

    pub fn held(&self, key: KeyCode) -> bool {
        self.held.contains(&key)
    }

    pub fn clear_frame(&mut self) {
        self.mouse_delta = Vec2::ZERO;
        self.scroll = 0.0;
    }

    pub fn release_all(&mut self) {
        self.held.clear();
    }
}

pub struct FlyCam {
    pub speed: f64,
    pub sensitivity: f32,
}

impl Default for FlyCam {
    fn default() -> Self {
        Self {
            speed: 8.0,
            sensitivity: 0.0022,
        }
    }
}

impl FlyCam {
    pub fn update(&mut self, cam: &mut Camera, input: &Input, dt: f32, look: bool) {
        if look {
            cam.yaw -= input.mouse_delta.x * self.sensitivity;
            cam.pitch = (cam.pitch - input.mouse_delta.y * self.sensitivity).clamp(-1.55, 1.55);
        }
        if input.scroll != 0.0 {
            self.speed = (self.speed * 1.2f64.powf(f64::from(input.scroll))).clamp(0.5, 500.0);
        }
        let f = cam.forward().as_dvec3();
        let r = cam.right().as_dvec3();
        let mut v = DVec3::ZERO;
        let axis = |pos: KeyCode, neg: KeyCode| {
            f64::from(u8::from(input.held(pos))) - f64::from(u8::from(input.held(neg)))
        };
        v += f * axis(KeyCode::KeyW, KeyCode::KeyS);
        v += r * axis(KeyCode::KeyD, KeyCode::KeyA);
        v += DVec3::Y * axis(KeyCode::Space, KeyCode::ControlLeft);
        let boost = if input.held(KeyCode::ShiftLeft) {
            4.0
        } else {
            1.0
        };
        if v.length_squared() > 0.0 {
            cam.position += v.normalize() * self.speed * boost * f64::from(dt);
        }
    }
}
