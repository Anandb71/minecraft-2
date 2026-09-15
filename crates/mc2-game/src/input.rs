//! Window-system independent input state, filled by the app each frame.

use bevy_ecs::prelude::*;
use glam::Vec2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Forward,
    Back,
    Left,
    Right,
    Jump,
    Crouch,
    Sprint,
    ToggleFly,
    ToggleView,
    ToggleMode,
    Slot(u8),
    Interact,
    Photo,
    Screenshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Button {
    Primary,
    Secondary,
    Middle,
}

#[derive(Resource, Default, Debug)]
pub struct Input {
    held_keys: Vec<Key>,
    pressed_keys: Vec<Key>,
    held_buttons: Vec<Button>,
    pressed_buttons: Vec<Button>,
    pub mouse_delta: Vec2,
    pub scroll: f32,
    /// False while the cursor is free: look and actions are ignored.
    pub captured: bool,
}

impl Input {
    pub fn key_down(&mut self, k: Key) {
        if !self.held_keys.contains(&k) {
            self.held_keys.push(k);
            self.pressed_keys.push(k);
        }
    }

    pub fn key_up(&mut self, k: Key) {
        self.held_keys.retain(|h| *h != k);
    }

    pub fn button_down(&mut self, b: Button) {
        if !self.held_buttons.contains(&b) {
            self.held_buttons.push(b);
            self.pressed_buttons.push(b);
        }
    }

    pub fn button_up(&mut self, b: Button) {
        self.held_buttons.retain(|h| *h != b);
    }

    pub fn held(&self, k: Key) -> bool {
        self.held_keys.contains(&k)
    }

    pub fn pressed(&self, k: Key) -> bool {
        self.pressed_keys.contains(&k)
    }

    pub fn button_held(&self, b: Button) -> bool {
        self.held_buttons.contains(&b)
    }

    pub fn button_pressed(&self, b: Button) -> bool {
        self.pressed_buttons.contains(&b)
    }

    /// Clears per-frame edges and deltas; call after the frame's systems ran.
    pub fn end_frame(&mut self) {
        self.pressed_keys.clear();
        self.pressed_buttons.clear();
        self.mouse_delta = Vec2::ZERO;
        self.scroll = 0.0;
    }

    pub fn release_all(&mut self) {
        self.held_keys.clear();
        self.held_buttons.clear();
        self.end_frame();
    }
}

/// Wall time and the fixed simulation clock.
#[derive(Resource, Debug)]
pub struct Time {
    pub dt: f32,
    pub elapsed: f64,
    pub fixed_dt: f64,
    accumulator: f64,
    pub ticks: u64,
    /// Fraction of a fixed step between the last tick and now, for rendering.
    pub alpha: f64,
}

impl Default for Time {
    fn default() -> Self {
        Self {
            dt: 0.0,
            elapsed: 0.0,
            fixed_dt: 1.0 / 120.0,
            accumulator: 0.0,
            ticks: 0,
            alpha: 0.0,
        }
    }
}

impl Time {
    /// Advances wall time; returns how many fixed ticks are due (capped so a
    /// stall does not spiral).
    pub fn advance(&mut self, dt: f32) -> u32 {
        self.dt = dt;
        self.elapsed += f64::from(dt);
        self.accumulator = (self.accumulator + f64::from(dt)).min(self.fixed_dt * 8.0);
        let mut n = 0;
        while self.accumulator >= self.fixed_dt {
            self.accumulator -= self.fixed_dt;
            self.ticks += 1;
            n += 1;
        }
        self.alpha = self.accumulator / self.fixed_dt;
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_last_one_frame() {
        let mut i = Input::default();
        i.key_down(Key::Jump);
        i.key_down(Key::Jump);
        assert!(i.pressed(Key::Jump) && i.held(Key::Jump));
        i.end_frame();
        assert!(!i.pressed(Key::Jump) && i.held(Key::Jump));
        i.key_up(Key::Jump);
        assert!(!i.held(Key::Jump));
    }

    #[test]
    fn fixed_clock_accumulates_and_caps() {
        let mut t = Time::default();
        assert_eq!(t.advance(1.0 / 60.0), 2);
        assert_eq!(t.advance(0.004), 0);
        assert!(t.alpha > 0.4 && t.alpha < 0.5);
        assert_eq!(t.advance(5.0), 8, "a long stall runs at most 8 ticks");
    }
}
