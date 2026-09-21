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
    /// Left stick, player space: x right, y forward, already deadzoned.
    pub move_axis: Vec2,
    /// Right stick look, -1..1 after deadzone and a quadratic curve.
    pub look_axis: Vec2,
    /// False while the cursor is free: look and actions are ignored.
    pub captured: bool,
    pad_keys: Vec<Key>,
    pad_pressed: Vec<Key>,
    pad_buttons: Vec<Button>,
    pad_button_pressed: Vec<Button>,
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
        self.held_keys.contains(&k) || self.pad_keys.contains(&k)
    }

    pub fn pressed(&self, k: Key) -> bool {
        self.pressed_keys.contains(&k) || self.pad_pressed.contains(&k)
    }

    pub fn button_held(&self, b: Button) -> bool {
        self.held_buttons.contains(&b) || self.pad_buttons.contains(&b)
    }

    pub fn button_pressed(&self, b: Button) -> bool {
        self.pressed_buttons.contains(&b) || self.pad_button_pressed.contains(&b)
    }

    /// Drops last frame's pad channels. The window rewrites them each poll.
    pub fn clear_pad(&mut self) {
        self.pad_keys.clear();
        self.pad_pressed.clear();
        self.pad_buttons.clear();
        self.pad_button_pressed.clear();
        self.move_axis = Vec2::ZERO;
        self.look_axis = Vec2::ZERO;
    }

    pub fn pad_hold_key(&mut self, k: Key) {
        if !self.pad_keys.contains(&k) {
            self.pad_keys.push(k);
        }
    }

    pub fn pad_press_key(&mut self, k: Key) {
        self.pad_hold_key(k);
        if !self.pad_pressed.contains(&k) {
            self.pad_pressed.push(k);
        }
    }

    pub fn pad_hold_button(&mut self, b: Button) {
        if !self.pad_buttons.contains(&b) {
            self.pad_buttons.push(b);
        }
    }

    pub fn pad_press_button(&mut self, b: Button) {
        self.pad_hold_button(b);
        if !self.pad_button_pressed.contains(&b) {
            self.pad_button_pressed.push(b);
        }
    }

    /// Clears per-frame edges and deltas; call after the frame's systems ran.
    pub fn end_frame(&mut self) {
        self.pressed_keys.clear();
        self.pressed_buttons.clear();
        self.pad_pressed.clear();
        self.pad_button_pressed.clear();
        self.mouse_delta = Vec2::ZERO;
        self.scroll = 0.0;
    }

    pub fn release_all(&mut self) {
        self.held_keys.clear();
        self.held_buttons.clear();
        self.clear_pad();
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

    #[test]
    fn pad_and_keyboard_merge_without_clobbering() {
        let mut i = Input::default();
        i.key_down(Key::Jump);
        i.pad_hold_key(Key::Crouch);
        i.pad_press_key(Key::ToggleFly);
        i.pad_press_button(Button::Primary);
        assert!(i.held(Key::Jump) && i.held(Key::Crouch));
        assert!(i.pressed(Key::ToggleFly));
        assert!(i.button_pressed(Button::Primary) && i.button_held(Button::Primary));
        i.end_frame();
        assert!(i.held(Key::Jump) && i.held(Key::Crouch));
        assert!(!i.pressed(Key::ToggleFly));
        assert!(i.button_held(Button::Primary) && !i.button_pressed(Button::Primary));
        i.clear_pad();
        assert!(i.held(Key::Jump));
        assert!(!i.held(Key::Crouch));
        assert!(!i.button_held(Button::Primary));
    }
}
