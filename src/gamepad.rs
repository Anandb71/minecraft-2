//! Xbox-layout gamepad. Polled each frame; the window writes it into Input.

use gilrs::{Axis, Button, EventType, Gamepad, Gilrs};
use glam::Vec2;

const DEADZONE: f32 = 0.18;
const TRIGGER: f32 = 0.45;

#[derive(Clone, Copy, Default)]
struct Digital {
    jump: bool,
    crouch: bool,
    sprint: bool,
    fly: bool,
    view: bool,
    mode: bool,
    interact: bool,
    break_b: bool,
    place: bool,
    preview: bool,
    lb: bool,
    rb: bool,
    start: bool,
    select: bool,
    up: bool,
    down: bool,
    left: bool,
    right: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Pad {
    pub connected: bool,
    pub move_axis: Vec2,
    pub look_axis: Vec2,
    /// Right trigger minus left, for photo zoom.
    pub zoom: f32,
    pub jump: bool,
    pub jump_pressed: bool,
    pub crouch: bool,
    pub crouch_pressed: bool,
    pub sprint: bool,
    pub fly: bool,
    pub view: bool,
    pub mode: bool,
    pub interact: bool,
    pub break_block: bool,
    pub break_pressed: bool,
    pub place: bool,
    pub place_pressed: bool,
    pub preview: bool,
    pub slot_next: bool,
    pub slot_prev: bool,
    pub start: bool,
    pub select: bool,
    pub dpad_up: bool,
    pub dpad_down: bool,
    pub dpad_left: bool,
    pub dpad_right: bool,
    pub lb: bool,
    pub rb: bool,
    pub stick_active: bool,
    pub any_edge: bool,
}

pub struct Gamepads {
    gilrs: Option<Gilrs>,
    prev: Digital,
    named: bool,
}

impl Gamepads {
    pub fn new() -> Self {
        match Gilrs::new() {
            Ok(gilrs) => {
                let n = gilrs.gamepads().filter(|(_, g)| g.is_connected()).count();
                if n > 0 {
                    log::info!("gamepad: {n} connected");
                }
                Self {
                    gilrs: Some(gilrs),
                    prev: Digital::default(),
                    named: false,
                }
            }
            Err(e) => {
                log::warn!("gamepad disabled: {e}");
                Self {
                    gilrs: None,
                    prev: Digital::default(),
                    named: false,
                }
            }
        }
    }

    pub fn poll(&mut self) -> Pad {
        let Some(gilrs) = self.gilrs.as_mut() else {
            return Pad::default();
        };
        while let Some(ev) = gilrs.next_event() {
            match ev.event {
                EventType::Connected => log::info!("gamepad connected"),
                EventType::Disconnected => log::info!("gamepad disconnected"),
                _ => {}
            }
        }
        let prev = self.prev;
        let found = gilrs
            .gamepads()
            .find(|(_, g)| g.is_connected())
            .map(|(_, gp)| {
                let name = gp.name().to_string();
                let now = Digital::read(&gp);
                (name, now, snapshot(&gp, now, prev))
            });
        let Some((name, now, pad)) = found else {
            self.prev = Digital::default();
            self.named = false;
            return Pad::default();
        };
        self.prev = now;
        if !self.named {
            log::info!("gamepad: {name}");
            self.named = true;
        }
        pad
    }
}

impl Digital {
    fn read(gp: &Gamepad<'_>) -> Self {
        Self {
            jump: gp.is_pressed(Button::South),
            crouch: gp.is_pressed(Button::East),
            sprint: gp.is_pressed(Button::LeftThumb),
            fly: gp.is_pressed(Button::North),
            view: gp.is_pressed(Button::RightThumb),
            mode: gp.is_pressed(Button::DPadUp),
            interact: gp.is_pressed(Button::West),
            break_b: trigger(gp, Button::RightTrigger2),
            place: trigger(gp, Button::LeftTrigger2),
            preview: gp.is_pressed(Button::DPadDown),
            lb: gp.is_pressed(Button::LeftTrigger),
            rb: gp.is_pressed(Button::RightTrigger),
            start: gp.is_pressed(Button::Start),
            select: gp.is_pressed(Button::Select),
            up: gp.is_pressed(Button::DPadUp),
            down: gp.is_pressed(Button::DPadDown),
            left: gp.is_pressed(Button::DPadLeft),
            right: gp.is_pressed(Button::DPadRight),
        }
    }
}

fn trigger(gp: &Gamepad<'_>, button: Button) -> bool {
    gp.is_pressed(button) || gp.button_data(button).is_some_and(|d| d.value() > TRIGGER)
}

fn trigger_value(gp: &Gamepad<'_>, button: Button) -> f32 {
    gp.button_data(button).map(|d| d.value()).unwrap_or(0.0)
}

fn edge(now: bool, was: bool) -> bool {
    now && !was
}

fn snapshot(gp: &Gamepad<'_>, now: Digital, prev: Digital) -> Pad {
    let move_axis = radial(gp.value(Axis::LeftStickX), gp.value(Axis::LeftStickY));
    // Stick up is positive Y; invert so look_axis uses the same sign as mouse.
    let look = radial(gp.value(Axis::RightStickX), gp.value(Axis::RightStickY));
    let look_axis = Vec2::new(look_curve(look.x), -look_curve(look.y));
    let zoom = trigger_value(gp, Button::RightTrigger2) - trigger_value(gp, Button::LeftTrigger2);
    let dpad_up = edge(now.up, prev.up);
    let dpad_down = edge(now.down, prev.down);
    let dpad_left = edge(now.left, prev.left);
    let dpad_right = edge(now.right, prev.right);
    let lb = edge(now.lb, prev.lb);
    let rb = edge(now.rb, prev.rb);
    let fly = edge(now.fly, prev.fly);
    let view = edge(now.view, prev.view);
    let mode = edge(now.mode, prev.mode);
    let interact = edge(now.interact, prev.interact);
    let start = edge(now.start, prev.start);
    let select = edge(now.select, prev.select);
    let break_edge = edge(now.break_b, prev.break_b);
    let place_edge = edge(now.place, prev.place);
    Pad {
        connected: true,
        move_axis,
        look_axis,
        zoom,
        jump: now.jump,
        jump_pressed: edge(now.jump, prev.jump),
        crouch: now.crouch,
        crouch_pressed: edge(now.crouch, prev.crouch),
        sprint: now.sprint,
        fly,
        view,
        mode,
        interact,
        break_block: now.break_b,
        break_pressed: break_edge,
        place: now.place,
        place_pressed: place_edge,
        preview: now.preview,
        slot_next: rb || dpad_right,
        slot_prev: lb || dpad_left,
        start,
        select,
        dpad_up,
        dpad_down,
        dpad_left,
        dpad_right,
        lb,
        rb,
        stick_active: move_axis.length_squared() > 0.0 || look_axis.length_squared() > 0.0,
        any_edge: break_edge
            || place_edge
            || fly
            || view
            || mode
            || interact
            || select
            || lb
            || rb
            || dpad_up
            || dpad_down
            || dpad_left
            || dpad_right
            || edge(now.jump, prev.jump)
            || edge(now.crouch, prev.crouch)
            || edge(now.sprint, prev.sprint),
    }
}

/// Radial deadzone, then rescale so the ring maps to 0..1.
pub(crate) fn radial(x: f32, y: f32) -> Vec2 {
    let v = Vec2::new(x, y);
    let m = v.length();
    if m < DEADZONE {
        return Vec2::ZERO;
    }
    v.normalize_or_zero() * ((m - DEADZONE) / (1.0 - DEADZONE)).min(1.0)
}

/// Quadratic on an already-deadzoned axis, so small deflections stay precise.
pub(crate) fn look_curve(v: f32) -> f32 {
    v * v.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadzone_swallows_noise_and_rescales() {
        assert_eq!(radial(0.05, 0.0), Vec2::ZERO);
        let full = radial(1.0, 0.0);
        assert!((full.x - 1.0).abs() < 1e-5 && full.y.abs() < 1e-5);
        let mid = radial(DEADZONE + (1.0 - DEADZONE) * 0.5, 0.0);
        assert!((mid.x - 0.5).abs() < 1e-5);
    }

    #[test]
    fn look_curve_is_soft_in_the_middle() {
        assert!((look_curve(0.5) - 0.25).abs() < 1e-6);
        assert_eq!(look_curve(1.0), 1.0);
        assert_eq!(look_curve(-1.0), -1.0);
    }
}
