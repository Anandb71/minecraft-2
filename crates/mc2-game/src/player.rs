//! Player controller: walking with step-up, jumping, crouching, flying.
//!
//! Runs on the fixed 120 Hz clock. The body is an axis-aligned box swept
//! through the voxel grid; small ledges (up to 0.55 m, common on 6.25 cm
//! terrain) are climbed automatically and the camera eases over the step
//! instead of snapping.

use crate::collide::{Aabb, SolidField, VOXEL_M, overlaps_solid, sweep};
use crate::input::{Input, Key, Time};
use crate::{Streaming, Voxels};
use bevy_ecs::prelude::*;
use glam::{DVec3, Vec3};
use mc2_voxel::coords::{self, ChunkPos};

pub const WIDTH: f64 = 0.6;
pub const HEIGHT: f64 = 1.8;
pub const CROUCH_HEIGHT: f64 = 1.45;
pub const EYE: f64 = 1.62;
pub const CROUCH_EYE: f64 = 1.27;
pub const STEP: f64 = 0.55;
const GRAVITY: f64 = 25.0;
const JUMP_SPEED: f64 = 7.6;
const WALK: f64 = 4.3;
const SPRINT: f64 = 6.8;
const CROUCH: f64 = 1.9;
const FLY: f64 = 12.0;
const FLY_SPRINT: f64 = 48.0;
const GROUND_ACCEL: f64 = 60.0;
const AIR_ACCEL: f64 = 12.0;
const MOUSE_SENSITIVITY: f32 = 0.0022;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveMode {
    Walk,
    Fly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    FirstPerson,
    ThirdPerson,
}

#[derive(Component, Debug)]
pub struct Player {
    pub yaw: f32,
    pub pitch: f32,
    pub mode: MoveMode,
    pub view: View,
    pub on_ground: bool,
    pub crouching: bool,
    /// Camera lag after a step-up, metres, decays to zero.
    pub step_smoothing: f64,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            mode: MoveMode::Walk,
            view: View::FirstPerson,
            on_ground: false,
            crouching: false,
            step_smoothing: 0.0,
        }
    }
}

#[derive(Component, Debug, Clone, Copy)]
pub struct Body {
    pub feet: DVec3,
    pub prev_feet: DVec3,
    pub velocity: DVec3,
}

impl Body {
    pub fn at(feet: DVec3) -> Self {
        Self {
            feet,
            prev_feet: feet,
            velocity: DVec3::ZERO,
        }
    }
}

impl Player {
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
    }

    pub fn height(&self) -> f64 {
        if self.crouching {
            CROUCH_HEIGHT
        } else {
            HEIGHT
        }
    }

    pub fn eye(&self) -> f64 {
        if self.crouching { CROUCH_EYE } else { EYE }
    }
}

/// One fixed step of movement. Pure, for tests and for the system below.
pub fn step_player(field: &impl SolidField, p: &mut Player, b: &mut Body, input: &Input, dt: f64) {
    b.prev_feet = b.feet;
    let yaw = f64::from(p.yaw);
    let forward = DVec3::new(yaw.sin(), 0.0, yaw.cos());
    let right = DVec3::new(-yaw.cos(), 0.0, yaw.sin());
    let axis = |pos: Key, neg: Key| {
        f64::from(u8::from(input.held(pos))) - f64::from(u8::from(input.held(neg)))
    };
    let mut wish = forward * axis(Key::Forward, Key::Back) + right * axis(Key::Right, Key::Left);
    if wish.length_squared() > 1.0 {
        wish = wish.normalize();
    }
    let sprint = input.held(Key::Sprint);

    match p.mode {
        MoveMode::Fly => {
            let vertical = axis(Key::Jump, Key::Crouch);
            let speed = if sprint { FLY_SPRINT } else { FLY };
            let target = (wish + DVec3::Y * vertical) * speed;
            b.velocity += (target - b.velocity) * (dt * 10.0).min(1.0);
            p.crouching = false;
        }
        MoveMode::Walk => {
            // Crouch only shrinks when there is room, and only stands when
            // the head has room too.
            let want_crouch = input.held(Key::Crouch);
            if !want_crouch && p.crouching {
                let tall = Aabb::standing(b.feet, WIDTH, HEIGHT);
                if !overlaps_solid(field, &tall) {
                    p.crouching = false;
                }
            } else if want_crouch {
                p.crouching = true;
            }
            let speed = if p.crouching {
                CROUCH
            } else if sprint {
                SPRINT
            } else {
                WALK
            };
            let target = wish * speed;
            let accel = if p.on_ground { GROUND_ACCEL } else { AIR_ACCEL };
            let mut horizontal = DVec3::new(b.velocity.x, 0.0, b.velocity.z);
            let diff = target - horizontal;
            let max = accel * dt;
            horizontal += if diff.length() > max {
                diff.normalize() * max
            } else {
                diff
            };
            b.velocity.x = horizontal.x;
            b.velocity.z = horizontal.z;
            b.velocity.y -= GRAVITY * dt;
            if p.on_ground && input.held(Key::Jump) {
                b.velocity.y = JUMP_SPEED;
                p.on_ground = false;
            }
        }
    }

    let height = p.height();
    let start = Aabb::standing(b.feet, WIDTH, height);
    let delta = b.velocity * dt;
    let (mut end, mut hit) = sweep(field, start, delta);

    // Step up small ledges when walking into them on the ground.
    if p.mode == MoveMode::Walk && p.on_ground && (hit.hit.x != 0 || hit.hit.z != 0) {
        let (raised, up) = sweep(field, start, DVec3::new(0.0, STEP, 0.0));
        let (across, _) = sweep(field, raised, DVec3::new(delta.x, 0.0, delta.z));
        let (down, _) = sweep(field, across, DVec3::new(0.0, -(up.moved.y + VOXEL_M), 0.0));
        let gained = (down.min - start.min).truncate_xz_len();
        let plain = (end.min - start.min).truncate_xz_len();
        if gained > plain + 1e-4 && down.min.y > start.min.y {
            p.step_smoothing += down.min.y - start.min.y;
            end = down;
            hit.hit.x = 0;
            hit.hit.z = 0;
            hit.hit.y = -1;
        }
    }

    b.feet = end.feet();
    for a in 0..3 {
        if hit.hit[a] != 0 {
            b.velocity[a] = 0.0;
        }
    }
    if p.mode == MoveMode::Walk {
        let probe = Aabb::standing(b.feet - DVec3::Y * (VOXEL_M * 0.5), WIDTH, height);
        p.on_ground = hit.hit.y == -1 || (b.velocity.y <= 0.0 && overlaps_solid(field, &probe));
    } else {
        p.on_ground = false;
    }
    p.step_smoothing *= (1.0 - dt * 14.0).max(0.0);
}

trait XzLen {
    fn truncate_xz_len(self) -> f64;
}

impl XzLen for DVec3 {
    fn truncate_xz_len(self) -> f64 {
        (self.x * self.x + self.z * self.z).sqrt()
    }
}

/// Mouse look, every frame.
pub fn look(input: Res<Input>, mut players: Query<&mut Player>) {
    if !input.captured {
        return;
    }
    for mut p in &mut players {
        p.yaw -= input.mouse_delta.x * MOUSE_SENSITIVITY;
        p.pitch = (p.pitch - input.mouse_delta.y * MOUSE_SENSITIVITY).clamp(-1.55, 1.55);
    }
}

/// Mode toggles, every frame.
pub fn toggles(input: Res<Input>, mut players: Query<(&mut Player, &mut Body)>) {
    for (mut p, mut b) in &mut players {
        if input.pressed(Key::ToggleFly) {
            p.mode = match p.mode {
                MoveMode::Walk => MoveMode::Fly,
                MoveMode::Fly => MoveMode::Walk,
            };
            b.velocity = DVec3::ZERO;
        }
        if input.pressed(Key::ToggleView) {
            p.view = match p.view {
                View::FirstPerson => View::ThirdPerson,
                View::ThirdPerson => View::FirstPerson,
            };
        }
    }
}

/// Movement, on the fixed clock.
pub fn movement(
    voxels: Res<Voxels>,
    streaming: Res<Streaming>,
    input: Res<Input>,
    time: Res<Time>,
    mut players: Query<(&mut Player, &mut Body)>,
) {
    for (mut p, mut b) in &mut players {
        // Hold still over terrain that has not streamed in yet.
        let feet_voxel = coords::metres_to_voxel(b.feet);
        let below = ChunkPos::of_voxel(feet_voxel - glam::IVec3::Y * 16);
        if let Some(s) = &streaming.0
            && s.lod_of(below).is_none()
            && s.lod_of(ChunkPos::of_voxel(feet_voxel)).is_none()
            && p.mode == MoveMode::Walk
        {
            b.prev_feet = b.feet;
            b.velocity = DVec3::ZERO;
            continue;
        }
        step_player(&voxels.0, &mut p, &mut b, &input, time.fixed_dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;
    use mc2_voxel::world::VoxelWorld;

    fn arena() -> VoxelWorld {
        let mut w = VoxelWorld::new();
        // Floor top at y = 10 m, a 0.5 m ledge from x = 4 m, a 1.5 m wall from x = 8 m.
        w.fill_box(
            glam::IVec3::new(0, 0, 0),
            glam::IVec3::new(255, 159, 63),
            ids::GRANITE,
        );
        w.fill_box(
            glam::IVec3::new(64, 160, 0),
            glam::IVec3::new(255, 167, 63),
            ids::GRANITE,
        );
        w.fill_box(
            glam::IVec3::new(128, 168, 0),
            glam::IVec3::new(135, 191, 63),
            ids::GRANITE,
        );
        w
    }

    fn run(w: &VoxelWorld, p: &mut Player, b: &mut Body, input: &Input, seconds: f64) {
        for _ in 0..(seconds * 120.0) as usize {
            step_player(w, p, b, input, 1.0 / 120.0);
        }
    }

    #[test]
    fn falls_lands_and_stands() {
        let w = arena();
        let mut p = Player::default();
        let mut b = Body::at(DVec3::new(1.0, 14.0, 2.0));
        run(&w, &mut p, &mut b, &Input::default(), 2.0);
        assert!(p.on_ground);
        assert!((b.feet.y - 10.0).abs() < 1e-6, "feet {}", b.feet.y);
    }

    #[test]
    fn walks_up_a_ledge_but_not_a_wall() {
        let w = arena();
        let mut p = Player {
            yaw: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        };
        let mut b = Body::at(DVec3::new(2.0, 10.0, 2.0));
        let mut input = Input::default();
        run(&w, &mut p, &mut b, &input, 0.5);
        input.key_down(Key::Forward);
        run(&w, &mut p, &mut b, &input, 3.0);
        assert!(
            (b.feet.y - 10.5).abs() < 1e-6,
            "climbed the 0.5 m ledge: feet {}",
            b.feet.y
        );
        assert!(
            b.feet.x < 8.0 - WIDTH * 0.5 + 1e-6 && b.feet.x > 7.0,
            "stopped at the wall: x {}",
            b.feet.x
        );
    }

    #[test]
    fn jump_clears_the_wall_top_when_high_enough() {
        let w = arena();
        let mut p = Player::default();
        let mut b = Body::at(DVec3::new(2.0, 10.0, 2.0));
        let mut input = Input::default();
        run(&w, &mut p, &mut b, &input, 0.5);
        input.key_down(Key::Jump);
        let mut peak: f64 = 0.0;
        for _ in 0..120 {
            step_player(&w, &mut p, &mut b, &input, 1.0 / 120.0);
            peak = peak.max(b.feet.y);
            input.key_up(Key::Jump);
        }
        assert!(peak > 11.0 && peak < 11.4, "jump apex {peak}");
    }

    #[test]
    fn crouch_keeps_crouched_under_a_low_ceiling() {
        let mut w = arena();
        // Ceiling at 11.5 m over x in 0..2 m.
        w.fill_box(
            glam::IVec3::new(0, 184, 0),
            glam::IVec3::new(31, 200, 63),
            ids::GRANITE,
        );
        let mut p = Player::default();
        let mut b = Body::at(DVec3::new(1.0, 10.0, 2.0));
        let mut input = Input::default();
        input.key_down(Key::Crouch);
        run(&w, &mut p, &mut b, &input, 0.2);
        assert!(p.crouching);
        input.key_up(Key::Crouch);
        run(&w, &mut p, &mut b, &input, 0.2);
        assert!(p.crouching, "no headroom to stand");
    }
}
