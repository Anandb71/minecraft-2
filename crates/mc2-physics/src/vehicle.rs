//! Wheeled vehicles: a rigid body carried on raycast wheels.
//!
//! Each wheel is a ray cast down the body's up axis from a mount on the
//! body. Where it meets the ground, a spring and damper push the body up at
//! that point, and the tyre pushes back against sliding sideways, drives
//! along the way it is turned, and brakes. The forces go in as velocity
//! impulses before each substep's integration, so the body's own contacts,
//! joints and sleeping carry on as for any other body. The wheels
//! themselves never touch the ground: the springs hold the body just above
//! the height at which they would.

use crate::body::{Body, BodyId};
use crate::collision::{CollisionWindow, SolidCells};
use glam::{DVec3, Vec3};

/// Ray march step along a wheel's ray, metres.
const RAY_STEP: f64 = 1.0 / 64.0;
/// How far below its resting reach a wheel still finds the ground, metres.
const DROOP: f32 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wheel {
    /// Where the suspension meets the body, principal frame, metres.
    pub mount: Vec3,
    /// Steered by the front axle's angle.
    pub steered: bool,
    /// Turned by the engine.
    pub driven: bool,
}

/// How a vehicle handles, per whole vehicle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Handling {
    /// From a wheel's mount to the ground at rest, metres.
    pub reach: f32,
    /// Spring and damper per wheel, N/m and N s/m.
    pub spring: f32,
    pub damper: f32,
    /// Engine force at full throttle, and braking force, N.
    pub drive: f32,
    pub brake: f32,
    /// Furthest the steered wheels turn, radians.
    pub max_steer: f32,
    /// Tyre friction: the most a tyre can push, as a share of its load.
    pub grip: f32,
}

impl Handling {
    /// Handling for a body of `mass` kg on `wheels` wheels that sits
    /// `sag` metres down on its springs at rest.
    pub fn for_mass(mass: f32, wheels: usize, reach: f32) -> Handling {
        let load = mass * 9.81 / wheels as f32;
        let sag = 0.08;
        let spring = load / sag;
        // A little under critical damping per corner.
        let damper = 0.5 * 2.0 * (spring * mass / wheels as f32).sqrt();
        Handling {
            reach: reach + sag,
            spring,
            damper,
            drive: mass * 4.5,
            brake: mass * 9.0,
            max_steer: 0.55,
            grip: 1.1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vehicle {
    pub body: BodyId,
    pub wheels: Vec<Wheel>,
    /// The body's up and forward axes, principal frame.
    pub up: Vec3,
    pub forward: Vec3,
    pub handling: Handling,
    /// Controls, each -1..1 (brake 0..1).
    pub throttle: f32,
    pub steer: f32,
    pub brake: f32,
}

impl Vehicle {
    pub fn controlled(&self) -> bool {
        self.throttle != 0.0 || self.brake != 0.0
    }
}

/// Where a ray from `from` down `dir` first enters a solid voxel of the
/// window within `max` metres.
fn ground<S: SolidCells>(
    window: &CollisionWindow<S>,
    from: DVec3,
    dir: DVec3,
    max: f64,
) -> Option<f64> {
    let mut t = 0.0;
    while t <= max {
        let v = ((from + dir * t) * 16.0).floor().as_ivec3();
        if window.solid(v) {
            return Some(t);
        }
        t += RAY_STEP;
    }
    None
}

/// One substep of `h` seconds of a vehicle's wheels acting on its body.
pub(crate) fn drive<S: SolidCells>(
    v: &Vehicle,
    b: &mut Body,
    window: Option<&CollisionWindow<S>>,
    h: f32,
) {
    let Some(window) = window else { return };
    if b.asleep {
        return;
    }
    let hd = v.handling;
    let up = b.rot * v.up;
    let down = -up.as_dvec3();
    let driven = v.wheels.iter().filter(|w| w.driven).count().max(1) as f32;
    let steer = v.steer.clamp(-1.0, 1.0) * hd.max_steer;
    for w in &v.wheels {
        let mount = b.world_point(w.mount);
        let Some(t) = ground(window, mount, down, f64::from(hd.reach + DROOP)) else {
            continue;
        };
        let contact = mount + down * t;
        let r = (contact - b.pos).as_vec3();
        let vel = b.point_velocity(r);
        // The spring and damper, pushing only.
        let squeeze = hd.reach - t as f32;
        let load = (hd.spring * squeeze - hd.damper * vel.dot(up)).max(0.0);
        if load == 0.0 {
            continue;
        }
        b.apply_velocity_impulse(r, up * (load * h), 1.0);
        // The tyre: its heading turned by the steering, flat to the body.
        let yaw = if w.steered { steer } else { 0.0 };
        let heading = glam::Quat::from_axis_angle(up, yaw) * (b.rot * v.forward);
        let side = up.cross(heading).normalize_or_zero();
        let limit = hd.grip * load * h;
        let vel = b.point_velocity(r);
        // Sideways: stop the slide, as far as the tyre can.
        let w_side = b.generalized_inverse_mass(r, side);
        let lateral = (-vel.dot(side) / w_side).clamp(-limit, limit);
        // Along the way it points: engine, brake, and the grip left.
        let along = vel.dot(heading);
        let mut push = if w.driven {
            v.throttle.clamp(-1.0, 1.0) * hd.drive / driven * h
        } else {
            0.0
        };
        if v.brake > 0.0 {
            let w_fwd = b.generalized_inverse_mass(r, heading);
            let stop = -along / w_fwd;
            let most = v.brake.min(1.0) * hd.brake / v.wheels.len() as f32 * h;
            push += stop.clamp(-most, most);
        }
        let room = (limit * limit - lateral * lateral).max(0.0).sqrt();
        let push = push.clamp(-room, room);
        b.apply_velocity_impulse(r, side * lateral + heading * push, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{BodyShape, VOXEL_M};
    use crate::world::PhysicsWorld;
    use glam::{IVec3, Quat};
    use mc2_voxel::material::ids;
    use mc2_voxel::world::VoxelWorld;
    use std::sync::Arc;

    fn ground() -> VoxelWorld {
        let mut w = VoxelWorld::new();
        // A 64 m square of granite, its top at 2 m.
        w.fill_box(
            IVec3::new(0, 0, 0),
            IVec3::new(1023, 31, 1023),
            ids::GRANITE,
        );
        w
    }

    /// A plank cart 1.2 m wide, 0.4 m deep and 2.4 m long (+z forward) on
    /// four wheels, spawned at `at`.
    fn cart(p: &mut PhysicsWorld, at: DVec3) -> BodyId {
        let size = IVec3::new(19, 6, 38);
        let shape = Arc::new(
            BodyShape::from_voxels(size, vec![ids::PLANKS; (size.x * size.y * size.z) as usize])
                .expect("cart"),
        );
        let id = p.spawn(shape, at, Quat::IDENTITY);
        let b = p.body(id).unwrap();
        let grid = |g: Vec3| {
            b.local_point(b.grid_origin() + (b.grid_rotation() * (g * VOXEL_M)).as_dvec3())
        };
        let corners = [
            (2.0, 4.0, true),
            (17.0, 4.0, true),
            (2.0, 34.0, false),
            (17.0, 34.0, false),
        ];
        let wheels = corners
            .iter()
            .map(|&(x, z, back)| Wheel {
                mount: grid(Vec3::new(x, 0.0, z)),
                steered: !back,
                driven: back,
            })
            .collect();
        let rot = b.grid_rotation();
        let up = b.rot.inverse() * (rot * Vec3::Y);
        let forward = b.rot.inverse() * (rot * -Vec3::Z);
        let mass = b.shape.mass;
        p.add_vehicle(Vehicle {
            body: id,
            wheels,
            up,
            forward,
            handling: Handling::for_mass(mass, 4, 0.3),
            throttle: 0.0,
            steer: 0.0,
            brake: 0.0,
        });
        id
    }

    fn run(p: &mut PhysicsWorld, w: &VoxelWorld, seconds: f32) {
        for _ in 0..(seconds * 120.0) as usize {
            p.step(1.0 / 120.0, w);
        }
    }

    fn heading(p: &PhysicsWorld, id: BodyId) -> Vec3 {
        let b = p.body(id).unwrap();
        b.grid_rotation() * -Vec3::Z
    }

    #[test]
    fn a_cart_settles_on_its_springs_level_and_clear_of_the_ground() {
        let w = ground();
        let mut p = PhysicsWorld::new();
        let id = cart(&mut p, DVec3::new(20.0, 3.0, 20.0));
        run(&mut p, &w, 3.0);
        let b = p.body(id).unwrap();
        let up = b.grid_rotation() * Vec3::Y;
        assert!(up.y > 0.995, "tilted {up}");
        // The chassis bottom rides above the ground on the springs.
        let bottom = b.grid_origin().y;
        assert!(bottom > 2.2 && bottom < 2.4, "bottom at {bottom}");
        assert!(b.vel.length() < 0.05, "still moving {}", b.vel);
    }

    #[test]
    fn throttle_drives_it_forward_steering_turns_it_and_the_brake_stops_it() {
        let w = ground();
        let mut p = PhysicsWorld::new();
        let id = cart(&mut p, DVec3::new(32.0, 2.8, 50.0));
        run(&mut p, &w, 1.5);
        let start = p.body(id).unwrap().pos;
        let facing = heading(&p, id);
        p.vehicles[0].throttle = 1.0;
        run(&mut p, &w, 2.0);
        let b = p.body(id).unwrap();
        let moved = (b.pos - start).as_vec3();
        assert!(moved.dot(facing) > 3.0, "only {moved}");
        assert!(moved.cross(facing).length() < 0.6, "drifted {moved}");
        // Steer left while driving: it turns.
        p.vehicles[0].steer = 1.0;
        run(&mut p, &w, 1.5);
        let turned = heading(&p, id).angle_between(facing);
        assert!(turned > 0.5, "turned only {turned}");
        // Brake: it stops, upright.
        p.vehicles[0].throttle = 0.0;
        p.vehicles[0].steer = 0.0;
        p.vehicles[0].brake = 1.0;
        run(&mut p, &w, 2.5);
        let b = p.body(id).unwrap();
        assert!(b.vel.length() < 0.2, "still going {}", b.vel);
        assert!((b.grid_rotation() * Vec3::Y).y > 0.98);
    }
}
