//! Cars: voxel bodies on wheels, parked by the villages, and driven.
//!
//! A car is one rigid body: a painted shell with glass above the waist,
//! seats inside, chrome bumpers, lamps that light the road, and wheels in
//! their arches. It rides on four raycast wheels (see `mc2_physics::
//! vehicle`); the player gets in and out with E, drives with the movement
//! keys and brakes with jump, watched from a camera behind.

use crate::input::{Input, Key};
use crate::interact::Interaction;
use crate::physics::Physics;
use crate::physics_host::WheelAt;
use crate::player::{Body, EYE, Player};
use crate::villagers::Trading;
use crate::{Voxels, nav};
use bevy_ecs::prelude::*;
use glam::{DVec3, IVec3, Quat, Vec3};
use mc2_physics::shape::VOXEL_M;
use mc2_physics::{BodyId, BodyShape};
use mc2_voxel::material::{MaterialId, ids};
use std::sync::Arc;

/// A car's grid, voxels: +x its left, +y up, +z forward. Four metres
/// long, the most a body may be.
const SIZE: IVec3 = IVec3::new(30, 24, mc2_physics::shape::MAX_EXTENT);
/// Wheel radius and the half-width of a tyre, voxels.
const WHEEL_R: f32 = 6.0;
/// Wheel centres along the car, voxels.
const AXLES: [f32; 2] = [51.0, 12.0];
/// How near the player must stand to a car to get in, metres.
pub const ENTER_M: f64 = 3.2;
/// The chase camera: behind the car and above it, metres.
pub const CHASE_BACK: f64 = 7.5;
pub const CHASE_UP: f64 = 2.6;

/// A car's shape and where things are on it (grid voxels).
pub struct CarModel {
    pub shape: Arc<BodyShape>,
    pub wheels: [WheelAt; 4],
    /// The driver's eyes.
    pub seat: Vec3,
    /// Where the driver steps out, beside the car.
    pub door: Vec3,
}

/// The material of voxel `v` of a car painted `paint` (air where there is
/// none).
fn car_voxel(paint: MaterialId, v: IVec3) -> MaterialId {
    let (x, y, z) = (v.x, v.y, v.z);
    let c = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
    // Wheels: a tyre round a chrome hub, in the outer four voxels each side.
    let side = !(4..SIZE.x - 4).contains(&x);
    for axle in AXLES {
        let r = ((c.y - WHEEL_R).powi(2) + (c.z - axle).powi(2)).sqrt();
        if side && r <= WHEEL_R {
            return if r < 2.2 && (x == 0 || x == SIZE.x - 1) {
                ids::CHROME
            } else {
                ids::RUBBER
            };
        }
        // The arch round it is open.
        if !(5..SIZE.x - 5).contains(&x) && r <= WHEEL_R + 1.5 && y < 14 {
            return MaterialId(0);
        }
    }
    let (x0, x1) = (1, SIZE.x - 1);
    // The lower body: a shell with a floor, from the sills to the waist.
    let lower = (4..13).contains(&y) && (x0..x1).contains(&x) && (1..SIZE.z - 1).contains(&z);
    if lower {
        let skin = x == x0 || x == x1 - 1 || y == 4 || y == 12 || z == 1 || z == SIZE.z - 2;
        if !skin {
            return seat(x, y, z);
        }
        // Lamps and bumpers on the ends.
        if z == SIZE.z - 2 {
            if (8..11).contains(&y) && ((3..7).contains(&x) || (23..27).contains(&x)) {
                return ids::HEADLIGHT;
            }
            if y < 6 {
                return ids::CHROME;
            }
        }
        if z == 1 {
            if (9..11).contains(&y) && ((2..6).contains(&x) || (24..28).contains(&x)) {
                return ids::TAILLIGHT;
            }
            if y < 6 {
                return ids::CHROME;
            }
        }
        return if y == 4 { ids::PAINT_BLACK } else { paint };
    }
    // The cabin: glass between pillars, under a painted roof.
    let cabin = (13..23).contains(&y) && (3..27).contains(&x) && (17..48).contains(&z);
    if cabin {
        let skin = x == 3 || x == 26 || y == 22 || z == 17 || z == 47;
        if !skin {
            return seat(x, y, z);
        }
        let pillar = matches!(z, 17 | 18 | 32 | 33 | 46 | 47) && (x == 3 || x == 26)
            || (matches!(x, 3 | 4 | 25 | 26) && (z == 17 || z == 47));
        return if y == 22 || pillar { paint } else { ids::GLASS };
    }
    MaterialId(0)
}

/// Seats inside: two in front, a bench behind, each with its back at its
/// rear edge.
fn seat(x: i32, y: i32, z: i32) -> MaterialId {
    let front = (28..34).contains(&z) && ((6..13).contains(&x) || (17..24).contains(&x));
    let back = (19..25).contains(&z) && (6..24).contains(&x);
    let cushion = (5..8).contains(&y) && (front || back);
    let rest = (8..15).contains(&y)
        && ((z == 28 && ((6..13).contains(&x) || (17..24).contains(&x)))
            || (z == 19 && (6..24).contains(&x)));
    if cushion || rest {
        ids::LEATHER
    } else {
        MaterialId(0)
    }
}

/// A car painted `paint`.
pub fn car(paint: MaterialId) -> CarModel {
    let mut voxels = Vec::with_capacity((SIZE.x * SIZE.y * SIZE.z) as usize);
    for z in 0..SIZE.z {
        for y in 0..SIZE.y {
            for x in 0..SIZE.x {
                voxels.push(car_voxel(paint, IVec3::new(x, y, z)));
            }
        }
    }
    // `from_voxels` wants x fastest, then y, then z: as built.
    let shape = BodyShape::from_voxels(SIZE, voxels).expect("a car has voxels");
    let wheel = |x: f32, z: f32, front: bool| WheelAt {
        mount: Vec3::new(x, WHEEL_R, z),
        steered: front,
        driven: !front,
    };
    let (l, r) = (2.0, SIZE.x as f32 - 2.0);
    CarModel {
        shape: Arc::new(shape),
        wheels: [
            wheel(l, AXLES[0], true),
            wheel(r, AXLES[0], true),
            wheel(l, AXLES[1], false),
            wheel(r, AXLES[1], false),
        ],
        seat: Vec3::new(20.0, 18.0, 30.0),
        door: Vec3::new(SIZE.x as f32 + 12.0, 0.0, 31.0),
    }
}

/// Paints the villages' cars come in.
pub const PAINTS: [MaterialId; 5] = [
    ids::PAINT_RED,
    ids::PAINT_BLUE,
    ids::PAINT_CREAM,
    ids::PAINT_GREEN,
    ids::PAINT_BLACK,
];

#[derive(Clone, Copy, Debug)]
pub struct Car {
    pub body: BodyId,
    pub seat: Vec3,
    pub door: Vec3,
}

/// Every car, which one the player is driving, and the villages that
/// have had theirs parked.
#[derive(Resource, Default)]
pub struct Garage {
    pub cars: Vec<Car>,
    pub driving: Option<usize>,
    pub parked: mc2_core::FxHashSet<(i32, i32)>,
}

impl Garage {
    /// Parks a new car with the corner of its grid at `corner`, facing
    /// `yaw` (radians about +y; 0 faces +z).
    pub fn park(
        &mut self,
        physics: &mut Physics,
        model: &CarModel,
        corner: DVec3,
        yaw: f32,
    ) -> BodyId {
        let body = physics.host.spawn_vehicle(
            model.shape.clone(),
            corner,
            Quat::from_rotation_y(yaw),
            &model.wheels,
            Vec3::Y,
            Vec3::Z,
            WHEEL_R * VOXEL_M + 0.03,
        );
        self.cars.push(Car {
            body,
            seat: model.seat,
            door: model.door,
        });
        body
    }

    /// Parks a car on level open ground a little way from `square`, if
    /// there is room for one: across the radius, like a car at a kerb.
    pub fn park_by(
        &mut self,
        physics: &mut Physics,
        world: &mc2_voxel::world::VoxelWorld,
        square: DVec3,
        seed: u64,
    ) -> Option<BodyId> {
        let half = SIZE.as_dvec3() * 0.5 * f64::from(VOXEL_M);
        for k in 0..24u64 {
            let a = (seed.wrapping_add(k * 7) % 24) as f64 / 24.0 * std::f64::consts::TAU;
            let d = 10.0 + (k % 3) as f64 * 2.5;
            let at = square + DVec3::new(a.cos() * d, 0.0, a.sin() * d);
            let yaw = (a + std::f64::consts::FRAC_PI_2) as f32;
            let rot = glam::DQuat::from_rotation_y(f64::from(yaw));
            let corners = [
                (-1.0, -1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (1.0, 1.0),
                (0.0, 0.0),
            ];
            let grounds: Option<Vec<f64>> = corners
                .iter()
                .map(|&(sx, sz)| {
                    let p = at + rot * DVec3::new(sx * half.x, 0.0, sz * half.z);
                    nav::stand(world, p.x, p.z, square.y)
                })
                .collect();
            let Some(grounds) = grounds else { continue };
            let (lo, hi) = grounds
                .iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), &g| (lo.min(g), hi.max(g)));
            if hi - lo > 0.3 {
                continue;
            }
            let corner =
                at + rot * DVec3::new(-half.x, 0.0, -half.z) + DVec3::Y * (hi + 0.08 - at.y);
            let model = car(PAINTS[(seed % PAINTS.len() as u64) as usize]);
            return Some(self.park(physics, &model, corner, yaw));
        }
        None
    }

    /// The car being driven, if any.
    pub fn driven(&self) -> Option<Car> {
        self.driving.map(|i| self.cars[i])
    }
}

/// A grid point of a body, in the world.
pub fn grid_point(body: &mc2_physics::Body, g: Vec3) -> DVec3 {
    body.grid_origin() + (body.grid_rotation() * (g * VOXEL_M)).as_dvec3()
}

/// Every frame: E beside a car gets in; E in one gets out beside it.
pub fn enter_cars(
    input: Res<Input>,
    interaction: Res<Interaction>,
    trading: Res<Trading>,
    voxels: Res<Voxels>,
    mut garage: ResMut<Garage>,
    mut physics: ResMut<Physics>,
    mut players: Query<(&mut Player, &mut Body)>,
) {
    if !input.captured
        || !input.pressed(Key::Interact)
        || interaction.opened.is_some()
        || trading.aimed.is_some()
    {
        return;
    }
    let Some((mut p, mut b)) = players.iter_mut().next() else {
        return;
    };
    if let Some(car) = garage.driven() {
        // Out on the driver's side, on the ground there.
        if let Some(body) = physics.host.body(car.body) {
            let door = grid_point(body, car.door);
            let y = nav::stand(&voxels.0, door.x, door.z, door.y + 0.5).unwrap_or(door.y);
            b.feet = DVec3::new(door.x, y, door.z);
            b.prev_feet = b.feet;
            b.velocity = DVec3::ZERO;
        }
        physics.host.drive(car.body, 0.0, 0.0, 1.0);
        garage.driving = None;
        p.seated = false;
        return;
    }
    let nearest = garage
        .cars
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let body = physics.host.body(c.body)?;
            let d = body.pos.distance(b.feet + DVec3::Y);
            (d < ENTER_M + f64::from(body.shape.radius) * 0.5).then_some((i, d))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1));
    if let Some((i, _)) = nearest {
        garage.driving = Some(i);
        p.seated = true;
        // Seen from behind to start with; F5 for the driver's seat.
        p.view = crate::player::View::ThirdPerson;
    }
}

/// Every fixed tick: the driver's keys to the car, and the driver to the
/// seat.
pub fn drive_cars(
    input: Res<Input>,
    garage: Res<Garage>,
    mut physics: ResMut<Physics>,
    mut players: Query<(&Player, &mut Body)>,
) {
    let Some(car) = garage.driven() else {
        return;
    };
    let key = |k: Key| if input.held(k) { 1.0 } else { 0.0 };
    // Keys or the left stick (x right, y forward).
    let stick = input.move_axis;
    let (throttle, steer, brake) = if input.captured {
        (
            (key(Key::Forward) - key(Key::Back) + stick.y).clamp(-1.0, 1.0),
            (key(Key::Left) - key(Key::Right) - stick.x).clamp(-1.0, 1.0),
            key(Key::Jump),
        )
    } else {
        (0.0, 0.0, 1.0)
    };
    physics.host.drive(car.body, throttle, steer, brake);
    let Some(seat) = physics.host.body(car.body).map(|b| grid_point(b, car.seat)) else {
        return;
    };
    for (p, mut b) in &mut players {
        if p.seated {
            b.prev_feet = b.feet;
            b.feet = seat - DVec3::Y * EYE;
            b.velocity = DVec3::ZERO;
        }
    }
}

/// Where the chase camera sits and looks while driving: behind the car
/// and above it, at `alpha` between physics steps.
pub fn chase(body: &mc2_physics::Body, alpha: f64) -> (DVec3, DVec3) {
    let (origin, rot) = body.grid_pose_at(alpha);
    let centre = origin + (rot * (SIZE.as_vec3() * 0.5 * VOXEL_M)).as_dvec3();
    let ahead = rot * Vec3::Z;
    let level = Vec3::new(ahead.x, 0.0, ahead.z)
        .normalize_or(Vec3::Z)
        .as_dvec3();
    let eye = centre - level * CHASE_BACK + DVec3::Y * CHASE_UP;
    (eye, centre + DVec3::Y * 0.8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_car_weighs_about_what_a_car_does_and_has_lamps() {
        let m = car(ids::PAINT_RED);
        assert!(
            (900.0..2600.0).contains(&m.shape.mass),
            "{} kg",
            m.shape.mass
        );
        assert!(m.shape.voxels.contains(&ids::HEADLIGHT));
        assert!(m.shape.voxels.contains(&ids::TAILLIGHT));
        assert!(m.shape.voxels.contains(&ids::GLASS));
        // The driver sits inside, on the left, and steps out clear of it.
        assert!(m.seat.x > 15.0 && m.seat.y < SIZE.y as f32);
        assert!(m.door.x > SIZE.x as f32);
    }

    #[test]
    fn a_parked_car_can_be_driven_off() {
        use mc2_voxel::world::VoxelWorld;
        let mut game = crate::Game::new();
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(1023, 31, 1023), ids::GRANITE);
        game.world.resource_mut::<Voxels>().0 = w;
        let model = car(ids::PAINT_BLUE);
        let body = {
            let mut garage = game.world.remove_resource::<Garage>().unwrap();
            let mut physics = game.world.resource_mut::<Physics>();
            let id = garage.park(&mut physics, &model, DVec3::new(30.0, 2.05, 20.0), 0.0);
            game.world.insert_resource(garage);
            id
        };
        game.spawn_player(DVec3::new(33.0, 2.0, 22.0), 0.0, 0.0);
        for _ in 0..90 {
            game.update(1.0 / 60.0);
        }
        let start = game
            .world
            .resource::<Physics>()
            .host
            .body(body)
            .unwrap()
            .pos;
        // In, and away.
        game.input().captured = true;
        game.input().key_down(Key::Interact);
        game.update(1.0 / 60.0);
        game.input().key_up(Key::Interact);
        assert!(
            game.world.resource::<Garage>().driving.is_some(),
            "did not get in"
        );
        game.input().key_down(Key::Forward);
        for _ in 0..150 {
            game.update(1.0 / 60.0);
        }
        let end = game
            .world
            .resource::<Physics>()
            .host
            .body(body)
            .unwrap()
            .pos;
        assert!(end.z - start.z > 3.0, "{start} -> {end}");
        // The driver went along, in the seat.
        let mut q = game.world.query::<(&Player, &Body)>();
        let (p, b) = q.iter(&game.world).next().unwrap();
        assert!(
            p.seated && (b.feet - end).length() < 3.0,
            "{} vs {end}",
            b.feet
        );
    }
}
