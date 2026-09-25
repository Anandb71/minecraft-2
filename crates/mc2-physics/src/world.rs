//! The body set and the substepped XPBD solver (Müller et al. 2020,
//! Algorithm 2 with one position iteration per substep).

use crate::body::{Body, BodyId};
use crate::collision::{CollisionWindow, SolidCells};
use crate::joint::{Joint, damp_joints, solve_joint_positions};
use crate::pair_contact::{
    candidate_pairs, find_pair_contacts, refresh_pair_contacts, solve_pair_positions,
    solve_pair_velocities,
};
use crate::probe::SAMPLE_RADIUS;
use crate::shape::{BodyShape, VOXEL_M};
use crate::vehicle::{Vehicle, drive};
use crate::world_contact::{
    Contact, find_world_contacts, refresh_contacts, solve_world_positions, solve_world_velocities,
};
use glam::{DVec3, IVec3, Quat, Vec3};
use rayon::prelude::*;
use std::sync::Arc;

pub const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
/// Bodies slower than this (m/s, including rotation at their radius) for
/// SLEEP_TIME seconds go to sleep.
const SLEEP_SPEED: f32 = 0.08;
const SLEEP_TIME: f32 = 0.5;

#[derive(Clone, Copy, Debug, Default)]
pub struct PhysicsStats {
    pub bodies: usize,
    pub awake: usize,
    pub world_contacts: usize,
    pub pair_contacts: usize,
    /// Candidate pairs from the broad phase.
    pub pairs: usize,
    pub step_ms: f32,
    /// Of which body-body contact search and solve.
    pub pair_ms: f32,
}

/// A box that pushes bodies without being pushed: the player, say.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obstacle {
    pub min: DVec3,
    pub max: DVec3,
    /// Metres per second; a moving obstacle wakes the bodies it touches.
    pub vel: Vec3,
}

pub struct PhysicsWorld {
    pub bodies: Vec<Body>,
    /// Kinematic boxes for the next step.
    pub obstacles: Vec<Obstacle>,
    /// Joints between bodies; one whose body is gone is dropped.
    pub joints: Vec<Joint>,
    /// Bodies on wheels; one whose body is gone is dropped.
    pub vehicles: Vec<Vehicle>,
    /// Brick cells the last step needed and the collision source lacked.
    /// Bodies that needed them were held in place.
    pub missing_cells: Vec<IVec3>,
    next_id: u64,
    pub substeps: u32,
    pub stats: PhysicsStats,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            obstacles: Vec::new(),
            joints: Vec::new(),
            vehicles: Vec::new(),
            missing_cells: Vec::new(),
            next_id: 1,
            substeps: 8,
            stats: PhysicsStats::default(),
        }
    }

    pub fn spawn(&mut self, shape: Arc<BodyShape>, pos: DVec3, rot: Quat) -> BodyId {
        let id = BodyId(self.next_id);
        self.spawn_with_id(id, shape, pos, rot);
        id
    }

    /// Spawns under an id chosen by the caller (who hands out ids before the
    /// physics thread sees the body). Later automatic ids skip past it.
    pub fn spawn_with_id(&mut self, id: BodyId, shape: Arc<BodyShape>, pos: DVec3, rot: Quat) {
        self.next_id = self.next_id.max(id.0 + 1);
        self.bodies.push(Body::new(id, shape, pos, rot));
    }

    /// The id the next automatic spawn will take.
    pub fn next_id(&self) -> BodyId {
        BodyId(self.next_id)
    }

    pub fn body(&self, id: BodyId) -> Option<&Body> {
        self.bodies.iter().find(|b| b.id == id)
    }

    pub fn body_mut(&mut self, id: BodyId) -> Option<&mut Body> {
        self.bodies.iter_mut().find(|b| b.id == id)
    }

    /// Joins two bodies and puts both in `group`, whose members do not
    /// collide with each other.
    pub fn add_joint(&mut self, joint: Joint, group: u32) {
        for id in [joint.a, joint.b] {
            if let Some(b) = self.body_mut(id) {
                b.group = group;
            }
        }
        self.joints.push(joint);
    }

    /// Puts a body on wheels.
    pub fn add_vehicle(&mut self, vehicle: Vehicle) {
        self.vehicles.retain(|v| v.body != vehicle.body);
        self.vehicles.push(vehicle);
    }

    pub fn vehicle_mut(&mut self, id: BodyId) -> Option<&mut Vehicle> {
        self.vehicles.iter_mut().find(|v| v.body == id)
    }

    pub fn remove(&mut self, id: BodyId) -> Option<Body> {
        self.joints.retain(|j| j.a != id && j.b != id);
        self.vehicles.retain(|v| v.body != id);
        let i = self.bodies.iter().position(|b| b.id == id)?;
        Some(self.bodies.swap_remove(i))
    }

    /// Wakes every body whose bounding sphere touches the box (world m).
    pub fn wake_region(&mut self, min: DVec3, max: DVec3) {
        for b in &mut self.bodies {
            let r = f64::from(b.shape.radius);
            let near = b.pos.cmpge(min - r).all() && b.pos.cmple(max + r).all();
            if near {
                b.wake();
            }
        }
    }

    /// Advances by `dt` seconds against the solid cells of `world`.
    pub fn step<S: SolidCells>(&mut self, dt: f32, world: &S) {
        let start = std::time::Instant::now();
        let substeps = self.substeps.max(1);
        let h = dt / substeps as f32;
        for b in &mut self.bodies {
            b.step_start_pos = b.pos;
            b.step_start_rot = b.rot;
            b.age += dt;
        }
        // Joints between bodies that still exist, by index. A body that
        // moved last step wakes the body it is jointed to; one merely not
        // yet asleep does not, so a still pair can fall asleep together.
        let index: std::collections::HashMap<BodyId, usize> = self
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| (b.id, i))
            .collect();
        self.joints
            .retain(|j| index.contains_key(&j.a) && index.contains_key(&j.b));
        let joints: Vec<(Joint, usize, usize)> = self
            .joints
            .iter()
            .map(|j| (*j, index[&j.a], index[&j.b]))
            .collect();
        for &(_, a, b) in &joints {
            let moving = |b: &Body| !b.asleep && b.still_time == 0.0;
            if moving(&self.bodies[a]) || moving(&self.bodies[b]) {
                for i in [a, b] {
                    if self.bodies[i].asleep {
                        self.bodies[i].wake();
                    }
                }
            }
        }
        // Vehicles by body; one being driven is awake.
        self.vehicles.retain(|v| index.contains_key(&v.body));
        let vehicles: Vec<(usize, usize)> = self
            .vehicles
            .iter()
            .enumerate()
            .map(|(vi, v)| (vi, index[&v.body]))
            .collect();
        for &(vi, bi) in &vehicles {
            if self.vehicles[vi].controlled() && self.bodies[bi].asleep {
                self.bodies[bi].wake();
            }
        }
        // Broad phase once per step: bounding sphere pairs, expanded by
        // how far the bodies can travel. Parts of one group pass through
        // each other.
        let mut pairs = candidate_pairs(&self.bodies, dt);
        pairs.retain(|&(i, j)| {
            let g = self.bodies[i].group;
            g == 0 || g != self.bodies[j].group
        });
        for o in self.obstacles.iter().filter(|o| o.vel.length() > 0.05) {
            for b in self.bodies.iter_mut().filter(|b| b.asleep) {
                let r = f64::from(b.shape.radius);
                if b.pos.clamp(o.min, o.max).distance(b.pos) <= r {
                    b.wake();
                }
            }
        }
        let obstacles = &self.obstacles;
        // The world around each awake body, over everywhere it can reach
        // this step, read once instead of per sample and substep.
        let windows: Vec<Option<CollisionWindow<S>>> = self
            .bodies
            .par_iter()
            .map(|b| (!b.asleep).then(|| reach_window(b, dt, world)))
            .collect();
        // A body whose surroundings are not known yet waits, held like a
        // sleeper, until they arrive.
        self.missing_cells.clear();
        let mut held = vec![false; self.bodies.len()];
        for (i, w) in windows.iter().enumerate() {
            if let Some(w) = w.as_ref().filter(|w| !w.missing.is_empty()) {
                self.missing_cells.extend_from_slice(&w.missing);
                held[i] = true;
                self.bodies[i].asleep = true;
            }
        }
        self.missing_cells.sort_unstable_by_key(|c| (c.x, c.y, c.z));
        self.missing_cells.dedup();
        let mut paired = vec![false; self.bodies.len()];
        for &(i, j) in &pairs {
            paired[i] = true;
            paired[j] = true;
        }
        for (_, i, j) in &joints {
            paired[*i] = true;
            paired[*j] = true;
        }
        // Vehicles take their wheels' pushes substep by substep with the
        // coupled bodies.
        for &(_, bi) in &vehicles {
            paired[bi] = true;
        }
        // A body near no other body is independent for the whole step: one
        // parallel task runs all of its substeps.
        let mut world_contacts: usize = self
            .bodies
            .par_iter_mut()
            .zip(windows.par_iter())
            .zip(paired.par_iter())
            .with_min_len(4)
            .filter(|((b, _), p)| !b.asleep && !**p)
            .map(|((b, window), _)| {
                let mut contacts = find_world(b, window.as_ref(), obstacles, dt);
                for _ in 0..substeps {
                    integrate(b, h);
                    refresh_contacts(b, &mut contacts);
                    solve_world_positions(b, &mut contacts, h);
                    derive_velocities(b, h);
                    solve_world_velocities(b, &contacts, h);
                }
                contacts.len()
            })
            .sum();
        // Bodies that may touch each other advance together, substep by
        // substep, with the pair constraints between their world solves.
        let mut contacts: Vec<Vec<Contact>> = vec![Vec::new(); self.bodies.len()];
        let mut pair_time = std::time::Duration::ZERO;
        let mut pair_contacts = 0usize;
        let group: Vec<usize> = (0..self.bodies.len()).filter(|&i| paired[i]).collect();
        // Small groups are cheaper serial than split across threads.
        let serial = group.len() < 24;
        if !group.is_empty() {
            // Narrow phase once for the step, then every substep solves the
            // same contacts against the poses it finds (Mueller et al. 2020,
            // Section 3.5).
            let find = |(b, window, contacts): (
                &mut Body,
                &Option<CollisionWindow<S>>,
                &mut Vec<Contact>,
            )| {
                contacts.clear();
                if !b.asleep {
                    *contacts = find_world(b, window.as_ref(), obstacles, dt);
                }
            };
            if serial {
                for &i in &group {
                    find((&mut self.bodies[i], &windows[i], &mut contacts[i]));
                }
            } else {
                self.bodies
                    .par_iter_mut()
                    .zip(windows.par_iter())
                    .zip(contacts.par_iter_mut())
                    .zip(paired.par_iter())
                    .with_min_len(8)
                    .filter(|(_, p)| **p)
                    .for_each(|(((b, w), c), _)| find((b, w, c)));
            }
            let pair_start = std::time::Instant::now();
            let mut pc = find_pair_contacts(&mut self.bodies, &pairs, step_margin(dt));
            pair_time += pair_start.elapsed();
            pair_contacts = pc.len();
            for _ in 0..substeps {
                for &(vi, bi) in &vehicles {
                    drive(
                        &self.vehicles[vi],
                        &mut self.bodies[bi],
                        windows[bi].as_ref(),
                        h,
                    );
                }
                let begin = |(b, contacts): (&mut Body, &mut Vec<Contact>)| {
                    if !b.asleep {
                        integrate(b, h);
                        refresh_contacts(b, contacts);
                        solve_world_positions(b, contacts, h);
                    }
                };
                if serial {
                    for &i in &group {
                        begin((&mut self.bodies[i], &mut contacts[i]));
                    }
                } else {
                    self.bodies
                        .par_iter_mut()
                        .zip(contacts.par_iter_mut())
                        .zip(paired.par_iter())
                        .with_min_len(8)
                        .filter(|(_, p)| **p)
                        .for_each(|((b, c), _)| begin((b, c)));
                }
                let pair_start = std::time::Instant::now();
                refresh_pair_contacts(&self.bodies, &mut pc);
                solve_pair_positions(&mut self.bodies, &mut pc);
                solve_joint_positions(&mut self.bodies, &joints);
                pair_time += pair_start.elapsed();
                let finish = |b: &mut Body, contacts: &Vec<Contact>| {
                    if !b.asleep {
                        derive_velocities(b, h);
                        solve_world_velocities(b, contacts, h);
                    }
                };
                if serial {
                    for &i in &group {
                        finish(&mut self.bodies[i], &contacts[i]);
                    }
                } else {
                    self.bodies
                        .par_iter_mut()
                        .zip(contacts.par_iter())
                        .zip(paired.par_iter())
                        .with_min_len(8)
                        .filter(|(_, p)| **p)
                        .for_each(|((b, c), _)| finish(b, c));
                }
                solve_pair_velocities(&mut self.bodies, &pc, h);
                damp_joints(&mut self.bodies, &joints, h);
            }
            world_contacts += contacts.iter().map(Vec::len).sum::<usize>();
        }
        for (b, held) in self.bodies.iter_mut().zip(&held) {
            if *held {
                b.asleep = false;
                continue;
            }
            if b.asleep {
                // Still counts how long the body has been at rest.
                b.still_time += dt;
                continue;
            }
            // Judge rest by how far the body actually moved over the step:
            // resting contact leaves substep velocities that flicker around
            // tiny position corrections without going anywhere.
            let moved = (b.pos - b.step_start_pos).length() as f32;
            let turned = (b.rot * b.step_start_rot.inverse())
                .to_scaled_axis()
                .length();
            let speed = (moved + turned * b.shape.radius) / dt;
            if speed < SLEEP_SPEED {
                b.still_time += dt;
                if b.still_time > SLEEP_TIME {
                    b.asleep = true;
                    b.vel = Vec3::ZERO;
                    b.ang_vel = Vec3::ZERO;
                }
            } else {
                b.still_time = 0.0;
            }
        }
        self.stats = PhysicsStats {
            bodies: self.bodies.len(),
            awake: self.bodies.iter().filter(|b| !b.asleep).count(),
            world_contacts,
            pair_contacts,
            pairs: pairs.len(),
            step_ms: start.elapsed().as_secs_f32() * 1000.0,
            pair_ms: pair_time.as_secs_f32() * 1000.0,
        };
    }
}

/// Voxels a body's samples can touch during a step of `dt`: its bounding
/// sphere at the start and at the ballistic end point, plus a sample radius
/// and a voxel of slack.
fn reach_window<'a, S: SolidCells>(b: &Body, dt: f32, world: &'a S) -> CollisionWindow<'a, S> {
    let end = b.pos + ((b.vel + GRAVITY * dt * 0.5) * dt).as_dvec3();
    let r = f64::from(b.shape.radius + SAMPLE_RADIUS) + 1.0 / 16.0;
    let lo = b.pos.min(end) - r;
    let hi = b.pos.max(end) + r;
    CollisionWindow::new(
        world,
        (lo * 16.0).floor().as_ivec3(),
        (hi * 16.0).floor().as_ivec3(),
    )
}

/// How far a contact may be from touching and still be worth finding at
/// the start of a step. Half a voxel: the sample spheres probe their
/// neighbouring cell, so a wider reach would look past it. A contact that
/// forms mid-step is found by the next one, an eighth of a second later.
fn step_margin(_dt: f32) -> f32 {
    VOXEL_M * 0.5
}

/// World and obstacle contacts of a body for the coming step.
fn find_world<S: SolidCells>(
    b: &mut Body,
    window: Option<&CollisionWindow<S>>,
    obstacles: &[Obstacle],
    dt: f32,
) -> Vec<Contact> {
    let window = window.filter(|w| w.any_solid());
    if window.is_none() && obstacles.is_empty() {
        return Vec::new();
    }
    find_world_contacts(b, window, obstacles, step_margin(dt))
}

fn integrate(b: &mut Body, h: f32) {
    b.prev_pos = b.pos;
    b.prev_rot = b.rot;
    b.vel += GRAVITY * h;
    b.pos += (b.vel * h).as_dvec3();
    // Gyroscopic term in the principal frame.
    let w = b.rot.inverse() * b.ang_vel;
    let inertia = b.shape.inertia;
    let gyro = -w.cross(inertia * w);
    let w = w + h * (b.inv_inertia * gyro);
    b.ang_vel = b.rot * w;
    // Exact rotation over the substep; the paper's linearised update loses
    // a little spin every substep.
    b.rot = (Quat::from_scaled_axis(b.ang_vel * h) * b.rot).normalize();
}

fn derive_velocities(b: &mut Body, h: f32) {
    b.vel = ((b.pos - b.prev_pos) / f64::from(h)).as_vec3();
    let dq = b.rot * b.prev_rot.inverse();
    let dq = if dq.w >= 0.0 { dq } else { -dq };
    b.ang_vel = dq.to_scaled_axis() / h;
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;
    use mc2_voxel::world::VoxelWorld;

    fn ground() -> VoxelWorld {
        let mut w = VoxelWorld::new();
        // A 16 m x 16 m granite floor, top surface at y = 2 m.
        w.fill_box(IVec3::new(0, 0, 0), IVec3::new(255, 31, 255), ids::GRANITE);
        w
    }

    fn cube(n: i32) -> Arc<BodyShape> {
        Arc::new(
            BodyShape::from_voxels(IVec3::splat(n), vec![ids::GRANITE; (n * n * n) as usize])
                .expect("cube"),
        )
    }

    #[test]
    fn a_dropped_cube_comes_to_rest_on_the_floor() {
        let w = ground();
        let mut p = PhysicsWorld::new();
        // 0.5 m cube dropped from 1 m above the floor.
        let id = p.spawn(cube(8), DVec3::new(8.0, 3.25, 8.0), Quat::IDENTITY);
        for _ in 0..600 {
            p.step(1.0 / 120.0, &w);
        }
        let b = p.body(id).unwrap();
        // Centre 0.25 m above the surface, give or take a sample radius.
        assert!((b.pos.y - 2.25).abs() < 0.05, "rest height {}", b.pos.y);
        assert!(b.asleep, "still moving at {} m/s", b.vel.length());
    }

    #[test]
    fn a_tumbling_cube_settles_flat() {
        let w = ground();
        let mut p = PhysicsWorld::new();
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.4, 0.7, 0.2);
        let id = p.spawn(cube(8), DVec3::new(8.0, 4.0, 8.0), rot);
        p.body_mut(id).unwrap().ang_vel = Vec3::new(3.0, 1.0, -2.0);
        for _ in 0..1200 {
            p.step(1.0 / 120.0, &w);
        }
        let b = p.body(id).unwrap();
        // Resting on a face: one grid axis is vertical.
        let up = b.grid_rotation().inverse() * Vec3::Y;
        assert!(up.abs().max_element() > 0.98, "tilted: {up}");
        assert!((b.pos.y - 2.25).abs() < 0.06, "rest height {}", b.pos.y);
    }

    #[test]
    fn friction_stops_a_spinning_sliding_cube() {
        let w = ground();
        let mut p = PhysicsWorld::new();
        let id = p.spawn(cube(6), DVec3::new(8.0, 2.1875, 8.0), Quat::IDENTITY);
        {
            let b = p.body_mut(id).unwrap();
            b.vel = Vec3::new(2.0, 0.0, 0.0);
            b.ang_vel = Vec3::new(0.0, 6.0, 0.0);
        }
        let mut trace = Vec::new();
        for i in 0..480 {
            p.step(1.0 / 120.0, &w);
            let b = p.body(id).unwrap();
            if i % 40 == 0 {
                trace.push((b.vel, b.ang_vel, b.pos.y));
            }
        }
        let b = p.body(id).unwrap();
        assert!(b.asleep, "still moving: {trace:?}");
        assert!((b.pos.y - 2.1875).abs() < 0.03, "{}", b.pos.y);
    }

    #[test]
    fn an_obstacle_shoves_a_resting_cube_aside() {
        let w = ground();
        let mut p = PhysicsWorld::new();
        let id = p.spawn(cube(4), DVec3::new(8.0, 2.125, 8.0), Quat::IDENTITY);
        for _ in 0..120 {
            p.step(1.0 / 120.0, &w);
        }
        assert!(p.body(id).unwrap().asleep);
        // A 0.6 m wide box walks through along +x at 3 m/s.
        let mut x = 7.0;
        for _ in 0..60 {
            x += 3.0 / 120.0;
            p.obstacles = vec![Obstacle {
                min: DVec3::new(x - 0.3, 2.0, 7.7),
                max: DVec3::new(x + 0.3, 3.8, 8.3),
                vel: Vec3::new(3.0, 0.0, 0.0),
            }];
            p.step(1.0 / 120.0, &w);
            let b = p.body(id).unwrap();
            let inside = b.pos.x > x - 0.3 && b.pos.x < x + 0.3;
            assert!(!inside, "cube at {} inside box at {x}", b.pos.x);
        }
        assert!(p.body(id).unwrap().pos.x > x + 0.3, "not pushed ahead");
    }

    #[test]
    fn bodies_wait_for_missing_collision_cells() {
        use crate::collision::{CollisionMap, world_occ};
        let w = ground();
        let mut map = CollisionMap::default();
        let mut p = PhysicsWorld::new();
        let id = p.spawn(cube(8), DVec3::new(8.0, 3.0, 8.0), Quat::IDENTITY);
        p.step(1.0 / 120.0, &map);
        assert!(!p.missing_cells.is_empty());
        assert_eq!(
            p.body(id).unwrap().pos.y,
            3.0,
            "moved without its surroundings"
        );
        // Hand over what each step asked for, as the world's thread would.
        for _ in 0..600 {
            for c in std::mem::take(&mut p.missing_cells) {
                map.insert(c, world_occ(&w, c).into_owned());
            }
            p.step(1.0 / 120.0, &map);
        }
        let b = p.body(id).unwrap();
        assert!((b.pos.y - 2.25).abs() < 0.05, "rest height {}", b.pos.y);
        assert!(b.asleep);
    }

    #[test]
    fn stacked_cubes_rest_on_each_other() {
        let w = ground();
        let mut p = PhysicsWorld::new();
        let low = p.spawn(cube(8), DVec3::new(8.0, 2.3, 8.0), Quat::IDENTITY);
        let high = p.spawn(cube(8), DVec3::new(8.0, 2.9, 8.0), Quat::IDENTITY);
        for _ in 0..600 {
            p.step(1.0 / 120.0, &w);
        }
        let (l, h) = (p.body(low).unwrap(), p.body(high).unwrap());
        assert!((l.pos.y - 2.25).abs() < 0.06, "low at {}", l.pos.y);
        assert!((h.pos.y - 2.75).abs() < 0.1, "high at {}", h.pos.y);
        assert!((h.pos.x - l.pos.x).abs() < 0.1 && (h.pos.z - l.pos.z).abs() < 0.1);
    }

    #[test]
    fn momentum_is_conserved_in_free_flight() {
        let w = VoxelWorld::new();
        let mut p = PhysicsWorld::new();
        let id = p.spawn(cube(8), DVec3::new(8.0, 100.0, 8.0), Quat::IDENTITY);
        p.body_mut(id).unwrap().vel = Vec3::new(3.0, 0.0, 0.0);
        p.body_mut(id).unwrap().ang_vel = Vec3::new(0.0, 5.0, 0.0);
        let e0 = p.body(id).unwrap().kinetic_energy();
        for _ in 0..120 {
            p.step(1.0 / 120.0, &w);
        }
        let b = p.body(id).unwrap();
        assert!((b.vel.x - 3.0).abs() < 1e-3);
        assert!((b.vel.y + 9.81).abs() < 0.02, "{}", b.vel.y);
        // Spin about a principal axis is steady.
        assert!((b.ang_vel.y - 5.0).abs() < 1e-2, "{}", b.ang_vel);
        let rot_e = b.kinetic_energy() - 0.5 * b.shape.mass * b.vel.length_squared();
        let rot_e0 = e0 - 0.5 * b.shape.mass * 9.0;
        assert!((rot_e - rot_e0).abs() / rot_e0 < 0.01);
    }
}
