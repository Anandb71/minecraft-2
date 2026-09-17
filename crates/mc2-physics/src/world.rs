//! The body set and the substepped XPBD solver (Müller et al. 2020,
//! Algorithm 2 with one position iteration per substep).

use crate::body::{Body, BodyId};
use crate::pair_contact::{
    candidate_pairs, find_pair_contacts, solve_pair_positions, solve_pair_velocities,
};
use crate::probe::SAMPLE_RADIUS;
use crate::shape::BodyShape;
use crate::world_contact::{
    Contact, find_world_contacts, solve_world_positions, solve_world_velocities,
};
use glam::{DVec3, Quat, Vec3};
use mc2_voxel::window::VoxelWindow;
use mc2_voxel::world::VoxelWorld;
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

pub struct PhysicsWorld {
    pub bodies: Vec<Body>,
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
            next_id: 1,
            substeps: 8,
            stats: PhysicsStats::default(),
        }
    }

    pub fn spawn(&mut self, shape: Arc<BodyShape>, pos: DVec3, rot: Quat) -> BodyId {
        let id = BodyId(self.next_id);
        self.next_id += 1;
        self.bodies.push(Body::new(id, shape, pos, rot));
        id
    }

    pub fn body(&self, id: BodyId) -> Option<&Body> {
        self.bodies.iter().find(|b| b.id == id)
    }

    pub fn body_mut(&mut self, id: BodyId) -> Option<&mut Body> {
        self.bodies.iter_mut().find(|b| b.id == id)
    }

    pub fn remove(&mut self, id: BodyId) -> Option<Body> {
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

    /// Advances by `dt` seconds.
    pub fn step(&mut self, dt: f32, world: &VoxelWorld) {
        let start = std::time::Instant::now();
        let substeps = self.substeps.max(1);
        let h = dt / substeps as f32;
        for b in &mut self.bodies {
            b.step_start_pos = b.pos;
            b.step_start_rot = b.rot;
            b.age += dt;
        }
        // Broad phase once per step: bounding sphere pairs, expanded by
        // how far the bodies can travel.
        let pairs = candidate_pairs(&self.bodies, dt);
        // The world around each awake body, over everywhere it can reach
        // this step, read once instead of per sample and substep.
        let windows: Vec<Option<VoxelWindow>> = self
            .bodies
            .par_iter()
            .map(|b| (!b.asleep).then(|| reach_window(b, dt, world)))
            .collect();
        let mut paired = vec![false; self.bodies.len()];
        for &(i, j) in &pairs {
            paired[i] = true;
            paired[j] = true;
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
                let mut found = 0;
                for _ in 0..substeps {
                    integrate(b, h);
                    let contacts = solve_world(b, window.as_ref());
                    derive_velocities(b, h);
                    solve_world_velocities(b, &contacts, h);
                    found += contacts.len();
                }
                found
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
        if !pairs.is_empty() {
            for _ in 0..substeps {
                let begin = |(b, window, contacts): (
                    &mut Body,
                    &Option<VoxelWindow>,
                    &mut Vec<Contact>,
                )| {
                    contacts.clear();
                    if !b.asleep {
                        integrate(b, h);
                        *contacts = solve_world(b, window.as_ref());
                    }
                };
                if serial {
                    for &i in &group {
                        begin((&mut self.bodies[i], &windows[i], &mut contacts[i]));
                    }
                } else {
                    self.bodies
                        .par_iter_mut()
                        .zip(windows.par_iter())
                        .zip(contacts.par_iter_mut())
                        .zip(paired.par_iter())
                        .with_min_len(8)
                        .filter(|(_, p)| **p)
                        .for_each(|(((b, w), c), _)| begin((b, w, c)));
                }
                let pair_start = std::time::Instant::now();
                let mut pc = find_pair_contacts(&mut self.bodies, &pairs);
                solve_pair_positions(&mut self.bodies, &mut pc);
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
                world_contacts += contacts.iter().map(Vec::len).sum::<usize>();
                pair_contacts += pc.len();
            }
        }
        for b in &mut self.bodies {
            if b.asleep {
                // Still counts how long the body has been at rest.
                b.still_time += dt;
                continue;
            }
            let speed = b.vel.length() + b.ang_vel.length() * b.shape.radius;
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
            world_contacts: world_contacts / substeps as usize,
            pair_contacts: pair_contacts / substeps as usize,
            pairs: pairs.len(),
            step_ms: start.elapsed().as_secs_f32() * 1000.0,
            pair_ms: pair_time.as_secs_f32() * 1000.0,
        };
    }
}

/// Voxels a body's samples can touch during a step of `dt`: its bounding
/// sphere at the start and at the ballistic end point, plus a sample radius
/// and a voxel of slack.
fn reach_window<'a>(b: &Body, dt: f32, world: &'a VoxelWorld) -> VoxelWindow<'a> {
    let end = b.pos + ((b.vel + GRAVITY * dt * 0.5) * dt).as_dvec3();
    let r = f64::from(b.shape.radius + SAMPLE_RADIUS) + 1.0 / 16.0;
    let lo = b.pos.min(end) - r;
    let hi = b.pos.max(end) + r;
    VoxelWindow::new(
        world,
        (lo * 16.0).floor().as_ivec3(),
        (hi * 16.0).floor().as_ivec3(),
    )
}

/// World contacts of a body this substep, already solved for position.
fn solve_world(b: &mut Body, window: Option<&VoxelWindow>) -> Vec<Contact> {
    let Some(window) = window.filter(|w| w.any_matter()) else {
        return Vec::new();
    };
    let mut contacts = find_world_contacts(b, window);
    solve_world_positions(b, &mut contacts);
    contacts
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
    use glam::IVec3;
    use mc2_voxel::material::ids;

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
