//! The body set and the substepped XPBD solver (Müller et al. 2020,
//! Algorithm 2 with one position iteration per substep).

use crate::body::{Body, BodyId};
use crate::shape::BodyShape;
use glam::{DVec3, Quat, Vec3};
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
    pub step_ms: f32,
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
    pub fn step(&mut self, dt: f32, _world: &VoxelWorld) {
        let start = std::time::Instant::now();
        let substeps = self.substeps.max(1);
        let h = dt / substeps as f32;
        for b in &mut self.bodies {
            b.step_start_pos = b.pos;
            b.step_start_rot = b.rot;
            b.age += dt;
        }
        for _ in 0..substeps {
            self.bodies.par_iter_mut().for_each(|b| {
                if !b.asleep {
                    integrate(b, h);
                }
            });
            self.bodies.par_iter_mut().for_each(|b| {
                if !b.asleep {
                    derive_velocities(b, h);
                }
            });
        }
        for b in &mut self.bodies {
            if b.asleep {
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
            world_contacts: 0,
            pair_contacts: 0,
            step_ms: start.elapsed().as_secs_f32() * 1000.0,
        };
    }
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

    fn cube(n: i32) -> Arc<BodyShape> {
        Arc::new(
            BodyShape::from_voxels(IVec3::splat(n), vec![ids::GRANITE; (n * n * n) as usize])
                .expect("cube"),
        )
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
