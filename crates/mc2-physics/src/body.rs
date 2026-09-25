//! Rigid body state and the two XPBD correction operations.

use crate::shape::{BodyShape, VOXEL_M};
use glam::{DVec3, IVec3, Quat, Vec3};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyId(pub u64);

#[derive(Clone, Debug)]
pub struct Body {
    pub id: BodyId,
    pub shape: Arc<BodyShape>,
    /// Centre of mass, world metres.
    pub pos: DVec3,
    /// Principal frame to world.
    pub rot: Quat,
    pub vel: Vec3,
    pub ang_vel: Vec3,
    pub prev_pos: DVec3,
    pub prev_rot: Quat,
    /// Pose at the start of the last fixed step, for render interpolation
    /// and motion vectors.
    pub step_start_pos: DVec3,
    pub step_start_rot: Quat,
    pub inv_mass: f32,
    /// Inverse principal moments.
    pub inv_inertia: Vec3,
    pub friction: f32,
    pub restitution: f32,
    pub asleep: bool,
    pub still_time: f32,
    /// Seconds since spawning.
    pub age: f32,
    /// Bodies sharing a non-zero group never collide with each other (the
    /// parts of one ragdoll).
    pub group: u32,
}

impl Body {
    pub fn new(id: BodyId, shape: Arc<BodyShape>, pos: DVec3, rot: Quat) -> Self {
        let inv_mass = 1.0 / shape.mass;
        let inv_inertia = Vec3::ONE / shape.inertia;
        Self {
            id,
            shape,
            pos,
            rot,
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            prev_pos: pos,
            prev_rot: rot,
            step_start_pos: pos,
            step_start_rot: rot,
            inv_mass,
            inv_inertia,
            friction: 0.6,
            restitution: 0.2,
            asleep: false,
            still_time: 0.0,
            age: 0.0,
            group: 0,
        }
    }

    /// World position of a principal-frame point.
    pub fn world_point(&self, local: Vec3) -> DVec3 {
        self.pos + (self.rot * local).as_dvec3()
    }

    /// Principal-frame position of a world point.
    pub fn local_point(&self, world: DVec3) -> Vec3 {
        self.rot.inverse() * (world - self.pos).as_vec3()
    }

    /// Material of the body at a world point.
    pub fn material_at(&self, world: DVec3) -> mc2_voxel::material::MaterialId {
        let g = self.shape.grid_of(self.local_point(world));
        self.shape.get(g.floor().as_ivec3())
    }

    /// Grid voxel containing a world point, and the point in grid space.
    pub fn grid_point(&self, world: DVec3) -> (IVec3, Vec3) {
        let g = self.shape.grid_of(self.local_point(world));
        (g.floor().as_ivec3(), g)
    }

    /// World-space rotation of the grid frame.
    pub fn grid_rotation(&self) -> Quat {
        self.rot * self.shape.principal.inverse()
    }

    /// World position of the grid origin (voxel (0, 0, 0)'s corner).
    pub fn grid_origin(&self) -> DVec3 {
        self.pos - (self.grid_rotation() * (self.shape.com * VOXEL_M)).as_dvec3()
    }

    /// Grid origin and rotation at `alpha` between the start of the last
    /// fixed step (0) and now (1), for drawing between steps.
    pub fn grid_pose_at(&self, alpha: f64) -> (DVec3, Quat) {
        let pos = self.step_start_pos.lerp(self.pos, alpha);
        let rot = self.step_start_rot.slerp(self.rot, alpha as f32);
        let grid_rot = rot * self.shape.principal.inverse();
        (
            pos - (grid_rot * (self.shape.com * VOXEL_M)).as_dvec3(),
            grid_rot,
        )
    }

    /// Generalised inverse mass for a correction along `n` (world) at
    /// offset `r` (world) from the centre of mass (Eq. 2).
    pub fn generalized_inverse_mass(&self, r: Vec3, n: Vec3) -> f32 {
        let rn = self.rot.inverse() * r.cross(n);
        self.inv_mass + rn.dot(self.inv_inertia * rn)
    }

    /// Applies a positional impulse `p` (world) at offset `r` (Eqs. 6, 8).
    pub fn apply_position_correction(&mut self, r: Vec3, p: Vec3, sign: f32) {
        self.pos += (p * (self.inv_mass * sign)).as_dvec3();
        let torque = self.rot.inverse() * r.cross(p);
        let omega = self.rot * (self.inv_inertia * torque) * sign;
        let dq = Quat::from_xyzw(omega.x, omega.y, omega.z, 0.0) * self.rot;
        self.rot = Quat::from_xyzw(
            self.rot.x + 0.5 * dq.x,
            self.rot.y + 0.5 * dq.y,
            self.rot.z + 0.5 * dq.z,
            self.rot.w + 0.5 * dq.w,
        )
        .normalize();
    }

    /// Generalised inverse mass for a rotation about `n` (world) (Eq. 3).
    pub fn angular_inverse_mass(&self, n: Vec3) -> f32 {
        let nl = self.rot.inverse() * n;
        nl.dot(self.inv_inertia * nl)
    }

    /// Applies an angular correction `p` (world) (Eqs. 7, 9).
    pub fn apply_rotation_correction(&mut self, p: Vec3, sign: f32) {
        let omega = self.rot * (self.inv_inertia * (self.rot.inverse() * p)) * sign;
        let dq = Quat::from_xyzw(omega.x, omega.y, omega.z, 0.0) * self.rot;
        self.rot = Quat::from_xyzw(
            self.rot.x + 0.5 * dq.x,
            self.rot.y + 0.5 * dq.y,
            self.rot.z + 0.5 * dq.z,
            self.rot.w + 0.5 * dq.w,
        )
        .normalize();
    }

    /// Applies an angular impulse `p` (world) to the spin.
    pub fn apply_angular_impulse(&mut self, p: Vec3, sign: f32) {
        self.ang_vel += self.rot * (self.inv_inertia * (self.rot.inverse() * p)) * sign;
    }

    /// Applies a velocity impulse `p` (world) at offset `r` (Eq. 33).
    pub fn apply_velocity_impulse(&mut self, r: Vec3, p: Vec3, sign: f32) {
        self.vel += p * (self.inv_mass * sign);
        let torque = self.rot.inverse() * r.cross(p);
        self.ang_vel += self.rot * (self.inv_inertia * torque) * sign;
    }

    /// Velocity of a point at world offset `r` from the centre of mass.
    pub fn point_velocity(&self, r: Vec3) -> Vec3 {
        self.vel + self.ang_vel.cross(r)
    }

    pub fn wake(&mut self) {
        self.asleep = false;
        self.still_time = 0.0;
    }

    pub fn kinetic_energy(&self) -> f32 {
        let w = self.rot.inverse() * self.ang_vel;
        0.5 * self.shape.mass * self.vel.length_squared() + 0.5 * w.dot(self.shape.inertia * w)
    }
}
