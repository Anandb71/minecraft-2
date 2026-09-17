//! Contacts between a body and the voxel world: found per substep from the
//! body's surface samples, solved as positional constraints with static
//! friction, then as velocity updates with dynamic friction and
//! restitution (Müller et al. 2020, Sections 3.5 and 3.6).

use crate::body::Body;
use crate::probe::sample_contacts;
use crate::world::GRAVITY;
use glam::{DVec3, Vec3};
use mc2_voxel::window::VoxelWindow;

/// Hard cap on contacts a body keeps per substep.
const MAX_CONTACTS: usize = 48;

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    /// Offset from the body's centre of mass (world), updated as the body
    /// moves during the solve.
    r: Vec3,
    /// The same offset in the principal frame.
    local: Vec3,
    /// Contact point when the contact was found.
    point: DVec3,
    /// Pointing out of the obstacle, toward the body.
    n: Vec3,
    depth: f32,
    /// Normal velocity before the substep's position solve.
    vn_before: f32,
    lambda_n: f32,
    /// The same body point at the start of the substep, for static friction.
    point_prev: DVec3,
}

pub fn find_world_contacts(b: &Body, world: &VoxelWindow) -> Vec<Contact> {
    let mut raw = Vec::new();
    let mut out = Vec::new();
    for &s in &b.shape.samples {
        raw.clear();
        let p = b.world_point(s);
        sample_contacts(world, p, &mut raw);
        let prev = b.prev_pos + (b.prev_rot * s).as_dvec3();
        for &(n, depth, point) in &raw {
            let r = (point - b.pos).as_vec3();
            out.push(Contact {
                r,
                local: b.rot.inverse() * r,
                point,
                n,
                depth,
                vn_before: n.dot(b.point_velocity(r)),
                lambda_n: 0.0,
                point_prev: prev + (point - p),
            });
        }
    }
    if out.len() > MAX_CONTACTS {
        out.sort_by(|a, c| c.depth.total_cmp(&a.depth));
        out.truncate(MAX_CONTACTS);
    }
    out
}

pub fn solve_world_positions(b: &mut Body, contacts: &mut [Contact]) {
    for c in contacts.iter_mut() {
        // Re-evaluate against the pose updated by earlier contacts: the
        // point has moved along the normal by however much they pushed it.
        let r_now = b.rot * c.local;
        let point_now = b.pos + r_now.as_dvec3();
        let depth = c.depth - c.n.dot((point_now - c.point).as_vec3());
        let w = b.generalized_inverse_mass(r_now, c.n);
        c.r = r_now;
        if w <= 0.0 || depth <= 0.0 {
            continue;
        }
        let lambda = depth / w;
        c.lambda_n += lambda;
        b.apply_position_correction(r_now, c.n * lambda, 1.0);
        // Static friction: undo tangential motion of the contact point
        // while the friction cone allows.
        let point_now = b.pos + r_now.as_dvec3();
        let dp = (point_now - c.point_prev).as_vec3();
        let dt_ = dp - c.n * dp.dot(c.n);
        let len = dt_.length();
        if len > 1e-7 {
            let t = dt_ / len;
            let wt = b.generalized_inverse_mass(r_now, t);
            let lambda_t = len / wt;
            if lambda_t < b.friction * c.lambda_n {
                b.apply_position_correction(r_now, -t * lambda_t, 1.0);
            }
        }
        c.r = b.rot * c.local;
    }
}

pub fn solve_world_velocities(b: &mut Body, contacts: &[Contact], h: f32) {
    let g = GRAVITY.length();
    for c in contacts {
        if c.lambda_n <= 0.0 {
            continue;
        }
        let v = b.point_velocity(c.r);
        let vn = c.n.dot(v);
        let vt = v - c.n * vn;
        let mut dv = Vec3::ZERO;
        let vt_len = vt.length();
        if vt_len > 1e-6 {
            let fn_ = c.lambda_n / (h * h);
            dv -= vt / vt_len * (h * b.friction * 0.8 * fn_ * b.inv_mass).min(vt_len);
        }
        let e = if c.vn_before.abs() <= 2.0 * g * h {
            0.0
        } else {
            b.restitution
        };
        dv += c.n * (-vn + (-e * c.vn_before).max(0.0));
        let len = dv.length();
        if len < 1e-7 {
            continue;
        }
        let w = b.generalized_inverse_mass(c.r, dv / len);
        b.apply_velocity_impulse(c.r, dv / w, 1.0);
    }
}
