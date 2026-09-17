//! Contacts between bodies: a bounding sphere broad phase, then each
//! body's surface samples probed against the other's voxel grid, solved
//! with both bodies moving.

use crate::body::Body;
use crate::probe::{SAMPLE_RADIUS, sphere_contacts};
use crate::shape::VOXEL_M;
use crate::world::GRAVITY;
use glam::{DVec3, Mat3, Vec3};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct PairContact {
    a: usize,
    b: usize,
    ra: Vec3,
    rb: Vec3,
    /// Offsets in each body's principal frame, and the contact point on
    /// each body when found, so later corrections see earlier ones.
    la: Vec3,
    lb: Vec3,
    pa: DVec3,
    pb: DVec3,
    /// Pointing from b toward a.
    n: Vec3,
    depth: f32,
    vn_before: f32,
    lambda_n: f32,
}

/// Body index pairs whose bounding spheres, grown by the distance each
/// can travel in `dt`, overlap. Sort and sweep along x: the spheres' x
/// intervals are sorted once and each is compared only with those that
/// start before it ends.
pub fn candidate_pairs(bodies: &[Body], dt: f32) -> Vec<(usize, usize)> {
    let reach = |b: &Body| f64::from(b.shape.radius + b.vel.length() * dt * 2.0);
    let mut order: Vec<(f64, f64, usize)> = bodies
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let r = reach(b);
            (b.pos.x - r, r, i)
        })
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut pairs = Vec::new();
    for (k, &(min_x, ra, i)) in order.iter().enumerate() {
        let a = &bodies[i];
        let max_x = min_x + 2.0 * ra;
        for &(other_min, rb, j) in &order[k + 1..] {
            if other_min > max_x {
                break;
            }
            let b = &bodies[j];
            if a.asleep && b.asleep {
                continue;
            }
            if a.pos.distance_squared(b.pos) < (ra + rb) * (ra + rb) {
                pairs.push((i.min(j), i.max(j)));
            }
        }
    }
    pairs
}

/// Speed at which a body wakes sleeping bodies it touches, m/s.
const WAKE_SPEED: f32 = 0.25;

/// Contacts between bodies, found once for a whole step with `margin`
/// metres of slack for the motion its substeps will make.
pub fn find_pair_contacts(
    bodies: &mut [Body],
    pairs: &[(usize, usize)],
    margin: f32,
) -> Vec<PairContact> {
    let r = f64::from((SAMPLE_RADIUS + margin) / VOXEL_M);
    // Pairs are independent until their contacts are solved, so search them
    // in parallel: a collapse brings thousands at once.
    let out: Vec<PairContact> = pairs
        .par_iter()
        .with_min_len(16)
        .flat_map_iter(|&(i, j)| {
            let mut out = Vec::new();
            let mut raw = Vec::new();
            for (x, y) in [(i, j), (j, i)] {
                let (sampler, other) = (&bodies[x], &bodies[y]);
                let grid_rot = other.grid_rotation();
                // Sampler principal frame (metres) to the other's grid (voxels):
                // g = m * s + t, one matrix product per sample.
                let to_grid = Mat3::from_quat(grid_rot.inverse()) * (1.0 / VOXEL_M);
                let m = to_grid * Mat3::from_quat(sampler.rot);
                let t = to_grid * (sampler.pos - other.grid_origin()).as_vec3();
                let reach = 1.0 + margin / VOXEL_M;
                let lo = Vec3::splat(-reach);
                let hi = other.shape.size.as_vec3() + reach;
                for &s in &sampler.shape.samples {
                    let g = m * s + t;
                    // Only samples within a voxel of the other grid can touch it.
                    if g.cmplt(lo).any() || g.cmpgt(hi).any() {
                        continue;
                    }
                    raw.clear();
                    sphere_contacts(g.as_dvec3(), r, |c| other.shape.get(c).is_solid(), &mut raw);
                    if raw.is_empty() {
                        continue;
                    }
                    let p = sampler.world_point(s);
                    for &(o, d, _) in &raw {
                        let n = grid_rot * o.as_vec3();
                        let depth = d as f32 * VOXEL_M - margin;
                        let point = p - (n * SAMPLE_RADIUS).as_dvec3();
                        let ra = (point - sampler.pos).as_vec3();
                        let rb = (point - other.pos).as_vec3();
                        let vn = n.dot(sampler.point_velocity(ra) - other.point_velocity(rb));
                        out.push(PairContact {
                            a: x,
                            b: y,
                            ra,
                            rb,
                            la: sampler.rot.inverse() * ra,
                            lb: other.rot.inverse() * rb,
                            pa: point,
                            pb: point,
                            n,
                            depth,
                            vn_before: vn,
                            lambda_n: 0.0,
                        });
                    }
                }
            }
            out
        })
        .collect();
    // A moving body wakes a sleeping one it touches; a merely awake
    // neighbour settling beside it does not.
    let moving = |b: &Body| {
        !b.asleep
            && b.still_time == 0.0
            && b.vel.length() + b.ang_vel.length() * b.shape.radius > WAKE_SPEED
    };
    for c in &out {
        if moving(&bodies[c.a]) || moving(&bodies[c.b]) {
            bodies[c.a].wake();
            bodies[c.b].wake();
        }
    }
    out
}

/// Prepares pair contacts for one substep, as `refresh_contacts` does for
/// world contacts.
pub fn refresh_pair_contacts(bodies: &[Body], contacts: &mut [PairContact]) {
    for c in contacts.iter_mut() {
        let (a, b) = (&bodies[c.a], &bodies[c.b]);
        c.ra = a.rot * c.la;
        c.rb = b.rot * c.lb;
        c.lambda_n = 0.0;
        c.vn_before = c.n.dot(a.point_velocity(c.ra) - b.point_velocity(c.rb));
    }
}

pub fn solve_pair_positions(bodies: &mut [Body], contacts: &mut [PairContact]) {
    for c in contacts.iter_mut() {
        let (a, b) = pair_mut(bodies, c.a, c.b);
        c.ra = a.rot * c.la;
        c.rb = b.rot * c.lb;
        let now_a = a.pos + c.ra.as_dvec3();
        let now_b = b.pos + c.rb.as_dvec3();
        let depth = c.depth - c.n.dot(((now_a - c.pa) - (now_b - c.pb)).as_vec3());
        if depth <= 0.0 {
            continue;
        }
        let wa = if a.asleep {
            0.0
        } else {
            a.generalized_inverse_mass(c.ra, c.n)
        };
        let wb = if b.asleep {
            0.0
        } else {
            b.generalized_inverse_mass(c.rb, c.n)
        };
        if wa + wb <= 0.0 {
            continue;
        }
        let lambda = depth / (wa + wb);
        c.lambda_n += lambda;
        let p = c.n * lambda;
        if !a.asleep {
            a.apply_position_correction(c.ra, p, 1.0);
        }
        if !b.asleep {
            b.apply_position_correction(c.rb, p, -1.0);
        }
        c.ra = a.rot * c.la;
        c.rb = b.rot * c.lb;
    }
}

pub fn solve_pair_velocities(bodies: &mut [Body], contacts: &[PairContact], h: f32) {
    for c in contacts {
        if c.lambda_n <= 0.0 {
            continue;
        }
        let (a, b) = pair_mut(bodies, c.a, c.b);
        let v = a.point_velocity(c.ra) - b.point_velocity(c.rb);
        let vn = c.n.dot(v);
        let vt = v - c.n * vn;
        let mu = 0.5 * (a.friction + b.friction);
        // No bounce from slow contact (Mueller et al. 2020, Eq. 34 note).
        let e = if c.vn_before.abs() <= 2.0 * GRAVITY.length() * h {
            0.0
        } else {
            0.5 * (a.restitution + b.restitution)
        };
        let mut dv = c.n * (-vn + (-e * c.vn_before).max(0.0));
        let vt_len = vt.length();
        if vt_len > 1e-6 {
            let fn_ = c.lambda_n / (h * h);
            dv -= vt / vt_len * (h * mu * fn_ * (a.inv_mass + b.inv_mass)).min(vt_len);
        }
        let len = dv.length();
        if len < 1e-7 {
            continue;
        }
        let dir = dv / len;
        let wa = if a.asleep {
            0.0
        } else {
            a.generalized_inverse_mass(c.ra, dir)
        };
        let wb = if b.asleep {
            0.0
        } else {
            b.generalized_inverse_mass(c.rb, dir)
        };
        if wa + wb <= 0.0 {
            continue;
        }
        let p = dv / (wa + wb);
        if !a.asleep {
            a.apply_velocity_impulse(c.ra, p, 1.0);
        }
        if !b.asleep {
            b.apply_velocity_impulse(c.rb, p, -1.0);
        }
    }
}

fn pair_mut(bodies: &mut [Body], i: usize, j: usize) -> (&mut Body, &mut Body) {
    assert_ne!(i, j);
    if i < j {
        let (lo, hi) = bodies.split_at_mut(j);
        (&mut lo[i], &mut hi[0])
    } else {
        let (lo, hi) = bodies.split_at_mut(i);
        (&mut hi[0], &mut lo[j])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::BodyShape;
    use glam::{IVec3, Quat};
    use mc2_voxel::material::ids;
    use std::sync::Arc;

    #[test]
    fn sweep_finds_exactly_the_overlapping_pairs() {
        let shape = Arc::new(
            BodyShape::from_voxels(IVec3::splat(4), vec![ids::GRANITE; 64]).expect("shape"),
        );
        // Pseudo-random bodies in a 6 m box; compare with all pairs.
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f64) / f64::from(1u32 << 31) * 6.0
        };
        let bodies: Vec<Body> = (0..200)
            .map(|i| {
                let pos = DVec3::new(next(), next(), next());
                Body::new(crate::BodyId(i), shape.clone(), pos, Quat::IDENTITY)
            })
            .collect();
        let mut swept = candidate_pairs(&bodies, 1.0 / 120.0);
        swept.sort_unstable();
        let r = f64::from(shape.radius) * 2.0;
        let mut brute = Vec::new();
        for i in 0..bodies.len() {
            for j in i + 1..bodies.len() {
                if bodies[i].pos.distance_squared(bodies[j].pos) < r * r {
                    brute.push((i, j));
                }
            }
        }
        assert!(!brute.is_empty());
        assert_eq!(swept, brute);
    }
}
