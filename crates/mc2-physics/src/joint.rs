//! Ball joints with a cone limit, for ragdolls (Müller et al. 2020,
//! Sections 3.3 and 3.4).
//!
//! A joint pins a point of one body to a point of another, and keeps an
//! axis fixed in the second within a cone around an axis fixed in the
//! first. Both are hard position constraints solved every substep; the
//! bodies' relative spin is damped at the velocity stage so limbs settle
//! instead of ringing.

use crate::body::{Body, BodyId};
use glam::{DVec3, Vec3};

/// Relative angular velocity damped away per second at a joint.
const DAMPING: f32 = 6.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Joint {
    pub a: BodyId,
    pub b: BodyId,
    /// The pinned point in each body's principal frame, metres.
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    /// The cone's axis in `a`'s frame, and the axis of `b` kept within
    /// `swing` radians of it.
    pub axis_a: Vec3,
    pub axis_b: Vec3,
    pub swing: f32,
}

impl Joint {
    /// A joint at world point `at` between `a` and `b` as they lie now.
    /// `b`'s `axis` (world, now) may swing up to `swing` radians from
    /// `cone` (world, now), which turns with `a`.
    pub fn new(a: &Body, b: &Body, at: DVec3, cone: Vec3, axis: Vec3, swing: f32) -> Self {
        Self {
            a: a.id,
            b: b.id,
            anchor_a: a.local_point(at),
            anchor_b: b.local_point(at),
            axis_a: (a.rot.inverse() * cone).normalize(),
            axis_b: (b.rot.inverse() * axis).normalize(),
            swing,
        }
    }

    /// How far apart the two pinned points are, metres.
    pub fn gap(&self, a: &Body, b: &Body) -> f32 {
        (a.world_point(self.anchor_a) - b.world_point(self.anchor_b)).length() as f32
    }

    /// The angle between the cone's axis and `b`'s, radians.
    pub fn angle(&self, a: &Body, b: &Body) -> f32 {
        let ca = a.rot * self.axis_a;
        let cb = b.rot * self.axis_b;
        ca.cross(cb).length().atan2(ca.dot(cb))
    }
}

/// Both bodies of a joint, mutably.
fn two(bodies: &mut [Body], i: usize, j: usize) -> (&mut Body, &mut Body) {
    debug_assert_ne!(i, j);
    if i < j {
        let (lo, hi) = bodies.split_at_mut(j);
        (&mut lo[i], &mut hi[0])
    } else {
        let (lo, hi) = bodies.split_at_mut(i);
        (&mut hi[0], &mut lo[j])
    }
}

/// Pulls each joint's pinned points together and its axes back inside
/// their cones. `index` gives each joint's two bodies' positions.
pub(crate) fn solve_joint_positions(bodies: &mut [Body], joints: &[(Joint, usize, usize)]) {
    for (j, ia, ib) in joints {
        let (a, b) = two(bodies, *ia, *ib);
        if a.asleep && b.asleep {
            continue;
        }
        // The point: a positional constraint of zero length.
        let ra = a.rot * j.anchor_a;
        let rb = b.rot * j.anchor_b;
        let d = ((b.pos + rb.as_dvec3()) - (a.pos + ra.as_dvec3())).as_vec3();
        let c = d.length();
        if c > 1e-6 {
            let n = d / c;
            let w = a.generalized_inverse_mass(ra, n) + b.generalized_inverse_mass(rb, n);
            let p = n * (c / w);
            a.apply_position_correction(ra, p, 1.0);
            b.apply_position_correction(rb, p, -1.0);
        }
        // The cone: turn the two axes toward each other by what exceeds it.
        let ca = a.rot * j.axis_a;
        let cb = b.rot * j.axis_b;
        let cross = ca.cross(cb);
        let s = cross.length();
        let angle = s.atan2(ca.dot(cb));
        if angle > j.swing && s > 1e-6 {
            let n = cross / s;
            let w = a.angular_inverse_mass(n) + b.angular_inverse_mass(n);
            let p = n * ((angle - j.swing) / w);
            a.apply_rotation_correction(p, 1.0);
            b.apply_rotation_correction(p, -1.0);
        }
    }
}

/// Damps the relative spin across each joint over a substep of `h`.
pub(crate) fn damp_joints(bodies: &mut [Body], joints: &[(Joint, usize, usize)], h: f32) {
    let k = (DAMPING * h).min(1.0);
    for (_, ia, ib) in joints {
        let (a, b) = two(bodies, *ia, *ib);
        if a.asleep || b.asleep {
            continue;
        }
        let dw = (b.ang_vel - a.ang_vel) * k;
        let len = dw.length();
        if len < 1e-6 {
            continue;
        }
        let n = dw / len;
        let w = a.angular_inverse_mass(n) + b.angular_inverse_mass(n);
        let p = dw / w;
        a.apply_angular_impulse(p, 1.0);
        b.apply_angular_impulse(p, -1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::BodyShape;
    use crate::world::PhysicsWorld;
    use glam::{IVec3, Quat};
    use mc2_voxel::material::ids;
    use mc2_voxel::world::VoxelWorld;
    use std::sync::Arc;

    fn ground() -> VoxelWorld {
        let mut w = VoxelWorld::new();
        // A granite floor, top surface at y = 2 m.
        w.fill_box(IVec3::new(0, 0, 0), IVec3::new(255, 31, 255), ids::GRANITE);
        w
    }

    /// A rod `n` voxels long along +Y, two voxels square.
    fn rod(n: i32) -> Arc<BodyShape> {
        Arc::new(
            BodyShape::from_voxels(IVec3::new(2, n, 2), vec![ids::PLANKS; (4 * n) as usize])
                .expect("rod"),
        )
    }

    /// Two rods end to end, standing up, jointed where they meet.
    fn chain(p: &mut PhysicsWorld, base: DVec3, swing: f32) -> (BodyId, BodyId) {
        let lower = p.spawn(rod(8), base + DVec3::Y * 0.25, Quat::IDENTITY);
        let upper = p.spawn(rod(8), base + DVec3::Y * 0.75, Quat::IDENTITY);
        let (a, b) = (p.body(lower).unwrap(), p.body(upper).unwrap());
        let j = Joint::new(a, b, base + DVec3::Y * 0.5, Vec3::Y, Vec3::Y, swing);
        p.add_joint(j, 1);
        (lower, upper)
    }

    #[test]
    fn jointed_bodies_fall_together_and_stay_pinned() {
        let w = ground();
        let mut p = PhysicsWorld::new();
        let (lower, upper) = chain(&mut p, DVec3::new(8.0, 3.0, 8.0), 1.2);
        // Knock the top sideways so the pair folds as it lands.
        p.body_mut(upper).unwrap().vel = Vec3::new(2.0, 0.0, 0.5);
        let mut worst = 0.0f32;
        for step in 0..480 {
            p.step(1.0 / 120.0, &w);
            let j = p.joints[0];
            let (a, b) = (p.body(lower).unwrap(), p.body(upper).unwrap());
            if step > 2 {
                worst = worst.max(j.gap(a, b));
            }
            assert!(
                j.angle(a, b) < j.swing + 0.1,
                "cone broken: {}",
                j.angle(a, b)
            );
        }
        assert!(worst < 0.02, "joint opened by {worst} m");
        let (a, b) = (p.body(lower).unwrap(), p.body(upper).unwrap());
        // Both came down onto the floor and stopped there.
        assert!(a.pos.y < 2.3 && b.pos.y < 2.3, "{} {}", a.pos.y, b.pos.y);
        assert!(a.asleep && b.asleep, "still moving");
    }

    #[test]
    fn a_tight_cone_holds_a_limb_nearly_straight() {
        let w = ground();
        let mut p = PhysicsWorld::new();
        let (lower, upper) = chain(&mut p, DVec3::new(8.0, 5.0, 8.0), 0.2);
        p.body_mut(upper).unwrap().ang_vel = Vec3::new(0.0, 0.0, 12.0);
        for _ in 0..240 {
            p.step(1.0 / 120.0, &w);
            let j = p.joints[0];
            let (a, b) = (p.body(lower).unwrap(), p.body(upper).unwrap());
            assert!(j.angle(a, b) < 0.3, "{}", j.angle(a, b));
        }
    }

    #[test]
    fn parts_of_one_group_pass_through_each_other() {
        let mut p = PhysicsWorld::new();
        let (lower, upper) = chain(&mut p, DVec3::new(8.0, 3.0, 8.0), 1.0);
        assert_eq!(p.body(lower).unwrap().group, p.body(upper).unwrap().group);
        assert_ne!(p.body(lower).unwrap().group, 0);
    }
}
