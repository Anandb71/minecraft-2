//! Procedural motion: a stride cycle whose speed blends walking into
//! running, idle breathing, legs tucked in the air and a crouch, each eased
//! in and out so nothing snaps.
//!
//! The stride advances with distance covered, not time, so feet do not
//! slide: a walking stride (two steps) is 1.3 m, a running one 2.4 m.

use crate::skeleton::{Bone, Pose};
use glam::Quat;
use std::f32::consts::TAU;

/// What a character is doing this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Motion {
    /// Horizontal speed, metres a second.
    pub speed: f32,
    pub on_ground: bool,
    pub crouching: bool,
}

/// The state of a character's motion from frame to frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gait {
    /// Stride phase, 0..1 a cycle: the left foot is forward at 0.25.
    pub phase: f32,
    /// Seconds running, for breathing.
    pub time: f32,
    /// Speed, eased.
    pub speed: f32,
    /// How far into the air pose, 0..1.
    pub airborne: f32,
    /// How far into the crouch, 0..1.
    pub crouch: f32,
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn ease(from: f32, to: f32, rate: f32, dt: f32) -> f32 {
    from + (to - from) * (1.0 - (-rate * dt).exp())
}

/// A limb swung forward by `a` radians about the character's left axis.
fn forward(a: f32) -> Quat {
    Quat::from_rotation_x(-a)
}

/// A shin or forearm folded by `a`: knees fold back, elbows forward.
fn fold(bone: Bone, a: f32) -> Quat {
    match bone {
        Bone::ShinL | Bone::ShinR => Quat::from_rotation_x(a),
        _ => Quat::from_rotation_x(-a),
    }
}

/// An arm held out to its side by `a` radians.
fn out(bone: Bone, a: f32) -> Quat {
    Quat::from_rotation_z(if bone.is_left() { a } else { -a })
}

impl Gait {
    pub fn advance(&mut self, m: Motion, dt: f32) {
        self.speed = ease(self.speed, m.speed, 8.0, dt);
        self.airborne = ease(self.airborne, if m.on_ground { 0.0 } else { 1.0 }, 7.0, dt);
        self.crouch = ease(self.crouch, if m.crouching { 1.0 } else { 0.0 }, 9.0, dt);
        self.time += dt;
        let run = smoothstep(2.5, 4.5, self.speed);
        let stride = 1.3 + 1.1 * run;
        if m.on_ground {
            self.phase = (self.phase + m.speed * dt / stride).fract();
        }
    }

    pub fn pose(&self) -> Pose {
        let run = smoothstep(2.5, 4.5, self.speed);
        let walk = smoothstep(0.05, 0.7, self.speed);
        let (s, c) = (TAU * self.phase).sin_cos();
        let mut pose = Pose::default();

        // Legs: thighs swing opposite ways; a knee folds as its leg swings
        // through, more when running.
        let swing = walk * (0.4 + 0.35 * run);
        let lift = walk * (0.25 + 0.95 * run);
        let mut thigh = [swing * s, -swing * s];
        let mut knee = [0.06 + lift * c.max(0.0), 0.06 + lift * (-c).max(0.0)];
        // Arms swing against the legs, elbows bent more when running.
        let arm_swing = walk * (0.35 + 0.35 * run);
        let mut shoulder = [-arm_swing * s, arm_swing * s];
        let mut elbow = 0.12 + 1.1 * run;
        let mut arm_out = 0.06;
        let mut lean = 0.04 * walk + 0.22 * run;
        // Breathing when standing still.
        let breath = (1.0 - walk) * 0.02 * (self.time * 1.7).sin();
        let mut hip_drop =
            walk * (0.015 + 0.045 * run) * (1.0 - (2.0 * TAU * self.phase).cos()) * 0.5;

        // In the air: knees up, one leg ahead, arms up and out.
        let a = self.airborne;
        thigh = [
            thigh[0] * (1.0 - a) + 0.75 * a,
            thigh[1] * (1.0 - a) + 0.3 * a,
        ];
        knee = [knee[0] * (1.0 - a) + 1.0 * a, knee[1] * (1.0 - a) + 0.6 * a];
        shoulder = [
            shoulder[0] * (1.0 - a) + 0.5 * a,
            shoulder[1] * (1.0 - a) - 0.3 * a,
        ];
        arm_out += 0.45 * a;
        elbow = elbow * (1.0 - a) + 0.6 * a;

        // Crouching: hips down, thighs forward, knees folded, torso over them.
        let k = self.crouch;
        hip_drop += 0.36 * k;
        for t in &mut thigh {
            *t += 1.0 * k;
        }
        for kn in &mut knee {
            *kn += 1.7 * k;
        }
        lean += 0.45 * k;
        elbow += 0.4 * k;

        pose.hip_drop = hip_drop;
        pose.set(Bone::ThighL, forward(thigh[0]));
        pose.set(Bone::ThighR, forward(thigh[1]));
        pose.set(Bone::ShinL, fold(Bone::ShinL, knee[0]));
        pose.set(Bone::ShinR, fold(Bone::ShinR, knee[1]));
        pose.set(
            Bone::UpperArmL,
            forward(shoulder[0]) * out(Bone::UpperArmL, arm_out),
        );
        pose.set(
            Bone::UpperArmR,
            forward(shoulder[1]) * out(Bone::UpperArmR, arm_out),
        );
        pose.set(Bone::ForeArmL, fold(Bone::ForeArmL, elbow));
        pose.set(Bone::ForeArmR, fold(Bone::ForeArmR, elbow));
        // The torso leans and twists a little against the hips.
        let twist = 0.08 * walk * s;
        pose.set(
            Bone::Torso,
            Quat::from_rotation_x(lean + breath) * Quat::from_rotation_y(twist),
        );
        // The head stays level: it takes back the lean.
        pose.set(Bone::Head, Quat::from_rotation_x(-(lean + breath) * 0.7));
        pose
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Root, place, tip};
    use glam::DVec3;

    fn run_for(g: &mut Gait, m: Motion, seconds: f32) {
        for _ in 0..(seconds * 60.0) as usize {
            g.advance(m, 1.0 / 60.0);
        }
    }

    #[test]
    fn walking_swings_opposite_legs_and_standing_still_does_not() {
        let mut g = Gait::default();
        run_for(
            &mut g,
            Motion {
                speed: 1.5,
                on_ground: true,
                crouching: false,
            },
            3.0,
        );
        g.phase = 0.25;
        let root = Root {
            feet: DVec3::ZERO,
            yaw: 0.0,
        };
        let placed = place(root, &g.pose());
        let (l, r) = (tip(Bone::ShinL, &placed), tip(Bone::ShinR, &placed));
        // At a quarter cycle the left foot is ahead, the right behind.
        assert!(l.z > 0.1 && r.z < -0.1, "left {l}, right {r}");
        let mut still = Gait::default();
        let standing = Motion {
            on_ground: true,
            ..Motion::default()
        };
        run_for(&mut still, standing, 3.0);
        let placed = place(root, &still.pose());
        let (l, r) = (tip(Bone::ShinL, &placed), tip(Bone::ShinR, &placed));
        assert!(l.z.abs() < 0.05 && r.z.abs() < 0.05, "left {l}, right {r}");
    }

    #[test]
    fn strides_follow_distance_so_feet_do_not_slide() {
        let mut g = Gait::default();
        let m = Motion {
            speed: 1.3,
            on_ground: true,
            crouching: false,
        };
        run_for(&mut g, m, 2.0);
        let before = g.phase;
        // One more second at 1.3 m/s is exactly one walking stride.
        run_for(&mut g, m, 1.0);
        let moved = (g.phase - before).rem_euclid(1.0);
        assert!(moved < 0.02 || moved > 0.98, "{moved}");
    }

    #[test]
    fn crouching_lowers_the_hips_and_keeps_the_feet_down() {
        let mut g = Gait::default();
        run_for(
            &mut g,
            Motion {
                speed: 0.0,
                on_ground: true,
                crouching: true,
            },
            2.0,
        );
        let pose = g.pose();
        assert!(pose.hip_drop > 0.3);
        let placed = place(
            Root {
                feet: DVec3::ZERO,
                yaw: 0.0,
            },
            &pose,
        );
        // Without IK the feet are only roughly under the body.
        let l = tip(Bone::ShinL, &placed);
        assert!(l.y.abs() < 0.25, "{l}");
    }
}
