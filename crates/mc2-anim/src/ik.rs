//! Inverse kinematics: feet planted on uneven ground, and a head that looks
//! at things.
//!
//! The gait poses legs over flat ground at the root's height. Planting
//! finds the ground under each foot, keeps each foot's lift above it, drops
//! the hips if one foot can no longer reach, then bends each leg to its
//! foot with an analytic two-bone solve, knees toward the way the
//! character faces.

use crate::skeleton::{Bone, Placed, Pose, Root, SHIN_VOXELS, THIGH_VOXELS, VOXEL_M, place, tip};
use glam::{DVec3, Quat, Vec3};

/// Legs never quite straighten, so the solve stays well conditioned.
const STRAIGHT: f64 = 0.995;

/// Bends the legs of `pose` so each foot keeps its lift above the ground
/// `ground(x, z)` gives (metres, or None where it is unknown), dropping
/// the hips as far as the lower foot needs.
pub fn plant_feet(root: Root, pose: &mut Pose, ground: impl Fn(f64, f64) -> Option<f64>) {
    let legs = [(Bone::ThighL, Bone::ShinL), (Bone::ThighR, Bone::ShinR)];
    let l1 = f64::from(THIGH_VOXELS * VOXEL_M);
    let l2 = f64::from(SHIN_VOXELS * VOXEL_M);
    let placed = place(root, pose);
    // Where each foot should be: its lift over flat ground, over the real
    // ground beneath it.
    let mut targets = [None; 2];
    for (i, &(_, shin)) in legs.iter().enumerate() {
        let foot = tip(shin, &placed);
        if let Some(g) = ground(foot.x, foot.z) {
            let lift = (foot.y - root.feet.y).max(0.0);
            targets[i] = Some(DVec3::new(foot.x, g + lift, foot.z));
        }
    }
    // Drop the hips until every target is within reach.
    let mut drop = 0.0f64;
    for (i, &(thigh, _)) in legs.iter().enumerate() {
        let Some(t) = targets[i] else { continue };
        let hip = placed[thigh.index()].joint;
        let reach = (l1 + l2) * STRAIGHT;
        let flat = DVec3::new(t.x - hip.x, 0.0, t.z - hip.z).length();
        if flat >= reach {
            continue;
        }
        let lowest = hip.y - (reach * reach - flat * flat).sqrt();
        drop = drop.max(lowest - t.y);
    }
    pose.hip_drop += drop.max(0.0) as f32;
    let placed = place(root, pose);
    let facing = Quat::from_rotation_y(root.yaw) * Vec3::Z;
    for (i, &(thigh, shin)) in legs.iter().enumerate() {
        let Some(t) = targets[i] else { continue };
        let (upper, lower) = two_bone(placed[thigh.index()].joint, t, l1, l2, facing);
        set_world(pose, &placed, thigh, upper);
        let placed_now = place(root, pose);
        set_world(pose, &placed_now, shin, lower);
    }
}

/// Directions of the upper and lower bone that carry a limb from `root` to
/// `target` (clamped to its reach), the joint between them bending toward
/// `pole`.
pub fn two_bone(root: DVec3, target: DVec3, l1: f64, l2: f64, pole: Vec3) -> (Vec3, Vec3) {
    let to = target - root;
    let d = to.length().clamp(1e-4, (l1 + l2) * STRAIGHT);
    let dir = if to.length() > 1e-6 {
        to.normalize()
    } else {
        -DVec3::Y
    };
    // Angle between the upper bone and the root-to-target line.
    let cos_a = ((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)).clamp(-1.0, 1.0);
    let a = cos_a.acos();
    let axis = dir.cross(pole.as_dvec3());
    let upper = if axis.length() > 1e-6 {
        (Quat::from_axis_angle(axis.normalize().as_vec3(), a as f32) * dir.as_vec3()).normalize()
    } else {
        dir.as_vec3()
    };
    let joint = root + upper.as_dvec3() * l1;
    let reach = root + dir * d;
    let lower = (reach - joint).normalize_or_zero().as_vec3();
    (upper, lower)
}

/// Sets `bone`'s local rotation so it points along `dir` (its rest pose
/// hangs down), keeping its parent's twist.
fn set_world(pose: &mut Pose, placed: &[Placed; crate::skeleton::BONES], bone: Bone, dir: Vec3) {
    let parent = bone
        .parent()
        .map_or(Quat::IDENTITY, |p| placed[p.index()].rot);
    let current = placed[bone.index()].rot;
    let now = current * -Vec3::Y;
    let world = Quat::from_rotation_arc(now, dir.normalize()) * current;
    pose.set(bone, parent.inverse() * world);
}

/// Turns the head toward `target` as far as a neck turns: 75 degrees
/// either way, 40 up or down.
pub fn look_at(root: Root, pose: &mut Pose, target: DVec3) {
    let placed = place(root, pose);
    let torso = placed[Bone::Torso.index()].rot;
    let eye = placed[Bone::Head.index()].joint + DVec3::Y * 0.15;
    let dir = (target - eye).as_vec3();
    if dir.length() < 1e-3 {
        return;
    }
    // The direction in the torso's frame: yaw about its up, then pitch.
    let local = torso.inverse() * dir.normalize();
    let yaw = local.x.atan2(local.z).clamp(-1.3, 1.3);
    let pitch = local.y.asin().clamp(-0.7, 0.7);
    pose.set(
        Bone::Head,
        Quat::from_rotation_y(yaw) * Quat::from_rotation_x(-pitch),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_bone_reaches_what_it_can_and_bends_toward_the_pole() {
        let root = DVec3::new(0.0, 1.0, 0.0);
        let target = DVec3::new(0.0, 0.4, 0.2);
        let (u, l) = two_bone(root, target, 0.45, 0.4, Vec3::Z);
        let end = root + u.as_dvec3() * 0.45 + l.as_dvec3() * 0.4;
        assert!(end.distance(target) < 1e-4, "{end}");
        // The knee sits ahead of the line from hip to foot.
        let knee = root + u.as_dvec3() * 0.45;
        assert!(knee.z > 0.1, "{knee}");
        // Out of reach: as close as a nearly straight limb gets.
        let (u, l) = two_bone(root, DVec3::new(0.0, -2.0, 0.0), 0.45, 0.4, Vec3::Z);
        let end = root + u.as_dvec3() * 0.45 + l.as_dvec3() * 0.4;
        assert!((end.y - (1.0 - 0.85 * STRAIGHT)).abs() < 1e-3, "{end}");
    }

    #[test]
    fn feet_find_a_step_and_the_hips_come_down_for_the_lower_one() {
        let root = Root {
            feet: DVec3::new(0.0, 10.0, 0.0),
            yaw: 0.0,
        };
        // Ground 25 cm higher under the left foot (+X) than under the right.
        let ground = |x: f64, _z: f64| Some(if x > 0.0 { 10.25 } else { 10.0 });
        let mut pose = Pose::default();
        plant_feet(root, &mut pose, ground);
        let placed = place(root, &pose);
        let (l, r) = (tip(Bone::ShinL, &placed), tip(Bone::ShinR, &placed));
        assert!((l.y - 10.25).abs() < 0.01, "left {l}");
        assert!((r.y - 10.0).abs() < 0.01, "right {r}");
        // Standing on a slope down to the right: the hips stayed where the
        // right leg could reach, the left knee bent.
        let lower = Root {
            feet: DVec3::new(0.0, 10.0, 0.0),
            yaw: 0.0,
        };
        let ground = |x: f64, _z: f64| Some(if x > 0.0 { 10.0 } else { 9.8 });
        let mut pose = Pose::default();
        plant_feet(lower, &mut pose, ground);
        assert!(pose.hip_drop > 0.15, "{}", pose.hip_drop);
        let placed = place(lower, &pose);
        assert!((tip(Bone::ShinR, &placed).y - 9.8).abs() < 0.01);
        assert!((tip(Bone::ShinL, &placed).y - 10.0).abs() < 0.01);
    }

    #[test]
    fn heads_turn_toward_things_but_not_all_the_way_round() {
        let root = Root {
            feet: DVec3::ZERO,
            yaw: 0.0,
        };
        let mut pose = Pose::default();
        look_at(root, &mut pose, DVec3::new(5.0, 1.7, 5.0));
        let head = place(root, &pose)[Bone::Head.index()].rot * Vec3::Z;
        assert!(head.x > 0.6 && head.z > 0.6, "{head}");
        look_at(root, &mut pose, DVec3::new(0.0, 1.7, -5.0));
        let head = place(root, &pose)[Bone::Head.index()].rot * Vec3::Z;
        // Straight behind: the neck stops short.
        assert!(head.z > 0.2, "{head}");
    }
}
