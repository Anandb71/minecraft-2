//! The humanoid skeleton: eleven bones, each a voxel part hung from a joint.
//!
//! Character space: +Y up, +Z the way the character faces, +X its left. A
//! bone's joint sits at a fixed offset from its parent's joint in the rest
//! pose (standing, arms down); its part hangs from that joint. Lengths are
//! in voxels of 6.25 cm, so a character stands 29 voxels, 1.81 m, tall.

use glam::{DVec3, Quat, Vec3};

/// Metres in a voxel.
pub const VOXEL_M: f32 = 1.0 / 16.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Bone {
    Pelvis,
    Torso,
    Head,
    UpperArmL,
    ForeArmL,
    UpperArmR,
    ForeArmR,
    ThighL,
    ShinL,
    ThighR,
    ShinR,
}

pub const BONES: usize = 11;

impl Bone {
    /// Every bone, parents before children.
    pub const ALL: [Bone; BONES] = [
        Bone::Pelvis,
        Bone::Torso,
        Bone::Head,
        Bone::UpperArmL,
        Bone::ForeArmL,
        Bone::UpperArmR,
        Bone::ForeArmR,
        Bone::ThighL,
        Bone::ShinL,
        Bone::ThighR,
        Bone::ShinR,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn parent(self) -> Option<Bone> {
        use Bone::*;
        match self {
            Pelvis => None,
            Torso | ThighL | ThighR => Some(Pelvis),
            Head | UpperArmL | UpperArmR => Some(Torso),
            ForeArmL => Some(UpperArmL),
            ForeArmR => Some(UpperArmR),
            ShinL => Some(ThighL),
            ShinR => Some(ThighR),
        }
    }

    /// The joint's offset from the parent's joint in the rest pose, voxels
    /// (for the pelvis: the hips above the feet).
    pub fn rest_offset(self) -> Vec3 {
        use Bone::*;
        match self {
            Pelvis => Vec3::new(0.0, LEG_VOXELS, 0.0),
            Torso => Vec3::new(0.0, 2.0, 0.0),
            Head => Vec3::new(0.0, 9.0, 0.0),
            UpperArmL => Vec3::new(5.0, 8.5, 0.0),
            UpperArmR => Vec3::new(-5.0, 8.5, 0.0),
            ForeArmL | ForeArmR => Vec3::new(0.0, -5.0, 0.0),
            ThighL => Vec3::new(1.5, 0.0, 0.0),
            ThighR => Vec3::new(-1.5, 0.0, 0.0),
            ShinL | ShinR => Vec3::new(0.0, -THIGH_VOXELS, 0.0),
        }
    }

    /// The part's length along its bone, voxels: from its joint to the next
    /// joint down (or its far end).
    pub fn length(self) -> f32 {
        use Bone::*;
        match self {
            Pelvis => 2.0,
            Torso => 9.0,
            Head => 5.0,
            UpperArmL | UpperArmR | ForeArmL | ForeArmR => 5.0,
            ThighL | ThighR => THIGH_VOXELS,
            ShinL | ShinR => SHIN_VOXELS,
        }
    }

    pub fn is_left(self) -> bool {
        matches!(
            self,
            Bone::UpperArmL | Bone::ForeArmL | Bone::ThighL | Bone::ShinL
        )
    }
}

pub const THIGH_VOXELS: f32 = 7.0;
/// Knee to sole, the foot included.
pub const SHIN_VOXELS: f32 = 6.0;
pub const LEG_VOXELS: f32 = THIGH_VOXELS + SHIN_VOXELS;
/// Feet to the crown, voxels.
pub const HEIGHT_VOXELS: f32 = LEG_VOXELS + 2.0 + 9.0 + 5.0;

/// Where a character stands and which way it faces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Root {
    /// Between the feet, on the ground, metres.
    pub feet: DVec3,
    /// Radians about +Y; 0 faces +Z.
    pub yaw: f32,
}

/// Each bone's rotation relative to its parent (identity: the rest pose),
/// and how far the hips sit below their rest height (crouching, a stride's
/// bob), metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub local: [Quat; BONES],
    pub hip_drop: f32,
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            local: [Quat::IDENTITY; BONES],
            hip_drop: 0.0,
        }
    }
}

impl Pose {
    pub fn get(&self, b: Bone) -> Quat {
        self.local[b.index()]
    }

    pub fn set(&mut self, b: Bone, q: Quat) {
        self.local[b.index()] = q;
    }
}

/// A bone placed in the world: its joint (metres) and its rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub joint: DVec3,
    pub rot: Quat,
}

/// Forward kinematics: every bone's joint and rotation in the world.
pub fn place(root: Root, pose: &Pose) -> [Placed; BONES] {
    let facing = Quat::from_rotation_y(root.yaw);
    let mut out = [Placed {
        joint: root.feet,
        rot: facing,
    }; BONES];
    for b in Bone::ALL {
        let local = pose.get(b);
        let offset = b.rest_offset() * VOXEL_M;
        out[b.index()] = match b.parent() {
            None => Placed {
                joint: root.feet + (facing * offset).as_dvec3()
                    - DVec3::Y * f64::from(pose.hip_drop),
                rot: facing * local,
            },
            Some(p) => {
                let parent = out[p.index()];
                Placed {
                    joint: parent.joint + (parent.rot * offset).as_dvec3(),
                    rot: parent.rot * local,
                }
            }
        };
    }
    out
}

/// Where a bone's far end is: its joint plus its length down the bone.
pub fn tip(b: Bone, placed: &[Placed; BONES]) -> DVec3 {
    let p = placed[b.index()];
    let down = if b == Bone::Head || b == Bone::Torso {
        Vec3::Y
    } else {
        -Vec3::Y
    };
    p.joint + (p.rot * down * (b.length() * VOXEL_M)).as_dvec3()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rest_pose_stands_its_height_with_feet_on_the_ground() {
        let root = Root {
            feet: DVec3::new(10.0, 5.0, -3.0),
            yaw: 0.0,
        };
        let placed = place(root, &Pose::default());
        let crown = tip(Bone::Head, &placed);
        let height = crown.y - root.feet.y;
        assert!((height - f64::from(HEIGHT_VOXELS * VOXEL_M)).abs() < 1e-5);
        assert!((height - 1.8125).abs() < 1e-5);
        for shin in [Bone::ShinL, Bone::ShinR] {
            assert!((tip(shin, &placed).y - root.feet.y).abs() < 1e-5);
        }
        // Left is +X when facing +Z.
        assert!(placed[Bone::UpperArmL.index()].joint.x > root.feet.x);
    }

    #[test]
    fn turning_the_root_turns_everything() {
        let pose = Pose::default();
        let a = place(
            Root {
                feet: DVec3::ZERO,
                yaw: 0.0,
            },
            &pose,
        );
        let b = place(
            Root {
                feet: DVec3::ZERO,
                yaw: std::f32::consts::FRAC_PI_2,
            },
            &pose,
        );
        // A quarter turn takes the left arm from +X to -Z.
        let l = b[Bone::UpperArmL.index()].joint;
        assert!(l.z < -0.25 && l.x.abs() < 1e-5, "{l}");
        assert!((a[Bone::Head.index()].joint.y - b[Bone::Head.index()].joint.y).abs() < 1e-5);
    }

    #[test]
    fn a_bent_knee_moves_the_foot_back_and_up() {
        let mut pose = Pose::default();
        pose.set(Bone::ShinL, Quat::from_rotation_x(1.0));
        let placed = place(
            Root {
                feet: DVec3::ZERO,
                yaw: 0.0,
            },
            &pose,
        );
        let foot = tip(Bone::ShinL, &placed);
        assert!(foot.z < -0.2 && foot.y > 0.1, "{foot}");
    }
}
