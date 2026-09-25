//! Characters: a look, a gait and a posed skeleton, moved by what their
//! body does.
//!
//! Every frame each character is posed from its body's interpolated feet:
//! the gait strides at the speed the body really moves, the feet are
//! planted on the voxels under them, and the head turns to what the eyes
//! are pointed at. The posed parts are left in `Drawn` for the renderer.

use crate::Voxels;
use crate::collide::SolidField;
use crate::input::Time;
use crate::player::{Body, Player, View};
use bevy_ecs::prelude::*;
use glam::{DVec3, IVec3, Quat};
pub use mc2_anim::Look;
use mc2_anim::{BONES, Gait, Motion, Part, Placed, Pose, Root};
use mc2_voxel::world::VoxelWorld;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Draw keys of character parts start here, clear of physics body ids.
pub const KEY_BASE: u64 = 1 << 40;
/// The ground under a foot is looked for this far above and below the
/// feet, metres.
const GROUND_UP: f64 = 0.5;
const GROUND_DOWN: f64 = 0.75;
/// Eye height above the feet, metres.
const EYES: f64 = 1.62;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Component)]
pub struct Character {
    /// Stable, for draw keys.
    pub id: u64,
    pub look: Look,
    /// Which way it faces when nobody steers it, radians about +Y.
    pub facing: f32,
    pub parts: Arc<[Part; BONES]>,
    pub gait: Gait,
    pub pose: Pose,
    pub placed: [Placed; BONES],
}

impl Character {
    pub fn new(look: Look) -> Self {
        let pose = Pose::default();
        let root = Root {
            feet: DVec3::ZERO,
            yaw: 0.0,
        };
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            look,
            facing: 0.0,
            parts: Arc::new(mc2_anim::look::parts(&look)),
            gait: Gait::default(),
            pose,
            placed: mc2_anim::place(root, &pose),
        }
    }

    /// Poses the character with its feet at `feet`, facing `yaw`, moving at
    /// `velocity`, over a frame of `dt` seconds; its head turns toward
    /// `look` when there is one.
    #[allow(clippy::too_many_arguments)]
    pub fn pose_at(
        &mut self,
        world: &VoxelWorld,
        feet: DVec3,
        yaw: f32,
        velocity: DVec3,
        on_ground: bool,
        crouching: bool,
        look: Option<DVec3>,
        dt: f32,
    ) {
        let motion = Motion {
            speed: velocity.x.hypot(velocity.z) as f32,
            on_ground,
            crouching,
        };
        self.gait.advance(motion, dt);
        let root = Root { feet, yaw };
        let mut pose = self.gait.pose();
        if on_ground {
            mc2_anim::ik::plant_feet(root, &mut pose, |x, z| ground(world, x, feet.y, z));
        }
        if let Some(target) = look {
            mc2_anim::ik::look_at(root, &mut pose, target);
        }
        self.pose = pose;
        self.placed = mc2_anim::place(root, &pose);
    }

    /// Each part's voxel grid in the world: its corner and rotation.
    pub fn grids(&self) -> [(DVec3, Quat); BONES] {
        mc2_anim::part_grids(&self.placed, &self.parts)
    }

    /// The draw key of one of this character's parts.
    pub fn key(&self, bone: usize) -> u64 {
        KEY_BASE + self.id * BONES as u64 + bone as u64
    }
}

/// The top of the solid voxel nearest height `y` under (x, z) with air
/// above it, metres, or None if there is none close.
pub fn ground(world: &VoxelWorld, x: f64, y: f64, z: f64) -> Option<f64> {
    let vx = (x * 16.0).floor() as i32;
    let vz = (z * 16.0).floor() as i32;
    let top = ((y + GROUND_UP) * 16.0).floor() as i32;
    let bottom = ((y - GROUND_DOWN) * 16.0).floor() as i32;
    (bottom..=top)
        .filter(|&vy| {
            world.solid(IVec3::new(vx, vy, vz)) && !world.solid(IVec3::new(vx, vy + 1, vz))
        })
        .map(|vy| f64::from(vy + 1) / 16.0)
        .min_by(|a, b| (a - y).abs().total_cmp(&(b - y).abs()))
}

/// One posed part, ready to draw.
#[derive(Clone)]
pub struct DrawnPart {
    pub key: u64,
    pub shape: Arc<mc2_physics::BodyShape>,
    pub corner: DVec3,
    pub rotation: Quat,
}

/// Every character part to draw this frame.
#[derive(Resource, Default)]
pub struct Drawn(pub Vec<DrawnPart>);

/// Poses every character from its body and lists its parts for drawing.
/// The player's own is drawn only when the camera is behind it.
pub fn pose_characters(
    time: Res<Time>,
    voxels: Res<Voxels>,
    mut drawn: ResMut<Drawn>,
    mut q: Query<(&Body, Option<&Player>, &mut Character)>,
) {
    drawn.0.clear();
    for (body, player, mut c) in &mut q {
        let feet = body.prev_feet.lerp(body.feet, time.alpha);
        let (yaw, on_ground, crouching, look, visible) = match player {
            Some(p) => (
                p.yaw,
                p.on_ground,
                p.crouching,
                Some(feet + DVec3::Y * EYES + p.forward().as_dvec3() * 8.0),
                p.view == View::ThirdPerson,
            ),
            None => (c.facing, true, false, None, true),
        };
        c.pose_at(
            &voxels.0,
            feet,
            yaw,
            body.velocity,
            on_ground,
            crouching,
            look,
            time.dt,
        );
        if visible {
            for (i, (corner, rotation)) in c.grids().into_iter().enumerate() {
                drawn.0.push(DrawnPart {
                    key: c.key(i),
                    shape: c.parts[i].shape.clone(),
                    corner,
                    rotation,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;

    fn slab(world: &mut VoxelWorld, top: i32) {
        for z in 0..48 {
            for x in 0..48 {
                for y in top - 4..top {
                    world.set_voxel(IVec3::new(x, y, z), ids::GRANITE);
                }
            }
        }
    }

    #[test]
    fn ground_is_the_nearest_top_with_air_above() {
        let mut world = VoxelWorld::new();
        slab(&mut world, 32);
        // Feet at 2 m stand on the slab's top at 2 m.
        assert_eq!(ground(&world, 0.3, 2.0, 0.3), Some(2.0));
        // A little above or below, the same top is found.
        assert_eq!(ground(&world, 0.3, 2.3, 0.3), Some(2.0));
        assert_eq!(ground(&world, 0.3, 1.6, 0.3), Some(2.0));
        // Nothing within reach far above it.
        assert_eq!(ground(&world, 0.3, 4.0, 0.3), None);
    }

    #[test]
    fn a_standing_character_has_its_feet_on_the_ground() {
        let mut world = VoxelWorld::new();
        slab(&mut world, 32);
        let mut c = Character::new(Look::from_seed(5));
        let feet = DVec3::new(0.2, 2.0, 0.1);
        for _ in 0..30 {
            c.pose_at(
                &world,
                feet,
                0.4,
                DVec3::ZERO,
                true,
                false,
                None,
                1.0 / 60.0,
            );
        }
        for shin in [mc2_anim::Bone::ShinL, mc2_anim::Bone::ShinR] {
            let foot = mc2_anim::skeleton::tip(shin, &c.placed);
            assert!((foot.y - 2.0).abs() < 0.03, "{foot}");
        }
        // Parts have distinct keys, clear of physics bodies.
        assert!(c.key(0) >= KEY_BASE && c.key(0) != c.key(1));
    }

    #[test]
    fn walking_moves_the_feet_apart() {
        let mut world = VoxelWorld::new();
        slab(&mut world, 32);
        let mut c = Character::new(Look::from_seed(6));
        let v = DVec3::new(0.0, 0.0, 1.4);
        let mut spread = 0.0f64;
        let mut feet = DVec3::new(1.5, 2.0, 0.5);
        for _ in 0..60 {
            feet += v / 60.0;
            c.pose_at(&world, feet, 0.0, v, true, false, None, 1.0 / 60.0);
            let l = mc2_anim::skeleton::tip(mc2_anim::Bone::ShinL, &c.placed);
            let r = mc2_anim::skeleton::tip(mc2_anim::Bone::ShinR, &c.placed);
            spread = spread.max((l.z - r.z).abs());
        }
        assert!(spread > 0.3, "{spread}");
    }
}
