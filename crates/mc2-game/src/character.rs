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
use crate::physics::Physics;
use crate::physics_host::{Hang, RagdollPart};
use crate::player::{Body, Player, View};
use bevy_ecs::prelude::*;
use glam::{DVec3, IVec3, Quat};
pub use mc2_anim::Look;
use mc2_anim::{BONES, Bone, Gait, Motion, Part, Placed, Pose, Root, VOXEL_M};
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
/// A blast throws a character within this many of its radii.
const BLAST_REACH: f64 = 2.5;
/// Speed a character is thrown at from the heart of a blast, m/s.
const BLAST_SPEED: f64 = 12.0;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Component)]
pub struct Character {
    /// Stable, for draw keys.
    pub id: u64,
    pub look: Look,
    /// Which way it faces when nobody steers it, radians about +Y.
    pub facing: f32,
    /// What it looks at when nobody steers it.
    pub look_at: Option<DVec3>,
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
            look_at: None,
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

    /// The character as it stands, as ragdoll parts: each part a body at
    /// its posed place, hung from its parent at the joint between them,
    /// free to swing about as far as that joint can.
    pub fn ragdoll(&self) -> Vec<RagdollPart> {
        let grids = self.grids();
        Bone::ALL
            .iter()
            .map(|&bone| {
                let i = bone.index();
                let shape = self.parts[i].shape.clone();
                let (corner, grid_rot) = grids[i];
                let hang = bone.parent().map(|p| {
                    let dir = along(bone);
                    let parent = self.placed[p.index()].rot;
                    Hang {
                        parent: p.index(),
                        at: self.placed[i].joint,
                        cone: parent * dir,
                        axis: self.placed[i].rot * dir,
                        swing: swing(bone),
                        // Knees and elbows turn about the limb's left axis:
                        // knees back, elbows forward.
                        hinge: bend(bone).map(|(min, max)| (parent * glam::Vec3::X, min, max)),
                    }
                });
                RagdollPart {
                    pos: corner + (grid_rot * (shape.com * VOXEL_M)).as_dvec3(),
                    rot: grid_rot * shape.principal,
                    vel: glam::Vec3::ZERO,
                    shape,
                    hang,
                }
            })
            .collect()
    }

    /// The draw key of one of this character's parts.
    pub fn key(&self, bone: usize) -> u64 {
        KEY_BASE + self.id * BONES as u64 + bone as u64
    }
}

/// Which way a bone runs from its joint in the rest pose.
fn along(bone: Bone) -> glam::Vec3 {
    match bone {
        Bone::Torso | Bone::Head => glam::Vec3::Y,
        _ => -glam::Vec3::Y,
    }
}

/// How far a bone may swing from its rest direction, radians.
fn swing(bone: Bone) -> f32 {
    match bone {
        Bone::Pelvis => 0.0,
        Bone::Torso => 0.5,
        Bone::Head => 0.7,
        Bone::UpperArmL | Bone::UpperArmR => 2.2,
        Bone::ThighL | Bone::ThighR => 1.3,
        // Hinged instead.
        Bone::ForeArmL | Bone::ForeArmR | Bone::ShinL | Bone::ShinR => 0.0,
    }
}

/// The bend a knee or elbow allows about the limb's left axis, radians.
fn bend(bone: Bone) -> Option<(f32, f32)> {
    match bone {
        Bone::ShinL | Bone::ShinR => Some((0.0, 2.4)),
        Bone::ForeArmL | Bone::ForeArmR => Some((-2.5, 0.0)),
        _ => None,
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
                p.view == View::ThirdPerson && !p.seated,
            ),
            None => (c.facing, true, false, c.look_at, true),
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

/// Characters caught in a blast go limp: each becomes a ragdoll of its
/// parts, every part thrown away from the blast and upward by how near it
/// was, so the nearer limbs fly first. Runs before the fire takes the
/// blasts.
pub fn blast_people(
    mut commands: Commands,
    mut physics: ResMut<Physics>,
    people: Query<(Entity, &Body, &Character), Without<Player>>,
) {
    if physics.blasts_out.is_empty() {
        return;
    }
    let blasts = physics.blasts_out.clone();
    for (e, body, c) in &people {
        let chest = body.feet + DVec3::Y * 1.1;
        let Some(&(centre, radius)) = blasts
            .iter()
            .find(|(centre, radius)| chest.distance(*centre) < f64::from(*radius) * BLAST_REACH)
        else {
            continue;
        };
        let reach = f64::from(radius) * BLAST_REACH;
        let mut parts = c.ragdoll();
        for p in &mut parts {
            let d = p.pos - centre;
            let near = (1.0 - d.length() / reach).max(0.0);
            let dir = (d.normalize_or_zero() + DVec3::Y * 0.7).normalize();
            p.vel = (dir * BLAST_SPEED * near.sqrt()).as_vec3();
        }
        physics.host.spawn_ragdoll(&parts);
        commands.entity(e).despawn();
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
    fn a_ragdoll_falls_in_one_piece() {
        let mut world = VoxelWorld::new();
        // A 16 m floor, its top at 2 m.
        world.fill_box(IVec3::ZERO, IVec3::new(255, 31, 255), ids::GRANITE);
        let mut c = Character::new(Look::from_seed(9));
        c.pose_at(
            &world,
            DVec3::new(8.0, 2.0, 8.0),
            0.3,
            DVec3::ZERO,
            true,
            false,
            None,
            0.0,
        );
        let parts = c.ragdoll();
        assert_eq!(parts.len(), BONES);
        let mut host = crate::physics_host::PhysicsHost::inline();
        let parts: Vec<RagdollPart> = parts
            .into_iter()
            .map(|p| RagdollPart {
                vel: glam::Vec3::new(1.5, 2.0, 0.0),
                ..p
            })
            .collect();
        let ids = host.spawn_ragdoll(&parts);
        for _ in 0..360 {
            host.tick(&world, 1.0 / 120.0);
        }
        let pelvis = host.body(ids[0]).expect("pelvis").pos;
        for (id, part) in ids.iter().zip(&parts) {
            let b = host.body(*id).expect("part");
            // Nothing flew off or sank: every part lies on the slab, near
            // the pelvis.
            assert!(
                b.pos.y > 1.95 && b.pos.y < 2.6,
                "{:?} at {}",
                part.hang,
                b.pos
            );
            assert!(
                b.pos.distance(pelvis) < 1.2,
                "{} from the pelvis",
                b.pos.distance(pelvis)
            );
        }
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
