//! Debris in the game: the rigid body world stepped on the fixed clock,
//! explosives with fuses, and settled debris baked back into terrain.
//!
//! TNT is a material. Using a TNT block (the interact key) lifts its voxels
//! out of the world as a body with a lit fuse; when the fuse runs out the
//! body detonates wherever it has been thrown. A blast lights every TNT
//! block it reaches with a short random fuse, so charges chain.

use crate::input::{Input, Key, Time};
use crate::interact::{Interaction, prepare_edit};
use crate::{Streaming, Voxels};
use bevy_ecs::prelude::*;
use glam::{DVec3, IVec3};
use mc2_core::FxHashSet;
use mc2_physics::explode::{ExplosionReport, bake_settled, explode};
use mc2_physics::{BodyId, BodyShape, PhysicsWorld};
use mc2_voxel::coords::{BlockPos, VOXELS_PER_BLOCK};
use mc2_voxel::material::{MaterialId, ids};
use mc2_voxel::world::VoxelWorld;
use std::sync::Arc;

/// Blast radius of one TNT block, metres.
pub const TNT_RADIUS_M: f32 = 3.0;
/// Fuse of a TNT block lit by hand, seconds.
pub const TNT_FUSE_S: f32 = 4.0;
/// Fragments thrown by one blast, at most.
pub const DEBRIS_PER_BLAST: usize = 48;
/// Bodies alive at once before the oldest sleepers are baked early.
pub const MAX_BODIES: usize = 600;
/// Debris at rest this long becomes terrain again, seconds.
pub const BAKE_AFTER_S: f32 = 15.0;
/// Baking runs this often, seconds.
const BAKE_INTERVAL_S: f64 = 0.5;

#[derive(Clone, Copy, Debug)]
pub struct Fuse {
    pub body: BodyId,
    pub remaining: f32,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Blast {
    pub centre: DVec3,
    pub radius: f32,
}

#[derive(Resource)]
pub struct Physics {
    pub world: PhysicsWorld,
    pub fuses: Vec<Fuse>,
    /// Blasts to set off on the next tick.
    pub blasts: Vec<Blast>,
    /// Voxel boxes edited since the last tick; bodies resting there wake.
    pub dirty: Vec<(IVec3, IVec3)>,
    /// Last blast, for the HUD and demos.
    pub last_blast: Option<ExplosionReport>,
    pub blasts_total: u64,
    next_bake: f64,
    seed: u64,
}

impl Default for Physics {
    fn default() -> Self {
        Self {
            world: PhysicsWorld::new(),
            fuses: Vec::new(),
            blasts: Vec::new(),
            dirty: Vec::new(),
            last_blast: None,
            blasts_total: 0,
            next_bake: 0.0,
            seed: 0x51f1_5eed,
        }
    }
}

impl Physics {
    fn next_seed(&mut self) -> u64 {
        self.seed = self
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.seed
    }

    /// Whether a body carries a lit fuse.
    pub fn is_primed(&self, id: BodyId) -> bool {
        self.fuses.iter().any(|f| f.body == id)
    }
}

/// Lifts the TNT voxels of a 1 m block out of the world as a body with a
/// lit fuse. `None` when the block holds no TNT.
pub fn ignite_block(
    world: &mut VoxelWorld,
    physics: &mut Physics,
    block: BlockPos,
    fuse_s: f32,
) -> Option<BodyId> {
    let o = block.origin();
    let n = VOXELS_PER_BLOCK;
    let mut voxels = vec![MaterialId(0); (n * n * n) as usize];
    let mut any = false;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let v = o + IVec3::new(x, y, z);
                if world.voxel(v) == ids::TNT {
                    voxels[(x + n * (y + n * z)) as usize] = ids::TNT;
                    world.set_voxel(v, ids::AIR);
                    any = true;
                }
            }
        }
    }
    if !any {
        return None;
    }
    let shape = Arc::new(BodyShape::from_voxels(IVec3::splat(n), voxels)?);
    let com = (o.as_dvec3() + shape.com.as_dvec3()) / f64::from(n);
    let rot = shape.principal;
    let id = physics.world.spawn(shape, com, rot);
    physics.fuses.push(Fuse {
        body: id,
        remaining: fuse_s,
        radius: TNT_RADIUS_M,
    });
    physics.dirty.push((o, o + n - 1));
    Some(id)
}

/// Sets off one blast: lights TNT in reach, restores generated detail in
/// the blast box, carves the crater and throws debris.
pub fn detonate(
    world: &mut VoxelWorld,
    streaming: &mut Streaming,
    interaction: &mut Interaction,
    physics: &mut Physics,
    blast: Blast,
) -> ExplosionReport {
    mc2_core::scope!("physics.detonate");
    let c = blast.centre * f64::from(VOXELS_PER_BLOCK);
    let r = f64::from(blast.radius) * f64::from(VOXELS_PER_BLOCK) * 1.3;
    let lo = (c - r).floor().as_ivec3();
    let hi = (c + r).ceil().as_ivec3();
    prepare_edit(
        world,
        streaming.0.as_mut(),
        &mut interaction.touched,
        lo,
        hi,
    );
    // Chain reaction: every TNT block the blast reaches is lit.
    let mut blocks = FxHashSet::default();
    for z in (lo.z >> 4)..=(hi.z >> 4) {
        for y in (lo.y >> 4)..=(hi.y >> 4) {
            for x in (lo.x >> 4)..=(hi.x >> 4) {
                let b = BlockPos(IVec3::new(x, y, z));
                let centre = (b.origin().as_dvec3() + 8.0) - c;
                if centre.length() <= r && block_has_tnt(world, b) {
                    blocks.insert(b);
                }
            }
        }
    }
    for b in blocks {
        let fuse = 0.15 + (physics.next_seed() >> 40) as f32 / (1u64 << 24) as f32 * 0.6;
        ignite_block(world, physics, b, fuse);
    }
    let seed = physics.next_seed();
    let report = explode(
        world,
        &mut physics.world,
        blast.centre,
        blast.radius,
        DEBRIS_PER_BLAST,
        seed,
    );
    physics.dirty.push((lo, hi));
    interaction.edits += 1;
    physics.blasts_total += 1;
    physics.last_blast = Some(report);
    report
}

fn block_has_tnt(world: &VoxelWorld, b: BlockPos) -> bool {
    let o = b.origin();
    // TNT blocks are placed whole; probe their eight interior points.
    (0..8).any(|i| {
        let p = o + IVec3::new(
            4 + 8 * (i & 1),
            4 + 8 * ((i >> 1) & 1),
            4 + 8 * ((i >> 2) & 1),
        );
        world.voxel(p) == ids::TNT
    })
}

/// The interact key lights a targeted TNT block.
pub fn light_fuses(
    mut voxels: ResMut<Voxels>,
    mut physics: ResMut<Physics>,
    mut interaction: ResMut<Interaction>,
    input: Res<Input>,
) {
    if !input.captured || !input.pressed(Key::Interact) {
        return;
    }
    let Some(t) = interaction.target else {
        return;
    };
    if t.hit.material != ids::TNT {
        return;
    }
    if ignite_block(&mut voxels.0, &mut physics, t.block, TNT_FUSE_S).is_some() {
        interaction.edits += 1;
    }
}

/// One fixed tick: wake bodies under edits, step, burn fuses, set off
/// blasts, and bake debris that has settled.
pub fn step_physics(
    mut voxels: ResMut<Voxels>,
    mut streaming: ResMut<Streaming>,
    mut interaction: ResMut<Interaction>,
    mut physics: ResMut<Physics>,
    time: Res<Time>,
) {
    mc2_core::scope!("physics.tick");
    let physics = &mut *physics;
    let world = &mut voxels.0;
    physics.dirty.append(&mut interaction.edited);
    for (lo, hi) in std::mem::take(&mut physics.dirty) {
        let m = f64::from(VOXELS_PER_BLOCK);
        physics
            .world
            .wake_region(lo.as_dvec3() / m, (hi + 1).as_dvec3() / m);
    }
    let dt = time.fixed_dt as f32;
    physics.world.step(dt, world);

    for f in &mut physics.fuses {
        f.remaining -= dt;
    }
    let (burnt, lit): (Vec<Fuse>, Vec<Fuse>) =
        physics.fuses.drain(..).partition(|f| f.remaining <= 0.0);
    physics.fuses = lit;
    for f in burnt {
        if let Some(body) = physics.world.remove(f.body) {
            physics.blasts.push(Blast {
                centre: body.pos,
                radius: f.radius,
            });
        }
    }
    for blast in std::mem::take(&mut physics.blasts) {
        detonate(world, &mut streaming, &mut interaction, physics, blast);
    }

    if time.elapsed >= physics.next_bake {
        physics.next_bake = time.elapsed + BAKE_INTERVAL_S;
        let primed: Vec<BodyId> = physics.fuses.iter().map(|f| f.body).collect();
        let baked = bake_settled(world, &mut physics.world, BAKE_AFTER_S, MAX_BODIES, |id| {
            primed.contains(&id)
        });
        if !baked.is_empty() {
            interaction.edits += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Game;

    fn ground_with_tnt(game: &mut Game, blocks: &[IVec3]) {
        let voxels = &mut game.world.resource_mut::<Voxels>().0;
        voxels.fill_box(IVec3::ZERO, IVec3::new(255, 63, 255), ids::GRANITE);
        for b in blocks {
            let o = BlockPos(*b).origin();
            voxels.fill_box(o, o + 15, ids::TNT);
        }
    }

    #[test]
    fn a_lit_block_flies_free_then_blows_a_crater() {
        let mut game = Game::new();
        ground_with_tnt(&mut game, &[IVec3::new(8, 4, 8)]);
        let block = BlockPos(IVec3::new(8, 4, 8));
        {
            let world = &mut game.world;
            let mut voxels = world.resource_mut::<Voxels>();
            let v = &mut voxels.0;
            let mut physics = Physics::default();
            let id = ignite_block(v, &mut physics, block, 0.5).expect("tnt");
            assert!(physics.is_primed(id));
            assert!(v.voxel(block.origin() + 8).is_air());
            world.insert_resource(physics);
        }
        // The block sits in its pit until the fuse burns, then detonates.
        for _ in 0..40 {
            game.update(1.0 / 60.0);
        }
        let physics = game.world.resource::<Physics>();
        assert_eq!(physics.blasts_total, 1);
        let report = physics.last_blast.expect("blast");
        assert!(report.removed_voxels > 20_000, "{report:?}");
        assert!(report.debris > 10, "{report:?}");
        let voxels = &game.world.resource::<Voxels>().0;
        // Granite around the block is gone.
        assert!(
            voxels
                .voxel(block.origin() + IVec3::new(8, -8, 30))
                .is_air()
        );
    }

    #[test]
    fn blasts_light_nearby_charges() {
        let mut game = Game::new();
        let charges = [
            IVec3::new(8, 4, 8),
            IVec3::new(10, 4, 8),
            IVec3::new(8, 4, 30),
        ];
        ground_with_tnt(&mut game, &charges);
        let mut physics = Physics::default();
        physics.blasts.push(Blast {
            centre: DVec3::new(8.5, 4.5, 8.5),
            radius: TNT_RADIUS_M,
        });
        game.world.insert_resource(physics);
        game.update(1.0 / 60.0);
        {
            let physics = game.world.resource::<Physics>();
            // The first charge was lit and thrown, the second lit, the far one untouched.
            assert_eq!(physics.fuses.len(), 2);
        }
        for _ in 0..90 {
            game.update(1.0 / 60.0);
        }
        let physics = game.world.resource::<Physics>();
        assert_eq!(physics.blasts_total, 3);
        let voxels = &game.world.resource::<Voxels>().0;
        assert_eq!(voxels.voxel(BlockPos(charges[2]).origin() + 8), ids::TNT);
    }
}
