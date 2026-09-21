//! Water that flows: the bridge between the voxel world and the fluid
//! simulation, which runs on the GPU in the renderer's hands.
//!
//! The simulation works in 0.5 m cells, eight voxels a side. The world
//! tells it which cells are solid; it tells the world, a frame or two
//! later, how full each cell is in eighths, and the cell's open voxels fill
//! with water to that height. Sea and lake water stays still until
//! something disturbs it: an edit beside it takes the water around into
//! the simulation, and the cells along the edge of what was taken are kept
//! full, as the rest of the sea would keep them.

use crate::Voxels;
use crate::physics::Physics;
use bevy_ecs::prelude::*;
use glam::IVec3;
use mc2_core::{FxHashMap, FxHashSet};
use mc2_fluid::{TILE, TILE_CELLS, Terrain, index_of, local_of};
use mc2_voxel::coords::ChunkPos;
use mc2_voxel::material::{Kind, MaterialId, ids};
use mc2_voxel::world::VoxelWorld;
use mc2_worldgen::stream::ChunkStreamer;

/// Voxels along a cell's edge.
pub const CELL_VOXELS: i32 = 8;
/// How far (cells) from an edit still water is taken into the simulation.
const PROMOTE_RADIUS: i32 = 16;
/// Most cells one edit takes in.
const PROMOTE_MAX: usize = 8_000;
/// Edits bigger than this many cells are searched for still water only on
/// their shell.
const SEARCH_CELLS: i64 = 32_768;
/// A reservoir cell is refilled once it has drained below this (eighths):
/// not at every ripple, which would keep the sea from ever sleeping.
const REFILL_BELOW: u8 = 6;
/// Frames between reservoir refills.
const REFILL_EVERY: u32 = 8;
/// Frames after still water joins the simulation during which levels read
/// back cannot lower it: they may predate the simulation taking it.
const SETTLE_FRAMES: u8 = 8;

#[derive(Resource, Default)]
pub struct Water {
    /// Levels written into the world, by tile, in eighths a cell.
    written: FxHashMap<IVec3, Box<[u8]>>,
    /// Still water taken into the simulation.
    promoted: FxHashSet<IVec3>,
    /// Promoted cells beside still water that stays: kept full.
    reservoir: Vec<IVec3>,
    /// Tiles that just took in still water, and frames left to settle.
    settling: FxHashMap<IVec3, u8>,
    frame: u32,
    /// Cells to fill with water, for the simulation to take.
    pub add: Vec<IVec3>,
    /// Inclusive cell boxes whose solidity may have changed.
    pub changed: Vec<(IVec3, IVec3)>,
    /// Cells whose voxels changed at the last `apply`.
    pub cells_written: usize,
}

/// The cell holding voxel `v`.
pub fn cell_of(v: IVec3) -> IVec3 {
    v.div_euclid(IVec3::splat(CELL_VOXELS))
}

fn is_water(m: MaterialId) -> bool {
    m == ids::WATER
}

impl Water {
    /// How full the simulation last left cell `c`, in eighths.
    pub fn level(&self, c: IVec3) -> u8 {
        let t = IVec3::splat(TILE);
        self.written
            .get(&c.div_euclid(t))
            .map_or(0, |l| l[index_of(c.rem_euclid(t))])
    }

    /// Whether cell `c`'s water, if any, belongs to the simulation.
    pub fn simulated(&self, c: IVec3) -> bool {
        self.promoted.contains(&c) || self.level(c) > 0
    }

    fn set_level(&mut self, c: IVec3, level: u8) {
        let t = IVec3::splat(TILE);
        let tile = self
            .written
            .entry(c.div_euclid(t))
            .or_insert_with(|| vec![0; TILE_CELLS].into_boxed_slice());
        tile[index_of(c.rem_euclid(t))] = level;
    }

    /// Pours water into the cells of voxel box `lo..=hi`.
    pub fn pour(&mut self, lo: IVec3, hi: IVec3) {
        let (a, b) = (cell_of(lo), cell_of(hi));
        for z in a.z..=b.z {
            for y in a.y..=b.y {
                for x in a.x..=b.x {
                    self.add.push(IVec3::new(x, y, z));
                }
            }
        }
    }

    /// Terrain changed in voxels `lo..=hi`: the simulation hears of it, and
    /// still water in or beside the change joins the simulation.
    pub fn edited(&mut self, world: &VoxelWorld, lo: IVec3, hi: IVec3) {
        let (a, b) = (cell_of(lo), cell_of(hi));
        self.changed.push((a, b));
        let mut seeds = Vec::new();
        let (lo, hi) = (a - 1, b + 1);
        let size = (hi - lo + 1).as_i64vec3();
        // A big edit is only searched on its shell: still water it cut into
        // meets the shell anyway.
        let shell_only = size.x * size.y * size.z > SEARCH_CELLS;
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let c = IVec3::new(x, y, z);
                    let on_shell = c.cmpeq(lo).any() || c.cmpeq(hi).any();
                    if shell_only && !on_shell {
                        continue;
                    }
                    if self.still_water(world, c) {
                        seeds.push(c);
                    }
                }
            }
        }
        if seeds.is_empty() {
            return;
        }
        // Flood out through still water from the edit, to a limit.
        let centre = (a + b) / 2;
        let mut taken = Vec::new();
        let mut queue = seeds;
        let mut seen: FxHashSet<IVec3> = queue.iter().copied().collect();
        while let Some(c) = queue.pop() {
            taken.push(c);
            if taken.len() >= PROMOTE_MAX {
                break;
            }
            for d in [
                IVec3::X,
                IVec3::NEG_X,
                IVec3::Y,
                IVec3::NEG_Y,
                IVec3::Z,
                IVec3::NEG_Z,
            ] {
                let n = c + d;
                let far = (n - centre).abs().max_element() > PROMOTE_RADIUS;
                if far || !seen.insert(n) || !self.still_water(world, n) {
                    continue;
                }
                queue.push(n);
            }
        }
        for &c in &taken {
            self.promoted.insert(c);
            // Its voxels are water already.
            self.set_level(c, 8);
            self.add.push(c);
            self.settling
                .insert(c.div_euclid(IVec3::splat(TILE)), SETTLE_FRAMES);
        }
        // Along the edge of what was taken, below its surface, the sea
        // stays full.
        for &c in &taken {
            let above = c + IVec3::Y;
            let submerged = self.promoted.contains(&above) || self.still_water(world, above);
            let edge = [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z, IVec3::NEG_Y]
                .iter()
                .any(|&d| self.still_water(world, c + d));
            if edge && submerged {
                self.reservoir.push(c);
            }
        }
    }

    /// Water the world generated, not yet simulated.
    fn still_water(&self, world: &VoxelWorld, c: IVec3) -> bool {
        !self.simulated(c) && is_water(world.voxel(c * CELL_VOXELS + CELL_VOXELS / 2))
    }

    /// Once a frame: settling tiles count down, and every few frames the
    /// reservoir cells that have drained ask to be filled again.
    pub fn tick(&mut self) {
        self.settling.retain(|_, n| {
            *n -= 1;
            *n > 0
        });
        self.frame = self.frame.wrapping_add(1);
        if self.frame.is_multiple_of(REFILL_EVERY) {
            self.refill();
        }
    }

    /// Asks for the reservoir cells that have drained to be filled again.
    pub fn refill(&mut self) {
        for i in 0..self.reservoir.len() {
            let c = self.reservoir[i];
            if self.level(c) < REFILL_BELOW {
                self.add.push(c);
            }
        }
    }

    /// Writes a tile's levels into the world: open voxels below the new
    /// level fill, water above it drains. Chunks written to are pinned so
    /// streaming never regenerates over the water.
    pub fn apply(
        &mut self,
        world: &mut VoxelWorld,
        mut streamer: Option<&mut ChunkStreamer>,
        tile: IVec3,
        eighths: &[u8],
    ) {
        let old = self
            .written
            .entry(tile)
            .or_insert_with(|| vec![0; TILE_CELLS].into_boxed_slice())
            .clone();
        let settling = self.settling.contains_key(&tile);
        let mut pinned: FxHashSet<ChunkPos> = FxHashSet::default();
        let mut kept = Vec::new();
        for (i, (&was, &now)) in old.iter().zip(eighths).enumerate() {
            let now = now.min(8);
            if was == now {
                continue;
            }
            if settling && now < was {
                kept.push(i);
                continue;
            }
            let c = tile * TILE + local_of(i);
            let base = c * CELL_VOXELS;
            let (from, to) = (was.min(now), was.max(now));
            for y in i32::from(from)..i32::from(to) {
                for z in 0..CELL_VOXELS {
                    for x in 0..CELL_VOXELS {
                        let v = base + IVec3::new(x, y, z);
                        let m = world.voxel(v);
                        if now > was && m.is_air() {
                            world.set_voxel(v, ids::WATER);
                        } else if now < was && is_water(m) {
                            world.set_voxel(v, ids::AIR);
                        }
                    }
                }
            }
            if let Some(s) = streamer.as_deref_mut()
                && pinned.insert(ChunkPos::of_voxel(base))
            {
                s.pin(ChunkPos::of_voxel(base));
            }
            self.cells_written += 1;
        }
        if let Some(l) = self.written.get_mut(&tile) {
            for (w, &e) in l.iter_mut().zip(eighths) {
                *w = e.min(8);
            }
            for i in kept {
                l[i] = old[i];
            }
        }
    }
}

/// The world as the simulation sees it: a cell is solid where the voxel at
/// its centre is, where the world has not loaded, and where still water
/// stands that the simulation does not own.
pub struct WorldTerrain<'a> {
    pub world: &'a VoxelWorld,
    pub water: &'a Water,
}

impl Terrain for WorldTerrain<'_> {
    fn solid(&self, c: IVec3) -> bool {
        let v = c * CELL_VOXELS + CELL_VOXELS / 2;
        if self.world.chunk(ChunkPos::of_voxel(v)).is_none() {
            return true;
        }
        let m = self.world.voxel(v);
        match m.get().kind {
            Kind::Solid | Kind::Transparent => true,
            Kind::Liquid => !(is_water(m) && self.water.simulated(c)),
            _ => false,
        }
    }
}

/// Hands this tick's terrain edits to the water. Every frame.
pub fn take_edits(voxels: Res<Voxels>, mut physics: ResMut<Physics>, mut water: ResMut<Water>) {
    for (lo, hi) in std::mem::take(&mut physics.water_out) {
        water.edited(&voxels.0, lo, hi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> VoxelWorld {
        // Granite below y = 64 voxels (4 m), sea water to 96 voxels.
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(511, 63, 511), ids::GRANITE);
        w.fill_box(IVec3::new(0, 64, 0), IVec3::new(511, 95, 511), ids::WATER);
        w
    }

    #[test]
    fn still_water_is_solid_until_taken_in() {
        let world = pool();
        let mut water = Water::default();
        let sea = IVec3::new(10, 10, 10);
        let t = WorldTerrain {
            world: &world,
            water: &water,
        };
        assert!(t.solid(sea));
        assert!(t.solid(IVec3::new(10, 5, 10)));
        assert!(!t.solid(IVec3::new(10, 14, 10)));
        // Dig beside the sea: the water around joins the simulation, and
        // the edge of what joined is kept full.
        water.edited(&world, IVec3::new(80, 64, 80), IVec3::new(95, 79, 95));
        assert!(water.simulated(sea));
        assert!(!water.add.is_empty());
        let t = WorldTerrain {
            world: &world,
            water: &water,
        };
        assert!(!t.solid(sea));
        // Sea beyond the reach of one edit stays still, and a wall.
        let far = IVec3::new(50, 10, 10);
        assert!(!water.simulated(far) && t.solid(far));
        assert!(!water.reservoir.is_empty());
        water.add.clear();
        water.refill();
        assert!(water.add.is_empty(), "full reservoir needs nothing");
        // Levels from before the simulation took the sea cannot drain it.
        let mut world = world;
        let tile = sea.div_euclid(IVec3::splat(TILE));
        water.apply(&mut world, None, tile, &vec![0; TILE_CELLS]);
        assert_eq!(water.level(sea), 8);
        assert_eq!(world.voxel(sea * CELL_VOXELS), ids::WATER);
    }

    #[test]
    fn levels_fill_and_drain_open_voxels_only() {
        let mut world = VoxelWorld::new();
        world.fill_box(IVec3::ZERO, IVec3::new(15, 7, 15), ids::GRANITE);
        // A stone in the way inside the cell above the floor.
        world.set_voxel(IVec3::new(2, 9, 2), ids::GRANITE);
        let mut water = Water::default();
        let mut eighths = vec![0u8; TILE_CELLS];
        let c = IVec3::new(0, 1, 0);
        eighths[index_of(c)] = 5;
        water.apply(&mut world, None, IVec3::ZERO, &eighths);
        assert_eq!(world.voxel(IVec3::new(3, 8, 3)), ids::WATER);
        assert_eq!(world.voxel(IVec3::new(3, 12, 3)), ids::WATER);
        assert!(world.voxel(IVec3::new(3, 13, 3)).is_air());
        assert_eq!(world.voxel(IVec3::new(2, 9, 2)), ids::GRANITE);
        assert_eq!(water.level(c), 5);
        eighths[index_of(c)] = 2;
        water.apply(&mut world, None, IVec3::ZERO, &eighths);
        assert_eq!(world.voxel(IVec3::new(3, 9, 3)), ids::WATER);
        assert!(world.voxel(IVec3::new(3, 10, 3)).is_air());
        assert_eq!(world.voxel(IVec3::new(2, 9, 2)), ids::GRANITE);
        assert_eq!(water.cells_written, 2);
    }
}
