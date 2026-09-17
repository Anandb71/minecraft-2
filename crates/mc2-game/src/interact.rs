//! Looking at, breaking, placing, carving and depositing.
//!
//! Block mode works on the 1 m macro grid: break returns the block's voxels
//! to the inventory, place writes a block model against the targeted face.
//! Carve mode works on 6.25 cm voxels: a sphere of adjustable radius is cut
//! out of whatever is under the crosshair, or material is deposited onto it.
//!
//! Edits restore full generated detail first. Rock buried under the surface
//! is stored at brick-cell resolution until something reaches it; the first
//! edit touching such a cell asks the generator for its voxel-level contents,
//! so a freshly dug tunnel shows real layer boundaries.

use crate::blocks::{BlockKind, BlockLayer};
use crate::collide::{Aabb, blocks_movement};
use crate::input::{Button, Input, Key, Time};
use crate::player::{Body, Player, WIDTH};
use crate::{Streaming, Voxels};
use bevy_ecs::prelude::*;
use glam::{DVec3, IVec3};
use mc2_core::{FxHashMap, FxHashSet};
use mc2_voxel::coords::{
    BRICK_SHIFT, BlockPos, CHUNK_SHIFT, ChunkPos, VOXELS_PER_BLOCK, chunk_local,
};
use mc2_voxel::march::{RayHit, raycast_filtered};
use mc2_voxel::material::{Kind, MaterialId, ids};
use mc2_voxel::tree::Cell;
use mc2_voxel::world::VoxelWorld;
use mc2_worldgen::stream::ChunkStreamer;

pub const REACH_M: f64 = 8.0;
const CARVE_INTERVAL_S: f64 = 0.06;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Block,
    Carve,
}

pub const HOTBAR: [BlockKind; 9] = [
    BlockKind::Solid(ids::STONE_BRICK),
    BlockKind::Solid(ids::PLANKS),
    BlockKind::Window,
    BlockKind::Torch,
    BlockKind::Lantern,
    BlockKind::Solid(ids::COBBLESTONE),
    BlockKind::Solid(ids::GLASS),
    BlockKind::Slab(ids::STONE_BRICK),
    BlockKind::Solid(ids::TNT),
];

#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub hit: RayHit,
    pub block: BlockPos,
    /// Block cell a placement against the hit face would occupy.
    pub place: BlockPos,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Preview {
    Box {
        min: DVec3,
        max: DVec3,
        valid: bool,
    },
    Sphere {
        centre: DVec3,
        radius: f64,
        deposit: bool,
    },
}

/// Material volumes in voxels (4096 to a block).
#[derive(Default, Debug)]
pub struct Inventory {
    pub volumes: FxHashMap<MaterialId, u64>,
    /// Creative mode never runs out.
    pub creative: bool,
}

impl Inventory {
    pub fn add(&mut self, m: MaterialId, n: u64) {
        if !m.is_air() {
            *self.volumes.entry(m).or_default() += n;
        }
    }

    pub fn can_afford(&self, bill: &[(MaterialId, u32)]) -> bool {
        self.creative
            || bill
                .iter()
                .all(|(m, n)| self.volumes.get(m).copied().unwrap_or(0) >= u64::from(*n))
    }

    pub fn spend(&mut self, bill: &[(MaterialId, u32)]) {
        if self.creative {
            return;
        }
        for (m, n) in bill {
            if let Some(v) = self.volumes.get_mut(m) {
                *v = v.saturating_sub(u64::from(*n));
            }
        }
    }
}

#[derive(Resource)]
pub struct Interaction {
    pub mode: Mode,
    pub slot: usize,
    pub radius_voxels: f64,
    pub target: Option<Target>,
    pub preview: Option<Preview>,
    pub inventory: Inventory,
    next_carve: f64,
    /// Brick cells already restored to generated detail (or edited since).
    pub(crate) touched: FxHashSet<(ChunkPos, IVec3)>,
    pub edits: u64,
    /// Voxel boxes edited since physics last looked, so resting debris
    /// there wakes up.
    pub edited: Vec<(IVec3, IVec3)>,
    /// Voxel boxes where the player placed material: structure, not rock.
    pub built: Vec<(IVec3, IVec3)>,
}

impl Default for Interaction {
    fn default() -> Self {
        Self {
            mode: Mode::Block,
            slot: 0,
            radius_voxels: 5.0,
            target: None,
            preview: None,
            inventory: Inventory {
                creative: true,
                ..Default::default()
            },
            next_carve: 0.0,
            touched: FxHashSet::default(),
            edits: 0,
            edited: Vec::new(),
            built: Vec::new(),
        }
    }
}

/// What an interaction may hit: anything but air and liquids.
fn targetable(m: MaterialId) -> bool {
    !m.is_air() && m.get().kind != Kind::Liquid
}

/// Readies an inclusive voxel box for editing: coarse cells that were never
/// touched get their full generated detail back, and the chunks are pinned
/// so streaming never regenerates over the edit.
pub fn prepare_edit(
    world: &mut VoxelWorld,
    streamer: Option<&mut ChunkStreamer>,
    touched: &mut FxHashSet<(ChunkPos, IVec3)>,
    min: IVec3,
    max: IVec3,
) {
    prepare_edit_where(world, streamer, touched, min, max, |_, _| true);
}

/// [`prepare_edit`] for an edit that does not need every cell of its box:
/// `detail` decides, per brick cell (given its minimum and maximum voxel),
/// whether the edit will read voxels there. A blast that clears whole cells
/// only needs detail where it cuts.
pub fn prepare_edit_where(
    world: &mut VoxelWorld,
    streamer: Option<&mut ChunkStreamer>,
    touched: &mut FxHashSet<(ChunkPos, IVec3)>,
    min: IVec3,
    max: IVec3,
    detail: impl Fn(IVec3, IVec3) -> bool,
) {
    let mut chunks = FxHashSet::default();
    let mut restore: FxHashMap<ChunkPos, Vec<IVec3>> = FxHashMap::default();
    let (lo_cell, hi_cell) = (min >> BRICK_SHIFT, max >> BRICK_SHIFT);
    for cz in lo_cell.z..=hi_cell.z {
        for cy in lo_cell.y..=hi_cell.y {
            for cx in lo_cell.x..=hi_cell.x {
                let cell_world = IVec3::new(cx, cy, cz);
                let pos = ChunkPos::of_voxel(cell_world << BRICK_SHIFT);
                let local = cell_world - (pos.0 << (CHUNK_SHIFT - BRICK_SHIFT));
                chunks.insert(pos);
                let cell_min = cell_world << BRICK_SHIFT;
                if !detail(cell_min, cell_min + (1 << BRICK_SHIFT) - 1) {
                    continue;
                }
                if !touched.insert((pos, local)) || streamer.is_none() {
                    continue;
                }
                if world
                    .chunk(pos)
                    .is_some_and(|tree| matches!(tree.cell(local), Cell::Uniform(_)))
                {
                    restore.entry(pos).or_default().push(local);
                }
            }
        }
    }
    // Coarse cells get their generated voxel detail back, one chunk sample
    // pass per chunk however many cells the edit touches.
    if let Some(s) = streamer.as_deref() {
        for (pos, cells) in restore {
            let bricks = s.generator().detail_bricks(pos, &cells);
            if let Some(tree) = world.chunk_mut(pos) {
                for (cell, brick) in cells.into_iter().zip(bricks) {
                    tree.set_brick(cell, brick);
                }
            }
        }
    }
    if let Some(s) = streamer {
        for pos in chunks {
            s.pin(pos);
        }
    }
}

/// Applies `f(voxel, old) -> new` over an inclusive voxel box after
/// [`prepare_edit`]. Returns (removed, added) material volumes.
pub fn edit_box(
    world: &mut VoxelWorld,
    streamer: Option<&mut ChunkStreamer>,
    touched: &mut FxHashSet<(ChunkPos, IVec3)>,
    min: IVec3,
    max: IVec3,
    mut f: impl FnMut(IVec3, MaterialId) -> MaterialId,
) -> (FxHashMap<MaterialId, u64>, FxHashMap<MaterialId, u64>) {
    prepare_edit(world, streamer, touched, min, max);
    let mut removed: FxHashMap<MaterialId, u64> = FxHashMap::default();
    let mut added: FxHashMap<MaterialId, u64> = FxHashMap::default();
    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let v = IVec3::new(x, y, z);
                let old = world.voxel(v);
                let new = f(v, old);
                if new == old {
                    continue;
                }
                world.set_voxel(v, new);
                if !old.is_air() {
                    *removed.entry(old).or_default() += 1;
                }
                if !new.is_air() {
                    *added.entry(new).or_default() += 1;
                }
            }
        }
    }
    (removed, added)
}

fn eye(p: &Player, b: &Body, alpha: f64) -> DVec3 {
    b.prev_feet.lerp(b.feet, alpha) + DVec3::Y * (p.eye() - p.step_smoothing)
}

/// Updates the target and preview, then applies clicks. Every frame.
pub fn interact(
    mut voxels: ResMut<Voxels>,
    mut streaming: ResMut<Streaming>,
    mut layer: ResMut<BlockLayer>,
    mut state: ResMut<Interaction>,
    input: Res<Input>,
    time: Res<Time>,
    players: Query<(&Player, &Body)>,
) {
    let Some((player, body)) = players.iter().next() else {
        return;
    };
    let state = &mut *state;
    if input.captured {
        if input.pressed(Key::ToggleMode) {
            state.mode = match state.mode {
                Mode::Block => Mode::Carve,
                Mode::Carve => Mode::Block,
            };
        }
        for i in 0..9 {
            if input.pressed(Key::Slot(i)) {
                state.slot = usize::from(i);
            }
        }
        if input.scroll != 0.0 {
            match state.mode {
                Mode::Carve => {
                    state.radius_voxels =
                        (state.radius_voxels + f64::from(input.scroll)).clamp(1.0, 16.0);
                }
                Mode::Block => {
                    let n = HOTBAR.len() as i32;
                    state.slot =
                        (state.slot as i32 - input.scroll.signum() as i32).rem_euclid(n) as usize;
                }
            }
        }
    }

    let origin = eye(player, body, time.alpha);
    let dir = player.forward().as_dvec3();
    state.target = raycast_filtered(&voxels.0, origin, dir, REACH_M, targetable).map(|hit| {
        let block = BlockPos::of_voxel(hit.voxel);
        Target {
            hit,
            block,
            place: BlockPos(block.0 + hit.normal),
        }
    });
    let feet_box = Aabb::standing(body.feet, WIDTH, player.height());

    state.preview = state.target.map(|t| match state.mode {
        Mode::Block => {
            let (place, valid) = if input.button_held(Button::Middle) {
                (t.block, true)
            } else {
                let o = t.place.origin().as_dvec3() / f64::from(VOXELS_PER_BLOCK);
                let b = Aabb {
                    min: o,
                    max: o + 1.0,
                };
                (t.place, !b.intersects(&feet_box))
            };
            let o = place.origin().as_dvec3() / f64::from(VOXELS_PER_BLOCK);
            Preview::Box {
                min: o,
                max: o + 1.0,
                valid,
            }
        }
        Mode::Carve => {
            let deposit = input.button_held(Button::Secondary);
            let hit_point = origin + dir * t.hit.t;
            let r = state.radius_voxels / f64::from(VOXELS_PER_BLOCK);
            let centre = if deposit {
                hit_point + t.hit.normal.as_dvec3() * r
            } else {
                hit_point
            };
            Preview::Sphere {
                centre,
                radius: r,
                deposit,
            }
        }
    });

    if !input.captured {
        return;
    }
    let Some(t) = state.target else {
        return;
    };
    let world = &mut voxels.0;
    let streamer = streaming.0.as_mut();
    match state.mode {
        Mode::Block => {
            if input.button_pressed(Button::Primary) {
                let o = t.block.origin();
                let (removed, _) =
                    edit_box(world, streamer, &mut state.touched, o, o + 15, |_, _| {
                        MaterialId(0)
                    });
                for (m, n) in removed {
                    state.inventory.add(m, n);
                }
                layer.set(t.block, None);
                state.edits += 1;
                state.edited.push((o, o + 15));
            } else if input.button_pressed(Button::Secondary) {
                let kind = HOTBAR[state.slot];
                let bill = kind.bill_of_materials();
                let o = t.place.origin();
                let cell = Aabb {
                    min: o.as_dvec3() / 16.0,
                    max: (o.as_dvec3() + 16.0) / 16.0,
                };
                let occupied = (0..4096).any(|i| {
                    blocks_movement(
                        world.voxel(o + IVec3::new(i & 15, (i >> 8) & 15, (i >> 4) & 15)),
                    )
                });
                if !cell.intersects(&feet_box) && !occupied && state.inventory.can_afford(&bill) {
                    edit_box(world, streamer, &mut state.touched, o, o + 15, |v, old| {
                        let m = kind.voxel(chunk_local(v) & 15);
                        if m.is_air() { old } else { m }
                    });
                    state.inventory.spend(&bill);
                    let explicit = !matches!(kind, BlockKind::Solid(_));
                    layer.set(t.place, explicit.then_some(kind));
                    state.edits += 1;
                    state.edited.push((o, o + 15));
                    state.built.push((o, o + 15));
                }
            }
        }
        Mode::Carve => {
            let carving = input.button_held(Button::Primary);
            let depositing = input.button_held(Button::Secondary);
            if !(carving || depositing) || time.elapsed < state.next_carve {
                return;
            }
            state.next_carve = time.elapsed + CARVE_INTERVAL_S;
            let Some(Preview::Sphere { centre, radius, .. }) = state.preview else {
                return;
            };
            let c = centre * f64::from(VOXELS_PER_BLOCK);
            let r = radius * f64::from(VOXELS_PER_BLOCK);
            let lo = (c - r).floor().as_ivec3();
            let hi = (c + r).ceil().as_ivec3();
            let paint = match HOTBAR[state.slot] {
                BlockKind::Solid(m) | BlockKind::Slab(m) => m,
                _ => ids::COBBLESTONE,
            };
            let player_box = feet_box;
            let (removed, added) =
                edit_box(world, streamer, &mut state.touched, lo, hi, |v, old| {
                    let inside = (v.as_dvec3() + 0.5 - c).length_squared() <= r * r;
                    if !inside {
                        return old;
                    }
                    if carving {
                        MaterialId(0)
                    } else {
                        let vm = v.as_dvec3() / 16.0;
                        let voxel_box = Aabb {
                            min: vm,
                            max: vm + 1.0 / 16.0,
                        };
                        if old.is_air() && !voxel_box.intersects(&player_box) {
                            paint
                        } else {
                            old
                        }
                    }
                });
            for (m, n) in removed {
                state.inventory.add(m, n);
            }
            let spent: Vec<(MaterialId, u32)> =
                added.into_iter().map(|(m, n)| (m, n as u32)).collect();
            state.inventory.spend(&spent);
            state.edits += 1;
            state.edited.push((lo, hi));
            if depositing && !carving {
                state.built.push((lo, hi));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_box_counts_removed_and_added() {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::splat(31), ids::GRANITE);
        let mut touched = FxHashSet::default();
        let (removed, added) = edit_box(
            &mut w,
            None,
            &mut touched,
            IVec3::ZERO,
            IVec3::splat(15),
            |_, _| MaterialId(0),
        );
        assert_eq!(removed.get(&ids::GRANITE), Some(&4096));
        assert!(added.is_empty());
        assert_eq!(w.voxel(IVec3::splat(3)), MaterialId(0));
        assert_eq!(w.voxel(IVec3::splat(16)), ids::GRANITE);
        let (_, added) = edit_box(
            &mut w,
            None,
            &mut touched,
            IVec3::ZERO,
            IVec3::splat(7),
            |_, _| ids::SAND,
        );
        assert_eq!(added.get(&ids::SAND), Some(&512));
    }

    #[test]
    fn inventory_spends_only_when_affordable() {
        let mut inv = Inventory::default();
        let bill = BlockKind::Solid(ids::PLANKS).bill_of_materials();
        assert!(!inv.can_afford(&bill));
        inv.add(ids::PLANKS, 5000);
        assert!(inv.can_afford(&bill));
        inv.spend(&bill);
        assert_eq!(inv.volumes[&ids::PLANKS], 904);
    }
}
