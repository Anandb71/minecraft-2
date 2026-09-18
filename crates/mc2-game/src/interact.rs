//! Looking at, breaking, placing, carving and depositing.
//!
//! Block mode works on the 1 m macro grid: holding break wears a block down
//! at a pace set by its hardness and the tool in hand, then drops what it
//! yields into the inventory; place writes the held block's model against
//! the targeted face, turned to face the player, and uses up one of it.
//! Using a crafting table or furnace opens it instead.
//! Carve mode works on 6.25 cm voxels: a sphere of adjustable radius is cut
//! out of whatever is under the crosshair, or material is deposited onto it.
//!
//! Edits restore full generated detail first. Rock buried under the surface
//! is stored at brick-cell resolution until something reaches it; the first
//! edit touching such a cell asks the generator for its voxel-level contents,
//! so a freshly dug tunnel shows real layer boundaries.

use crate::blocks::{BlockKind, BlockLayer};
use crate::collide::{Aabb, blocks_movement};
use crate::crafting::Station;
use crate::input::{Button, Input, Key, Time};
use crate::inventory::{HOTBAR_SLOTS, Inventory};
use crate::items::{Item, Tier, ToolKind, break_time, drop_for};
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

/// What a creative hotbar starts with.
pub const CREATIVE_HOTBAR: [BlockKind; 9] = [
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

/// Seconds to break a block holding `tool` (`None`: bare hands).
pub fn block_break_time(kind: BlockKind, tool: Option<(ToolKind, Tier)>) -> f32 {
    match kind {
        BlockKind::Solid(m) => break_time(m, tool),
        BlockKind::Slab(m) => break_time(m, tool) * 0.6,
        BlockKind::Torch => 0.05,
        BlockKind::Lantern | BlockKind::Window => break_time(ids::GLASS, tool),
        BlockKind::CraftingTable => break_time(ids::PLANKS, tool),
        BlockKind::Furnace => break_time(ids::COBBLESTONE, tool),
    }
}

/// What breaking a block gives. Placed things come back whole; a furnace
/// needs a pickaxe like the stone it is made of.
pub fn block_drop(
    kind: BlockKind,
    tool: Option<(ToolKind, Tier)>,
    roll: f32,
) -> Option<(Item, u32)> {
    match kind {
        BlockKind::Solid(m) => drop_for(m, tool, roll),
        BlockKind::Slab(m) => drop_for(m, tool, roll).map(|(item, n)| {
            if item == Item::solid(m) {
                (Item::Block(kind), n)
            } else {
                (item, n)
            }
        }),
        BlockKind::Furnace => tool
            .is_some_and(|(k, _)| k == ToolKind::Pickaxe)
            .then_some((Item::Block(kind), 1)),
        _ => Some((Item::Block(kind), 1)),
    }
}

/// How long a line of news stays on screen, seconds.
pub const MESSAGE_S: f64 = 4.0;

#[derive(Resource)]
pub struct Interaction {
    pub mode: Mode,
    pub slot: usize,
    pub radius_voxels: f64,
    pub target: Option<Target>,
    pub preview: Option<Preview>,
    pub inventory: Inventory,
    /// The block being broken and how far along (0..1).
    pub breaking: Option<(BlockPos, f32)>,
    /// A station the player just used; the interface takes it to open.
    pub opened: Option<Station>,
    /// Recent news for the HUD: when, and what.
    pub messages: Vec<(f64, String)>,
    rng: u64,
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
        Self::new(true)
    }
}

impl Interaction {
    /// Creative play starts with a hotbar of building blocks and never runs
    /// out; survival starts with nothing.
    pub fn new(creative: bool) -> Self {
        let mut inventory = Inventory {
            creative,
            ..Default::default()
        };
        if creative {
            for (i, kind) in CREATIVE_HOTBAR.iter().enumerate() {
                inventory.slots[i] = Some(crate::inventory::Stack::new(Item::Block(*kind), 1));
            }
        }
        Self {
            mode: Mode::Block,
            slot: 0,
            radius_voxels: 5.0,
            target: None,
            preview: None,
            inventory,
            breaking: None,
            opened: None,
            messages: Vec::new(),
            rng: 0x9e37_79b9_7f4a_7c15,
            next_carve: 0.0,
            touched: FxHashSet::default(),
            edits: 0,
            edited: Vec::new(),
            built: Vec::new(),
        }
    }

    /// The item in the selected hotbar slot.
    pub fn held(&self) -> Option<Item> {
        self.inventory.slots[self.slot].map(|s| s.item)
    }

    pub fn say(&mut self, now: f64, text: impl Into<String>) {
        self.messages.retain(|(t, _)| now - t < MESSAGE_S);
        self.messages.push((now, text.into()));
        if self.messages.len() > 5 {
            self.messages.remove(0);
        }
    }

    fn roll(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Quarter turns that face a model's front toward a player looking along
/// `yaw`.
pub fn facing_turns(yaw: f32) -> u8 {
    (yaw / std::f32::consts::FRAC_PI_2).round().rem_euclid(4.0) as u8
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
    mut water: ResMut<crate::water::Water>,
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
                    let n = HOTBAR_SLOTS as i32;
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
    if state.mode == Mode::Block
        && input.button_pressed(Button::Secondary)
        && let Some(held @ (Item::Bucket | Item::WaterBucket)) = state.held()
    {
        use_bucket(
            state,
            &voxels.0,
            &mut water,
            origin,
            dir,
            held,
            time.elapsed,
        );
        return;
    }
    let Some(t) = state.target else {
        return;
    };
    let world = &mut voxels.0;
    let streamer = streaming.0.as_mut();
    let creative = state.inventory.creative;
    if !input.button_held(Button::Primary) || state.mode != Mode::Block {
        state.breaking = None;
    }
    match state.mode {
        Mode::Block => {
            let tool = state.inventory.tool_in(state.slot);
            let kind = layer.block_at(world, t.block);
            // Creative breaks on the click; survival wears the block down
            // while the button is held.
            let broken = match kind {
                Some(_) if creative => input.button_pressed(Button::Primary),
                Some(k) if input.button_held(Button::Primary) => {
                    let secs = block_break_time(k, tool);
                    let so_far = match state.breaking {
                        Some((b, p)) if b == t.block => p,
                        _ => 0.0,
                    };
                    let progress = so_far + time.dt / secs;
                    state.breaking = Some((t.block, progress));
                    progress >= 1.0
                }
                _ => false,
            };
            if broken && let Some(k) = kind {
                state.breaking = None;
                let o = t.block.origin();
                edit_box(world, streamer, &mut state.touched, o, o + 15, |_, _| {
                    MaterialId(0)
                });
                if !creative {
                    collect(state, k, tool, time.elapsed);
                }
                layer.set(t.block, None);
                state.edits += 1;
                state.edited.push((o, o + 15));
            } else if input.button_pressed(Button::Secondary)
                && let Some(station) = match kind {
                    Some(BlockKind::CraftingTable) => Some(Station::Table),
                    Some(BlockKind::Furnace) => Some(Station::Furnace),
                    _ => None,
                }
            {
                state.opened = Some(station);
            } else if input.button_pressed(Button::Secondary)
                && let Some(Item::Block(kind)) = state.held()
            {
                let turns = if kind.faces() {
                    facing_turns(player.yaw)
                } else {
                    0
                };
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
                if !cell.intersects(&feet_box) && !occupied {
                    edit_box(world, streamer, &mut state.touched, o, o + 15, |v, old| {
                        let m = kind.voxel_turned(chunk_local(v) & 15, turns);
                        if m.is_air() { old } else { m }
                    });
                    state.inventory.take_from(state.slot);
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
            let paint = match state.held() {
                Some(Item::Block(BlockKind::Solid(m) | BlockKind::Slab(m))) => Some(m),
                _ if creative => Some(ids::COBBLESTONE),
                _ => None,
            };
            if depositing && !carving && paint.is_none() {
                return;
            }
            let paint = paint.unwrap_or(ids::COBBLESTONE);
            // Depositing spends what is carried, and stops when it runs out.
            let mut budget = if creative {
                u64::MAX
            } else {
                state.inventory.volume(paint)
            };
            let player_box = feet_box;
            let (removed, added) =
                edit_box(world, streamer, &mut state.touched, lo, hi, |v, old| {
                    let inside = (v.as_dvec3() + 0.5 - c).length_squared() <= r * r;
                    if !inside {
                        return old;
                    }
                    if carving {
                        // Bedrock stays.
                        if old.get().hardness.is_finite() {
                            MaterialId(0)
                        } else {
                            old
                        }
                    } else {
                        let vm = v.as_dvec3() / 16.0;
                        let voxel_box = Aabb {
                            min: vm,
                            max: vm + 1.0 / 16.0,
                        };
                        if old.is_air() && !voxel_box.intersects(&player_box) && budget > 0 {
                            budget -= 1;
                            paint
                        } else {
                            old
                        }
                    }
                });
            for (m, n) in removed {
                state.inventory.add_loose(m, n);
            }
            for (m, n) in added {
                state.inventory.take_loose(m, n);
            }
            state.edits += 1;
            state.edited.push((lo, hi));
            if depositing && !carving {
                state.built.push((lo, hi));
            }
        }
    }
}

/// Right click with a bucket: an empty one fills from water in reach (water
/// is endless, as a spring would be); a full one pours a cubic metre into
/// the block it is aimed at.
fn use_bucket(
    state: &mut Interaction,
    world: &VoxelWorld,
    water: &mut crate::water::Water,
    origin: DVec3,
    dir: DVec3,
    held: Item,
    now: f64,
) {
    let creative = state.inventory.creative;
    let (from, to) = match held {
        Item::Bucket => {
            let wet = raycast_filtered(world, origin, dir, REACH_M, |m| !m.is_air())
                .is_some_and(|hit| hit.material == ids::WATER);
            if !wet {
                return;
            }
            state.say(now, "filled a bucket");
            (Item::Bucket, Item::WaterBucket)
        }
        _ => {
            let Some(t) = state.target else {
                return;
            };
            let o = t.place.origin();
            water.pour(o, o + (VOXELS_PER_BLOCK - 1));
            state.say(now, "poured the water out");
            (Item::WaterBucket, Item::Bucket)
        }
    };
    if creative {
        return;
    }
    let inv = &mut state.inventory;
    match inv.slots[state.slot] {
        Some(s) if s.item == from && s.count == 1 => {
            inv.slots[state.slot] = Some(crate::inventory::Stack::new(to, 1));
        }
        Some(s) if s.item == from => {
            inv.take_from(state.slot);
            if inv.add(to, 1) > 0 {
                state.say(now, "inventory full");
            }
        }
        _ => {}
    }
}

/// A block of `kind` broke with `tool` in the selected slot: take what it
/// drops, say so, and wear the tool.
fn collect(state: &mut Interaction, kind: BlockKind, tool: Option<(ToolKind, Tier)>, now: f64) {
    let roll = state.roll();
    match block_drop(kind, tool, roll) {
        Some((item, n)) => {
            if state.inventory.add(item, n) > 0 {
                state.say(now, "inventory full");
            } else {
                state.say(now, format!("+{n} {}", item.name()));
            }
        }
        None => {
            if let BlockKind::Solid(m) = kind
                && let Some((need, Some(least))) = crate::items::suited_tool(m)
            {
                let tool = Item::Tool(need, least).name();
                state.say(now, format!("{} needs a {tool}", m.get().name));
            }
        }
    }
    if let Some((k, tier)) = tool
        && state.inventory.wear(state.slot)
    {
        state.say(now, format!("your {} broke", Item::Tool(k, tier).name()));
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
    fn placed_things_come_back_whole() {
        let pick = Some((ToolKind::Pickaxe, Tier::Wood));
        assert_eq!(
            block_drop(BlockKind::Torch, None, 0.5),
            Some((Item::Block(BlockKind::Torch), 1))
        );
        assert_eq!(block_drop(BlockKind::Furnace, None, 0.5), None);
        assert_eq!(
            block_drop(BlockKind::Furnace, pick, 0.5),
            Some((Item::Block(BlockKind::Furnace), 1))
        );
        assert_eq!(
            block_drop(BlockKind::Slab(ids::STONE_BRICK), pick, 0.5),
            Some((Item::Block(BlockKind::Slab(ids::STONE_BRICK)), 1))
        );
        assert!(block_break_time(BlockKind::Torch, None) < 0.1);
    }

    #[test]
    fn models_turn_to_face_the_player() {
        use std::f32::consts::{FRAC_PI_2, PI};
        assert_eq!(facing_turns(0.0), 0);
        assert_eq!(facing_turns(FRAC_PI_2), 1);
        assert_eq!(facing_turns(PI), 2);
        assert_eq!(facing_turns(-FRAC_PI_2), 3);
        assert_eq!(facing_turns(2.0 * PI + 0.1), 0);
    }
}
