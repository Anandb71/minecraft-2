//! The macro layer: 1 m blocks with identity.
//!
//! Natural terrain needs no explicit block ids: a block of generated rock *is*
//! its dominant material, derived from the voxels on demand. Only placed
//! blocks whose identity is more than a material (a torch, a lantern, a
//! window) are recorded explicitly. Placing writes the block's voxel model;
//! breaking returns every voxel's material to the inventory, so a half-carved
//! block yields half a block of stone.

use bevy_ecs::prelude::*;
use glam::IVec3;
use mc2_core::FxHashMap;
use mc2_voxel::coords::{BlockPos, VOXELS_PER_BLOCK};
use mc2_voxel::material::{MaterialId, ids};
use mc2_voxel::world::VoxelWorld;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockKind {
    /// A full cube of one material.
    Solid(MaterialId),
    /// Plank stick with an emissive flame, the classic light source.
    Torch,
    /// Glass box in a plank frame with a glowing core.
    Lantern,
    /// Glass pane in a plank frame, 2 voxels thick.
    Window,
    /// A lower half slab.
    Slab(MaterialId),
    /// Plank workbench with a grid inlaid in its top and tools laid on it.
    CraftingTable,
    /// Cobblestone oven on a stone-brick plinth, embers glowing in its mouth.
    Furnace,
}

impl BlockKind {
    pub fn name(&self) -> String {
        match self {
            BlockKind::Solid(m) => m.get().name.to_owned(),
            BlockKind::Torch => "torch".into(),
            BlockKind::Lantern => "lantern".into(),
            BlockKind::Window => "window".into(),
            BlockKind::Slab(m) => format!("{} slab", m.get().name),
            BlockKind::CraftingTable => "crafting table".into(),
            BlockKind::Furnace => "furnace".into(),
        }
    }

    /// Material at block-local voxel `l` of the block turned `turns` quarter
    /// turns about the vertical, its front (model -z) then facing -x, +z and
    /// +x in turn.
    pub fn voxel_turned(&self, l: IVec3, turns: u8) -> MaterialId {
        let mut m = l;
        for _ in 0..turns % 4 {
            m = IVec3::new(15 - m.z, m.y, m.x);
        }
        self.voxel(m)
    }

    /// Whether the block has a front worth turning toward whoever places it.
    pub fn faces(&self) -> bool {
        matches!(
            self,
            BlockKind::Window | BlockKind::CraftingTable | BlockKind::Furnace
        )
    }

    /// Material at block-local voxel `l` (each component 0..16).
    pub fn voxel(&self, l: IVec3) -> MaterialId {
        let n = VOXELS_PER_BLOCK;
        match *self {
            BlockKind::Solid(m) => m,
            BlockKind::Slab(m) => {
                if l.y < n / 2 {
                    m
                } else {
                    MaterialId(0)
                }
            }
            BlockKind::Torch => {
                let centre = l.x >= 7 && l.x <= 8 && l.z >= 7 && l.z <= 8;
                if centre && l.y < 10 {
                    ids::PLANKS
                } else if l.x >= 6 && l.x <= 9 && l.z >= 6 && l.z <= 9 && (10..13).contains(&l.y) {
                    ids::TORCH_FLAME
                } else {
                    MaterialId(0)
                }
            }
            BlockKind::Lantern => {
                let edge = |v: i32| v == 3 || v == 12;
                let inside = |v: i32| (3..=12).contains(&v);
                if !(inside(l.x) && inside(l.y) && inside(l.z)) {
                    MaterialId(0)
                } else if (edge(l.x) as u8 + edge(l.y) as u8 + edge(l.z) as u8) >= 2 {
                    ids::PLANKS
                } else if (5..=10).contains(&l.x)
                    && (5..=10).contains(&l.y)
                    && (5..=10).contains(&l.z)
                {
                    ids::GLOWSTONE
                } else if edge(l.x) || edge(l.y) || edge(l.z) {
                    ids::GLASS
                } else {
                    MaterialId(0)
                }
            }
            BlockKind::Window => {
                if !(7..=8).contains(&l.z) {
                    MaterialId(0)
                } else if l.x <= 1 || l.x >= 14 || l.y <= 1 || l.y >= 14 {
                    ids::PLANKS
                } else {
                    ids::GLASS
                }
            }
            BlockKind::CraftingTable => crafting_table(l),
            BlockKind::Furnace => furnace(l),
        }
    }

    /// Voxel volume of each material this block consumes.
    pub fn bill_of_materials(&self) -> Vec<(MaterialId, u32)> {
        let mut out: Vec<(MaterialId, u32)> = Vec::new();
        for i in 0..4096 {
            let m = self.voxel(IVec3::new(i & 15, (i >> 8) & 15, (i >> 4) & 15));
            if m.is_air() {
                continue;
            }
            match out.iter_mut().find(|(k, _)| *k == m) {
                Some((_, c)) => *c += 1,
                None => out.push((m, 1)),
            }
        }
        out
    }
}

/// Workbench: a three-voxel top on corner legs, a shelf low between them, a
/// dark grid inlaid in the top and a saw and hammer resting on it.
fn crafting_table(l: IVec3) -> MaterialId {
    let (x, y, z) = (l.x, l.y, l.z);
    let leg = |v: i32| (1..=3).contains(&v) || (12..=14).contains(&v);
    let inner = |v: i32| (1..=14).contains(&v);
    match y {
        15 => {
            // Saw: steel blade, plank grip. Hammer: iron head, log handle.
            if z == 11 && (2..=8).contains(&x) {
                ids::STEEL
            } else if z == 11 && (9..=10).contains(&x) {
                ids::PLANKS
            } else if (11..=13).contains(&x) && (2..=3).contains(&z) {
                ids::IRON
            } else if x == 12 && (4..=8).contains(&z) {
                ids::OAK_LOG
            } else {
                MaterialId(0)
            }
        }
        12..=14 => {
            let grid = y == 14
                && (3..=12).contains(&x)
                && (3..=12).contains(&z)
                && (x == 3 || x == 6 || x == 9 || x == 12 || z == 3 || z == 6 || z == 9 || z == 12);
            if grid { ids::DARK_PLANKS } else { ids::PLANKS }
        }
        10..=11 => {
            // Apron under the top, set in from its edge.
            let edge = inner(x) && inner(z) && (x == 1 || x == 14 || z == 1 || z == 14);
            if edge || (leg(x) && leg(z)) {
                ids::DARK_PLANKS
            } else {
                MaterialId(0)
            }
        }
        _ => {
            if leg(x) && leg(z) {
                ids::DARK_PLANKS
            } else if (2..=3).contains(&y) && inner(x) && inner(z) {
                ids::PLANKS
            } else {
                MaterialId(0)
            }
        }
    }
}

/// Oven: stone-brick plinth and cap, cobblestone walls, an iron-framed
/// mouth in its front (model -z) with embers and a low flame inside.
fn furnace(l: IVec3) -> MaterialId {
    let (x, y, z) = (l.x, l.y, l.z);
    if y <= 1 || y >= 14 {
        // Cap is inset a voxel over a plinth that is not.
        let inset = y >= 14 && (x == 0 || x == 15 || z == 0 || z == 15);
        return if inset {
            MaterialId(0)
        } else {
            ids::STONE_BRICK
        };
    }
    let mouth_x = (4..=11).contains(&x);
    let mouth_y = (3..=8).contains(&y);
    if mouth_x && mouth_y && z <= 4 {
        return match (y, z) {
            (3, 3..=4) => ids::EMBER,
            (4, 4) if (x + z) % 3 != 0 => ids::TORCH_FLAME,
            (5, 4) if (5..=10).contains(&x) && x % 2 == 0 => ids::TORCH_FLAME,
            _ => MaterialId(0),
        };
    }
    let frame = z == 0 && (3..=12).contains(&x) && (2..=9).contains(&y);
    if frame {
        return ids::IRON;
    }
    // Vent above the mouth.
    if z == 0 && (6..=9).contains(&x) && (11..=12).contains(&y) {
        return ids::BASALT;
    }
    ids::COBBLESTONE
}

/// Explicitly placed blocks. Everything else is implied by voxels.
#[derive(Resource, Default)]
pub struct BlockLayer {
    placed: FxHashMap<BlockPos, BlockKind>,
}

impl BlockLayer {
    pub fn set(&mut self, pos: BlockPos, kind: Option<BlockKind>) {
        match kind {
            Some(k) => {
                self.placed.insert(pos, k);
            }
            None => {
                self.placed.remove(&pos);
            }
        }
    }

    pub fn placed(&self) -> impl Iterator<Item = (&BlockPos, &BlockKind)> {
        self.placed.iter()
    }

    /// The block at `pos`: an explicit placement, or a solid block of the
    /// dominant material among eight interior probes, or `None` for air.
    pub fn block_at(&self, world: &VoxelWorld, pos: BlockPos) -> Option<BlockKind> {
        if let Some(k) = self.placed.get(&pos) {
            return Some(*k);
        }
        let o = pos.origin();
        let mut counts: Vec<(MaterialId, u32)> = Vec::new();
        for i in 0..8 {
            let probe = o + IVec3::new(
                4 + 8 * (i & 1),
                4 + 8 * ((i >> 1) & 1),
                4 + 8 * ((i >> 2) & 1),
            );
            let m = world.voxel(probe);
            if m.is_air() {
                continue;
            }
            match counts.iter_mut().find(|(k, _)| *k == m) {
                Some((_, c)) => *c += 1,
                None => counts.push((m, 1)),
            }
        }
        counts
            .into_iter()
            .max_by_key(|(m, c)| (*c, std::cmp::Reverse(m.0)))
            .map(|(m, _)| BlockKind::Solid(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_have_expected_volumes() {
        assert_eq!(
            BlockKind::Solid(ids::GRANITE).bill_of_materials(),
            vec![(ids::GRANITE, 4096)]
        );
        assert_eq!(
            BlockKind::Slab(ids::PLANKS).bill_of_materials(),
            vec![(ids::PLANKS, 2048)]
        );
        let torch = BlockKind::Torch.bill_of_materials();
        assert!(torch.iter().any(|(m, _)| *m == ids::TORCH_FLAME));
        for kind in [BlockKind::CraftingTable, BlockKind::Furnace] {
            let bill = kind.bill_of_materials();
            let total: u32 = bill.iter().map(|(_, n)| n).sum();
            assert!(total > 1500 && total < 4096, "{kind:?} {total}");
            assert!(bill.iter().any(|(m, _)| m.is_emissive()) == (kind == BlockKind::Furnace));
        }
        let window = BlockKind::Window.bill_of_materials();
        let glass = window.iter().find(|(m, _)| *m == ids::GLASS).unwrap().1;
        assert_eq!(glass, 12 * 12 * 2);
    }

    #[test]
    fn turning_keeps_the_model_and_moves_its_front() {
        let f = BlockKind::Furnace;
        // The mouth's iron frame, on the front.
        assert_eq!(f.voxel(IVec3::new(3, 6, 0)), ids::IRON);
        assert!(f.voxel(IVec3::new(7, 6, 0)).is_air());
        // A quarter turn puts the front on the -x face, and so on round.
        assert_eq!(f.voxel_turned(IVec3::new(0, 6, 12), 1), ids::IRON);
        assert_eq!(f.voxel_turned(IVec3::new(12, 6, 15), 2), ids::IRON);
        assert_eq!(f.voxel_turned(IVec3::new(15, 6, 3), 3), ids::IRON);
        for t in 0..4 {
            let mut n = 0;
            for i in 0..4096 {
                let l = IVec3::new(i & 15, (i >> 8) & 15, (i >> 4) & 15);
                n += u32::from(!f.voxel_turned(l, t).is_air());
            }
            let total: u32 = f.bill_of_materials().iter().map(|(_, n)| n).sum();
            assert_eq!(n, total);
        }
    }

    #[test]
    fn implicit_blocks_follow_voxels() {
        let mut w = VoxelWorld::new();
        let pos = BlockPos(IVec3::new(3, 4, 5));
        let o = pos.origin();
        w.fill_box(o, o + IVec3::splat(15), ids::BASALT);
        let mut layer = BlockLayer::default();
        assert_eq!(layer.block_at(&w, pos), Some(BlockKind::Solid(ids::BASALT)));
        assert_eq!(layer.block_at(&w, BlockPos(IVec3::new(3, 5, 5))), None);
        layer.set(pos, Some(BlockKind::Lantern));
        assert_eq!(layer.block_at(&w, pos), Some(BlockKind::Lantern));
    }
}
