//! Slots of stacked items: a hotbar of nine over a pack of twenty-seven,
//! plus the voxel volumes carving has knocked loose and the furnace's heat.

use crate::items::{Item, ToolKind, drop_for};
use mc2_core::FxHashMap;
use mc2_voxel::material::MaterialId;

pub const HOTBAR_SLOTS: usize = 9;
pub const SLOTS: usize = 36;
/// Voxels in a whole block.
pub const BLOCK_VOXELS: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stack {
    pub item: Item,
    pub count: u32,
    /// Uses left of something that wears (0 for everything else).
    pub life: u32,
}

impl Stack {
    pub fn new(item: Item, count: u32) -> Self {
        Self {
            item,
            count,
            life: item.uses().unwrap_or(0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Inventory {
    pub slots: [Option<Stack>; SLOTS],
    /// Voxels carved loose, by material, short of a whole block.
    pub loose: FxHashMap<MaterialId, u64>,
    /// Smelts the furnace fire has left before it needs more fuel.
    pub heat: u32,
    /// Creative play: nothing is used up and anything can be made.
    pub creative: bool,
}

impl Default for Inventory {
    fn default() -> Self {
        Self {
            slots: [None; SLOTS],
            loose: FxHashMap::default(),
            heat: 0,
            creative: false,
        }
    }
}

impl Inventory {
    /// Adds `count` of `item`, topping up stacks of it before filling empty
    /// slots, hotbar first. Returns what did not fit.
    pub fn add(&mut self, item: Item, mut count: u32) -> u32 {
        let max = item.max_stack();
        for slot in self.slots.iter_mut().flatten() {
            if count == 0 {
                return 0;
            }
            if slot.item == item && slot.count < max {
                let n = (max - slot.count).min(count);
                slot.count += n;
                count -= n;
            }
        }
        for slot in self.slots.iter_mut() {
            if count == 0 {
                break;
            }
            if slot.is_none() {
                let n = max.min(count);
                *slot = Some(Stack::new(item, n));
                count -= n;
            }
        }
        count
    }

    /// How many items `wanted` accepts.
    pub fn count(&self, wanted: impl Fn(Item) -> bool) -> u32 {
        if self.creative {
            return u32::MAX;
        }
        self.slots
            .iter()
            .flatten()
            .filter(|s| wanted(s.item))
            .map(|s| s.count)
            .sum()
    }

    /// Takes `n` items `wanted` accepts, from the pack before the hotbar
    /// and the hotbar from its right, so what is at hand goes last. Takes
    /// nothing and returns the items it would have taken as None if there
    /// are too few.
    pub fn take(&mut self, wanted: impl Fn(Item) -> bool, n: u32) -> Option<Vec<(Item, u32)>> {
        if self.creative {
            return Some(Vec::new());
        }
        if self.count(&wanted) < n {
            return None;
        }
        let mut left = n;
        let mut taken: Vec<(Item, u32)> = Vec::new();
        let order = (HOTBAR_SLOTS..SLOTS).chain((0..HOTBAR_SLOTS).rev());
        for i in order {
            if left == 0 {
                break;
            }
            let Some(s) = &mut self.slots[i] else {
                continue;
            };
            if !wanted(s.item) {
                continue;
            }
            let k = s.count.min(left);
            s.count -= k;
            left -= k;
            match taken.iter_mut().find(|(it, _)| *it == s.item) {
                Some((_, c)) => *c += k,
                None => taken.push((s.item, k)),
            }
            if s.count == 0 {
                self.slots[i] = None;
            }
        }
        Some(taken)
    }

    /// Takes one item from a given slot.
    pub fn take_from(&mut self, slot: usize) -> Option<Item> {
        if self.creative {
            return self.slots[slot].map(|s| s.item);
        }
        let s = self.slots[slot].as_mut()?;
        let item = s.item;
        s.count -= 1;
        if s.count == 0 {
            self.slots[slot] = None;
        }
        Some(item)
    }

    /// The tool in `slot`, if it holds one.
    pub fn tool_in(&self, slot: usize) -> Option<(ToolKind, crate::items::Tier)> {
        self.slots.get(slot).copied().flatten()?.item.tool()
    }

    /// Wears the tool in `slot` by one use; true when that broke it.
    pub fn wear(&mut self, slot: usize) -> bool {
        if self.creative {
            return false;
        }
        let Some(s) = &mut self.slots[slot] else {
            return false;
        };
        if s.item.uses().is_none() {
            return false;
        }
        s.life = s.life.saturating_sub(1);
        if s.life == 0 {
            self.slots[slot] = None;
            return true;
        }
        false
    }

    /// Voxels of `m` carved loose. Every whole block's worth becomes what
    /// breaking such a block with the right tool gives.
    pub fn add_loose(&mut self, m: MaterialId, voxels: u64) {
        if self.creative || m.is_air() {
            return;
        }
        let v = self.loose.entry(m).or_default();
        *v += voxels;
        let whole = *v / BLOCK_VOXELS;
        *v %= BLOCK_VOXELS;
        let best = crate::items::suited_tool(m).map(|(k, _)| (k, crate::items::Tier::Steel));
        for _ in 0..whole {
            if let Some((item, n)) = drop_for(m, best, 1.0) {
                self.add(item, n);
            }
        }
    }

    /// Spends `voxels` of `m` for depositing: loose volume first, then whole
    /// blocks of it broken open. False, spending nothing, if there is not
    /// that much.
    pub fn take_loose(&mut self, m: MaterialId, voxels: u64) -> bool {
        if self.creative {
            return true;
        }
        let have = self.loose.get(&m).copied().unwrap_or(0);
        if have < voxels {
            let blocks = (voxels - have).div_ceil(BLOCK_VOXELS);
            if self.take(|i| i == Item::solid(m), blocks as u32).is_none() {
                return false;
            }
            *self.loose.entry(m).or_default() += blocks * BLOCK_VOXELS;
        }
        *self.loose.entry(m).or_default() -= voxels;
        true
    }

    /// Total voxels of `m` on hand, loose and in whole blocks.
    pub fn volume(&self, m: MaterialId) -> u64 {
        let blocks = u64::from(self.count(|i| i == Item::solid(m)));
        self.loose.get(&m).copied().unwrap_or(0) + blocks.saturating_mul(BLOCK_VOXELS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::Tier;
    use mc2_voxel::material::ids;

    #[test]
    fn stacks_fill_then_spill() {
        let mut inv = Inventory::default();
        let dirt = Item::solid(ids::DIRT);
        assert_eq!(inv.add(dirt, 100), 0);
        assert_eq!(inv.slots[0].unwrap().count, 64);
        assert_eq!(inv.slots[1].unwrap().count, 36);
        assert_eq!(inv.add(Item::Stick, 3), 0);
        assert_eq!(inv.add(dirt, 30), 0);
        assert_eq!(inv.slots[1].unwrap().count, 64);
        assert_eq!(inv.slots[3].unwrap().count, 2);
        assert_eq!(inv.count(|i| i == dirt), 130);
        // Tools never stack.
        let pick = Item::Tool(ToolKind::Pickaxe, Tier::Wood);
        inv.add(pick, 2);
        assert_eq!(inv.count(|i| i == pick), 2);
    }

    #[test]
    fn take_is_all_or_nothing() {
        let mut inv = Inventory::default();
        inv.add(Item::Coal, 5);
        assert!(inv.take(|i| i == Item::Coal, 6).is_none());
        assert_eq!(inv.count(|i| i == Item::Coal), 5);
        assert_eq!(
            inv.take(|i| i == Item::Coal, 5),
            Some(vec![(Item::Coal, 5)])
        );
        assert!(inv.slots.iter().all(Option::is_none));
    }

    #[test]
    fn tools_wear_out() {
        let mut inv = Inventory::default();
        inv.add(Item::Tool(ToolKind::Shovel, Tier::Wood), 1);
        for _ in 0..59 {
            assert!(!inv.wear(0));
        }
        assert!(inv.wear(0));
        assert!(inv.slots[0].is_none());
    }

    #[test]
    fn carving_loose_makes_blocks_and_deposits_spend_them() {
        let mut inv = Inventory::default();
        inv.add_loose(ids::GRANITE, 5000);
        assert_eq!(inv.count(|i| i == Item::solid(ids::GRANITE)), 1);
        assert_eq!(inv.volume(ids::GRANITE), 5000);
        inv.add_loose(ids::COAL_ORE, 4096);
        assert_eq!(inv.count(|i| i == Item::Coal), 1);
        assert!(inv.take_loose(ids::GRANITE, 2000));
        assert_eq!(inv.volume(ids::GRANITE), 3000);
        assert!(!inv.take_loose(ids::GRANITE, 3001));
        assert_eq!(inv.volume(ids::GRANITE), 3000);
    }
}
