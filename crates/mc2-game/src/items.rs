//! Items: what an inventory slot holds, what breaking a block yields, and
//! how fast which tool breaks what.
//!
//! A placeable block is an item (`Item::Block`); so are the things only
//! crafting makes (sticks, ingots, tools) and the things only mining finds
//! (coal, raw ore, flint). Tools wear with use and break at their tier's
//! durability.

use crate::blocks::BlockKind;
use mc2_voxel::material::{MaterialId, ids};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolKind {
    Pickaxe,
    Axe,
    Shovel,
}

/// Tool material, weakest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    Wood,
    Stone,
    Iron,
    Steel,
}

impl Tier {
    pub const ALL: [Tier; 4] = [Tier::Wood, Tier::Stone, Tier::Iron, Tier::Steel];

    pub fn name(self) -> &'static str {
        match self {
            Tier::Wood => "wooden",
            Tier::Stone => "stone",
            Tier::Iron => "iron",
            Tier::Steel => "steel",
        }
    }

    /// Break-speed multiplier on the materials the tool suits.
    pub fn speed(self) -> f32 {
        match self {
            Tier::Wood => 2.0,
            Tier::Stone => 4.0,
            Tier::Iron => 6.0,
            Tier::Steel => 9.0,
        }
    }

    /// Blocks broken before the tool falls apart.
    pub fn durability(self) -> u32 {
        match self {
            Tier::Wood => 60,
            Tier::Stone => 132,
            Tier::Iron => 251,
            Tier::Steel => 1200,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Item {
    Block(BlockKind),
    Stick,
    Coal,
    Charcoal,
    Flint,
    Gunpowder,
    RawIron,
    RawCopper,
    RawGold,
    IronIngot,
    CopperIngot,
    GoldIngot,
    SteelIngot,
    Tool(ToolKind, Tier),
}

impl Item {
    pub const fn solid(m: MaterialId) -> Item {
        Item::Block(BlockKind::Solid(m))
    }

    pub fn name(&self) -> String {
        match self {
            Item::Block(b) => b.name(),
            Item::Stick => "stick".into(),
            Item::Coal => "coal".into(),
            Item::Charcoal => "charcoal".into(),
            Item::Flint => "flint".into(),
            Item::Gunpowder => "gunpowder".into(),
            Item::RawIron => "raw iron".into(),
            Item::RawCopper => "raw copper".into(),
            Item::RawGold => "raw gold".into(),
            Item::IronIngot => "iron ingot".into(),
            Item::CopperIngot => "copper ingot".into(),
            Item::GoldIngot => "gold ingot".into(),
            Item::SteelIngot => "steel ingot".into(),
            Item::Tool(kind, tier) => format!(
                "{} {}",
                tier.name(),
                match kind {
                    ToolKind::Pickaxe => "pickaxe",
                    ToolKind::Axe => "axe",
                    ToolKind::Shovel => "shovel",
                }
            ),
        }
    }

    /// Most of the item one slot holds.
    pub fn max_stack(&self) -> u32 {
        match self {
            Item::Tool(..) => 1,
            _ => 64,
        }
    }

    pub fn tool(&self) -> Option<(ToolKind, Tier)> {
        match *self {
            Item::Tool(k, t) => Some((k, t)),
            _ => None,
        }
    }

    /// Furnace heat one of this item gives, in smelts.
    pub fn fuel(&self) -> u32 {
        match self {
            Item::Coal | Item::Charcoal => 8,
            Item::Block(BlockKind::Solid(m)) if is_log(*m) => 3,
            Item::Block(BlockKind::Solid(m)) if is_planks(*m) => 1,
            Item::Block(BlockKind::CraftingTable) => 2,
            _ => 0,
        }
    }
}

pub fn is_log(m: MaterialId) -> bool {
    matches!(m, ids::OAK_LOG | ids::PINE_LOG | ids::BIRCH_LOG)
}

pub fn is_planks(m: MaterialId) -> bool {
    matches!(m, ids::PLANKS | ids::DARK_PLANKS)
}

/// Natural rock: needs a pickaxe, crushes to cobblestone, dresses to brick.
pub fn is_rock(m: MaterialId) -> bool {
    matches!(
        m,
        ids::GRANITE
            | ids::BASALT
            | ids::LIMESTONE
            | ids::SANDSTONE
            | ids::SHALE
            | ids::MARBLE
            | ids::SLATE
            | ids::RHYOLITE
            | ids::CONGLOMERATE
            | ids::CHALK
            | ids::GNEISS
    )
}

/// The tool that suits a material and the least tier that yields a drop
/// (`None`: anything, even a bare hand, gets the drop).
pub fn suited_tool(m: MaterialId) -> Option<(ToolKind, Option<Tier>)> {
    use ToolKind::*;
    Some(match m {
        _ if is_rock(m) => (Pickaxe, Some(Tier::Wood)),
        ids::COBBLESTONE
        | ids::STONE_BRICK
        | ids::COAL_ORE
        | ids::RED_BRICK
        | ids::CONCRETE
        | ids::ASPHALT
        | ids::ROOF_TILE
        | ids::GLOWING_ROCK => (Pickaxe, Some(Tier::Wood)),
        ids::IRON_ORE | ids::COPPER_ORE | ids::IRON | ids::STEEL => (Pickaxe, Some(Tier::Stone)),
        ids::GOLD_ORE => (Pickaxe, Some(Tier::Iron)),
        ids::OBSIDIAN => (Pickaxe, Some(Tier::Steel)),
        ids::GLASS | ids::ICE | ids::GLOWSTONE | ids::WHITEWASH => (Pickaxe, None),
        _ if is_log(m) || is_planks(m) => (Axe, None),
        ids::CACTUS => (Axe, None),
        ids::DIRT
        | ids::GRASS
        | ids::SAND
        | ids::GRAVEL
        | ids::CLAY
        | ids::SNOW
        | ids::FARMLAND
        | ids::WET_DIRT
        | ids::ASH
        | ids::THATCH
        | ids::CHARCOAL => (Shovel, None),
        _ => return None,
    })
}

/// Seconds to break a block of `m` holding `tool` (`None`: bare hands).
/// `f32::INFINITY` for what cannot be broken.
pub fn break_time(m: MaterialId, tool: Option<(ToolKind, Tier)>) -> f32 {
    let hardness = m.get().hardness;
    if !hardness.is_finite() {
        return f32::INFINITY;
    }
    let base = 0.2 + 0.45 * hardness;
    let Some((kind, least)) = suited_tool(m) else {
        return base;
    };
    match tool {
        Some((k, tier)) if k == kind => {
            if least.is_some_and(|l| tier < l) {
                base * 3.3 / tier.speed()
            } else {
                base / tier.speed()
            }
        }
        // Without the tool it needs, rock takes an age and gives nothing.
        _ if least.is_some() => base * 3.3,
        _ => base,
    }
}

/// What breaking a block of `m` gives, if anything, for the hand or tool
/// that broke it. `roll` in `0..1` decides chance drops.
pub fn drop_for(m: MaterialId, tool: Option<(ToolKind, Tier)>, roll: f32) -> Option<(Item, u32)> {
    if let Some((kind, Some(least))) = suited_tool(m) {
        let ok = tool.is_some_and(|(k, t)| k == kind && t >= least);
        if !ok {
            return None;
        }
    }
    Some(match m {
        ids::GRASS | ids::FARMLAND => (Item::solid(ids::DIRT), 1),
        ids::GLOWING_ROCK => (Item::solid(ids::BASALT), 1),
        ids::COAL_ORE => (Item::Coal, 1 + u32::from(roll < 0.3)),
        ids::CHARCOAL => (Item::Charcoal, 1),
        ids::IRON_ORE => (Item::RawIron, 1),
        ids::COPPER_ORE => (Item::RawCopper, 1 + u32::from(roll < 0.5)),
        ids::GOLD_ORE => (Item::RawGold, 1),
        ids::GRAVEL if roll < 0.25 => (Item::Flint, 1),
        ids::ICE => return None,
        ids::LEAVES | ids::PINE_NEEDLES | ids::BIRCH_LEAVES => {
            return (roll < 0.12).then_some((Item::Stick, 1));
        }
        _ if m.get().kind == mc2_voxel::material::Kind::Foliage => return None,
        m if m.is_air() || !m.get().hardness.is_finite() => return None,
        m => (Item::solid(m), 1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rock_needs_a_pickaxe() {
        let wood = Some((ToolKind::Pickaxe, Tier::Wood));
        assert_eq!(drop_for(ids::GRANITE, None, 0.5), None);
        assert_eq!(
            drop_for(ids::GRANITE, wood, 0.5),
            Some((Item::solid(ids::GRANITE), 1))
        );
        assert_eq!(drop_for(ids::IRON_ORE, wood, 0.5), None);
        assert_eq!(
            drop_for(ids::IRON_ORE, Some((ToolKind::Pickaxe, Tier::Stone)), 0.5),
            Some((Item::RawIron, 1))
        );
        assert!(break_time(ids::GRANITE, None) > 4.0 * break_time(ids::GRANITE, wood));
        assert!(break_time(ids::BEDROCK, wood).is_infinite());
    }

    #[test]
    fn hands_fell_trees_and_dig_dirt() {
        assert_eq!(
            drop_for(ids::OAK_LOG, None, 0.9),
            Some((Item::solid(ids::OAK_LOG), 1))
        );
        assert_eq!(
            drop_for(ids::GRASS, None, 0.9),
            Some((Item::solid(ids::DIRT), 1))
        );
        assert!(break_time(ids::OAK_LOG, None) < 1.5);
        let axe = Some((ToolKind::Axe, Tier::Stone));
        assert!(break_time(ids::OAK_LOG, axe) < break_time(ids::OAK_LOG, None) / 3.0);
        assert_eq!(drop_for(ids::TALL_GRASS, None, 0.0), None);
        assert_eq!(drop_for(ids::GRAVEL, None, 0.1), Some((Item::Flint, 1)));
    }

    #[test]
    fn fuels_and_stacks() {
        assert_eq!(Item::Coal.fuel(), 8);
        assert_eq!(Item::solid(ids::PINE_LOG).fuel(), 3);
        assert_eq!(Item::solid(ids::GRANITE).fuel(), 0);
        assert_eq!(Item::Tool(ToolKind::Axe, Tier::Iron).max_stack(), 1);
        assert_eq!(
            Item::Tool(ToolKind::Pickaxe, Tier::Steel).name(),
            "steel pickaxe"
        );
    }
}
