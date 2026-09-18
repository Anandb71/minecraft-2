//! Recipes: what can be made from what, and where.
//!
//! Hand recipes work anywhere; table recipes need a crafting table within
//! reach, furnace recipes a furnace and fuel. The furnace burns one fuel
//! item at a time into heat (coal smelts eight things, a log three) and
//! each smelt spends one heat.

use crate::blocks::BlockKind;
use crate::inventory::Inventory;
use crate::items::{Item, Tier, ToolKind, is_log, is_planks, is_rock};
use mc2_voxel::material::{MaterialId, ids};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Station {
    Hand,
    Table,
    Furnace,
}

impl Station {
    pub fn name(self) -> &'static str {
        match self {
            Station::Hand => "hand",
            Station::Table => "crafting table",
            Station::Furnace => "furnace",
        }
    }
}

/// Stations within reach of the crafter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Near {
    pub table: bool,
    pub furnace: bool,
}

impl Near {
    pub fn has(self, s: Station) -> bool {
        match s {
            Station::Hand => true,
            Station::Table => self.table,
            Station::Furnace => self.furnace,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ingredient {
    Item(Item),
    AnyLog,
    AnyPlanks,
    /// Cobblestone or any natural rock.
    AnyStone,
    /// Coal or charcoal.
    AnyCoal,
    /// Chalk, limestone or marble: lime for plaster and cement.
    AnyLime,
}

impl Ingredient {
    pub fn matches(self, item: Item) -> bool {
        let solid = match item {
            Item::Block(BlockKind::Solid(m)) => Some(m),
            _ => None,
        };
        match self {
            Ingredient::Item(i) => i == item,
            Ingredient::AnyLog => solid.is_some_and(is_log),
            Ingredient::AnyPlanks => solid.is_some_and(is_planks),
            Ingredient::AnyStone => solid.is_some_and(|m| m == ids::COBBLESTONE || is_rock(m)),
            Ingredient::AnyCoal => matches!(item, Item::Coal | Item::Charcoal),
            Ingredient::AnyLime => {
                solid.is_some_and(|m| matches!(m, ids::CHALK | ids::LIMESTONE | ids::MARBLE))
            }
        }
    }

    pub fn name(self) -> String {
        match self {
            Ingredient::Item(i) => i.name(),
            Ingredient::AnyLog => "any log".into(),
            Ingredient::AnyPlanks => "any planks".into(),
            Ingredient::AnyStone => "any stone".into(),
            Ingredient::AnyCoal => "coal or charcoal".into(),
            Ingredient::AnyLime => "chalk, limestone or marble".into(),
        }
    }

    /// An item that satisfies it, to show.
    pub fn example(self) -> Item {
        match self {
            Ingredient::Item(i) => i,
            Ingredient::AnyLog => Item::solid(ids::OAK_LOG),
            Ingredient::AnyPlanks => Item::solid(ids::PLANKS),
            Ingredient::AnyStone => Item::solid(ids::COBBLESTONE),
            Ingredient::AnyCoal => Item::Coal,
            Ingredient::AnyLime => Item::solid(ids::LIMESTONE),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Recipe {
    pub output: Item,
    pub count: u32,
    pub inputs: Vec<(Ingredient, u32)>,
    pub station: Station,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CraftError {
    /// The station it needs is not within reach.
    Station(Station),
    /// Short of an ingredient.
    Missing(Ingredient),
    /// The furnace has no heat and there is nothing to burn.
    NoFuel,
    /// The pack is full.
    NoRoom,
}

fn solid(m: MaterialId) -> Ingredient {
    Ingredient::Item(Item::solid(m))
}

fn recipe(output: Item, count: u32, station: Station, inputs: &[(Ingredient, u32)]) -> Recipe {
    Recipe {
        output,
        count,
        inputs: inputs.to_vec(),
        station,
    }
}

/// Every recipe, roughly in the order a new world needs them.
pub fn recipes() -> &'static [Recipe] {
    static ALL: OnceLock<Vec<Recipe>> = OnceLock::new();
    ALL.get_or_init(build)
}

fn build() -> Vec<Recipe> {
    use Ingredient::{AnyCoal, AnyLime, AnyLog, AnyPlanks, AnyStone};
    use Station::*;
    let it = Ingredient::Item;
    let stick = it(Item::Stick);
    let mut r = vec![
        recipe(
            Item::solid(ids::PLANKS),
            4,
            Hand,
            &[(solid(ids::OAK_LOG), 1)],
        ),
        recipe(
            Item::solid(ids::PLANKS),
            4,
            Hand,
            &[(solid(ids::BIRCH_LOG), 1)],
        ),
        recipe(
            Item::solid(ids::DARK_PLANKS),
            4,
            Hand,
            &[(solid(ids::PINE_LOG), 1)],
        ),
        recipe(Item::Stick, 4, Hand, &[(AnyPlanks, 2)]),
        recipe(
            Item::Block(BlockKind::CraftingTable),
            1,
            Hand,
            &[(AnyPlanks, 4)],
        ),
        recipe(
            Item::Block(BlockKind::Torch),
            4,
            Hand,
            &[(stick, 1), (AnyCoal, 1)],
        ),
    ];
    // Tools: a head of three (a shovel's of one) on two sticks.
    for tier in Tier::ALL {
        let head = match tier {
            Tier::Wood => AnyPlanks,
            Tier::Stone => AnyStone,
            Tier::Iron => it(Item::IronIngot),
            Tier::Steel => it(Item::SteelIngot),
        };
        for (kind, n) in [
            (ToolKind::Pickaxe, 3),
            (ToolKind::Axe, 3),
            (ToolKind::Shovel, 1),
        ] {
            r.push(recipe(
                Item::Tool(kind, tier),
                1,
                Table,
                &[(head, n), (stick, 2)],
            ));
        }
    }
    r.extend([
        recipe(
            Item::Block(BlockKind::Furnace),
            1,
            Table,
            &[(solid(ids::COBBLESTONE), 8)],
        ),
        recipe(Item::solid(ids::COBBLESTONE), 1, Table, &[(AnyStone, 1)]),
        recipe(Item::solid(ids::STONE_BRICK), 4, Table, &[(AnyStone, 4)]),
        recipe(
            Item::Gunpowder,
            2,
            Table,
            &[(AnyCoal, 1), (it(Item::Flint), 1)],
        ),
        recipe(
            Item::solid(ids::TNT),
            1,
            Table,
            &[(it(Item::Gunpowder), 5), (solid(ids::SAND), 4)],
        ),
        recipe(
            Item::Block(BlockKind::Window),
            2,
            Table,
            &[(solid(ids::GLASS), 1), (stick, 2)],
        ),
        recipe(
            Item::Block(BlockKind::Lantern),
            1,
            Table,
            &[
                (it(Item::Block(BlockKind::Torch)), 1),
                (solid(ids::GLASS), 1),
                (AnyPlanks, 1),
            ],
        ),
        recipe(
            Item::solid(ids::CONCRETE),
            4,
            Table,
            &[(solid(ids::GRAVEL), 2), (solid(ids::SAND), 2), (AnyLime, 1)],
        ),
        recipe(
            Item::solid(ids::WHITEWASH),
            4,
            Table,
            &[(AnyLime, 1), (solid(ids::SAND), 2)],
        ),
        recipe(
            Item::solid(ids::IRON),
            1,
            Table,
            &[(it(Item::IronIngot), 9)],
        ),
        recipe(Item::Bucket, 1, Table, &[(it(Item::IronIngot), 3)]),
        recipe(
            Item::solid(ids::STEEL),
            1,
            Table,
            &[(it(Item::SteelIngot), 9)],
        ),
    ]);
    for m in [
        ids::STONE_BRICK,
        ids::COBBLESTONE,
        ids::PLANKS,
        ids::DARK_PLANKS,
        ids::RED_BRICK,
        ids::CONCRETE,
        ids::MARBLE,
    ] {
        r.push(recipe(
            Item::Block(BlockKind::Slab(m)),
            2,
            Table,
            &[(solid(m), 1)],
        ));
    }
    let smelt = |out: Item, from: Ingredient| recipe(out, 1, Furnace, &[(from, 1)]);
    r.extend([
        smelt(Item::solid(ids::GLASS), solid(ids::SAND)),
        smelt(Item::IronIngot, it(Item::RawIron)),
        smelt(Item::CopperIngot, it(Item::RawCopper)),
        smelt(Item::GoldIngot, it(Item::RawGold)),
        smelt(Item::Charcoal, AnyLog),
        smelt(Item::solid(ids::RED_BRICK), solid(ids::CLAY)),
        smelt(Item::solid(ids::ROOF_TILE), solid(ids::RED_BRICK)),
        recipe(
            Item::SteelIngot,
            1,
            Furnace,
            &[(it(Item::IronIngot), 1), (AnyCoal, 1)],
        ),
    ]);
    r
}

/// Fuel burns in this order: coal first, the wood it would be a shame to
/// burn last.
const FUEL_ORDER: [fn(Item) -> bool; 4] = [
    |i| matches!(i, Item::Coal | Item::Charcoal),
    |i| matches!(i, Item::Block(BlockKind::Solid(m)) if is_planks(m)),
    |i| matches!(i, Item::Block(BlockKind::Solid(m)) if is_log(m)),
    |i| i.fuel() > 0,
];

/// Whether `recipe` could be made now, and if not, why.
pub fn check(inv: &Inventory, recipe: &Recipe, near: Near) -> Result<(), CraftError> {
    let mut trial = inv.clone();
    apply(&mut trial, recipe, near)
}

/// Makes `recipe` once: takes its inputs (and fuel), gives its output.
pub fn craft(inv: &mut Inventory, recipe: &Recipe, near: Near) -> Result<(), CraftError> {
    let mut trial = inv.clone();
    apply(&mut trial, recipe, near)?;
    *inv = trial;
    Ok(())
}

fn apply(inv: &mut Inventory, recipe: &Recipe, near: Near) -> Result<(), CraftError> {
    if !inv.creative && !near.has(recipe.station) {
        return Err(CraftError::Station(recipe.station));
    }
    for &(ing, n) in &recipe.inputs {
        if inv.take(|i| ing.matches(i), n).is_none() {
            return Err(CraftError::Missing(ing));
        }
    }
    if recipe.station == Station::Furnace && !inv.creative {
        if inv.heat == 0 {
            let fuel = FUEL_ORDER
                .iter()
                .find_map(|wanted| inv.take(*wanted, 1))
                .and_then(|t| t.first().copied())
                .ok_or(CraftError::NoFuel)?;
            inv.heat += fuel.0.fuel();
        }
        inv.heat -= 1;
    }
    if inv.creative {
        return Ok(());
    }
    if inv.add(recipe.output, recipe.count) > 0 {
        return Err(CraftError::NoRoom);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(output: Item) -> &'static Recipe {
        recipes().iter().find(|r| r.output == output).unwrap()
    }

    #[test]
    fn a_tree_becomes_a_pickaxe() {
        let mut inv = Inventory::default();
        inv.add(Item::solid(ids::OAK_LOG), 3);
        let hand = Near::default();
        let planks = &recipes()[0];
        for _ in 0..3 {
            craft(&mut inv, planks, hand).unwrap();
        }
        assert_eq!(inv.count(|i| i == Item::solid(ids::PLANKS)), 12);
        craft(&mut inv, find(Item::Stick), hand).unwrap();
        let table = find(Item::Block(BlockKind::CraftingTable));
        craft(&mut inv, table, hand).unwrap();
        let pick = find(Item::Tool(ToolKind::Pickaxe, Tier::Wood));
        assert_eq!(
            check(&inv, pick, hand),
            Err(CraftError::Station(Station::Table))
        );
        let at_table = Near {
            table: true,
            furnace: false,
        };
        craft(&mut inv, pick, at_table).unwrap();
        assert_eq!(inv.count(|i| i.tool().is_some()), 1);
        // 12 planks - 2 sticks - 4 table - 3 head = 3 left, 2 sticks left.
        assert_eq!(inv.count(|i| i == Item::solid(ids::PLANKS)), 3);
        assert_eq!(inv.count(|i| i == Item::Stick), 2);
    }

    #[test]
    fn smelting_burns_fuel_into_heat() {
        let mut inv = Inventory::default();
        inv.add(Item::solid(ids::SAND), 10);
        let glass = find(Item::solid(ids::GLASS));
        let near = Near {
            table: false,
            furnace: true,
        };
        assert_eq!(check(&inv, glass, near), Err(CraftError::NoFuel));
        inv.add(Item::Coal, 1);
        inv.add(Item::solid(ids::PLANKS), 2);
        for _ in 0..8 {
            craft(&mut inv, glass, near).unwrap();
        }
        // Coal burned first; planks untouched until it ran out.
        assert_eq!(inv.count(|i| i == Item::Coal), 0);
        assert_eq!(inv.count(|i| i == Item::solid(ids::PLANKS)), 2);
        craft(&mut inv, glass, near).unwrap();
        assert_eq!(inv.count(|i| i == Item::solid(ids::PLANKS)), 1);
        assert_eq!(inv.count(|i| i == Item::solid(ids::GLASS)), 9);
    }

    #[test]
    fn failed_crafts_take_nothing() {
        let mut inv = Inventory::default();
        inv.add(Item::Gunpowder, 5);
        inv.add(Item::solid(ids::SAND), 3);
        let tnt = find(Item::solid(ids::TNT));
        let near = Near {
            table: true,
            furnace: false,
        };
        assert_eq!(
            craft(&mut inv, tnt, near),
            Err(CraftError::Missing(solid(ids::SAND)))
        );
        assert_eq!(inv.count(|i| i == Item::Gunpowder), 5);
        inv.add(Item::solid(ids::SAND), 1);
        craft(&mut inv, tnt, near).unwrap();
        assert_eq!(inv.count(|i| i == Item::solid(ids::TNT)), 1);
    }

    #[test]
    fn every_recipe_names_its_inputs() {
        for r in recipes() {
            assert!(!r.inputs.is_empty(), "{:?}", r.output);
            for (ing, n) in &r.inputs {
                assert!(*n > 0);
                assert!(ing.matches(ing.example()), "{}", ing.name());
            }
        }
    }
}
