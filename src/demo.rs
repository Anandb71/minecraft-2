//! Scripted play for headless captures: the same game systems a player
//! drives, fed synthetic input, so screenshots and regression captures
//! exercise real interaction code.

use mc2_game::Game;
use mc2_game::blocks::BlockKind;
use mc2_game::crafting;
use mc2_game::input::{Button, Key};
use mc2_game::interact::Interaction;
use mc2_game::items::{Item, Tier, ToolKind};
use mc2_render::hud::HudCanvas;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Demo {
    /// Stand, build a small wall with a window and a torch, carve a crater.
    Build,
    /// Ring the player with lanterns and torches on the ground: emitters
    /// for ReSTIR, meant to be captured at night.
    Lights,
    /// Lay polished slabs (steel, obsidian, ice) and a wall in front of the
    /// player: glossy reflections to look at.
    Mirror,
    /// Place two TNT blocks down the view, light one, and stop while the
    /// debris of both blasts is in the air.
    Blast,
    /// Raise a stone tower, blow its base out and stop while it comes down.
    Collapse,
    /// A pool of water over a gravel bed and a glass wall with coloured
    /// blocks behind it: refraction and absorption to look at.
    Glass,
    /// Survival: turn a few logs and some stone into a crafting table,
    /// tools and a furnace, set them down, and stop with the inventory open
    /// at the table.
    Craft,
    /// The craft demo's workshop from outside: table and furnace set down,
    /// a torch beside them, and a block of ground half broken.
    Workshop,
    /// A stone tank ahead, open on the near side, poured three metres deep:
    /// the water collapses across the ground toward the camera.
    Flood,
    /// Standing on a beach facing the sea: dig a trench 2 m wide from the
    /// player's feet to the water, below sea level, and let the sea in.
    Shore,
    /// A tree ahead set alight with flint and steel, caught seven seconds
    /// later with its canopy ablaze.
    Wildfire,
    /// A storm, and a bolt brought down forty metres ahead.
    Lightning,
}

impl Demo {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "build" => Some(Self::Build),
            "lights" => Some(Self::Lights),
            "mirror" => Some(Self::Mirror),
            "blast" => Some(Self::Blast),
            "collapse" => Some(Self::Collapse),
            "glass" => Some(Self::Glass),
            "craft" => Some(Self::Craft),
            "workshop" => Some(Self::Workshop),
            "flood" => Some(Self::Flood),
            "shore" => Some(Self::Shore),
            "wildfire" => Some(Self::Wildfire),
            "lightning" => Some(Self::Lightning),
            _ => None,
        }
    }
}

fn tick(game: &mut Game, frames: u32) {
    let threaded = game
        .world
        .resource::<mc2_game::physics::Physics>()
        .host
        .is_threaded();
    for _ in 0..frames {
        game.update(1.0 / 60.0);
        mc2_core::profiler::end_frame();
        if threaded {
            // Physics runs beside the script in real time, as in the window.
            std::thread::sleep(std::time::Duration::from_secs_f32(1.0 / 60.0));
        }
    }
}

fn click(game: &mut Game, button: Button) {
    game.input().button_down(button);
    tick(game, 1);
    game.input().button_up(button);
    tick(game, 1);
}

fn select(game: &mut Game, slot: u8) {
    game.input().key_down(Key::Slot(slot));
    tick(game, 1);
    game.input().key_up(Key::Slot(slot));
    tick(game, 1);
}

fn look(game: &mut Game, dx: f32, dy: f32) {
    game.input().mouse_delta += glam::Vec2::new(dx, dy);
    tick(game, 1);
}

/// Makes `output` from what is carried, if it can, and says so.
fn make(game: &mut Game, output: Item, times: u32) {
    let near = crafting::Near {
        table: true,
        furnace: true,
    };
    let now = game.world.resource::<mc2_game::input::Time>().elapsed;
    let recipe = crafting::recipes()
        .iter()
        .find(|r| r.output == output)
        .expect("recipe");
    let mut state = game.world.resource_mut::<Interaction>();
    let mut made = 0;
    for _ in 0..times {
        if crafting::craft(&mut state.inventory, recipe, near).is_ok() {
            made += recipe.count;
        }
    }
    state.say(now, format!("made {made} {}", output.name()));
}

/// Selects the hotbar slot holding `item`, moving it there if need be.
fn hold(game: &mut Game, item: Item) {
    let slot = {
        let mut state = game.world.resource_mut::<Interaction>();
        let inv = &mut state.inventory;
        let at = inv
            .slots
            .iter()
            .position(|s| s.is_some_and(|s| s.item == item));
        match at {
            Some(i) if i < 9 => i,
            Some(i) => {
                inv.slots.swap(i, 8);
                8
            }
            None => return,
        }
    };
    select(game, slot as u8);
}

/// Survival from a handful of materials: planks, sticks, a crafting table,
/// tools, a furnace, glass, iron, torches and gunpowder, with the table and
/// furnace set down side by side ahead.
fn workshop(game: &mut Game) {
    use mc2_voxel::material::ids;
    *game.world.resource_mut::<Interaction>() = Interaction::new(false);
    {
        let mut state = game.world.resource_mut::<Interaction>();
        let inv = &mut state.inventory;
        for (item, n) in [
            (Item::solid(ids::OAK_LOG), 7),
            (Item::solid(ids::COBBLESTONE), 24),
            (Item::Coal, 6),
            (Item::solid(ids::SAND), 12),
            (Item::RawIron, 4),
            (Item::Flint, 3),
            (Item::solid(ids::GRAVEL), 4),
        ] {
            inv.add(item, n);
        }
    }
    make(game, Item::solid(ids::PLANKS), 7);
    make(game, Item::Stick, 3);
    make(game, Item::Block(BlockKind::CraftingTable), 1);
    make(game, Item::Tool(ToolKind::Pickaxe, Tier::Wood), 1);
    make(game, Item::Tool(ToolKind::Pickaxe, Tier::Stone), 1);
    make(game, Item::Tool(ToolKind::Axe, Tier::Stone), 1);
    make(game, Item::Block(BlockKind::Furnace), 1);
    make(game, Item::solid(ids::GLASS), 3);
    make(game, Item::IronIngot, 3);
    make(game, Item::Block(BlockKind::Torch), 2);
    make(game, Item::Gunpowder, 2);
    // Set the table and furnace down side by side ahead.
    look(game, 0.0, 150.0);
    hold(game, Item::Block(BlockKind::CraftingTable));
    click(game, Button::Secondary);
    look(game, -60.0, 0.0);
    hold(game, Item::Block(BlockKind::Furnace));
    click(game, Button::Secondary);
    look(game, 30.0, -60.0);
}

/// Screens a demo shows over its capture: the craft demo's inventory, open
/// at the crafting table with the cursor over a recipe.
pub fn overlay(game: &mut Game, hud: &mut HudCanvas, screen: (u32, u32), demo: Option<Demo>) {
    if demo != Some(Demo::Craft) {
        return;
    }
    let mut inv = crate::inventory_ui::Screen::default();
    inv.open_at(Some(crafting::Station::Table));
    let (w, h) = (screen.0 as f32, screen.1 as f32);
    inv.mouse_move(w * 0.72, h * 0.5 - 150.0);
    // Once aside to lay out, then for real with the cursor's hover.
    inv.draw(game, &mut HudCanvas::default(), screen);
    inv.draw(game, hud, screen);
}

/// Runs the script. Streaming must already cover the player.
pub fn run(game: &mut Game, demo: Demo) {
    game.input().captured = true;
    tick(game, 60);
    match demo {
        Demo::Build => {
            // Look down at the ground a few metres ahead.
            look(game, 0.0, 180.0);
            for slot in [0u8, 0, 2, 0, 0] {
                game.input().key_down(Key::Slot(slot));
                tick(game, 1);
                game.input().key_up(Key::Slot(slot));
                click(game, Button::Secondary);
                look(game, -90.0, 0.0);
            }
            look(game, 250.0, -60.0);
            game.input().key_down(Key::Slot(3));
            tick(game, 1);
            game.input().key_up(Key::Slot(3));
            click(game, Button::Secondary);
            // Carve a crater to the side.
            game.input().key_down(Key::ToggleMode);
            tick(game, 1);
            game.input().key_up(Key::ToggleMode);
            game.input().scroll = 10.0;
            look(game, 300.0, 40.0);
            game.input().button_down(Button::Primary);
            tick(game, 40);
            game.input().button_up(Button::Primary);
            tick(game, 2);
            look(game, -280.0, -20.0);
        }
        Demo::Lights => {
            // Place on the ground a few metres out, turning between each.
            look(game, 0.0, 140.0);
            for (i, slot) in [4u8, 3, 4, 3, 4, 3].into_iter().enumerate() {
                select(game, slot);
                click(game, Button::Secondary);
                look(game, if i % 2 == 0 { 170.0 } else { 150.0 }, 0.0);
            }
            // Face the first lantern again, looking down the row.
            look(game, 120.0, -110.0);
        }
        Demo::Mirror => {
            use mc2_voxel::material::ids;
            let feet = {
                let mut q = game.world.query::<&mc2_game::player::Body>();
                q.iter(&game.world).next().map(|b| b.feet)
            };
            if let Some(feet) = feet {
                let (yaw, _) = {
                    let view = game.view();
                    (view.yaw, view.pitch)
                };
                // Ahead of the player along the view's horizontal direction.
                let ahead = glam::DVec3::new(f64::from(yaw).sin(), 0.0, f64::from(yaw).cos());
                let centre = feet + ahead * 5.0;
                let v = |p: glam::DVec3| (p * 16.0).floor().as_ivec3();
                let floor = v(centre) - glam::IVec3::new(0, 1, 0);
                let mut voxels = game.world.resource_mut::<mc2_game::Voxels>();
                let w = &mut voxels.0;
                // Clear air above, then three 2 m slabs side by side.
                w.fill_box(
                    floor + glam::IVec3::new(-48, 1, -16),
                    floor + glam::IVec3::new(47, 64, 31),
                    ids::AIR,
                );
                for (i, m) in [ids::STEEL, ids::OBSIDIAN, ids::ICE]
                    .into_iter()
                    .enumerate()
                {
                    let x = -48 + i as i32 * 32;
                    w.fill_box(
                        floor + glam::IVec3::new(x, -3, -16),
                        floor + glam::IVec3::new(x + 31, 0, 15),
                        m,
                    );
                }
                // A marble wall behind them to reflect.
                w.fill_box(
                    floor + glam::IVec3::new(-48, 1, 16),
                    floor + glam::IVec3::new(47, 48, 31),
                    ids::MARBLE,
                );
            }
            look(game, 0.0, 60.0);
        }
        Demo::Glass => {
            use glam::IVec3;
            use mc2_voxel::material::ids;
            let feet = {
                let mut q = game.world.query::<&mc2_game::player::Body>();
                q.iter(&game.world).next().map(|b| b.feet)
            };
            if let Some(feet) = feet {
                let yaw = game.view().yaw;
                let ahead = glam::DVec3::new(f64::from(yaw).sin(), 0.0, f64::from(yaw).cos());
                let side = glam::DVec3::new(ahead.z, 0.0, -ahead.x);
                let v = |p: glam::DVec3| (p * 16.0).floor().as_ivec3();
                let ground = v(feet) - IVec3::new(0, 1, 0);
                let mut voxels = game.world.resource_mut::<mc2_game::Voxels>();
                let w = &mut voxels.0;
                let block = |w: &mut mc2_voxel::world::VoxelWorld,
                             at: glam::DVec3,
                             dy: i32,
                             m: mc2_voxel::material::MaterialId| {
                    let c: IVec3 = (v(at) >> 4) << 4;
                    let lo = IVec3::new(c.x, ground.y + 1 + dy * 16, c.z);
                    w.fill_box(lo, lo + 15, m);
                };
                // A pool 6 m by 4 m, 1.5 m deep: gravel and sand bed, water,
                // on the block grid a little ahead.
                let centre: IVec3 = v(feet + ahead * 4.0) >> 4;
                let top = ground.y;
                for bx in -3..3 {
                    for bz in -2..2 {
                        let c: IVec3 = (centre + IVec3::new(bx, 0, bz)) << 4;
                        let bed = if (bx + bz).rem_euclid(3) == 0 {
                            ids::SAND
                        } else {
                            ids::GRAVEL
                        };
                        w.fill_box(
                            IVec3::new(c.x, top - 31, c.z),
                            IVec3::new(c.x + 15, top - 24, c.z + 15),
                            bed,
                        );
                        w.fill_box(
                            IVec3::new(c.x, top - 23, c.z),
                            IVec3::new(c.x + 15, top - 1, c.z + 15),
                            ids::WATER,
                        );
                        w.fill_box(
                            IVec3::new(c.x, top, c.z),
                            IVec3::new(c.x + 15, top + 48, c.z + 15),
                            ids::AIR,
                        );
                    }
                }
                // Behind the pool: glass blocks in front of coloured ones.
                for i in -3i32..3 {
                    let at = feet + side * f64::from(i) + ahead * 7.0;
                    for dy in 0..3 {
                        block(w, at, dy, ids::GLASS);
                    }
                    let behind = at + ahead * 2.0;
                    let m = [
                        ids::GOLD_ORE,
                        ids::ROOF_TILE,
                        ids::COPPER_ORE,
                        ids::WOOL,
                        ids::TNT,
                        ids::PLANKS,
                    ][(i + 3) as usize];
                    for dy in 0..2 {
                        block(w, behind, dy, m);
                    }
                }
            }
            look(game, 0.0, 70.0);
        }
        Demo::Collapse => {
            use mc2_game::physics::{Blast, Physics};
            use mc2_game::structure::Structure;
            use mc2_voxel::material::ids;
            let (feet, yaw) = {
                let mut q = game.world.query::<&mc2_game::player::Body>();
                let feet = q.iter(&game.world).next().map(|b| b.feet);
                (feet, game.view().yaw)
            };
            let Some(feet) = feet else {
                return;
            };
            let ahead = glam::DVec3::new(f64::from(yaw).sin(), 0.0, f64::from(yaw).cos());
            let base = feet + ahead * 20.0;
            let b0: glam::IVec3 = (base * 16.0).floor().as_ivec3() >> 4;
            // A 3 m square stone tower 14 m tall under a 5 m cap.
            let tower = [
                (
                    b0 + glam::IVec3::new(-1, 0, -1),
                    b0 + glam::IVec3::new(1, 13, 1),
                ),
                (
                    b0 + glam::IVec3::new(-2, 14, -2),
                    b0 + glam::IVec3::new(2, 14, 2),
                ),
            ];
            for (lo, hi) in tower {
                let (a, b) = (
                    lo * mc2_voxel::coords::VOXELS_PER_BLOCK,
                    (hi + 1) * mc2_voxel::coords::VOXELS_PER_BLOCK - 1,
                );
                game.world
                    .resource_mut::<mc2_game::Voxels>()
                    .0
                    .fill_box(a, b, ids::STONE_BRICK);
                let s = &mut *game.world.resource_mut::<Structure>();
                s.mark_built(a, b);
                s.edited(a, b);
            }
            look(game, 0.0, -60.0);
            // Let it settle, then blow the base out from under it.
            tick(game, 120);
            let standing = game.world.resource::<Physics>().host.body_count();
            game.world.resource_mut::<Physics>().blasts.push(Blast {
                centre: (b0.as_dvec3() + glam::DVec3::new(0.5, 1.0, 0.5)),
                radius: 3.5,
            });
            let bodies = |g: &Game| g.world.resource::<Physics>().host.body_count();
            let mut waited = 0;
            while bodies(game) < standing + 40 && waited < 600 {
                tick(game, 1);
                waited += 1;
            }
            // Long enough for the pieces to be visibly on their way down.
            tick(game, 30);
            let s = game.world.resource::<Structure>().stats;
            eprintln!(
                "collapse demo: {} bodies after {:.2} s, {} failures, {} collapses, {} pieces, worst ratio {:.2}",
                bodies(game),
                f64::from(waited) / 60.0,
                s.failures,
                s.islands,
                s.pieces,
                s.worst_ratio
            );
        }
        Demo::Blast => {
            // TNT sits in hotbar slot 9. Two blocks about two metres apart,
            // five to seven metres out.
            look(game, 0.0, 130.0);
            select(game, 8);
            click(game, Button::Secondary);
            look(game, 120.0, 0.0);
            click(game, Button::Secondary);
            // Raise the crosshair onto the second charge and light it by
            // hand; the first goes off when the blast reaches it.
            let on_tnt = |g: &Game| {
                g.world
                    .resource::<Interaction>()
                    .target
                    .is_some_and(|t| t.hit.material == mc2_voxel::material::ids::TNT)
            };
            for _ in 0..80 {
                if on_tnt(game) {
                    break;
                }
                look(game, 0.0, -2.0);
            }
            game.input().key_down(Key::Interact);
            tick(game, 1);
            game.input().key_up(Key::Interact);
            look(game, -60.0, -60.0);
            let blasts = |g: &Game| {
                g.world
                    .resource::<mc2_game::physics::Physics>()
                    .blasts_total
            };
            let mut waited = 0;
            while blasts(game) < 2 && waited < 600 {
                tick(game, 1);
                waited += 1;
            }
            // Debris mid-flight.
            tick(game, 9);
            let p = game.world.resource::<mc2_game::physics::Physics>();
            eprintln!(
                "blast demo: {} blasts after {:.2} s, {} bodies, last {:?}",
                p.blasts_total,
                f64::from(waited) / 60.0,
                p.host.body_count(),
                p.last_blast
            );
        }
        Demo::Flood => {
            use glam::IVec3;
            use mc2_voxel::material::ids;
            look(game, 0.0, 60.0);
            let feet = {
                let mut q = game.world.query::<&mc2_game::player::Body>();
                q.iter(&game.world).next().map(|b| b.feet)
            };
            if let Some(feet) = feet {
                let yaw = game.view().yaw;
                let ahead = glam::DVec3::new(f64::from(yaw).sin(), 0.0, f64::from(yaw).cos());
                let ground = (feet * 16.0).floor().as_ivec3().y;
                let centre: IVec3 = (feet + ahead * 14.0).floor().as_ivec3();
                {
                    let mut voxels = game.world.resource_mut::<mc2_game::Voxels>();
                    let w = &mut voxels.0;
                    // Walls 4 m high round a 7 m square, the side nearest the
                    // camera left out; the ground inside cleared of plants.
                    for bz in -4..=4 {
                        for bx in -4..=4 {
                            let b = centre + IVec3::new(bx, 0, bz);
                            let lo = IVec3::new(b.x * 16, ground, b.z * 16);
                            let ring = bx.abs() == 4 || bz.abs() == 4;
                            let near = glam::DVec3::new(f64::from(bx), 0.0, f64::from(bz))
                                .dot(ahead)
                                < -2.5;
                            let m = if ring && !near {
                                ids::STONE_BRICK
                            } else {
                                ids::AIR
                            };
                            w.fill_box(lo, lo + IVec3::new(15, 16 * 4 - 1, 15), m);
                        }
                    }
                }
                let lo = IVec3::new((centre.x - 3) * 16, ground, (centre.z - 3) * 16);
                let hi = IVec3::new(
                    (centre.x + 4) * 16 - 1,
                    ground + 16 * 3 - 1,
                    (centre.z + 4) * 16 - 1,
                );
                game.world
                    .resource_mut::<mc2_game::water::Water>()
                    .pour(lo, hi);
            }
        }
        Demo::Shore => {
            use glam::{DVec3, IVec3};
            use mc2_voxel::material::ids;
            let feet = {
                let mut q = game.world.query::<&mc2_game::player::Body>();
                q.iter(&game.world).next().map(|b| b.feet)
            };
            if let Some(feet) = feet {
                let yaw = game.view().yaw;
                let ahead = DVec3::new(f64::from(yaw).sin(), 0.0, f64::from(yaw).cos());
                let side = DVec3::new(ahead.z, 0.0, -ahead.x);
                let sea = 96.0 * 16.0;
                let floor = (sea - 20.0) as i32;
                let top = (feet.y * 16.0) as i32 + 32;
                game.world.resource_scope(
                    |world, mut state: bevy_ecs::prelude::Mut<Interaction>| {
                        world.resource_scope(
                            |world, mut streaming: bevy_ecs::prelude::Mut<mc2_game::Streaming>| {
                                let mut voxels = world.resource_mut::<mc2_game::Voxels>();
                                // Half a metre at a time toward the sea, until two
                                // metres past the first water.
                                let mut past_water = None;
                                for step in 2..120 {
                                    let at = feet + ahead * (f64::from(step) * 0.5);
                                    let a = ((at - side) * 16.0).floor().as_ivec3();
                                    let b = ((at + side) * 16.0).floor().as_ivec3();
                                    let lo = IVec3::new(a.x.min(b.x), floor, a.z.min(b.z));
                                    let hi = IVec3::new(a.x.max(b.x), top, a.z.max(b.z)) + 7;
                                    let probe = IVec3::new(
                                        (lo.x + hi.x) / 2,
                                        sea as i32 - 8,
                                        (lo.z + hi.z) / 2,
                                    );
                                    if past_water.is_none() && voxels.0.voxel(probe) == ids::WATER {
                                        past_water = Some(step + 4);
                                    }
                                    if past_water.is_some_and(|last| step > last) {
                                        break;
                                    }
                                    state.edit(
                                        &mut voxels.0,
                                        streaming.0.as_mut(),
                                        lo,
                                        hi,
                                        |_, old| {
                                            if old == ids::WATER { old } else { ids::AIR }
                                        },
                                    );
                                }
                            },
                        );
                    },
                );
                // Fly up and back to look down the trench.
                for (k, frames) in [(Key::ToggleFly, 1), (Key::Jump, 40), (Key::Back, 30)] {
                    game.input().key_down(k);
                    tick(game, frames);
                    game.input().key_up(k);
                }
                look(game, 0.0, 110.0);
            }
        }
        Demo::Wildfire => {
            use mc2_game::blocks::BlockLayer;
            use mc2_voxel::coords::BlockPos;
            let feet = {
                let mut q = game.world.query::<&mc2_game::player::Body>();
                q.iter(&game.world).next().map(|b| b.feet)
            };
            if let Some(feet) = feet {
                // The tree nearest a point 18 m ahead, trunk with leaves
                // above it: in view from where the player stands.
                let yaw = f64::from(game.view().yaw);
                let ahead = feet + glam::DVec3::new(yaw.sin(), 0.0, yaw.cos()) * 18.0;
                let f = ahead.floor().as_ivec3();
                let mut best: Option<(i32, BlockPos)> = None;
                {
                    let layer = game.world.resource::<BlockLayer>();
                    let v = &game.world.resource::<mc2_game::Voxels>().0;
                    for dz in -30..=30 {
                        for dx in -30..=30 {
                            for dy in -3..=8 {
                                let b = BlockPos(f + glam::IVec3::new(dx, dy, dz));
                                let d = dx * dx + dz * dz;
                                if best.is_some_and(|(bd, _)| bd <= d) {
                                    continue;
                                }
                                let leafy = |b: BlockPos| {
                                    matches!(
                                        layer.block_at(v, b),
                                        Some(mc2_game::blocks::BlockKind::Solid(m))
                                            if m.get().kind == mc2_voxel::material::Kind::Foliage
                                    )
                                };
                                if let Some(mc2_game::blocks::BlockKind::Solid(m)) =
                                    layer.block_at(v, b)
                                    && mc2_game::items::is_log(m)
                                    && (2..10).any(|k| leafy(BlockPos(b.0 + glam::IVec3::Y * k)))
                                {
                                    best = Some((d, b));
                                }
                            }
                        }
                    }
                }
                if let Some((_, trunk)) = best {
                    // Look at it.
                    let to = trunk.0.as_dvec3() + glam::DVec3::new(0.5, 3.0, 0.5)
                        - (feet + glam::DVec3::Y * mc2_game::player::EYE);
                    let mut players = game.world.query::<&mut mc2_game::player::Player>();
                    if let Some(mut p) = players.iter_mut(&mut game.world).next() {
                        p.yaw = to.x.atan2(to.z) as f32;
                        p.pitch = (to.y / to.length()).asin() as f32;
                    }
                    let mut fire = game.world.resource_mut::<mc2_game::fire::Fire>();
                    fire.ignite.push((trunk, true));
                    fire.ignite.push((BlockPos(trunk.0 + glam::IVec3::Y), true));
                    eprintln!("wildfire: trunk at {:?}", trunk.0);
                }
                tick(game, 60 * 7);
                let fire = game.world.resource::<mc2_game::fire::Fire>();
                eprintln!(
                    "wildfire: {} burning, {} caught, {} burnt out",
                    fire.burning(),
                    fire.caught,
                    fire.burnt_out
                );
            }
        }
        Demo::Lightning => {
            let view = game.view();
            let ahead = glam::DVec3::new(f64::from(view.yaw).sin(), 0.0, f64::from(view.yaw).cos());
            {
                let mut w = game.world.resource_mut::<mc2_game::weather::Weather>();
                w.set(mc2_game::weather::Sky::Storm);
                w.frozen = true;
                w.aim = Some(view.position + ahead * 40.0);
            }
            look(game, 0.0, -40.0);
            // The flash is fading, the bolt still there.
            tick(game, 6);
        }
        Demo::Craft => {
            workshop(game);
            hold(game, Item::Tool(ToolKind::Pickaxe, Tier::Stone));
        }
        Demo::Workshop => {
            workshop(game);
            // A torch on the ground to the left, then break the ground
            // between it and the table, stopping a third of the way.
            look(game, -70.0, 10.0);
            hold(game, Item::Block(BlockKind::Torch));
            click(game, Button::Secondary);
            look(game, 110.0, -30.0);
            hold(game, Item::Tool(ToolKind::Pickaxe, Tier::Stone));
            look(game, -20.0, 30.0);
            game.input().button_down(Button::Primary);
            tick(game, 9);
        }
    }
    mc2_core::profiler::end_frame();
    let edits = game.world.resource::<Interaction>().edits;
    let update = mc2_core::profiler::rows()
        .into_iter()
        .find(|r| r.name == "game.update")
        .map(|r| (r.stats.mean(), r.stats.max()))
        .unwrap_or_default();
    eprintln!(
        "demo {demo:?}: {edits} edits, game.update mean {:.3} ms, worst frame {:.2} ms",
        update.0, update.1
    );
    let mut slow: Vec<(String, f32)> = mc2_core::profiler::rows()
        .into_iter()
        .filter(|r| r.stats.max() > 5.0)
        .map(|r| (r.name.to_string(), r.stats.max()))
        .collect();
    slow.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (name, max) in slow {
        eprintln!("  slowest {name}: {max:.2} ms");
    }
}
