//! Scripted play for headless captures: the same game systems a player
//! drives, fed synthetic input, so screenshots and regression captures
//! exercise real interaction code.

use mc2_game::Game;
use mc2_game::input::{Button, Key};
use mc2_game::interact::Interaction;

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
}

impl Demo {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "build" => Some(Self::Build),
            "lights" => Some(Self::Lights),
            "mirror" => Some(Self::Mirror),
            "blast" => Some(Self::Blast),
            "collapse" => Some(Self::Collapse),
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
