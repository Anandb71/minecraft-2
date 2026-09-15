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
}

impl Demo {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "build" => Some(Self::Build),
            _ => None,
        }
    }
}

fn tick(game: &mut Game, frames: u32) {
    for _ in 0..frames {
        game.update(1.0 / 60.0);
        mc2_core::profiler::end_frame();
    }
}

fn click(game: &mut Game, button: Button) {
    game.input().button_down(button);
    tick(game, 1);
    game.input().button_up(button);
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
}
