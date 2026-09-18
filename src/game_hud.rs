//! In-game overlay: crosshair, hotbar, mode, target and gizmo previews.

use mc2_game::Game;
use mc2_game::blocks::BlockLayer;
use mc2_game::interact::{Interaction, Mode, Preview};
use mc2_game::player::{MoveMode, Player};
use mc2_game::{Voxels, view::ViewCamera};
use mc2_render::gizmo::{AMBER, Gizmos, RED, WHITE};
use mc2_render::hud::{HudCanvas, rgba};

pub fn draw(game: &mut Game, hud: &mut HudCanvas, gizmos: &mut Gizmos, screen: (u32, u32)) {
    let view: ViewCamera = game.view();
    gizmos.begin(view.position);
    let (w, h) = (screen.0 as f32, screen.1 as f32);

    // Crosshair.
    let c = rgba(255, 255, 255, 200);
    hud.rect(w * 0.5 - 8.0, h * 0.5 - 1.0, 16.0, 2.0, c);
    hud.rect(w * 0.5 - 1.0, h * 0.5 - 8.0, 2.0, 16.0, c);

    let state = game.world.resource::<Interaction>();
    match state.preview {
        Some(Preview::Box { min, max, valid }) => {
            gizmos.aabb(min, max, if valid { WHITE } else { RED });
        }
        Some(Preview::Sphere {
            centre,
            radius,
            deposit,
        }) => gizmos.sphere(centre, radius, if deposit { AMBER } else { WHITE }),
        None => {}
    }

    // Hotbar, held item, break progress and news.
    let (x0, y0) = crate::inventory_ui::draw_hotbar(game, hud, screen);
    let mode = match state.mode {
        Mode::Block => "BLOCK  (Tab: carve)".to_owned(),
        Mode::Carve => format!(
            "CARVE r={:.0} voxels  (wheel: size, Tab: block)",
            state.radius_voxels
        ),
    };
    hud.text(x0, y0 + 14.0, 1.0, rgba(200, 220, 255, 255), &mode);

    let target = state.target;
    let edits = state.edits;
    let carried: u32 = state
        .inventory
        .slots
        .iter()
        .flatten()
        .map(|s| s.count)
        .sum();
    let player_line = game
        .world
        .query::<&Player>()
        .iter(&game.world)
        .next()
        .map(|p| {
            format!(
                "{}{}",
                if p.mode == MoveMode::Fly {
                    "flying (F)"
                } else {
                    "walking (F: fly)"
                },
                if p.crouching { ", crouched" } else { "" }
            )
        })
        .unwrap_or_default();
    if let Some(t) = target {
        let voxels = &game.world.resource::<Voxels>().0;
        let block = game
            .world
            .resource::<BlockLayer>()
            .block_at(voxels, t.block)
            .map(|b| b.name())
            .unwrap_or_else(|| "air".into());
        let line = format!(
            "{block}  voxel {}  {:.1} m  {}  edits {edits}  carried {carried}",
            t.hit.material.get().name,
            t.hit.t,
            player_line,
        );
        let tw = HudCanvas::text_width(&line, 1.0);
        hud.text(
            (w - tw) * 0.5,
            y0 - 22.0,
            1.0,
            rgba(255, 255, 255, 230),
            &line,
        );
    }
}
