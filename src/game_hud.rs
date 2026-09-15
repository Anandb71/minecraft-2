//! In-game overlay: crosshair, hotbar, mode, target and gizmo previews.

use mc2_game::Game;
use mc2_game::blocks::BlockLayer;
use mc2_game::interact::{HOTBAR, Interaction, Mode, Preview};
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

    // Hotbar.
    let slot_w = 150.0;
    let total = slot_w * HOTBAR.len() as f32;
    let x0 = (w - total) * 0.5;
    let y0 = h - 44.0;
    hud.rect(x0 - 6.0, y0 - 6.0, total + 12.0, 40.0, rgba(0, 0, 0, 140));
    for (i, kind) in HOTBAR.iter().enumerate() {
        let x = x0 + i as f32 * slot_w;
        let selected = i == state.slot;
        if selected {
            hud.rect(x, y0 - 4.0, slot_w - 6.0, 36.0, rgba(255, 255, 255, 60));
        }
        let color = if selected {
            rgba(255, 230, 120, 255)
        } else {
            rgba(220, 220, 220, 255)
        };
        hud.text(
            x + 4.0,
            y0,
            1.0,
            color,
            &format!("{} {}", i + 1, kind.name()),
        );
    }
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
    let carried: u64 = state.inventory.volumes.values().sum();
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
            "{block}  voxel {}  {:.1} m  {}  edits {edits}  carried {:.2} blocks",
            t.hit.material.get().name,
            t.hit.t,
            player_line,
            carried as f64 / 4096.0
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
