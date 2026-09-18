//! The hotbar: item icons with their counts and wear, the held item's
//! name, break progress under the crosshair and recent news.

use crate::icons::icons;
use mc2_game::Game;
use mc2_game::input::Time;
use mc2_game::interact::{Interaction, MESSAGE_S};
use mc2_game::inventory::{HOTBAR_SLOTS, Stack};
use mc2_render::hud::{HudCanvas, rgba};

const INK: u32 = 0xffff_ffff;

fn stack_name(s: &Stack) -> String {
    match s.item.tool() {
        Some((_, tier)) => format!("{} ({}/{})", s.item.name(), s.life, tier.durability()),
        None => s.item.name(),
    }
}

fn slot_frame(hud: &mut HudCanvas, x: f32, y: f32, size: f32, selected: bool, over: bool) {
    let bg = if over {
        rgba(255, 255, 255, 48)
    } else {
        rgba(255, 255, 255, 16)
    };
    hud.rect(x, y, size, size, bg);
    let edge = if selected {
        rgba(255, 255, 255, 235)
    } else {
        rgba(255, 255, 255, 34)
    };
    if selected {
        let t = 3.0;
        hud.rect(x, y, size, t, edge);
        hud.rect(x, y + size - t, size, t, edge);
        hud.rect(x, y, t, size, edge);
        hud.rect(x + size - t, y, t, size, edge);
    } else {
        // Sunk into the panel: shade above and left, light below and right.
        let (dark, light) = (rgba(0, 0, 0, 110), rgba(255, 255, 255, 38));
        hud.rect(x, y, size, 2.0, dark);
        hud.rect(x, y, 2.0, size, dark);
        hud.rect(x, y + size - 1.0, size, 1.0, light);
        hud.rect(x + size - 1.0, y, 1.0, size, light);
    }
}

/// An item's icon with its count and, for tools, how worn they are.
fn draw_stack(hud: &mut HudCanvas, s: Stack, x: f32, y: f32, size: f32, creative: bool) {
    match icons().get(s.item) {
        Some(idx) => hud.icon(x, y, size, idx, INK),
        None => {
            hud.text(x, y + size * 0.4, 1.0, INK, &s.item.name());
        }
    }
    if s.count > 1 && !creative {
        let label = s.count.to_string();
        let lw = HudCanvas::text_width(&label, 2.0);
        let (tx, ty) = (x + size - lw + 2.0, y + size - 14.0);
        hud.text(tx + 2.0, ty + 2.0, 2.0, rgba(0, 0, 0, 200), &label);
        hud.text(tx, ty, 2.0, INK, &label);
    }
    if let Some((_, tier)) = s.item.tool()
        && s.life < tier.durability()
    {
        let f = s.life as f32 / tier.durability() as f32;
        let (bx, by, bw) = (x + 4.0, y + size - 3.0, size - 8.0);
        hud.rect(bx, by, bw, 3.0, rgba(0, 0, 0, 200));
        let c = rgba((255.0 * (1.0 - f)) as u8, (220.0 * f + 35.0) as u8, 40, 255);
        hud.rect(bx, by, bw * f, 3.0, c);
    }
}

/// The hotbar along the bottom edge, what is held named above it, break
/// progress under the crosshair and recent news. Returns the hotbar's left
/// edge and a height above it for the HUD's own lines (mode 14 px below it,
/// the target 22 px above).
pub fn draw_hotbar(game: &Game, hud: &mut HudCanvas, screen: (u32, u32)) -> (f32, f32) {
    let state = game.world.resource::<Interaction>();
    let now = game.world.resource::<Time>().elapsed;
    let (w, h) = (screen.0 as f32, screen.1 as f32);
    let size = 56.0;
    let gap = 4.0;
    let total = HOTBAR_SLOTS as f32 * (size + gap) - gap;
    let x0 = (w - total) * 0.5;
    let y0 = h - size - 14.0;
    hud.rect(
        x0 - 6.0,
        y0 - 6.0,
        total + 12.0,
        size + 12.0,
        rgba(0, 0, 0, 120),
    );
    for i in 0..HOTBAR_SLOTS {
        let x = x0 + i as f32 * (size + gap);
        let selected = i == state.slot;
        slot_frame(hud, x, y0, size, selected, false);
        if let Some(s) = state.inventory.slots[i] {
            draw_stack(
                hud,
                s,
                x + 6.0,
                y0 + 6.0,
                size - 12.0,
                state.inventory.creative,
            );
        }
        hud.text(
            x + 4.0,
            y0 + 4.0,
            1.0,
            rgba(255, 255, 255, 90),
            &format!("{}", i + 1),
        );
    }
    // The held item's name.
    if let Some(s) = state.inventory.slots[state.slot] {
        let name = stack_name(&s);
        let nw = HudCanvas::text_width(&name, 2.0);
        hud.text(
            (w - nw) * 0.5,
            y0 - 28.0,
            2.0,
            rgba(255, 255, 255, 220),
            &name,
        );
    }
    // Break progress under the crosshair.
    if let Some((_, p)) = state.breaking {
        let bw = 44.0;
        let (bx, by) = (w * 0.5 - bw * 0.5, h * 0.5 + 16.0);
        hud.rect(bx - 1.0, by - 1.0, bw + 2.0, 6.0, rgba(0, 0, 0, 170));
        hud.rect(
            bx,
            by,
            bw * p.clamp(0.0, 1.0),
            4.0,
            rgba(255, 255, 255, 230),
        );
    }
    // News, fading, stacked up above the lines the HUD writes over the
    // hotbar.
    let mut ny = y0 - 120.0;
    for (t, msg) in state.messages.iter().rev() {
        let age = now - t;
        if age >= MESSAGE_S {
            continue;
        }
        let fade = (1.0 - (age / MESSAGE_S).powi(4)).clamp(0.0, 1.0);
        let a = (255.0 * fade) as u8;
        let mw = HudCanvas::text_width(msg, 2.0);
        hud.text(
            (w - mw) * 0.5 + 2.0,
            ny + 2.0,
            2.0,
            rgba(0, 0, 0, a / 2),
            msg,
        );
        hud.text((w - mw) * 0.5, ny, 2.0, rgba(255, 230, 160, a), msg);
        ny -= 22.0;
    }
    (x0, y0 - 70.0)
}
