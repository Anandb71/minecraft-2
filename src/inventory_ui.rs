//! The hotbar, and the inventory screen: the pack's slots to move stacks
//! between, and the recipe book to craft from.
//!
//! I opens the screen, as does using a crafting table or furnace; the
//! cursor is free while it is open. Left click picks a stack up or puts it
//! down (swapping with what is there), right click takes half or puts one,
//! shift click moves a stack between hotbar and pack. Recipes craftable now
//! list first; click one to make it, shift click to make up to eight.

use crate::icons::{all_items, icons};
use glam::Vec2;
use mc2_game::Game;
use mc2_game::blocks::{BlockKind, BlockLayer};
use mc2_game::crafting::{self, CraftError, Near, Recipe, Station};
use mc2_game::input::Time;
use mc2_game::interact::{Interaction, MESSAGE_S};
use mc2_game::inventory::{HOTBAR_SLOTS, Inventory, SLOTS, Stack};
use mc2_game::items::Item;
use mc2_render::hud::{HudCanvas, rgba};

/// Reach of a crafting table or furnace, metres: as far as the player can
/// use one, to its centre.
const STATION_REACH_M: f64 = mc2_game::interact::REACH_M + 1.0;
const SLOT: f32 = 54.0;
const PITCH: f32 = 58.0;
const ROW: f32 = 52.0;

const PANEL: u32 = 0xf81a_1612; // rgba(18, 22, 26, 248), little endian
const INK: u32 = 0xffff_ffff;

fn dim() -> u32 {
    rgba(150, 156, 168, 255)
}

fn gold() -> u32 {
    rgba(255, 222, 140, 255)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    All,
    At(Station),
    Catalogue,
}

impl Tab {
    fn name(self) -> &'static str {
        match self {
            Tab::All => "all",
            Tab::At(Station::Hand) => "hand",
            Tab::At(Station::Table) => "table",
            Tab::At(Station::Furnace) => "furnace",
            Tab::At(Station::Trade(_)) => "trade",
            Tab::Catalogue => "catalogue",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Hit {
    Slot(usize),
    Recipe(usize),
    Tab(Tab),
    Catalogue(Item),
    Panel,
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl Rect {
    fn contains(&self, p: Vec2) -> bool {
        p.x >= self.x && p.y >= self.y && p.x < self.x + self.w && p.y < self.y + self.h
    }
}

pub struct Screen {
    pub open: bool,
    cursor: Vec2,
    /// A stack picked up, following the cursor.
    carried: Option<Stack>,
    tab: Tab,
    scroll: f32,
    /// Where things were drawn last frame, for clicks; later wins.
    hits: Vec<(Rect, Hit)>,
    /// What the cursor is over and a name to show for it.
    tooltip: Option<String>,
}

impl Default for Screen {
    fn default() -> Self {
        Self {
            open: false,
            cursor: Vec2::ZERO,
            carried: None,
            tab: Tab::All,
            scroll: 0.0,
            hits: Vec::new(),
            tooltip: None,
        }
    }
}

/// Stations within reach of the player.
fn near(game: &Game) -> Near {
    let eye = game.view().position;
    let mut n = Near::default();
    for (pos, kind) in game.world.resource::<BlockLayer>().placed() {
        let centre = pos.0.as_dvec3() + 0.5;
        if centre.distance(eye) > STATION_REACH_M {
            continue;
        }
        match kind {
            BlockKind::CraftingTable => n.table = true,
            BlockKind::Furnace => n.furnace = true,
            _ => {}
        }
    }
    n
}

impl Screen {
    /// Opens at a station's recipes (None: all of them).
    pub fn open_at(&mut self, station: Option<Station>) {
        self.open = true;
        self.tab = station.map_or(Tab::All, Tab::At);
        self.scroll = 0.0;
    }

    /// Closes; whatever the cursor carries goes back into the pack.
    pub fn close(&mut self, game: &mut Game) {
        self.open = false;
        if let Some(s) = self.carried.take() {
            let mut state = game.world.resource_mut::<Interaction>();
            if !state.inventory.creative {
                state.inventory.add(s.item, s.count);
            }
        }
    }

    pub fn mouse_move(&mut self, x: f32, y: f32) {
        self.cursor = Vec2::new(x, y);
    }

    pub fn center(&mut self, screen: (u32, u32)) {
        self.cursor = Vec2::new(screen.0 as f32 * 0.5, screen.1 as f32 * 0.5);
    }

    /// Left stick, player-space (x right, y forward): moves the cursor.
    pub fn stick_move(&mut self, axis: Vec2, dt: f32, screen: (u32, u32)) {
        let speed = 1100.0;
        self.cursor.x = (self.cursor.x + axis.x * speed * dt).clamp(0.0, screen.0 as f32);
        self.cursor.y = (self.cursor.y - axis.y * speed * dt).clamp(0.0, screen.1 as f32);
    }

    pub fn wheel(&mut self, lines: f32) {
        self.scroll = (self.scroll - lines * ROW).max(0.0);
    }

    fn hit(&self) -> Option<Hit> {
        self.hits
            .iter()
            .rev()
            .find(|(r, _)| r.contains(self.cursor))
            .map(|(_, h)| *h)
    }

    /// A click at the cursor: `secondary` for the right button.
    pub fn click(&mut self, game: &mut Game, secondary: bool, shift: bool) {
        let now = game.world.resource::<Time>().elapsed;
        let near = near(game);
        let hit = self.hit();
        let mut state = game.world.resource_mut::<Interaction>();
        let state = &mut *state;
        match hit {
            Some(Hit::Slot(i)) => {
                let inv = &mut state.inventory;
                if shift && !secondary {
                    quick_move(inv, i);
                } else {
                    self.carried = slot_click(inv, i, self.carried.take(), secondary);
                }
            }
            Some(Hit::Recipe(r)) => {
                let recipe = &crafting::recipes()[r];
                let times = if shift { 8 } else { 1 };
                let mut made = 0;
                let mut err = None;
                for _ in 0..times {
                    match crafting::craft(&mut state.inventory, recipe, near) {
                        Ok(()) => made += 1,
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
                if made > 0 {
                    let name = recipe.output.name();
                    state.say(now, format!("made {} {name}", made * recipe.count));
                } else if let Some(e) = err {
                    state.say(now, why(e));
                }
            }
            Some(Hit::Tab(t)) => {
                self.tab = t;
                self.scroll = 0.0;
            }
            Some(Hit::Catalogue(item)) => {
                self.carried = Some(Stack::new(item, item.max_stack()));
            }
            Some(Hit::Panel) => {}
            None => {
                // Outside the screen: what is carried goes back.
                if let Some(s) = self.carried.take()
                    && !state.inventory.creative
                {
                    state.inventory.add(s.item, s.count);
                }
            }
        }
    }

    /// A number key while the screen is open: swap the slot under the
    /// cursor with that hotbar slot.
    pub fn number(&mut self, game: &mut Game, n: usize) {
        if let Some(Hit::Slot(i)) = self.hit()
            && n < HOTBAR_SLOTS
        {
            let mut state = game.world.resource_mut::<Interaction>();
            state.inventory.slots.swap(i, n);
        }
    }

    /// Draws the screen over everything else.
    pub fn draw(&mut self, game: &mut Game, hud: &mut HudCanvas, screen: (u32, u32), pad: bool) {
        self.hits.clear();
        self.tooltip = None;
        if !self.open {
            return;
        }
        let near = near(game);
        let now = game.world.resource::<Time>().elapsed;
        let state = game.world.resource::<Interaction>();
        let inv = &state.inventory;
        let (w, h) = (screen.0 as f32, screen.1 as f32);

        // Dim the world, then the panel.
        hud.rect(0.0, 0.0, w, h, rgba(0, 0, 0, 110));
        let pw = 1180.0f32.min(w - 32.0);
        let ph = 640.0f32.min(h - 32.0);
        let (px, py) = ((w - pw) * 0.5, (h - ph) * 0.5);
        panel(hud, px, py, pw, ph);
        self.hits.push((
            Rect {
                x: px,
                y: py,
                w: pw,
                h: ph,
            },
            Hit::Panel,
        ));

        // The pack: three rows over the hotbar.
        let lx = px + 24.0;
        hud.text(lx, py + 22.0, 2.0, gold(), "INVENTORY");
        let grid_y = py + 64.0;
        for i in HOTBAR_SLOTS..SLOTS {
            let k = i - HOTBAR_SLOTS;
            let (x, y) = (lx + (k % 9) as f32 * PITCH, grid_y + (k / 9) as f32 * PITCH);
            self.slot(hud, inv, i, x, y, false);
        }
        let bar_y = grid_y + 3.0 * PITCH + 14.0;
        for i in 0..HOTBAR_SLOTS {
            let x = lx + i as f32 * PITCH;
            self.slot(hud, inv, i, x, bar_y, i == state.slot);
        }

        // Stations and the furnace's fire.
        let mut y = bar_y + PITCH + 18.0;
        let status = |ok: bool| if ok { rgba(140, 230, 140, 255) } else { dim() };
        let line = |hud: &mut HudCanvas, y: f32, c: u32, s: &str| hud.text(lx, y, 2.0, c, s);
        if inv.creative {
            line(hud, y, gold(), "creative: nothing runs out");
            y += 20.0;
        }
        line(
            hud,
            y,
            status(near.table),
            if near.table {
                "crafting table: in reach"
            } else {
                "crafting table: none in reach"
            },
        );
        y += 20.0;
        let fire = if near.furnace {
            format!("furnace: in reach, heat {}", inv.heat)
        } else {
            "furnace: none in reach".to_owned()
        };
        line(hud, y, status(near.furnace), &fire);
        y += 30.0;
        for hint in [
            "left: pick up / put down   right: half / one",
            "shift: move to or from the hotbar   1-9: swap",
            "E on TNT lights it   I or Esc closes   pad: LS cursor  A pick  X half  B close",
        ] {
            hud.text(lx, y, 1.0, dim(), hint);
            y += 14.0;
        }
        // News, newest at the bottom.
        let mut ny = py + ph - 26.0;
        for (t, msg) in state.messages.iter().rev() {
            if now - t < MESSAGE_S {
                hud.text(lx, ny, 2.0, gold(), msg);
                ny -= 22.0;
            }
        }

        // The recipe book (or the catalogue).
        let rx = lx + 9.0 * PITCH + 28.0;
        let rw = px + pw - 24.0 - rx;
        let mut tx = rx;
        let mut tabs = vec![
            Tab::All,
            Tab::At(Station::Hand),
            Tab::At(Station::Table),
            Tab::At(Station::Furnace),
        ];
        if let Some(p) = near.trader {
            tabs.push(Tab::At(Station::Trade(p)));
        }
        if inv.creative {
            tabs.push(Tab::Catalogue);
        }
        for t in tabs {
            let label = t.name();
            let tw = HudCanvas::text_width(label, 2.0) + 20.0;
            let r = Rect {
                x: tx,
                y: py + 16.0,
                w: tw,
                h: 30.0,
            };
            let on = self.tab == t;
            let bg = if on {
                rgba(255, 222, 140, 60)
            } else if r.contains(self.cursor) {
                rgba(255, 255, 255, 30)
            } else {
                rgba(255, 255, 255, 12)
            };
            hud.rect(r.x, r.y, r.w, r.h, bg);
            if on {
                hud.rect(r.x, r.y + r.h - 2.0, r.w, 2.0, gold());
            }
            hud.text(
                tx + 10.0,
                py + 23.0,
                2.0,
                if on { gold() } else { INK },
                label,
            );
            self.hits.push((r, Hit::Tab(t)));
            tx += tw + 6.0;
        }
        let list = Rect {
            x: rx,
            y: py + 58.0,
            w: rw,
            h: ph - 58.0 - 16.0,
        };
        if self.tab == Tab::Catalogue {
            self.catalogue(hud, list);
        } else {
            self.recipes(hud, inv, near, list);
        }

        // Pad users have no OS cursor; draw one where clicks land.
        if pad {
            hud.rect(self.cursor.x - 7.0, self.cursor.y - 1.0, 14.0, 2.0, gold());
            hud.rect(self.cursor.x - 1.0, self.cursor.y - 7.0, 2.0, 14.0, gold());
        }

        // What the cursor carries, and a name for what it is over.
        if let Some(s) = self.carried {
            draw_stack(
                hud,
                s,
                self.cursor.x - 24.0,
                self.cursor.y - 24.0,
                48.0,
                false,
            );
        } else if let Some(tip) = &self.tooltip {
            let tw = HudCanvas::text_width(tip, 2.0) + 16.0;
            let (x, y) = (
                (self.cursor.x + 16.0).min(w - tw - 4.0),
                self.cursor.y + 18.0,
            );
            hud.rect(x, y, tw, 26.0, rgba(8, 8, 12, 235));
            hud.rect(x, y, tw, 1.0, rgba(255, 222, 140, 120));
            hud.text(x + 8.0, y + 7.0, 2.0, INK, tip);
        }
    }

    fn slot(
        &mut self,
        hud: &mut HudCanvas,
        inv: &Inventory,
        i: usize,
        x: f32,
        y: f32,
        selected: bool,
    ) {
        let r = Rect {
            x,
            y,
            w: SLOT,
            h: SLOT,
        };
        let over = r.contains(self.cursor);
        slot_frame(hud, x, y, SLOT, selected, over);
        if let Some(s) = inv.slots[i] {
            draw_stack(hud, s, x + 5.0, y + 5.0, SLOT - 10.0, inv.creative);
            if over {
                self.tooltip = Some(stack_name(&s));
            }
        }
        self.hits.push((r, Hit::Slot(i)));
    }

    fn recipes(&mut self, hud: &mut HudCanvas, inv: &Inventory, near: Near, list: Rect) {
        let all = crafting::recipes();
        let mut shown: Vec<(usize, &Recipe, bool)> = all
            .iter()
            .enumerate()
            .filter(|(_, r)| match self.tab {
                Tab::At(s) => r.station == s,
                // Trades belong to their traders, not the book.
                _ => !matches!(r.station, Station::Trade(_)) || near.has(r.station),
            })
            .map(|(i, r)| (i, r, crafting::check(inv, r, near).is_ok()))
            .collect();
        // Craftable first, then the book's order.
        shown.sort_by_key(|(i, _, ok)| (!ok, *i));
        let max_scroll = (shown.len() as f32 * ROW - list.h).max(0.0);
        self.scroll = self.scroll.min(max_scroll);
        let first = (self.scroll / ROW) as usize;
        let visible = (list.h / ROW) as usize;
        for (k, &(index, recipe, ok)) in shown.iter().skip(first).take(visible).enumerate() {
            let y = list.y + k as f32 * ROW;
            let r = Rect {
                x: list.x,
                y,
                w: list.w,
                h: ROW - 4.0,
            };
            let over = r.contains(self.cursor);
            let bg = match (ok, over) {
                (true, true) => rgba(255, 222, 140, 50),
                (true, false) => rgba(255, 255, 255, 16),
                (false, true) => rgba(255, 255, 255, 14),
                (false, false) => rgba(255, 255, 255, 5),
            };
            hud.rect(r.x, r.y, r.w, r.h, bg);
            let ink = if ok { INK } else { dim() };
            let out = Stack::new(recipe.output, recipe.count);
            draw_stack(hud, out, r.x + 4.0, y + 2.0, 44.0, false);
            let name = recipe.output.name();
            hud.text(r.x + 56.0, y + 8.0, 2.0, ink, &name);
            let station_ok = inv.creative || near.has(recipe.station);
            let where_ = match recipe.station {
                Station::Hand => "by hand".to_owned(),
                Station::Table => "at a crafting table".to_owned(),
                Station::Furnace => "in a furnace, with fuel".to_owned(),
                Station::Trade(p) => format!("traded with a {}", p.name()),
            };
            let where_colour = if station_ok {
                rgba(140, 200, 140, 255)
            } else {
                rgba(230, 150, 90, 255)
            };
            hud.text(r.x + 56.0, y + 28.0, 1.0, where_colour, &where_);
            // Ingredients from the right: icon and have/need.
            let mut ix = r.x + r.w - 8.0;
            for &(ing, need) in recipe.inputs.iter().rev() {
                let have = inv.count(|i| ing.matches(i));
                let label = if inv.creative {
                    format!("{need}")
                } else {
                    format!("{}/{need}", have.min(999))
                };
                let lw = HudCanvas::text_width(&label, 1.0);
                ix -= lw.max(36.0) + 10.0;
                let icon = Rect {
                    x: ix,
                    y: y + 2.0,
                    w: 36.0,
                    h: 36.0,
                };
                if let Some(idx) = icons().get(ing.example()) {
                    hud.icon(icon.x, icon.y, 36.0, idx, INK);
                }
                let enough = inv.creative || have >= need;
                let c = if enough {
                    INK
                } else {
                    rgba(255, 110, 100, 255)
                };
                hud.text(ix + (36.0 - lw) * 0.5, y + 39.0, 1.0, c, &label);
                if icon.contains(self.cursor) {
                    self.tooltip = Some(ing.name());
                }
            }
            if over && self.tooltip.is_none() {
                self.tooltip = Some(if ok {
                    format!("click: make {}   shift: make more", recipe.count)
                } else {
                    why(crafting::check(inv, recipe, near).unwrap_err())
                });
            }
            self.hits.push((r, Hit::Recipe(index)));
        }
        if shown.len() > visible {
            let bar_h = list.h * visible as f32 / shown.len() as f32;
            let bar_y = list.y + (list.h - bar_h) * self.scroll / max_scroll.max(1.0);
            hud.rect(
                list.x + list.w + 6.0,
                bar_y,
                4.0,
                bar_h,
                rgba(255, 255, 255, 60),
            );
        }
    }

    fn catalogue(&mut self, hud: &mut HudCanvas, list: Rect) {
        let items = all_items();
        let cols = ((list.w / PITCH) as usize).max(1);
        let rows = items.len().div_ceil(cols);
        let max_scroll = (rows as f32 * PITCH - list.h).max(0.0);
        self.scroll = self.scroll.min(max_scroll);
        let first_row = (self.scroll / PITCH) as usize;
        let visible_rows = (list.h / PITCH) as usize;
        for (k, item) in items.iter().enumerate().skip(first_row * cols) {
            let (row, col) = (k / cols - first_row, k % cols);
            if row >= visible_rows {
                break;
            }
            let (x, y) = (list.x + col as f32 * PITCH, list.y + row as f32 * PITCH);
            let r = Rect {
                x,
                y,
                w: SLOT,
                h: SLOT,
            };
            let over = r.contains(self.cursor);
            slot_frame(hud, x, y, SLOT, false, over);
            if let Some(idx) = icons().get(*item) {
                hud.icon(x + 5.0, y + 5.0, SLOT - 10.0, idx, INK);
            }
            if over {
                self.tooltip = Some(item.name());
            }
            self.hits.push((r, Hit::Catalogue(*item)));
        }
    }
}

/// Why a recipe cannot be made, for people.
fn why(e: CraftError) -> String {
    match e {
        CraftError::Station(s) => format!("needs a {} in reach", s.name()),
        CraftError::Missing(i) => format!("not enough {}", i.name()),
        CraftError::NoFuel => "the furnace needs fuel: coal, charcoal or wood".into(),
        CraftError::NoRoom => "no room in the pack".into(),
    }
}

fn stack_name(s: &Stack) -> String {
    match s.item.uses() {
        Some(uses) => format!("{} ({}/{uses})", s.item.name(), s.life),
        None => s.item.name(),
    }
}

/// Left or right click on slot `i` holding `carried`; returns what the
/// cursor holds afterwards.
fn slot_click(inv: &mut Inventory, i: usize, carried: Option<Stack>, one: bool) -> Option<Stack> {
    let here = inv.slots[i];
    match (carried, here) {
        (None, None) => None,
        (None, Some(s)) => {
            // Take all, or half (rounded up).
            let take = if one { s.count.div_ceil(2) } else { s.count };
            let left = s.count - take;
            inv.slots[i] = (left > 0).then_some(Stack { count: left, ..s });
            Some(Stack { count: take, ..s })
        }
        (Some(c), None) => {
            let put = if one { 1 } else { c.count };
            inv.slots[i] = Some(Stack { count: put, ..c });
            (c.count > put).then_some(Stack {
                count: c.count - put,
                ..c
            })
        }
        (Some(c), Some(s)) if c.item == s.item && c.item.tool().is_none() => {
            let room = s.item.max_stack() - s.count;
            let put = if one { 1.min(room) } else { c.count.min(room) };
            inv.slots[i] = Some(Stack {
                count: s.count + put,
                ..s
            });
            (c.count > put).then_some(Stack {
                count: c.count - put,
                ..c
            })
        }
        (Some(c), Some(s)) => {
            inv.slots[i] = Some(c);
            Some(s)
        }
    }
}

/// Moves slot `i`'s stack between hotbar and pack.
fn quick_move(inv: &mut Inventory, i: usize) {
    let Some(s) = inv.slots[i].take() else {
        return;
    };
    let targets: Vec<usize> = if i < HOTBAR_SLOTS {
        (HOTBAR_SLOTS..SLOTS).collect()
    } else {
        (0..HOTBAR_SLOTS).collect()
    };
    let mut left = s.count;
    // Top up matching stacks, then take an empty slot.
    for &t in &targets {
        if let Some(o) = &mut inv.slots[t]
            && o.item == s.item
            && o.item.tool().is_none()
        {
            let put = (o.item.max_stack() - o.count).min(left);
            o.count += put;
            left -= put;
        }
    }
    if left > 0
        && let Some(&t) = targets.iter().find(|&&t| inv.slots[t].is_none())
    {
        inv.slots[t] = Some(Stack { count: left, ..s });
        left = 0;
    }
    if left > 0 {
        inv.slots[i] = Some(Stack { count: left, ..s });
    }
}

fn panel(hud: &mut HudCanvas, x: f32, y: f32, w: f32, h: f32) {
    hud.rect(x + 4.0, y + 6.0, w, h, rgba(0, 0, 0, 90));
    hud.rect(x, y, w, h, PANEL);
    hud.rect(x, y, w, 2.0, rgba(255, 222, 140, 150));
    hud.rect(x, y + h - 1.0, w, 1.0, rgba(255, 255, 255, 30));
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
    if let Some(uses) = s.item.uses()
        && s.life < uses
    {
        let f = s.life as f32 / uses as f32;
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

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;

    fn inv_with(slots: &[(usize, Item, u32)]) -> Inventory {
        let mut inv = Inventory::default();
        for &(i, item, n) in slots {
            inv.slots[i] = Some(Stack::new(item, n));
        }
        inv
    }

    #[test]
    fn clicks_pick_up_put_down_split_and_swap() {
        let dirt = Item::solid(ids::DIRT);
        let mut inv = inv_with(&[(0, dirt, 10), (1, Item::Coal, 3)]);
        // Right click takes half.
        let c = slot_click(&mut inv, 0, None, true);
        assert_eq!(c.unwrap().count, 5);
        assert_eq!(inv.slots[0].unwrap().count, 5);
        // Right click with it puts one down in an empty slot.
        let c = slot_click(&mut inv, 5, c, true);
        assert_eq!(inv.slots[5].unwrap().count, 1);
        assert_eq!(c.unwrap().count, 4);
        // Left click merges into the same item.
        let c = slot_click(&mut inv, 0, c, false);
        assert!(c.is_none());
        assert_eq!(inv.slots[0].unwrap().count, 9);
        // Left click on a different item swaps.
        let picked = slot_click(&mut inv, 0, None, false);
        let c = slot_click(&mut inv, 1, picked, false);
        assert_eq!(inv.slots[1].unwrap().item, dirt);
        assert_eq!(c.unwrap().item, Item::Coal);
    }

    #[test]
    fn shift_click_moves_between_hotbar_and_pack() {
        let sand = Item::solid(ids::SAND);
        let mut inv = inv_with(&[(2, sand, 20), (12, sand, 60)]);
        quick_move(&mut inv, 2);
        // Tops up the pack's stack, the rest to the first empty pack slot.
        assert_eq!(inv.slots[12].unwrap().count, 64);
        assert_eq!(inv.slots[9].unwrap().count, 16);
        assert!(inv.slots[2].is_none());
        quick_move(&mut inv, 9);
        assert_eq!(inv.slots[0].unwrap().count, 16);
    }
}
