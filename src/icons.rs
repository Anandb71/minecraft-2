//! Item icons, painted once at start-up by ray casting each item's voxel
//! model into a cell of an atlas the HUD draws from.
//!
//! Blocks sit in three-quarter view, lit from above and the front, with a
//! texture per material: planks show their boards, bricks their mortar,
//! logs their rings, ore its seams, glass its frame and glare. Tools, sticks
//! and flint lie face-on, tilted a little so their thickness shows; lumps,
//! ingots and powder sit in three-quarter view like blocks. Every icon gets
//! a dark outline and a soft shadow so it reads on any background.

use glam::{IVec3, Vec2, Vec3, Vec4};
use mc2_core::FxHashMap;
use mc2_game::blocks::BlockKind;
use mc2_game::items::{Item, Tier, ToolKind};
use mc2_render::hud::IconAtlas;
use mc2_voxel::material::{self, Kind, MaterialId, ids};
use std::sync::Arc;

/// Icon size in the atlas, pixels.
pub const CELL: u32 = 64;
/// Rays per pixel along each axis.
const SS: u32 = 3;
const ATLAS: u32 = 1024;

pub struct Icons {
    atlas: Arc<IconAtlas>,
    index: FxHashMap<Item, u32>,
}

/// The icons, painted on first use.
pub fn icons() -> &'static Icons {
    static ICONS: std::sync::OnceLock<Icons> = std::sync::OnceLock::new();
    ICONS.get_or_init(Icons::build)
}

impl Icons {
    /// Paints every item's icon, spread over the machine's cores.
    pub fn build() -> Self {
        let items = all_items();
        let per_row = ATLAS / CELL;
        assert!(items.len() as u32 <= per_row * per_row);
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let chunk = items.len().div_ceil(threads);
        let painted: Vec<Vec<u8>> = std::thread::scope(|s| {
            let handles: Vec<_> = items
                .chunks(chunk)
                .map(|part| s.spawn(move || part.iter().map(|i| paint(*i)).collect::<Vec<_>>()))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("icon thread"))
                .collect()
        });
        let mut pixels = vec![0u8; (ATLAS * ATLAS * 4) as usize];
        let mut index = FxHashMap::default();
        for (i, (item, icon)) in items.iter().zip(painted).enumerate() {
            let (cx, cy) = (i as u32 % per_row, i as u32 / per_row);
            for y in 0..CELL {
                let src = (y * CELL * 4) as usize;
                let dst = (((cy * CELL + y) * ATLAS + cx * CELL) * 4) as usize;
                pixels[dst..dst + (CELL * 4) as usize]
                    .copy_from_slice(&icon[src..src + (CELL * 4) as usize]);
            }
            index.insert(*item, i as u32);
        }
        Self {
            atlas: Arc::new(IconAtlas {
                size: ATLAS,
                cell: CELL,
                pixels,
            }),
            index,
        }
    }

    pub fn get(&self, item: Item) -> Option<u32> {
        self.index.get(&item).copied()
    }

    pub fn atlas(&self) -> Arc<IconAtlas> {
        self.atlas.clone()
    }
}

/// Materials a player can hold as a block.
pub fn holdable(m: MaterialId) -> bool {
    let mat = m.get();
    match mat.kind {
        Kind::Air | Kind::Liquid => false,
        Kind::Foliage => matches!(
            m,
            ids::LEAVES | ids::PINE_NEEDLES | ids::BIRCH_LEAVES | ids::MOSS
        ),
        _ => {
            mat.hardness.is_finite()
                && !matches!(
                    m,
                    ids::GLOWING_ROCK | ids::EMBER | ids::ASH | ids::GLOWSTONE
                )
        }
    }
}

/// Slab materials (as the recipes make them).
pub const SLABS: [MaterialId; 7] = [
    ids::STONE_BRICK,
    ids::COBBLESTONE,
    ids::PLANKS,
    ids::DARK_PLANKS,
    ids::RED_BRICK,
    ids::CONCRETE,
    ids::MARBLE,
];

/// Every item with an icon, blocks first.
pub fn all_items() -> Vec<Item> {
    let mut v: Vec<Item> = (1..material::count() as u16)
        .map(MaterialId)
        .filter(|m| holdable(*m))
        .map(Item::solid)
        .collect();
    v.extend(SLABS.map(|m| Item::Block(BlockKind::Slab(m))));
    v.extend(
        [
            BlockKind::Torch,
            BlockKind::Lantern,
            BlockKind::Window,
            BlockKind::CraftingTable,
            BlockKind::Furnace,
        ]
        .map(Item::Block),
    );
    v.extend([
        Item::Stick,
        Item::Coal,
        Item::Charcoal,
        Item::Flint,
        Item::Gunpowder,
        Item::RawIron,
        Item::RawCopper,
        Item::RawGold,
        Item::IronIngot,
        Item::CopperIngot,
        Item::GoldIngot,
        Item::SteelIngot,
        Item::Bucket,
        Item::WaterBucket,
    ]);
    for tier in Tier::ALL {
        for kind in [ToolKind::Pickaxe, ToolKind::Axe, ToolKind::Shovel] {
            v.push(Item::Tool(kind, tier));
        }
    }
    v
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Look {
    Matte,
    Metal,
    Glow,
    Glass,
}

#[derive(Clone, Copy, Debug)]
enum Cell {
    /// A material, textured by where the voxel is.
    Mat(MaterialId),
    /// A plain colour (linear) with a look.
    Tint(Vec3, Look),
}

impl Cell {
    fn look(self) -> Look {
        match self {
            Cell::Tint(_, look) => look,
            Cell::Mat(m) => {
                let mat = m.get();
                if mat.kind == Kind::Transparent {
                    Look::Glass
                } else if m.is_emissive() {
                    Look::Glow
                } else if mat.metallic > 0.5 {
                    Look::Metal
                } else {
                    Look::Matte
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum View {
    /// Three-quarter view from above, front (-z) on the left.
    Iso,
    /// Face-on along -z, tilted a touch to show depth.
    Flat,
}

struct Model {
    n: i32,
    view: View,
    /// Frame the whole grid (full blocks all the same size) rather than
    /// just what is filled.
    whole: bool,
    cells: Vec<Option<Cell>>,
}

impl Model {
    fn new(n: i32, view: View, whole: bool, f: impl Fn(IVec3) -> Option<Cell>) -> Self {
        let mut cells = Vec::with_capacity((n * n * n) as usize);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    cells.push(f(IVec3::new(x, y, z)));
                }
            }
        }
        Self {
            n,
            view,
            whole,
            cells,
        }
    }

    fn at(&self, v: IVec3) -> Option<Cell> {
        let n = self.n;
        if v.cmplt(IVec3::ZERO).any() || v.cmpge(IVec3::splat(n)).any() {
            return None;
        }
        self.cells[(v.x + n * (v.y + n * v.z)) as usize]
    }

    fn opaque(&self, v: IVec3) -> bool {
        self.at(v).is_some_and(|c| c.look() != Look::Glass)
    }

    /// Filled bounds, inclusive-exclusive, in voxels.
    fn bounds(&self) -> (Vec3, Vec3) {
        if self.whole {
            return (Vec3::ZERO, Vec3::splat(self.n as f32));
        }
        let (mut lo, mut hi) = (IVec3::splat(self.n), IVec3::ZERO);
        for (i, c) in self.cells.iter().enumerate() {
            if c.is_some() {
                let i = i as i32;
                let v = IVec3::new(i % self.n, (i / self.n) % self.n, i / (self.n * self.n));
                lo = lo.min(v);
                hi = hi.max(v + 1);
            }
        }
        (lo.as_vec3(), hi.as_vec3())
    }
}

/// View direction, screen right and screen up.
fn basis(view: View) -> (Vec3, Vec3, Vec3) {
    let d = match view {
        View::Iso => Vec3::new(1.0, -0.82, 1.0).normalize(),
        View::Flat => Vec3::new(0.16, -0.22, -1.0).normalize(),
    };
    let r = d.cross(Vec3::Y).normalize();
    let u = r.cross(d);
    (d, r, u)
}

/// Light direction (toward the light) for each view.
fn light(view: View) -> Vec3 {
    match view {
        View::Iso => Vec3::new(-0.35, 1.0, -0.6).normalize(),
        View::Flat => Vec3::new(-0.4, 0.9, 0.6).normalize(),
    }
}

fn hash(v: IVec3, salt: u32) -> f32 {
    let mut h = (v.x as u32).wrapping_mul(0x8da6_b343)
        ^ (v.y as u32).wrapping_mul(0xd816_3841)
        ^ (v.z as u32).wrapping_mul(0xcb1a_b31f)
        ^ salt.wrapping_mul(0x9e37_79b9);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

fn rgb(m: MaterialId) -> Vec3 {
    Vec3::from(m.get().albedo)
}

/// Face-plane coordinates of voxel `v` seen through a face with normal
/// `nrm`: (across, up) on walls, (x, z) on tops and bottoms.
fn plane(v: IVec3, nrm: IVec3) -> (i32, i32) {
    if nrm.y != 0 {
        (v.x, v.z)
    } else if nrm.x != 0 {
        (v.z, v.y)
    } else {
        (v.x, v.y)
    }
}

/// Running bond: bricks `w` by `h` voxels, alternate rows offset half a
/// brick; returns (in mortar, brick id).
fn bond(u: i32, w: i32, h: i32, wv: i32) -> (bool, IVec3) {
    let row = wv.div_euclid(h);
    let shifted = u + (row & 1) * (w / 2);
    let col = shifted.div_euclid(w);
    let mortar = wv.rem_euclid(h) == 0 || shifted.rem_euclid(w) == 0;
    (mortar, IVec3::new(col, row, 7))
}

/// "TNT" in a 12 x 4 bitmap, top row first.
const TNT_ROWS: [&str; 4] = [
    "###.#..#.###",
    ".#..##.#..#.",
    ".#..#.##..#.",
    ".#..#..#..#.",
];

/// Colour (linear) of material `m` at voxel `v` of a 16-voxel block,
/// seen through a face with normal `nrm`.
fn texture(m: MaterialId, v: IVec3, nrm: IVec3) -> Vec3 {
    let base = rgb(m);
    let (u, w) = plane(v, nrm);
    let n1 = hash(v, 1) - 0.5;
    let grain = |amp: f32| base * (1.0 + n1 * amp);
    match m {
        ids::GRASS => {
            let blade = hash(v, 2);
            base * (0.85 + 0.35 * blade)
        }
        ids::DIRT | ids::FARMLAND | ids::WET_DIRT => {
            let clump = hash(v / 2, 3);
            base * (0.8 + 0.3 * clump + n1 * 0.2)
        }
        ids::OAK_LOG | ids::PINE_LOG | ids::BIRCH_LOG => {
            if nrm.y != 0 {
                // End grain: bark ring, then growth rings round a pale core.
                let d = Vec2::new(u as f32 - 7.5, w as f32 - 7.5).length();
                if d > 6.6 {
                    return base * 0.75;
                }
                let wood = rgb(ids::PLANKS) * if m == ids::PINE_LOG { 0.8 } else { 1.0 };
                wood * (0.82 + 0.18 * (d * 1.9).sin()) * (1.0 + n1 * 0.08)
            } else if m == ids::BIRCH_LOG {
                let dash = hash(IVec3::new(u / 3, w / 2, 0), 4) > 0.8;
                if dash {
                    Vec3::new(0.05, 0.05, 0.05)
                } else {
                    base * (0.92 + n1 * 0.12)
                }
            } else {
                // Bark: vertical ridges.
                let ridge = hash(IVec3::new(u, w / 5, 0), 5);
                base * (0.65 + 0.55 * ridge + n1 * 0.1)
            }
        }
        ids::PLANKS | ids::DARK_PLANKS => {
            // Boards four voxels wide, running across the face.
            let board = w.div_euclid(4);
            let gap = w.rem_euclid(4) == 0;
            let tone = 0.85 + 0.3 * hash(IVec3::new(board, 0, nrm.y), 6);
            let streak = 1.0 + 0.1 * (hash(IVec3::new(u / 3, board, 0), 7) - 0.5);
            base * if gap { 0.62 } else { tone * streak }
        }
        ids::STONE_BRICK | ids::RED_BRICK => {
            let (bw, bh) = if m == ids::STONE_BRICK {
                (8, 4)
            } else {
                (6, 3)
            };
            let (mortar, id) = bond(u, bw, bh, w);
            if mortar {
                if m == ids::RED_BRICK {
                    Vec3::new(0.55, 0.53, 0.5)
                } else {
                    base * 0.55
                }
            } else {
                base * (0.85 + 0.3 * hash(id, 8) + n1 * 0.08)
            }
        }
        ids::COBBLESTONE => {
            // Rounded stones in darker mortar.
            let cell = IVec3::new(u.div_euclid(5), w.div_euclid(4), 0);
            let off = IVec3::new(u.rem_euclid(5), w.rem_euclid(4), 0);
            let edge = off.x == 0 || off.y == 0;
            if edge && hash(IVec3::new(u, w, 3), 9) > 0.25 {
                base * 0.5
            } else {
                base * (0.75 + 0.5 * hash(cell, 10) + n1 * 0.1)
            }
        }
        ids::COAL_ORE | ids::IRON_ORE | ids::COPPER_ORE | ids::GOLD_ORE => {
            let host = match m {
                ids::COAL_ORE => rgb(ids::SHALE) * 1.4,
                _ => rgb(ids::SANDSTONE) * 0.8,
            };
            let seam = hash(v / 2, 11) > 0.72 && hash(v, 12) > 0.25;
            if seam {
                match m {
                    ids::COAL_ORE => Vec3::splat(0.02),
                    ids::IRON_ORE => Vec3::new(0.62, 0.36, 0.22),
                    ids::COPPER_ORE => Vec3::new(0.25, 0.55, 0.40),
                    _ => Vec3::new(0.95, 0.72, 0.2),
                }
            } else {
                host * (0.85 + n1 * 0.2)
            }
        }
        ids::TNT => {
            if nrm.y != 0 {
                // Top: paper over the charges, the fuse in the middle.
                let d = Vec2::new(u as f32 - 7.5, w as f32 - 7.5).length();
                return if d < 1.5 {
                    Vec3::splat(0.03)
                } else if d < 3.0 {
                    Vec3::new(0.5, 0.45, 0.4)
                } else {
                    base * (0.8 + n1 * 0.1)
                };
            }
            // Wall: a paper band with the letters on it.
            let across = if nrm.z < 0 || nrm.x > 0 { 15 - u } else { u };
            if (5..=10).contains(&w) {
                let row = 9 - w;
                let col = across - 2;
                let ink = (0..4).contains(&row)
                    && (0..12).contains(&col)
                    && TNT_ROWS[row as usize].as_bytes()[col as usize] == b'#';
                if ink {
                    Vec3::splat(0.02)
                } else {
                    Vec3::new(0.85, 0.83, 0.78)
                }
            } else {
                // Paper-wrapped sticks.
                let stick = across.rem_euclid(4) == 0;
                base * if stick { 0.7 } else { 1.0 + n1 * 0.1 }
            }
        }
        ids::IRON | ids::STEEL => {
            // Plates with a rivet in each corner.
            let (pu, pw) = (u.rem_euclid(8), w.rem_euclid(8));
            let seam = pu == 0 || pw == 0;
            let rivet = (pu == 1 || pu == 6) && (pw == 1 || pw == 6);
            base * if seam {
                0.6
            } else if rivet {
                1.3
            } else {
                1.0 + n1 * 0.05
            }
        }
        ids::ROOF_TILE => {
            let (mortar, id) = bond(u, 4, 3, w);
            base * if mortar && w.rem_euclid(3) == 0 {
                0.55
            } else {
                0.85 + 0.3 * hash(id, 13)
            }
        }
        ids::THATCH => base * (0.7 + 0.5 * hash(IVec3::new(u, w / 4, 0), 14)),
        ids::CACTUS => {
            let rib = u.rem_euclid(4) == 0;
            let spine = hash(v, 15) > 0.93;
            if spine {
                Vec3::new(0.85, 0.8, 0.6)
            } else {
                base * if rib { 1.3 } else { 0.9 + n1 * 0.1 }
            }
        }
        ids::SAND | ids::SNOW | ids::CHALK | ids::CONCRETE | ids::WHITEWASH | ids::CLAY => {
            grain(0.12)
        }
        ids::GRAVEL => {
            let pebble = hash(v / 2, 16);
            base * (0.6 + 0.8 * pebble)
        }
        ids::ASPHALT => base * (0.8 + 0.6 * hash(v, 17).powi(4)),
        ids::MARBLE => {
            let vein = ((u as f32 * 0.7 + w as f32 * 0.45 + 3.0 * hash(v / 4, 18)).sin()).abs();
            base * if vein < 0.12 { 0.72 } else { 1.0 + n1 * 0.05 }
        }
        ids::GRANITE | ids::RHYOLITE => {
            let fleck = hash(v, 19);
            if fleck > 0.9 {
                Vec3::splat(0.08)
            } else if fleck < 0.1 {
                base * 1.5
            } else {
                grain(0.2)
            }
        }
        ids::SANDSTONE | ids::LIMESTONE | ids::SHALE | ids::SLATE | ids::GNEISS => {
            // Bedding: bands of slightly different stone.
            let band = hash(IVec3::new(0, v.y / 3, 0), 20);
            base * (0.85 + 0.3 * band + n1 * 0.08)
        }
        ids::OBSIDIAN => {
            if hash(v, 21) > 0.9 {
                Vec3::new(0.3, 0.18, 0.45)
            } else {
                grain(0.3)
            }
        }
        ids::WOOL => grain(0.18),
        ids::LEAVES | ids::PINE_NEEDLES | ids::BIRCH_LEAVES | ids::MOSS => grain(0.6),
        _ => grain(0.16),
    }
}

/// The voxel model of a block item.
fn block_model(kind: BlockKind) -> Model {
    let whole = !matches!(kind, BlockKind::Torch | BlockKind::Lantern);
    Model::new(16, View::Iso, whole, |v| match kind {
        // Grass: a turf top over dirt, its edge ragged.
        BlockKind::Solid(ids::GRASS) => {
            let turf = v.y >= 13 - i32::from(hash(IVec3::new(v.x, 0, v.z), 22) > 0.5);
            Some(Cell::Mat(if turf { ids::GRASS } else { ids::DIRT }))
        }
        // Foliage blocks are leafy all through, with gaps.
        BlockKind::Solid(m) if m.get().kind == Kind::Foliage => {
            (hash(v, 23) > 0.18).then_some(Cell::Mat(m))
        }
        _ => {
            let m = kind.voxel(v);
            (!m.is_air()).then_some(Cell::Mat(m))
        }
    })
}

const HANDLE: Vec3 = Vec3::new(0.40, 0.26, 0.13);
const GRIP: Vec3 = Vec3::new(0.16, 0.08, 0.05);

/// Distance from `p` to the segment `a..b`, and how far along it (0..1).
fn segment(p: Vec2, a: Vec2, b: Vec2) -> (f32, f32) {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    ((p - (a + ab * t)).length(), t)
}

fn tier_paint(tier: Tier) -> (Vec3, Look) {
    match tier {
        Tier::Wood => (rgb(ids::PLANKS), Look::Matte),
        Tier::Stone => (Vec3::new(0.36, 0.35, 0.34), Look::Matte),
        Tier::Iron => (Vec3::new(0.62, 0.63, 0.64), Look::Metal),
        Tier::Steel => (Vec3::new(0.46, 0.53, 0.64), Look::Metal),
    }
}

/// A tool face-on in a 32-voxel grid: its handle from bottom left to top
/// right, its head at the top.
fn tool_model(kind: ToolKind, tier: Tier) -> Model {
    let (head_rgb, head_look) = tier_paint(tier);
    let diag = Vec2::new(1.0, 1.0).normalize();
    let across = Vec2::new(-diag.y, diag.x);
    let (a, b) = match kind {
        ToolKind::Shovel => (Vec2::new(4.0, 4.0), Vec2::new(17.0, 17.0)),
        _ => (Vec2::new(5.0, 5.0), Vec2::new(21.0, 21.0)),
    };
    Model::new(32, View::Flat, false, move |v| {
        let p = Vec2::new(v.x as f32 + 0.5, v.y as f32 + 0.5);
        let z = v.z as f32 + 0.5;
        let head = |thick: f32| (z - 16.0).abs() < thick;
        let q = p - b;
        let (s, t) = (q.dot(diag), q.dot(across));
        let speck = 1.0 + (hash(v, 30) - 0.5) * if tier == Tier::Stone { 0.5 } else { 0.1 };
        let head_cell = |bright: f32| Some(Cell::Tint(head_rgb * speck * bright, head_look));
        let in_head = match kind {
            ToolKind::Pickaxe => {
                let c = b - diag * 13.0;
                let d = p - c;
                let r = d.length();
                let ang = d.y.atan2(d.x).to_degrees() - 45.0;
                let spread = (ang / 62.0).clamp(-1.0, 1.0);
                let half = 1.2 + 1.9 * (spread * std::f32::consts::FRAC_PI_2).cos();
                ang.abs() <= 62.0 && (r - 14.5).abs() < half && head(3.0)
            }
            ToolKind::Axe => {
                // A bearded blade fanning out to a curved edge, a short
                // poll behind the eye.
                let blade = (-8.0..3.0).contains(&s)
                    && (0.5..11.5).contains(&t)
                    && (s + 2.2).abs() <= 2.0 + 0.62 * t - 0.02 * t * t;
                let poll = (-2.4..2.4).contains(&s) && (-3.5..0.5).contains(&t);
                (blade || poll) && head(if t > 8.5 { 1.4 } else { 2.4 })
            }
            ToolKind::Shovel => {
                let blade =
                    (0.5..12.5).contains(&s) && t.abs() <= 5.2 * ((12.5 - s) / 4.0).min(1.0);
                let socket = (-1.5..0.5).contains(&s) && t.abs() <= 2.0;
                (blade && head(1.5)) || (socket && head(2.2))
            }
        };
        if in_head {
            // The cutting edge is ground bright.
            let edge = match kind {
                ToolKind::Axe => t > 9.0,
                ToolKind::Shovel => s > 10.0,
                ToolKind::Pickaxe => false,
            };
            return head_cell(if edge { 1.35 } else { 1.0 });
        }
        let (d, along) = segment(p, a, b);
        if d < 1.7 && (z - 16.0).abs() < 1.7 {
            let wrap = along < 0.3 && (along * 40.0).fract() < 0.7;
            let wood = HANDLE * (0.85 + 0.3 * hash(IVec3::new((along * 20.0) as i32, 0, 0), 31));
            return Some(Cell::Tint(if wrap { GRIP } else { wood }, Look::Matte));
        }
        None
    })
}

fn stick_model() -> Model {
    Model::new(32, View::Flat, false, |v| {
        let p = Vec2::new(v.x as f32 + 0.5, v.y as f32 + 0.5);
        let (d, along) = segment(p, Vec2::new(7.0, 7.0), Vec2::new(25.0, 25.0));
        let z = v.z as f32 + 0.5;
        (d < 1.9 && (z - 16.0).abs() < 1.9).then(|| {
            let knot = hash(IVec3::new((along * 12.0) as i32, 0, 0), 32);
            Cell::Tint(HANDLE * (0.8 + 0.4 * knot), Look::Matte)
        })
    })
}

/// A rough lump in three-quarter view, coloured by `paint(voxel)`.
fn lump_model(stretch: Vec3, paint: impl Fn(IVec3) -> Cell) -> Model {
    Model::new(32, View::Iso, false, |v| {
        let d = (v.as_vec3() + 0.5 - Vec3::new(16.0, 12.0, 16.0)) / stretch;
        let dir = d.normalize_or_zero();
        let bumps =
            (dir.x * 4.1 + 1.3).sin() * (dir.y * 3.3 + 0.2).sin() * (dir.z * 3.7 + 2.1).sin();
        let r = 8.5 + 2.2 * bumps + 0.8 * (hash(v / 3, 33) - 0.5);
        (d.length() < r).then(|| paint(v))
    })
}

fn ingot_model(metal: Vec3) -> Model {
    Model::new(32, View::Iso, false, |v| {
        if !(9..16).contains(&v.y) {
            return None;
        }
        let shrink = (v.y - 9) as f32 * 0.6;
        let (x, z) = (v.x as f32 + 0.5, v.z as f32 + 0.5);
        let inside = x > 5.0 + shrink
            && x < 27.0 - shrink
            && z > 10.0 + shrink * 0.8
            && z < 22.0 - shrink * 0.8;
        inside.then(|| Cell::Tint(metal * (1.0 + (hash(v, 34) - 0.5) * 0.06), Look::Metal))
    })
}

fn powder_model() -> Model {
    Model::new(32, View::Iso, false, |v| {
        let r = Vec2::new(v.x as f32 + 0.5 - 16.0, v.z as f32 + 0.5 - 16.0).length();
        let height = 9.0 * (1.0 - r / 12.0).max(0.0).powf(1.2) + 4.0;
        let y = v.y as f32 + 0.5;
        if r > 12.0 || y < 4.0 || y > height + hash(v, 35) {
            return None;
        }
        let g = hash(v, 36);
        let c = if g > 0.9 {
            Vec3::splat(0.5)
        } else if g < 0.2 {
            Vec3::splat(0.03)
        } else {
            Vec3::splat(0.12)
        };
        Some(Cell::Tint(c, Look::Matte))
    })
}

fn flint_model() -> Model {
    let pts = [
        Vec2::new(9.0, 5.0),
        Vec2::new(24.0, 9.0),
        Vec2::new(23.0, 22.0),
        Vec2::new(13.0, 27.0),
        Vec2::new(6.0, 15.0),
    ];
    Model::new(32, View::Flat, false, move |v| {
        let p = Vec2::new(v.x as f32 + 0.5, v.y as f32 + 0.5);
        // Signed distance inside the convex outline (positive inside).
        let mut inside = f32::MAX;
        for i in 0..pts.len() {
            let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
            let e = (b - a).normalize();
            inside = inside.min(e.perp_dot(p - a));
        }
        let chip = hash(v / 2, 37) * 1.5;
        let z = v.z as f32 + 0.5;
        if inside < chip || (z - 16.0).abs() > 1.2 + (inside * 0.3).min(1.6) {
            return None;
        }
        let ripple = 0.8 + 0.25 * (inside * 1.3).sin();
        let c = if inside < 2.6 {
            Vec3::new(0.32, 0.31, 0.3)
        } else {
            Vec3::new(0.07, 0.07, 0.08) * ripple
        };
        Some(Cell::Tint(c, Look::Matte))
    })
}

/// An iron pail, tapering to its base, its handle arched over the rim;
/// full of water or empty.
fn bucket_model(full: bool) -> Model {
    let iron = Vec3::new(0.58, 0.59, 0.61);
    let water = Vec3::new(0.10, 0.30, 0.52);
    Model::new(32, View::Iso, false, move |v| {
        let p = v.as_vec3() + 0.5;
        let r = Vec2::new(p.x - 16.0, p.z - 16.0).length();
        let (bottom, rim) = (5.0, 21.0);
        // The handle: a thin arc across the top, in the x-y plane.
        let arc = Vec2::new(p.x - 16.0, p.y - rim).length();
        if p.y > rim && (arc - 9.2).abs() < 0.8 && (p.z - 16.0).abs() < 0.8 {
            return Some(Cell::Tint(iron * 0.8, Look::Metal));
        }
        if p.y < bottom || p.y > rim {
            return None;
        }
        let radius = 7.0 + 2.2 * (p.y - bottom) / (rim - bottom);
        if r > radius {
            return None;
        }
        let wall = r > radius - 1.4 || p.y < bottom + 1.0;
        // A rolled rim and two hoops round the pail.
        let hoop = (p.y - 9.0).abs() < 0.7 || (p.y - 16.0).abs() < 0.7 || p.y > rim - 1.0;
        if wall {
            let c = if hoop { iron * 0.75 } else { iron };
            return Some(Cell::Tint(
                c * (1.0 + (hash(v, 40) - 0.5) * 0.06),
                Look::Metal,
            ));
        }
        (full && p.y < rim - 1.5).then(|| {
            let glint = hash(v, 41) > 0.93;
            Cell::Tint(if glint { water * 2.2 } else { water }, Look::Matte)
        })
    })
}

fn item_model(item: Item) -> Model {
    let rough = |base: Vec3, spot: Vec3, look: Look| {
        move |v: IVec3| {
            let c = if hash(v / 3, 38) > 0.68 { spot } else { base };
            Cell::Tint(c * (1.0 + (hash(v, 39) - 0.5) * 0.25), look)
        }
    };
    match item {
        Item::Block(kind) => block_model(kind),
        Item::Stick => stick_model(),
        Item::Tool(kind, tier) => tool_model(kind, tier),
        Item::Coal => lump_model(
            Vec3::ONE,
            rough(Vec3::splat(0.025), Vec3::splat(0.14), Look::Metal),
        ),
        Item::Charcoal => lump_model(
            Vec3::new(1.4, 0.8, 0.8),
            rough(
                Vec3::new(0.05, 0.04, 0.035),
                Vec3::new(0.11, 0.08, 0.06),
                Look::Matte,
            ),
        ),
        Item::RawIron => lump_model(
            Vec3::ONE,
            rough(
                Vec3::new(0.55, 0.38, 0.27),
                Vec3::new(0.34, 0.21, 0.14),
                Look::Matte,
            ),
        ),
        Item::RawCopper => lump_model(
            Vec3::ONE,
            rough(
                Vec3::new(0.66, 0.33, 0.16),
                Vec3::new(0.24, 0.52, 0.36),
                Look::Matte,
            ),
        ),
        Item::RawGold => lump_model(
            Vec3::ONE,
            rough(
                Vec3::new(0.85, 0.6, 0.13),
                Vec3::new(0.55, 0.4, 0.25),
                Look::Metal,
            ),
        ),
        Item::IronIngot => ingot_model(Vec3::new(0.66, 0.67, 0.68)),
        Item::CopperIngot => ingot_model(Vec3::new(0.80, 0.42, 0.24)),
        Item::GoldIngot => ingot_model(Vec3::new(0.95, 0.72, 0.2)),
        Item::SteelIngot => ingot_model(Vec3::new(0.50, 0.57, 0.68)),
        Item::Flint => flint_model(),
        Item::Bucket => bucket_model(false),
        Item::WaterBucket => bucket_model(true),
        Item::Gunpowder => powder_model(),
    }
}

/// Colour (linear, premultiplied) and coverage along one ray.
fn trace(model: &Model, origin: Vec3, dir: Vec3) -> Vec4 {
    let n = model.n as f32;
    // Enter the grid's box.
    let inv = dir.recip();
    let t0 = (Vec3::ZERO - origin) * inv;
    let t1 = (Vec3::splat(n) - origin) * inv;
    let tmin = t0.min(t1);
    let tmax = t0.max(t1);
    let enter = tmin.max_element().max(0.0);
    let exit = tmax.min_element();
    if enter >= exit {
        return Vec4::ZERO;
    }
    let mut axis = if tmin.x >= tmin.y && tmin.x >= tmin.z {
        0
    } else if tmin.y >= tmin.z {
        1
    } else {
        2
    };
    let p = origin + dir * (enter + 1e-4);
    let mut v = p
        .floor()
        .as_ivec3()
        .clamp(IVec3::ZERO, IVec3::splat(model.n - 1));
    let step = dir.signum().as_ivec3();
    let delta = inv.abs();
    let next = (v.as_vec3() + (step.max(IVec3::ZERO)).as_vec3() - origin) * inv;
    let mut side = next;
    let (lit, view) = (light(model.view), model.view);
    let mut acc = Vec4::ZERO;
    let mut in_glass = false;
    for _ in 0..(model.n * 3 + 3) {
        if let Some(cell) = model.at(v) {
            let mut nrm = IVec3::ZERO;
            nrm[axis] = -step[axis];
            let look = cell.look();
            if look == Look::Glass {
                if !in_glass {
                    let c = glass(model, cell, v, nrm, lit);
                    acc += c * (1.0 - acc.w);
                }
                in_glass = true;
            } else {
                let c = shade(model, cell, v, nrm, lit, view);
                acc += c.extend(1.0) * (1.0 - acc.w);
                return acc;
            }
        } else {
            in_glass = false;
        }
        if acc.w > 0.995 {
            return acc;
        }
        // Step to the next voxel.
        axis = if side.x < side.y && side.x < side.z {
            0
        } else if side.y < side.z {
            1
        } else {
            2
        };
        v[axis] += step[axis];
        side[axis] += delta[axis];
        if v[axis] < 0 || v[axis] >= model.n {
            break;
        }
    }
    acc
}

/// Ambient occlusion: how open the space in front of a face is.
fn occlusion(model: &Model, v: IVec3, nrm: IVec3) -> f32 {
    let f = v + nrm;
    let (a, b) = match (nrm.x != 0, nrm.y != 0) {
        (true, _) => (IVec3::Y, IVec3::Z),
        (_, true) => (IVec3::X, IVec3::Z),
        _ => (IVec3::X, IVec3::Y),
    };
    let mut shut = 0.0;
    for (o, w) in [
        (a, 0.08),
        (-a, 0.08),
        (b, 0.08),
        (-b, 0.08),
        (a + b, 0.04),
        (a - b, 0.04),
        (b - a, 0.04),
        (-a - b, 0.04),
    ] {
        if model.opaque(f + o) {
            shut += w;
        }
    }
    1.0 - shut
}

fn shade(model: &Model, cell: Cell, v: IVec3, nrm: IVec3, lit: Vec3, view: View) -> Vec3 {
    let (base, look) = match cell {
        Cell::Mat(m) => (texture(m, v, nrm), cell.look()),
        Cell::Tint(c, look) => (c, look),
    };
    if look == Look::Glow {
        let e = match cell {
            Cell::Mat(m) => Vec3::from(m.get().emission),
            Cell::Tint(c, _) => c,
        };
        return base.lerp(e / e.max_element().max(1e-6), 0.7);
    }
    let n = nrm.as_vec3();
    let diffuse = 0.42 + 0.58 * n.dot(lit).max(0.0);
    let mut c = base * diffuse * occlusion(model, v, nrm) * 1.15;
    if look == Look::Metal {
        // A soft sheen toward the light, stronger on tops.
        let sheen = n.dot(lit).max(0.0).powi(3) * 0.35 + if nrm.y > 0 { 0.12 } else { 0.0 };
        let fade = (v.y as f32 + 0.5) / model.n as f32;
        c = c * (0.8 + 0.4 * fade) + Vec3::splat(sheen);
    }
    // Flat items: a lighter rim where the front meets the bevel.
    if view == View::Flat && nrm.z > 0 && !model.opaque(v + IVec3::Y) {
        c *= 1.12;
    }
    c
}

/// A glass face: faint tint, a bright frame at the block's edges, and a
/// diagonal glare.
fn glass(model: &Model, cell: Cell, v: IVec3, nrm: IVec3, lit: Vec3) -> Vec4 {
    let tint = match cell {
        Cell::Mat(ids::ICE) => Vec3::new(0.55, 0.75, 0.95),
        Cell::Mat(m) => rgb(m) * Vec3::new(0.78, 0.9, 0.94),
        Cell::Tint(c, _) => c,
    };
    let edge_axes = (0..3)
        .filter(|&i| nrm[i] == 0)
        .filter(|&i| v[i] == 0 || v[i] == model.n - 1)
        .count();
    let (u, w) = plane(v, nrm);
    let glare = ((u + w).rem_euclid(11) < 2) as u8 as f32;
    let light = 0.6 + 0.4 * nrm.as_vec3().dot(lit).max(0.0);
    let (c, a) = if edge_axes > 0 {
        (Vec3::new(0.9, 0.96, 1.0) * light, 0.85)
    } else {
        (tint * light + Vec3::splat(0.5 * glare), 0.22 + 0.25 * glare)
    };
    (c * a).extend(a)
}

fn to_srgb(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let s = if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// One item's icon: CELL x CELL sRGB RGBA, straight alpha.
pub fn paint(item: Item) -> Vec<u8> {
    let model = item_model(item);
    let (d, r, u) = basis(model.view);
    let (lo, hi) = model.bounds();
    // Frame the filled bounds: centre them, fit the larger extent.
    let mut span = Vec2::splat(f32::MAX);
    let mut top = Vec2::splat(f32::MIN);
    for i in 0..8 {
        let c = Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        );
        let s = Vec2::new(c.dot(r), c.dot(u));
        span = span.min(s);
        top = top.max(s);
    }
    let mid = (span + top) * 0.5;
    let half = ((top - span) * 0.5).max_element() * 1.16;
    let centre = (lo + hi) * 0.5;
    let centre_on_screen = Vec2::new(centre.dot(r), centre.dot(u));
    let back = centre + r * (mid.x - centre_on_screen.x) + u * (mid.y - centre_on_screen.y)
        - d * (model.n as f32 * 2.0);

    let size = CELL * SS;
    let mut coverage = vec![Vec4::ZERO; (CELL * CELL) as usize];
    for py in 0..size {
        for px in 0..size {
            let a = (px as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let b = 1.0 - (py as f32 + 0.5) / size as f32 * 2.0;
            let origin = back + r * (a * half) + u * (b * half);
            let c = trace(&model, origin, d);
            coverage[((py / SS) * CELL + px / SS) as usize] += c / (SS * SS) as f32;
        }
    }
    finish(&coverage)
}

/// Outline and drop shadow under the painted coverage; to sRGB bytes.
fn finish(icon: &[Vec4]) -> Vec<u8> {
    let n = CELL as i32;
    let alpha = |x: i32, y: i32| {
        if x < 0 || y < 0 || x >= n || y >= n {
            0.0
        } else {
            icon[(y * n + x) as usize].w
        }
    };
    let mut out = vec![0u8; (CELL * CELL * 4) as usize];
    for y in 0..n {
        for x in 0..n {
            // Shadow: the silhouette blurred, down and to the right.
            let mut shadow = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    shadow += alpha(x - 1 + dx, y - 2 + dy);
                }
            }
            let shadow = shadow / 9.0 * 0.35;
            // Outline: dark where the silhouette's edge is next door.
            let near = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .map(|(dx, dy)| alpha(x + dx, y + dy))
                .fold(0.0, f32::max);
            let outline = (near - alpha(x, y)).max(0.0) * 0.7;
            // Composite: shadow, then outline, then the icon over both.
            let under_a = outline + shadow * (1.0 - outline);
            let under_c = Vec3::splat(0.02) * under_a;
            let top = icon[(y * n + x) as usize];
            let a = top.w + under_a * (1.0 - top.w);
            let c = top.truncate() + under_c * (1.0 - top.w);
            let straight = if a > 0.0 { c / a } else { Vec3::ZERO };
            let o = ((y * n + x) * 4) as usize;
            out[o] = to_srgb(straight.x);
            out[o + 1] = to_srgb(straight.y);
            out[o + 2] = to_srgb(straight.z);
            out[o + 3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coverage(px: &[u8]) -> f32 {
        px.chunks(4).map(|p| f32::from(p[3]) / 255.0).sum::<f32>() / (CELL * CELL) as f32
    }

    #[test]
    fn every_item_paints_something_that_fits() {
        for item in all_items() {
            let px = paint(item);
            let c = coverage(&px);
            assert!(c > 0.08 && c < 0.8, "{} covers {c}", item.name());
            // Nothing touches the cell's border.
            let n = CELL as usize;
            for i in 0..n {
                for (x, y) in [(i, 0), (i, n - 1), (0, i), (n - 1, i)] {
                    assert!(px[(y * n + x) * 4 + 3] < 40, "{} at edge", item.name());
                }
            }
        }
    }

    /// `MC2_DUMP_ICONS=atlas.png cargo test -p minecraft2 icons` saves the
    /// atlas to look at.
    #[test]
    fn dump_atlas_when_asked() {
        let Ok(path) = std::env::var("MC2_DUMP_ICONS") else {
            return;
        };
        let icons = Icons::build();
        let a = icons.atlas();
        mc2_gpu::capture::save_png(std::path::Path::new(&path), a.size, a.size, &a.pixels)
            .expect("save atlas");
    }

    #[test]
    fn items_are_unique_and_fit_the_atlas() {
        let items = all_items();
        let mut seen = std::collections::HashSet::new();
        assert!(items.iter().all(|i| seen.insert(*i)));
        assert!(items.len() as u32 <= (ATLAS / CELL).pow(2));
        // Everything a recipe makes has an icon.
        for r in mc2_game::crafting::recipes() {
            assert!(seen.contains(&r.output), "{}", r.output.name());
        }
    }
}
