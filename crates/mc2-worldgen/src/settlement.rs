//! Settlements: villages on open ground, built at voxel resolution.
//!
//! A village is a plaza with a covered well, two to four streets running
//! out along the axes, and houses facing them: timber-framed cottages in
//! whitewash, stone houses, brick townhouses of two or three storeys and
//! thatched cottages, each on a foundation stepped into the slope, with
//! glazed windows in plank frames, a doorway, a pitched roof of tiles or
//! thatch overhanging the walls, a chimney, and lanterns by the door and
//! inside. Lamp posts line the streets; wheat grows in fields behind.
//!
//! Everything is an ordered list of operations (boxes, gable prisms, roof
//! slabs, cylinders, paths that follow the ground); the last operation
//! containing a voxel sets it, so later ones carve rooms, windows and
//! doorways out of earlier walls. A village is a pure function of its grid
//! cell, cached once built, and stamped into whichever chunks it touches:
//! voxel by voxel only in brick cells an operation's edge crosses, whole
//! uniform cells elsewhere, one sample per cell or node farther out.

use crate::amplify::Surface;
use crate::flora::Biome;
use crate::noise::{hash_unit, hash3};
use glam::{IVec3, Vec2, Vec3};
use mc2_core::FxHashMap;
use mc2_voxel::material::{MaterialId, ids};
use mc2_voxel::tree::{Cell, ChunkTree};
use std::sync::{Arc, Mutex};

const VOXEL_M: f32 = 1.0 / 16.0;
/// One village at most per cell of this grid.
pub const VILLAGE_CELL_M: f32 = 480.0;
/// Farthest anything of a village or town lies from its centre.
pub const VILLAGE_RADIUS_M: f32 = 150.0;
/// Share of settlement sites flat enough for one that are towns.
const TOWN_SHARE: f32 = 0.35;
/// Clearing above paths: the grass, flowers and bumps of ground they cut.
const CLEAR_M: f32 = 1.5;
/// Paths lie on the ground in straight runs this long at most.
const PATH_RUN_M: f32 = 4.0;

/// Where a street's markings go: its run starts `along0` metres down the
/// street, junctions are `pitch` apart from `junction0`, each `width` wide.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Marks {
    pub along0: f32,
    pub pitch: f32,
    pub junction0: f32,
    pub width: f32,
}

impl Marks {
    /// Paint at `along` metres down the street, `lateral` metres off its
    /// centre line (half width `half`): a dashed centre line between
    /// junctions, zebra stripes just outside them.
    fn paint(&self, along: f32, lateral: f32, half: f32) -> bool {
        let from_junction = (along - self.junction0).rem_euclid(self.pitch);
        let to_next = self.pitch - from_junction;
        let gap = from_junction.min(to_next) - self.width * 0.5;
        if gap < 0.0 {
            return false;
        }
        if (0.6..3.4).contains(&gap) {
            return lateral.abs() < half - 0.6 && (lateral + 20.0).rem_euclid(1.0) < 0.45;
        }
        gap > 4.5 && lateral.abs() < 0.07 && along.rem_euclid(6.0) < 3.0
    }
}

/// Window patterns for a wall layer (see `Shape::Facade`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FacadeStyle {
    /// Floor-to-ceiling glass between steel mullions, a steel spandrel at
    /// every floor.
    Curtain,
    /// Windows punched in the wall, a sill under each.
    Punched,
    /// Continuous bands of glass along each floor.
    Ribbon,
    /// Shop windows between pillars, the ground floor only.
    Shopfront,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Shape {
    /// Axis-aligned box, `lo` inclusive, `hi` exclusive.
    Box { lo: Vec3, hi: Vec3 },
    /// Everything under a gable roof surface over the footprint `lo..hi`:
    /// eaves at `lo.y`, the ridge along x (or z) at `hi.y`.
    Prism { lo: Vec3, hi: Vec3, ridge_x: bool },
    /// The roof surface of such a gable, `thick` metres deep.
    Slab {
        lo: Vec3,
        hi: Vec3,
        ridge_x: bool,
        thick: f32,
    },
    /// Upright cylinder.
    Cylinder { c: Vec3, r: f32, h: f32 },
    /// A straight run `half` metres either side of `a..b` (square ends, `a`
    /// in and `b` out, so runs abut), over a surface rising evenly from `g0`
    /// at `a` to `g1` at `b` and level across: everything from `from` to
    /// `to` metres above that surface. Road markings, if any, are painted
    /// on its top voxel.
    Ramp {
        a: Vec2,
        b: Vec2,
        half: f32,
        g0: f32,
        g1: f32,
        from: f32,
        to: f32,
        marks: Option<Marks>,
    },
    /// A wall layer facing along `face` (0: x, 2: z), its outside toward
    /// `out` (+1 or -1), windowed in a style; floors `storey` metres apart
    /// from `base`. Where the pattern leaves no wall it is glass (one voxel,
    /// set back) or air.
    Facade {
        lo: Vec3,
        hi: Vec3,
        face: u32,
        out: f32,
        style: FacadeStyle,
        base: f32,
        storey: f32,
    },
    /// Floor slabs `thick` deep every `storey` metres from `lo.y`, with
    /// light panels under the slabs of the floors `lit` picks.
    Floors {
        lo: Vec3,
        hi: Vec3,
        storey: f32,
        thick: f32,
        lit: u32,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Op {
    shape: Shape,
    m: MaterialId,
    lo: Vec3,
    hi: Vec3,
}

impl Op {
    pub(crate) fn new(shape: Shape, m: MaterialId) -> Self {
        let (lo, hi) = match shape {
            Shape::Box { lo, hi }
            | Shape::Prism { lo, hi, .. }
            | Shape::Facade { lo, hi, .. }
            | Shape::Floors { lo, hi, .. } => (lo, hi),
            Shape::Slab { lo, hi, thick, .. } => (lo - Vec3::Y * thick, hi + Vec3::Y * 0.01),
            Shape::Cylinder { c, r, h } => (c - Vec3::new(r, 0.0, r), c + Vec3::new(r, h, r)),
            Shape::Ramp {
                a,
                b,
                half,
                g0,
                g1,
                from,
                to,
                ..
            } => (
                Vec3::new(
                    a.x.min(b.x) - half,
                    g0.min(g1) + from - 0.1,
                    a.y.min(b.y) - half,
                ),
                Vec3::new(
                    a.x.max(b.x) + half,
                    g0.max(g1) + to + 0.1,
                    a.y.max(b.y) + half,
                ),
            ),
        };
        Self { shape, m, lo, hi }
    }
}

/// Roof surface height over `p` for a gable over `lo..hi`.
fn gable_height(p: Vec3, lo: Vec3, hi: Vec3, ridge_x: bool) -> f32 {
    let (a, b, q) = if ridge_x {
        (lo.z, hi.z, p.z)
    } else {
        (lo.x, hi.x, p.x)
    };
    let half = (b - a) * 0.5;
    let from_ridge = (q - (a + half)).abs();
    lo.y + (hi.y - lo.y) * (1.0 - from_ridge / half)
}

fn in_footprint(p: Vec3, lo: Vec3, hi: Vec3) -> bool {
    p.x >= lo.x && p.x < hi.x && p.z >= lo.z && p.z < hi.z
}

impl Op {
    /// Material this operation gives point `p`. `coarse` is the cell size
    /// when sampling
    /// once per coarse cell (0 at voxel resolution): thin things thicken to
    /// show at all, a pane of glass fills its opening.
    fn at(&self, p: Vec3, coarse: f32) -> Option<MaterialId> {
        match self.shape {
            Shape::Box { lo, hi } => (p.cmpge(lo).all() && p.cmplt(hi).all()).then_some(self.m),
            Shape::Prism { lo, hi, ridge_x } => {
                (in_footprint(p, lo, hi) && p.y >= lo.y && p.y < gable_height(p, lo, hi, ridge_x))
                    .then_some(self.m)
            }
            Shape::Slab {
                lo,
                hi,
                ridge_x,
                thick,
            } => {
                if !in_footprint(p, lo, hi) {
                    return None;
                }
                let top = gable_height(p, lo, hi, ridge_x);
                (p.y < top && p.y >= top - thick).then_some(self.m)
            }
            Shape::Cylinder { c, r, h } => {
                let d = Vec2::new(p.x - c.x, p.z - c.z);
                (d.length_squared() < r * r && p.y >= c.y && p.y < c.y + h).then_some(self.m)
            }
            Shape::Facade {
                lo,
                hi,
                face,
                out,
                style,
                base,
                storey,
            } => {
                if !(p.cmpge(lo).all() && p.cmplt(hi).all()) {
                    return None;
                }
                let (along, depth) = if face == 0 {
                    (p.z, if out > 0.0 { hi.x - p.x } else { p.x - lo.x })
                } else {
                    (p.x, if out > 0.0 { hi.z - p.z } else { p.z - lo.z })
                };
                let up = p.y - base;
                let level = up.rem_euclid(storey);
                let pane = |at: f32| {
                    if coarse > 0.0 || (depth >= at && depth < at + VOXEL_M) {
                        ids::GLASS
                    } else {
                        ids::AIR
                    }
                };
                Some(match style {
                    FacadeStyle::Curtain => {
                        if along.rem_euclid(1.5) < 0.12 || level < 0.45 {
                            ids::STEEL
                        } else {
                            pane(0.06)
                        }
                    }
                    FacadeStyle::Punched => {
                        let a = along.rem_euclid(3.0);
                        if (0.9..2.1).contains(&a) && (0.9..2.5).contains(&level) {
                            pane(0.12)
                        } else if (0.8..2.2).contains(&a) && (0.78..0.9).contains(&level) {
                            ids::CONCRETE
                        } else {
                            self.m
                        }
                    }
                    FacadeStyle::Ribbon => {
                        if (1.0..2.6).contains(&level) {
                            if along.rem_euclid(3.0) < 0.1 {
                                ids::STEEL
                            } else {
                                pane(0.1)
                            }
                        } else {
                            self.m
                        }
                    }
                    FacadeStyle::Shopfront => {
                        if along.rem_euclid(6.0) < 0.45 || !(0.3..3.1).contains(&up) {
                            self.m
                        } else {
                            pane(0.1)
                        }
                    }
                })
            }
            Shape::Floors {
                lo,
                hi,
                storey,
                thick,
                lit,
            } => {
                if !(p.cmpge(lo).all() && p.cmplt(hi).all()) {
                    return None;
                }
                let up = p.y - lo.y;
                let level = up.rem_euclid(storey);
                if level < thick {
                    return Some(self.m);
                }
                // Light panels hang under the slab above, on lit floors.
                let floor = (up / storey).floor() as i32;
                let on = hash_unit(hash3(floor, 0, lit as i32, 0x0011_9e00)) < 0.55;
                (on && level >= storey - 0.06
                    && p.x.rem_euclid(3.0) < 0.6
                    && p.z.rem_euclid(3.0) < 0.6)
                    .then_some(ids::GLOWSTONE)
            }
            Shape::Ramp {
                a,
                b,
                half,
                g0,
                g1,
                from,
                to,
                marks,
            } => {
                let (t, lateral) = ramp_coords(Vec2::new(p.x, p.z), a, b)?;
                if lateral.abs() > half {
                    return None;
                }
                let g = g0 + (g1 - g0) * t;
                // Coarse cells reach down a whole cell so thin runs show.
                let from = if coarse > 0.0 && !self.m.is_air() {
                    from.min(-coarse)
                } else {
                    from
                };
                if p.y < g + from || p.y >= g + to {
                    return None;
                }
                if let Some(mk) = marks
                    && p.y >= g + to - VOXEL_M
                    && mk.paint(mk.along0 + t * (b - a).length(), lateral, half)
                {
                    return Some(ids::WHITEWASH);
                }
                Some(self.m)
            }
        }
    }

    /// Whether a cube (centre `c`, half size `h`) lies wholly inside, wholly
    /// outside, or across this operation's edge. Only boxes and cylinders
    /// answer exactly; the rest report an edge wherever their bounds reach.
    fn relate(&self, c: Vec3, h: f32) -> Option<bool> {
        let (clo, chi) = (c - h, c + h);
        if chi.cmple(self.lo).any() || clo.cmpge(self.hi).any() {
            return Some(false);
        }
        let within = |lo: Vec3, hi: Vec3| clo.cmpge(lo).all() && chi.cmple(hi).all();
        match self.shape {
            Shape::Box { lo, hi } => within(lo, hi).then_some(true),
            Shape::Floors {
                lo,
                hi,
                storey,
                thick,
                ..
            } => {
                // Between two slabs and clear of the lights: untouched.
                let (y0, y1) = (clo.y - lo.y, chi.y - lo.y);
                let k = (y0 / storey).floor();
                if (y1 / storey).floor() != k {
                    return None;
                }
                let (a, b) = (y0 - k * storey, y1 - k * storey);
                if a >= thick && b <= storey - 0.07 {
                    return Some(false);
                }
                (b <= thick && within(lo, hi)).then_some(true)
            }
            Shape::Ramp {
                a,
                b,
                half,
                g0,
                g1,
                from,
                to,
                marks,
            } => {
                // The cell's corners in the run's frame.
                let mut inside = true;
                let (mut t_lo, mut t_hi) = (f32::MAX, f32::MIN);
                for (x, z) in [
                    (clo.x, clo.z),
                    (chi.x, clo.z),
                    (clo.x, chi.z),
                    (chi.x, chi.z),
                ] {
                    match ramp_coords(Vec2::new(x, z), a, b) {
                        Some((t, lateral)) if lateral.abs() <= half => {
                            t_lo = t_lo.min(t);
                            t_hi = t_hi.max(t);
                        }
                        _ => inside = false,
                    }
                }
                if !inside {
                    return None;
                }
                let (ga, gb) = (g0 + (g1 - g0) * t_lo, g0 + (g1 - g0) * t_hi);
                let (g_lo, g_hi) = (ga.min(gb), ga.max(gb));
                if clo.y >= g_hi + to || chi.y <= g_lo + from {
                    return Some(false);
                }
                let painted = marks.is_some() && chi.y > g_lo + to - VOXEL_M;
                (clo.y >= g_hi + from && chi.y <= g_lo + to && !painted).then_some(true)
            }
            _ => None,
        }
    }
}

/// Side of the squares operations are bucketed in.
const TILE_M: f32 = 8.0;

/// Operations reaching one square of the village, in order, and the height
/// range they span.
struct Tile {
    ops: Vec<u32>,
    lo_y: f32,
    hi_y: f32,
}

pub struct Village {
    pub centre: Vec3,
    /// A town of streets and tall buildings rather than a village.
    pub town: bool,
    ops: Vec<Op>,
    tiles: FxHashMap<(i32, i32), Tile>,
    pub lo: Vec3,
    pub hi: Vec3,
    /// Footprints (x, z min and max) of everything built, where no tree
    /// may grow.
    pub claimed: Vec<(Vec2, Vec2)>,
}

struct Rand(u64);

impl Rand {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        hash_unit(self.0 ^ (self.0 >> 29))
    }

    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next()
    }

    fn int(&mut self, a: i32, b: i32) -> i32 {
        a + (self.next() * (b - a + 1) as f32) as i32
    }
}

/// A path from `a` to `b` following the ground under `surface`, `rise`
/// metres proud of it.
pub(crate) fn path_op(
    surface: &Surface,
    a: Vec2,
    b: Vec2,
    half: f32,
    depth: f32,
    rise: f32,
    m: MaterialId,
) -> Vec<Op> {
    runs(surface, a, b, half, depth, rise, m, None)
}

/// A street from `a` to `b`: `path_op` with markings painted on.
pub(crate) fn street_op(
    surface: &Surface,
    a: Vec2,
    b: Vec2,
    half: f32,
    marks: Marks,
    m: MaterialId,
) -> Vec<Op> {
    runs(surface, a, b, half, 0.3, 0.0, m, Some(marks))
}

/// Straight runs between ground samples along the centre line: an even
/// grade rather than every step of the ground, filled at least a metre down
/// so no edge floats, and cleared above.
#[allow(clippy::too_many_arguments)]
fn runs(
    surface: &Surface,
    a: Vec2,
    b: Vec2,
    half: f32,
    depth: f32,
    rise: f32,
    m: MaterialId,
    marks: Option<Marks>,
) -> Vec<Op> {
    let count = ((b - a).length() / PATH_RUN_M).ceil().max(1.0) as i32;
    let depth = depth.max(1.0);
    let ground = |q: Vec2| snap(surface.sample(q.x, q.y).height);
    let mut ops = Vec::with_capacity(count as usize * 2);
    let mut start = a;
    let mut g_start = ground(a);
    let length = (b - a).length();
    for i in 1..=count {
        let end = a.lerp(b, i as f32 / count as f32);
        let g_end = ground(end);
        let run_marks = marks.map(|mk| Marks {
            along0: mk.along0 + length * (i - 1) as f32 / count as f32,
            ..mk
        });
        let ramp = |from: f32, to: f32, marks: Option<Marks>| Shape::Ramp {
            a: start,
            b: end,
            half,
            g0: g_start,
            g1: g_end,
            from,
            to,
            marks,
        };
        ops.push(Op::new(ramp(-depth, rise, run_marks), m));
        ops.push(Op::new(ramp(rise, rise + CLEAR_M, None), ids::AIR));
        start = end;
        g_start = g_end;
    }
    ops
}

/// A point's position along a run (0 at `a`, 1 at `b`; None beyond its
/// square ends, `b` excluded) and its signed distance from the centre line.
fn ramp_coords(q: Vec2, a: Vec2, b: Vec2) -> Option<(f32, f32)> {
    let ab = b - a;
    let len2 = ab.length_squared().max(1e-6);
    let t = (q - a).dot(ab) / len2;
    if !(0.0..1.0).contains(&t) {
        return None;
    }
    Some((t, ab.perp_dot(q - a) / len2.sqrt()))
}

pub(crate) fn b(lo: Vec3, hi: Vec3, m: MaterialId) -> Op {
    Op::new(Shape::Box { lo, hi }, m)
}

/// Snaps to the voxel grid.
pub(crate) fn snap(v: f32) -> f32 {
    (v / VOXEL_M).round() * VOXEL_M
}

#[derive(Clone, Copy)]
enum Style {
    Timber,
    Stone,
    Brick,
    Thatch,
}

/// Which way a house faces: the wall its door is in.
#[derive(Clone, Copy)]
enum Facing {
    NegX,
    PosX,
    NegZ,
    PosZ,
}

struct Builder<'a> {
    surface: &'a Surface,
    ops: Vec<Op>,
    claimed: Vec<(Vec2, Vec2)>,
}

impl Builder<'_> {
    fn ground(&self, x: f32, z: f32) -> f32 {
        self.surface.sample(x, z).height
    }

    /// A path from `a` to `b` that follows the ground.
    fn path(&mut self, a: Vec2, b: Vec2, half: f32, depth: f32, m: MaterialId) {
        self.ops
            .extend(path_op(self.surface, a, b, half, depth, 0.0, m));
    }

    fn free(&self, lo: Vec2, hi: Vec2) -> bool {
        self.claimed
            .iter()
            .all(|(a, b)| hi.x <= a.x || lo.x >= b.x || hi.y <= a.y || lo.y >= b.y)
    }

    /// A house on footprint `lo..hi` (x, z), its door in the `facing` wall.
    fn house(&mut self, lo: Vec2, hi: Vec2, facing: Facing, style: Style, rng: &mut Rand) -> bool {
        let pad = Vec2::splat(1.2);
        if !self.free(lo - pad, hi + pad) {
            return false;
        }
        let corners = [
            self.ground(lo.x, lo.y),
            self.ground(hi.x, lo.y),
            self.ground(lo.x, hi.y),
            self.ground(hi.x, hi.y),
            self.ground((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5),
        ];
        let gmin = corners.iter().copied().fold(f32::MAX, f32::min);
        let gmax = corners.iter().copied().fold(f32::MIN, f32::max);
        let sea = self.surface.terrain.params.sea_level;
        if gmax - gmin > 3.0 || gmin < sea + 0.5 {
            return false;
        }
        self.claimed.push((lo - pad, hi + pad));
        let floor = snap(gmax + 0.3);
        let storeys = match style {
            Style::Brick => rng.int(2, 3),
            Style::Stone => rng.int(1, 2),
            Style::Timber => rng.int(1, 2),
            Style::Thatch => 1,
        };
        let storey = 3.0;
        let top = floor + storeys as f32 * storey;
        let t = 0.25;
        let (wall, frame, roof, found) = match style {
            Style::Timber => (
                ids::WHITEWASH,
                ids::DARK_PLANKS,
                ids::ROOF_TILE,
                ids::COBBLESTONE,
            ),
            Style::Stone => (
                ids::COBBLESTONE,
                ids::STONE_BRICK,
                ids::SLATE,
                ids::STONE_BRICK,
            ),
            Style::Brick => (
                ids::RED_BRICK,
                ids::STONE_BRICK,
                ids::ROOF_TILE,
                ids::STONE_BRICK,
            ),
            Style::Thatch => (ids::PLANKS, ids::OAK_LOG, ids::THATCH, ids::COBBLESTONE),
        };
        let (x0, x1, z0, z1) = (lo.x, hi.x, lo.y, hi.y);
        let v3 = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        // Foundation down into the slope, walls, the room carved out.
        self.ops.push(b(
            v3(x0 - 0.12, gmin - 1.5, z0 - 0.12),
            v3(x1 + 0.12, floor, z1 + 0.12),
            found,
        ));
        self.ops.push(b(v3(x0, floor, z0), v3(x1, top, z1), wall));
        for s in 0..storeys {
            let y = floor + s as f32 * storey;
            self.ops.push(b(
                v3(x0 + t, y + t, z0 + t),
                v3(x1 - t, y + storey, z1 - t),
                ids::AIR,
            ));
            // Floorboards and, above the ground floor, the beams under them.
            self.ops.push(b(
                v3(x0 + t, y, z0 + t),
                v3(x1 - t, y + t, z1 - t),
                ids::PLANKS,
            ));
        }
        // Frame: corner posts, and for timber framing studs and rails.
        let posts: Vec<(f32, f32)> = vec![(x0, z0), (x1 - t, z0), (x0, z1 - t), (x1 - t, z1 - t)];
        for (px, pz) in posts {
            self.ops.push(b(
                v3(px - 0.04, floor, pz - 0.04),
                v3(px + t + 0.04, top, pz + t + 0.04),
                frame,
            ));
        }
        if matches!(style, Style::Timber | Style::Brick) {
            for s in 0..=storeys {
                let y = floor + s as f32 * storey;
                self.ops.push(b(
                    v3(x0 - 0.04, y - 0.12, z0 - 0.04),
                    v3(x1 + 0.04, y + 0.12, z1 + 0.04),
                    frame,
                ));
            }
        }
        if matches!(style, Style::Timber) {
            let mut x = x0 + 1.5;
            while x < x1 - 1.0 {
                for z in [z0 - 0.04, z1 - t + 0.04] {
                    self.ops
                        .push(b(v3(x, floor, z), v3(x + 0.2, top, z + t), frame));
                }
                x += 1.5;
            }
            let mut z = z0 + 1.5;
            while z < z1 - 1.0 {
                for x in [x0 - 0.04, x1 - t + 0.04] {
                    self.ops
                        .push(b(v3(x, floor, z), v3(x + t, top, z + 0.2), frame));
                }
                z += 1.5;
            }
        }
        // Windows: openings with a pane in the middle of the wall, a sill
        // and a lintel; one per bay on every storey.
        let window = |b0: &mut Vec<Op>, along_x: bool, fixed: f32, from: f32, to: f32, y: f32| {
            let mut s = from + 1.0;
            while s + 1.0 < to {
                let (wlo, whi) = if along_x {
                    (
                        v3(s, y + 0.9, fixed - 0.1),
                        v3(s + 0.75, y + 2.1, fixed + t + 0.1),
                    )
                } else {
                    (
                        v3(fixed - 0.1, y + 0.9, s),
                        v3(fixed + t + 0.1, y + 2.1, s + 0.75),
                    )
                };
                b0.push(b(wlo, whi, ids::AIR));
                let pane = if along_x {
                    (
                        v3(s, y + 0.9, fixed + 0.09),
                        v3(s + 0.75, y + 2.1, fixed + 0.16),
                    )
                } else {
                    (
                        v3(fixed + 0.09, y + 0.9, s),
                        v3(fixed + 0.16, y + 2.1, s + 0.75),
                    )
                };
                b0.push(b(pane.0, pane.1, ids::GLASS));
                // Mullion and transom.
                if along_x {
                    b0.push(b(
                        v3(s + 0.34, y + 0.9, fixed + 0.06),
                        v3(s + 0.41, y + 2.1, fixed + 0.19),
                        ids::DARK_PLANKS,
                    ));
                    b0.push(b(
                        v3(s, y + 1.62, fixed + 0.06),
                        v3(s + 0.75, y + 1.69, fixed + 0.19),
                        ids::DARK_PLANKS,
                    ));
                    b0.push(b(
                        v3(s - 0.06, y + 0.8, fixed - 0.12),
                        v3(s + 0.81, y + 0.9, fixed + t + 0.12),
                        ids::STONE_BRICK,
                    ));
                } else {
                    b0.push(b(
                        v3(fixed + 0.06, y + 0.9, s + 0.34),
                        v3(fixed + 0.19, y + 2.1, s + 0.41),
                        ids::DARK_PLANKS,
                    ));
                    b0.push(b(
                        v3(fixed + 0.06, y + 1.62, s),
                        v3(fixed + 0.19, y + 1.69, s + 0.75),
                        ids::DARK_PLANKS,
                    ));
                    b0.push(b(
                        v3(fixed - 0.12, y + 0.8, s - 0.06),
                        v3(fixed + t + 0.12, y + 0.9, s + 0.81),
                        ids::STONE_BRICK,
                    ));
                }
                s += 2.0;
            }
        };
        for s in 0..storeys {
            let y = floor + s as f32 * storey;
            window(&mut self.ops, true, z0, x0, x1, y);
            window(&mut self.ops, true, z1 - t, x0, x1, y);
            window(&mut self.ops, false, x0, z0, z1, y);
            window(&mut self.ops, false, x1 - t, z0, z1, y);
        }
        // The doorway in the facing wall, and a lantern beside it.
        let (door_lo, door_hi, lamp) = match facing {
            Facing::NegZ => {
                let c = snap((x0 + x1) * 0.5);
                (
                    v3(c - 0.5, floor + t, z0 - 0.2),
                    v3(c + 0.5, floor + 2.3, z0 + t + 0.2),
                    v3(c + 0.8, floor + 2.0, z0 - 0.25),
                )
            }
            Facing::PosZ => {
                let c = snap((x0 + x1) * 0.5);
                (
                    v3(c - 0.5, floor + t, z1 - t - 0.2),
                    v3(c + 0.5, floor + 2.3, z1 + 0.2),
                    v3(c + 0.8, floor + 2.0, z1),
                )
            }
            Facing::NegX => {
                let c = snap((z0 + z1) * 0.5);
                (
                    v3(x0 - 0.2, floor + t, c - 0.5),
                    v3(x0 + t + 0.2, floor + 2.3, c + 0.5),
                    v3(x0 - 0.25, floor + 2.0, c + 0.8),
                )
            }
            Facing::PosX => {
                let c = snap((z0 + z1) * 0.5);
                (
                    v3(x1 - t - 0.2, floor + t, c - 0.5),
                    v3(x1 + 0.2, floor + 2.3, c + 0.5),
                    v3(x1, floor + 2.0, c + 0.8),
                )
            }
        };
        self.ops.push(b(door_lo, door_hi, ids::AIR));
        self.ops.push(b(
            door_lo - Vec3::new(0.06, 0.0, 0.06),
            Vec3::new(door_hi.x + 0.06, door_lo.y + 0.06, door_hi.z + 0.06),
            ids::STONE_BRICK,
        ));
        self.lantern(lamp);
        // Step in front of the door down to the ground.
        let (slo, shi) = match facing {
            Facing::NegZ => (
                v3(door_lo.x - 0.2, gmin - 0.5, z0 - 1.0),
                v3(door_hi.x + 0.2, floor, z0),
            ),
            Facing::PosZ => (
                v3(door_lo.x - 0.2, gmin - 0.5, z1),
                v3(door_hi.x + 0.2, floor, z1 + 1.0),
            ),
            Facing::NegX => (
                v3(x0 - 1.0, gmin - 0.5, door_lo.z - 0.2),
                v3(x0, floor, door_hi.z + 0.2),
            ),
            Facing::PosX => (
                v3(x1, gmin - 0.5, door_lo.z - 0.2),
                v3(x1 + 1.0, floor, door_hi.z + 0.2),
            ),
        };
        self.ops.push(b(slo, shi, ids::STONE_BRICK));
        // Roof: the gable over the longer side, walls up into its ends, the
        // attic hollow, then the covering overhanging every wall.
        let ridge_x = (x1 - x0) >= (z1 - z0);
        let span = if ridge_x { z1 - z0 } else { x1 - x0 };
        let pitch = match style {
            Style::Thatch => rng.range(1.0, 1.25),
            Style::Brick => rng.range(0.6, 0.8),
            _ => rng.range(0.7, 0.95),
        };
        let over = 0.45;
        let ridge = snap(top + (span * 0.5 + over) * pitch);
        let eave = top - over * pitch;
        let gable_lo = v3(x0, top, z0);
        let gable_hi = v3(x1, ridge - over * pitch, z1);
        self.ops.push(Op::new(
            Shape::Prism {
                lo: gable_lo,
                hi: gable_hi,
                ridge_x,
            },
            wall,
        ));
        let inset = Vec3::new(t, 0.0, t);
        self.ops.push(Op::new(
            Shape::Prism {
                lo: gable_lo + inset,
                hi: gable_hi - inset - Vec3::Y * t,
                ridge_x,
            },
            ids::AIR,
        ));
        let thick = if matches!(style, Style::Thatch) {
            0.45
        } else {
            0.28
        };
        let (rlo, rhi) = (
            v3(x0 - over, eave, z0 - over),
            v3(x1 + over, ridge, z1 + over),
        );
        self.ops.push(Op::new(
            Shape::Slab {
                lo: rlo,
                hi: rhi,
                ridge_x,
                thick,
            },
            roof,
        ));
        // Ridge board.
        if ridge_x {
            let c = (z0 + z1) * 0.5;
            self.ops.push(b(
                v3(x0 - over, ridge - 0.1, c - 0.12),
                v3(x1 + over, ridge + 0.06, c + 0.12),
                frame,
            ));
        } else {
            let c = (x0 + x1) * 0.5;
            self.ops.push(b(
                v3(c - 0.12, ridge - 0.1, z0 - over),
                v3(c + 0.12, ridge + 0.06, z1 + over),
                frame,
            ));
        }
        // Chimney through the roof near one end.
        if !matches!(style, Style::Thatch) || rng.next() < 0.5 {
            let (cx, cz) = if ridge_x {
                (x0 + 0.8, (z0 + z1) * 0.5 + rng.range(-1.0, 0.2))
            } else {
                ((x0 + x1) * 0.5 + rng.range(-1.0, 0.2), z0 + 0.8)
            };
            let chimney = if matches!(style, Style::Brick) {
                ids::RED_BRICK
            } else {
                ids::COBBLESTONE
            };
            self.ops.push(b(
                v3(cx - 0.4, floor, cz - 0.4),
                v3(cx + 0.4, ridge + 1.2, cz + 0.4),
                chimney,
            ));
            self.ops.push(b(
                v3(cx - 0.5, ridge + 1.2, cz - 0.5),
                v3(cx + 0.5, ridge + 1.35, cz + 0.5),
                ids::STONE_BRICK,
            ));
            self.ops.push(b(
                v3(cx - 0.2, ridge + 0.4, cz - 0.2),
                v3(cx + 0.2, ridge + 1.4, cz + 0.2),
                ids::AIR,
            ));
        }
        // A lantern hung in each room.
        for s in 0..storeys {
            let y = floor + (s + 1) as f32 * storey - 0.55;
            self.lantern(v3((x0 + x1) * 0.5, y, (z0 + z1) * 0.5));
        }
        true
    }

    /// A small lantern: glowing glass in an iron frame, `at` its base.
    fn lantern(&mut self, at: Vec3) {
        let s = 0.14;
        self.ops.push(b(
            at - Vec3::new(s, 0.0, s),
            at + Vec3::new(s, 0.3, s),
            ids::IRON,
        ));
        self.ops.push(b(
            at - Vec3::new(s - 0.04, -0.04, s + 0.01),
            at + Vec3::new(s - 0.04, 0.26, s + 0.01),
            ids::GLOWSTONE,
        ));
        self.ops.push(b(
            at - Vec3::new(s + 0.01, -0.04, s - 0.04),
            at + Vec3::new(s + 0.01, 0.26, s - 0.04),
            ids::GLOWSTONE,
        ));
    }

    fn lamp_post(&mut self, x: f32, z: f32) {
        let g = self.ground(x, z);
        let base = snap(g - 0.5);
        self.ops.push(b(
            Vec3::new(x - 0.2, base, z - 0.2),
            Vec3::new(x + 0.2, g + 0.3, z + 0.2),
            ids::STONE_BRICK,
        ));
        self.ops.push(b(
            Vec3::new(x - 0.09, g + 0.3, z - 0.09),
            Vec3::new(x + 0.09, g + 3.0, z + 0.09),
            ids::DARK_PLANKS,
        ));
        self.lantern(Vec3::new(x, g + 3.0, z));
        self.ops.push(b(
            Vec3::new(x - 0.2, g + 3.3, z - 0.2),
            Vec3::new(x + 0.2, g + 3.38, z + 0.2),
            ids::IRON,
        ));
    }

    fn field(&mut self, lo: Vec2, hi: Vec2) {
        if !self.free(lo, hi) {
            return;
        }
        let g0 = self.ground(lo.x, lo.y).min(self.ground(hi.x, hi.y));
        let g1 = self.ground(lo.x, lo.y).max(self.ground(hi.x, hi.y));
        if g1 - g0 > 1.5 {
            return;
        }
        self.claimed.push((lo, hi));
        self.path(
            Vec2::new(lo.x, (lo.y + hi.y) * 0.5),
            Vec2::new(hi.x, (lo.y + hi.y) * 0.5),
            (hi.y - lo.y) * 0.5,
            0.3,
            ids::FARMLAND,
        );
    }
}

fn plan(surface: &Surface, centre: Vec3, seed: u64) -> Village {
    let mut rng = Rand(seed);
    let mut bld = Builder {
        surface,
        ops: Vec::new(),
        claimed: Vec::new(),
    };
    let c = Vec2::new(centre.x, centre.z);
    // Plaza: cobbles, and a well with a roof on four posts.
    let plaza = 7.0;
    let g = centre.y;
    bld.ops.push(b(
        Vec3::new(c.x - plaza, g - 2.5, c.y - plaza),
        Vec3::new(c.x + plaza, g, c.y + plaza),
        ids::STONE_BRICK,
    ));
    bld.ops.push(b(
        Vec3::new(c.x - plaza + 0.5, g - 0.25, c.y - plaza + 0.5),
        Vec3::new(c.x + plaza - 0.5, g, c.y + plaza - 0.5),
        ids::COBBLESTONE,
    ));
    bld.ops.push(b(
        Vec3::new(c.x - plaza, g, c.y - plaza),
        Vec3::new(c.x + plaza, g + 3.0, c.y + plaza),
        ids::AIR,
    ));
    bld.claimed
        .push((c - Vec2::splat(plaza), c + Vec2::splat(plaza)));
    let well = Vec3::new(c.x, g - 3.0, c.y);
    bld.ops.push(Op::new(
        Shape::Cylinder {
            c: well,
            r: 1.3,
            h: 3.8,
        },
        ids::STONE_BRICK,
    ));
    bld.ops.push(Op::new(
        Shape::Cylinder {
            c: well,
            r: 0.95,
            h: 3.8,
        },
        ids::AIR,
    ));
    bld.ops.push(Op::new(
        Shape::Cylinder {
            c: well - Vec3::Y * 2.0,
            r: 0.95,
            h: 4.3,
        },
        ids::WATER,
    ));
    for (dx, dz) in [(-1.1, -1.1), (0.9, -1.1), (-1.1, 0.9), (0.9, 0.9)] {
        bld.ops.push(b(
            Vec3::new(c.x + dx, g + 0.8, c.y + dz),
            Vec3::new(c.x + dx + 0.2, g + 3.0, c.y + dz + 0.2),
            ids::DARK_PLANKS,
        ));
    }
    bld.ops.push(Op::new(
        Shape::Slab {
            lo: Vec3::new(c.x - 1.7, g + 2.7, c.y - 1.7),
            hi: Vec3::new(c.x + 1.7, g + 3.9, c.y + 1.7),
            ridge_x: true,
            thick: 0.25,
        },
        ids::ROOF_TILE,
    ));
    // Streets out along the axes, houses facing them on both sides.
    let dirs = [Vec2::X, -Vec2::X, Vec2::Y, -Vec2::Y];
    let skip = rng.int(0, 5);
    for (i, dir) in dirs.into_iter().enumerate() {
        if skip < 4 && i as i32 == skip {
            continue;
        }
        let length = rng.range(45.0, 75.0);
        let start = c + dir * plaza;
        let end = c + dir * (plaza + length);
        bld.path(start, end, 1.6, 0.25, ids::GRAVEL);
        let side = Vec2::new(-dir.y, dir.x);
        bld.claimed.push((
            start.min(end) - Vec2::splat(1.6),
            start.max(end) + Vec2::splat(1.6),
        ));
        let mut d = plaza + 4.0;
        let mut lamp = 0;
        while d < plaza + length - 4.0 {
            for sgn in [1.0f32, -1.0] {
                let w = rng.int(6, 9) as f32;
                let depth = rng.int(5, 8) as f32;
                // Footprint centred off the street, door facing it.
                let along = c + dir * (d + w * 0.5);
                let off = along + side * sgn * (3.2 + depth * 0.5);
                let half = if dir.x != 0.0 {
                    Vec2::new(w * 0.5, depth * 0.5)
                } else {
                    Vec2::new(depth * 0.5, w * 0.5)
                };
                let lo = (off - half).floor();
                let hi = lo + half * 2.0;
                // The door is in the wall toward the street.
                let away = side * sgn;
                let face = if dir.x != 0.0 {
                    if away.y > 0.0 {
                        Facing::NegZ
                    } else {
                        Facing::PosZ
                    }
                } else if away.x > 0.0 {
                    Facing::NegX
                } else {
                    Facing::PosX
                };
                let style = match rng.int(0, 9) {
                    0..=3 => Style::Timber,
                    4..=5 => Style::Stone,
                    6..=7 => Style::Brick,
                    _ => Style::Thatch,
                };
                if bld.house(lo, hi, face, style, &mut rng) && rng.next() < 0.35 {
                    // A field behind.
                    let back = off + side * sgn * (depth * 0.5 + 6.0);
                    bld.field(back - Vec2::splat(4.0), back + Vec2::splat(4.0));
                }
            }
            if lamp % 2 == 0 {
                let p = c + dir * (d + 1.5) + side * 2.1;
                bld.lamp_post(p.x, p.y);
            }
            lamp += 1;
            d += rng.range(10.0, 13.0);
        }
    }
    assemble(centre, bld.ops, bld.claimed, false)
}

/// A settlement from its operations: bounds, and the operations bucketed
/// by the squares they reach.
fn assemble(centre: Vec3, ops: Vec<Op>, claimed: Vec<(Vec2, Vec2)>, town: bool) -> Village {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    let mut tiles: FxHashMap<(i32, i32), Tile> = FxHashMap::default();
    for (n, op) in ops.iter().enumerate() {
        lo = lo.min(op.lo);
        hi = hi.max(op.hi);
        let t0 = (op.lo / TILE_M).floor().as_ivec3();
        let t1 = (op.hi / TILE_M).floor().as_ivec3();
        for tz in t0.z..=t1.z {
            for tx in t0.x..=t1.x {
                let t = tiles.entry((tx, tz)).or_insert(Tile {
                    ops: Vec::new(),
                    lo_y: f32::MAX,
                    hi_y: f32::MIN,
                });
                t.ops.push(n as u32);
                t.lo_y = t.lo_y.min(op.lo.y);
                t.hi_y = t.hi_y.max(op.hi.y);
            }
        }
    }
    Village {
        centre,
        town,
        ops,
        tiles,
        lo,
        hi,
        claimed,
    }
}

/// Villages by grid cell, built on first use.
pub struct Settlements {
    seed: u64,
    cache: Mutex<VillageCache>,
}

type VillageCache = FxHashMap<(i32, i32), Option<Arc<Village>>>;

impl Settlements {
    pub fn new(seed: u64) -> Self {
        Self {
            seed: seed ^ 0x7111_a6e5,
            cache: Mutex::new(FxHashMap::default()),
        }
    }

    /// The village of grid cell (i, k), if it has one.
    pub fn village(&self, surface: &Surface, i: i32, k: i32) -> Option<Arc<Village>> {
        if let Some(v) = self.cache.lock().expect("cache").get(&(i, k)) {
            return v.clone();
        }
        let v = self.site(surface, i, k).map(|(centre, h, town)| {
            Arc::new(if town {
                let (ops, claimed) = crate::town::plan(surface, centre, h);
                assemble(centre, ops, claimed, true)
            } else {
                plan(surface, centre, h)
            })
        });
        self.cache.lock().expect("cache").insert((i, k), v.clone());
        v
    }

    /// Where a cell's settlement stands (flat open ground well above the
    /// sea), its seed, and whether it is a town (which needs its whole
    /// street grid flat).
    fn site(&self, surface: &Surface, i: i32, k: i32) -> Option<(Vec3, u64, bool)> {
        let cell = hash3(i, 3, k, self.seed);
        if hash_unit(cell) > 0.75 {
            return None;
        }
        let wants_town = hash_unit(cell.rotate_left(41)) < TOWN_SHARE;
        let sea = surface.terrain.params.sea_level;
        // A few candidate spots in the cell; the first that suits wins.
        for attempt in 0..6 {
            let h = hash3(i, 4 + attempt, k, self.seed);
            let x = (i as f32 + 0.2 + 0.6 * hash_unit(h.rotate_left(13))) * VILLAGE_CELL_M;
            let z = (k as f32 + 0.2 + 0.6 * hash_unit(h.rotate_left(29))) * VILLAGE_CELL_M;
            let s = surface.sample(x, z);
            let open = matches!(
                s.biome(sea),
                Biome::Plains | Biome::Forest | Biome::BirchForest | Biome::Taiga
            );
            if !open || s.slope > 0.2 || s.height < sea + 3.0 || s.river > 0.2 {
                continue;
            }
            // Flat enough across the plaza.
            let around = [(-12.0, 0.0), (12.0, 0.0), (0.0, -12.0), (0.0, 12.0)];
            let spread = around
                .iter()
                .map(|(dx, dz)| (surface.sample(x + dx, z + dz).height - s.height).abs())
                .fold(0.0, f32::max);
            if spread > 2.5 {
                continue;
            }
            let centre = Vec3::new(x, snap(s.height), z);
            if !wants_town {
                return Some((centre, h, false));
            }
            // A town needs its whole street grid near one level and dry.
            let (_, extent) = crate::town::size(h);
            let half = extent * 0.5 + 6.0;
            let mut flat = true;
            for gz in -2..=2 {
                for gx in -2..=2 {
                    let (px, pz) = (x + gx as f32 * half * 0.5, z + gz as f32 * half * 0.5);
                    let q = surface.sample(px, pz);
                    if (q.height - s.height).abs() > 7.0 || q.height < sea + 2.0 || q.river > 0.3 {
                        flat = false;
                    }
                }
            }
            if flat {
                return Some((centre, h, true));
            }
        }
        None
    }

    /// Villages whose bounds may reach the box `lo..hi` (metres).
    pub fn near(&self, surface: &Surface, lo: Vec3, hi: Vec3) -> Vec<Arc<Village>> {
        let c0 = ((lo.x - VILLAGE_RADIUS_M) / VILLAGE_CELL_M).floor() as i32;
        let c1 = ((hi.x + VILLAGE_RADIUS_M) / VILLAGE_CELL_M).floor() as i32;
        let k0 = ((lo.z - VILLAGE_RADIUS_M) / VILLAGE_CELL_M).floor() as i32;
        let k1 = ((hi.z + VILLAGE_RADIUS_M) / VILLAGE_CELL_M).floor() as i32;
        let mut out = Vec::new();
        for k in k0..=k1 {
            for i in c0..=c1 {
                if let Some(v) = self.village(surface, i, k)
                    && v.hi.cmpge(lo).all()
                    && v.lo.cmple(hi).all()
                {
                    out.push(v);
                }
            }
        }
        out
    }
}

impl Village {
    /// Whether anything of the village stands at (x, z), with room for a
    /// crown: no tree grows there.
    pub fn claims(&self, x: f32, z: f32) -> bool {
        self.claimed
            .iter()
            .any(|(a, b)| x >= a.x - 3.0 && x < b.x + 3.0 && z >= a.y - 3.0 && z < b.y + 3.0)
    }

    /// Material at `p` from every operation that contains it, last wins.
    fn at(&self, ops: &[&Op], p: Vec3, coarse: f32) -> Option<MaterialId> {
        let mut found = None;
        for op in ops {
            if let Some(m) = op.at(p, coarse) {
                found = Some(m);
            }
        }
        found
    }

    /// The brick cells of a chunk (origin in metres, `size` metres a cell,
    /// `per_axis` to a side) each tile of the village reaches, with that
    /// tile's operations.
    fn tiles_in(&self, origin_m: Vec3, size: f32, per_axis: i32) -> Vec<(IVec3, IVec3, &Tile)> {
        let span = size * per_axis as f32;
        let t0 = (origin_m / TILE_M).floor().as_ivec3();
        let t1 = ((origin_m + span - 0.001) / TILE_M).floor().as_ivec3();
        let mut out = Vec::new();
        for tz in t0.z..=t1.z {
            for tx in t0.x..=t1.x {
                let Some(tile) = self.tiles.get(&(tx, tz)) else {
                    continue;
                };
                let tlo = Vec3::new(tx as f32 * TILE_M, tile.lo_y, tz as f32 * TILE_M);
                let thi = Vec3::new(
                    (tx + 1) as f32 * TILE_M,
                    tile.hi_y,
                    (tz + 1) as f32 * TILE_M,
                );
                let lo = ((tlo - origin_m) / size)
                    .floor()
                    .as_ivec3()
                    .max(IVec3::ZERO);
                let hi = ((thi - origin_m) / size - 0.001)
                    .floor()
                    .as_ivec3()
                    .min(IVec3::splat(per_axis - 1));
                if lo.cmple(hi).all() {
                    out.push((lo, hi, tile));
                }
            }
        }
        out
    }

    /// Stamps the village into a chunk generated voxel by voxel.
    pub fn stamp_full(&self, tree: &mut ChunkTree, origin: IVec3) {
        let origin_m = origin.as_vec3() * VOXEL_M;
        for (lo, hi, tile) in self.tiles_in(origin_m, 0.5, 64) {
            for cz in lo.z..=hi.z {
                for cx in lo.x..=hi.x {
                    for cy in lo.y..=hi.y {
                        let cell = IVec3::new(cx, cy, cz);
                        let centre = origin_m + (cell.as_vec3() + 0.5) * 0.5;
                        let ops: Vec<&Op> = tile
                            .ops
                            .iter()
                            .map(|&n| &self.ops[n as usize])
                            .filter(|op| op.relate(centre, 0.25) != Some(false))
                            .collect();
                        if ops.is_empty() {
                            continue;
                        }
                        // The last operation wholly containing the cell with
                        // none after it touching the cell: one material.
                        let whole = ops
                            .iter()
                            .rposition(|op| op.relate(centre, 0.25) == Some(true));
                        if whole == Some(ops.len() - 1) {
                            let m = ops[ops.len() - 1].m;
                            tree.set_cell(
                                cell,
                                if m.is_air() {
                                    Cell::Empty
                                } else {
                                    Cell::Uniform(m)
                                },
                            );
                            continue;
                        }
                        let ops = &ops[whole.unwrap_or(0)..];
                        let base = origin + cell * 8;
                        tree.edit_brick(cell, |brick| {
                            for i in 0..512 {
                                let l = IVec3::new(i & 7, (i >> 6) & 7, (i >> 3) & 7);
                                let v = base + l;
                                let p = (v.as_vec3() + 0.5) * VOXEL_M;

                                if let Some(m) = self.at(ops, p, 0.0) {
                                    brick.set(l, m);
                                }
                            }
                        });
                    }
                }
            }
        }
    }

    /// Stamps the village into a chunk kept at one material per cell of
    /// `cells` brick cells.
    pub fn stamp_coarse(&self, tree: &mut ChunkTree, origin: IVec3, cells: i32) {
        let origin_m = origin.as_vec3() * VOXEL_M;
        let size = cells as f32 * 0.5;
        for (lo, hi, tile) in self.tiles_in(origin_m, size, 64 / cells) {
            for z in lo.z..=hi.z {
                for x in lo.x..=hi.x {
                    let cx = origin_m.x + (x as f32 + 0.5) * size;
                    let cz = origin_m.z + (z as f32 + 0.5) * size;

                    for y in lo.y..=hi.y {
                        let c = IVec3::new(x, y, z);
                        let centre = Vec3::new(cx, origin_m.y + (y as f32 + 0.5) * size, cz);
                        let ops: Vec<&Op> = tile
                            .ops
                            .iter()
                            .map(|&n| &self.ops[n as usize])
                            .filter(|op| op.relate(centre, size * 0.5) != Some(false))
                            .collect();
                        let Some(m) = self.at(&ops, centre, size) else {
                            continue;
                        };
                        if cells == 1 {
                            tree.set_cell(
                                c,
                                if m.is_air() {
                                    Cell::Empty
                                } else {
                                    Cell::Uniform(m)
                                },
                            );
                        } else if !m.is_air() {
                            tree.set_node(1, c, m);
                        }
                    }
                }
            }
        }
    }
}
