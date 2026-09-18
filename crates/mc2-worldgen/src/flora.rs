//! Life on the land: a climate and the biomes it implies, then trees,
//! bushes, cacti, fallen logs and boulders on a jittered grid, each built
//! from a handful of primitives at voxel resolution: round trunks that
//! flare at the root and lean, branches, crowns of noisy leaf clusters with
//! gaps near their skin, whorled pines with snow on every tier.
//!
//! Every plant is a pure function of its grid cell, so any chunk can be
//! generated alone and neighbours agree on what crosses their seam. Plants
//! are stamped after the terrain, only into air: a voxel at a time where a
//! brick cell holds an edge, whole uniform cells inside crowns (which is
//! what keeps a forest's voxel count down), and at coarse levels of detail
//! one sample per cell or node, so forests stay on distant hills.

use crate::amplify::{SNOWLINE_M, Surface, SurfaceSample, TREELINE_M};
use crate::noise::{Perlin, hash_unit, hash3};
use glam::{IVec3, Vec3};
use mc2_voxel::material::{Kind, MaterialId, ids};
use mc2_voxel::tree::{Cell, ChunkTree};

const VOXEL_M: f32 = 1.0 / 16.0;
/// Plants stand one per cell of this grid at most.
pub const FLORA_CELL_M: f32 = 4.0;
/// Farthest a plant reaches from its root, sideways.
pub const FLORA_REACH_M: f32 = 6.0;
/// Tallest plant.
pub const FLORA_HEIGHT_M: f32 = 22.0;
/// Tallest grass blade or flower above the ground.
pub const GROUND_COVER_M: f32 = 0.6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Biome {
    Ocean,
    Beach,
    Plains,
    Forest,
    BirchForest,
    Taiga,
    SnowyTaiga,
    Desert,
    Alpine,
    Snow,
}

/// Temperature and moisture fields. Temperature falls with altitude.
pub struct Climate {
    temp: Perlin,
    wet: Perlin,
}

impl Climate {
    pub fn new(seed: u64) -> Self {
        Self {
            temp: Perlin::new(seed ^ 0x7e3a_11c5),
            wet: Perlin::new(seed ^ 0x51d0_f00d),
        }
    }

    /// (temperature, moisture), each roughly 0..1, at a point `height`
    /// metres up by water carrying `river` (0..1).
    pub fn at(&self, x: f32, z: f32, height: f32, sea: f32, river: f32) -> (f32, f32) {
        let t = 0.5
            + 0.6
                * self
                    .temp
                    .fbm2(x / 4200.0 + 7.0, z / 4200.0 - 3.0, 3, 2.0, 0.5)
            - (height - sea).max(0.0) / 700.0;
        let w = 0.5
            + 0.65
                * self
                    .wet
                    .fbm2(x / 2600.0 - 11.0, z / 2600.0 + 5.0, 3, 2.0, 0.5)
            + river * 0.3;
        (t.clamp(0.0, 1.0), w.clamp(0.0, 1.0))
    }
}

impl SurfaceSample {
    pub fn biome(&self, sea: f32) -> Biome {
        if self.height < sea - 0.3 {
            return Biome::Ocean;
        }
        if self.height < sea + 1.5 && self.slope < 0.25 {
            return Biome::Beach;
        }
        if self.height > SNOWLINE_M {
            return Biome::Snow;
        }
        if self.height > TREELINE_M {
            return Biome::Alpine;
        }
        let (t, w) = (self.temp, self.wet);
        if t < 0.3 {
            return Biome::SnowyTaiga;
        }
        if t < 0.42 {
            return Biome::Taiga;
        }
        if t > 0.64 && w < 0.4 {
            return Biome::Desert;
        }
        if w > 0.58 {
            return if t < 0.52 {
                Biome::BirchForest
            } else {
                Biome::Forest
            };
        }
        Biome::Plains
    }
}

fn unit(h: u64, k: u32) -> f32 {
    hash_unit(h.rotate_left(k))
}

/// Material of the ground cover (grass blades and flowers) at a voxel
/// `above` metres over grassy ground at world voxel column (vx, vz).
pub fn ground_cover(
    sample: &SurfaceSample,
    sea: f32,
    vx: i32,
    vz: i32,
    above: f32,
    seed: u64,
) -> MaterialId {
    let biome = sample.biome(sea);
    let (density, tallest) = match biome {
        Biome::Plains => (0.38, 9),
        Biome::Forest => (0.22, 6),
        Biome::BirchForest => (0.3, 7),
        Biome::Taiga => (0.12, 4),
        _ => return ids::AIR,
    };
    let h = hash3(vx, 0, vz, seed ^ 0x6a55);
    if unit(h, 0) > density {
        return ids::AIR;
    }
    let height = 1 + (unit(h, 13) * unit(h, 29) * tallest as f32) as i32;
    let level = (above / VOXEL_M) as i32;
    if level > height {
        return ids::AIR;
    }
    // A flower now and then, at the top of a taller stem.
    let flower = unit(h, 41);
    if level == height && flower < 0.04 && matches!(biome, Biome::Plains | Biome::BirchForest) {
        return match (flower * 100.0) as u32 {
            0 => ids::FLOWER_RED,
            1 => ids::FLOWER_BLUE,
            2 | 3 => ids::FLOWER_WHITE,
            _ => ids::FLOWER_YELLOW,
        };
    }
    if level >= height {
        return ids::AIR;
    }
    ids::TALL_GRASS
}

/// A part of a plant.
#[derive(Clone, Copy, Debug)]
enum Prim {
    /// A tapered rod with round ends, from `a` (radius `ra`) to `b` (`rb`).
    Wood {
        a: Vec3,
        b: Vec3,
        ra: f32,
        rb: f32,
        m: MaterialId,
    },
    /// An ellipsoid with a noisy edge. `fill` is the share of voxels kept
    /// in its outer skin (leaves have gaps, stone none); `cap` covers its
    /// top skin (snow on pine tiers, moss on stones).
    Blob {
        c: Vec3,
        r: Vec3,
        m: MaterialId,
        fill: f32,
        cap: Option<MaterialId>,
        salt: u32,
    },
}

enum Relation {
    Outside,
    Inside(MaterialId),
    Edge,
}

/// Distance from `p` to segment `ab` and the parameter of the nearest point.
fn segment(p: Vec3, a: Vec3, b: Vec3) -> (f32, f32) {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
    ((p - (a + ab * t)).length(), t)
}

/// How far into a blob `p` is, in units of its radii, with the noisy edge
/// at 1 (between 0.85 and 1.15 of the ellipsoid).
fn blob_depth(noise: &Perlin, p: Vec3, c: Vec3, r: Vec3, salt: u32) -> f32 {
    let q = (p - c) / r;
    let d = q.length();
    if d > 1.3 {
        return d;
    }
    let s = salt as f32 * 17.13;
    let n = noise.noise3(p.x * 0.9 + s, p.y * 0.9 - s, p.z * 0.9 + s * 0.5);
    d - n * 0.15
}

impl Prim {
    fn sample(&self, noise: &Perlin, p: Vec3, voxel: IVec3) -> Option<MaterialId> {
        match *self {
            Prim::Wood { a, b, ra, rb, m } => {
                let (d, t) = segment(p, a, b);
                (d <= ra + (rb - ra) * t).then_some(m)
            }
            Prim::Blob {
                c,
                r,
                m,
                fill,
                cap,
                salt,
            } => {
                let depth = blob_depth(noise, p, c, r, salt);
                if depth > 1.0 {
                    return None;
                }
                if depth > 0.72 {
                    let h = hash3(voxel.x, voxel.y, voxel.z, salt as u64 ^ 0x1eaf);
                    if hash_unit(h) > fill {
                        return None;
                    }
                }
                if let Some(cap) = cap {
                    let above = p + Vec3::Y * 0.14;
                    if blob_depth(noise, above, c, r, salt) > 1.0 {
                        return Some(cap);
                    }
                }
                Some(m)
            }
        }
    }

    /// Relation of a cube (centre `p`, half size `h`) to this part.
    fn relate(&self, p: Vec3, h: f32) -> Relation {
        let reach = h * 1.74;
        match *self {
            Prim::Wood { a, b, ra, rb, m } => {
                let (d, t) = segment(p, a, b);
                let r = ra + (rb - ra) * t;
                if d > r + reach {
                    Relation::Outside
                } else if d < r - reach {
                    Relation::Inside(m)
                } else {
                    Relation::Edge
                }
            }
            Prim::Blob { c, r, m, cap, .. } => {
                let q = (p - c) / r;
                let d = q.length();
                let k = reach / r.min_element();
                if d - k > 1.2 {
                    Relation::Outside
                } else if d + k < 0.6 && cap.is_none() {
                    Relation::Inside(m)
                } else {
                    Relation::Edge
                }
            }
        }
    }

    fn bounds(&self) -> (Vec3, Vec3) {
        match *self {
            Prim::Wood { a, b, ra, rb, .. } => {
                let r = ra.max(rb);
                (a.min(b) - r, a.max(b) + r)
            }
            Prim::Blob { c, r, .. } => (c - r * 1.2, c + r * 1.2),
        }
    }

    fn is_wood(&self) -> bool {
        matches!(self, Prim::Wood { .. })
    }
}

#[derive(Clone)]
pub struct Plant {
    prims: Vec<Prim>,
    pub lo: Vec3,
    pub hi: Vec3,
}

impl Plant {
    fn new(prims: Vec<Prim>) -> Self {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in &prims {
            let (a, b) = p.bounds();
            lo = lo.min(a);
            hi = hi.max(b);
        }
        Self { prims, lo, hi }
    }

    /// Material at world point `p` (metres) in world voxel `voxel`: wood
    /// before leaves.
    pub fn sample(&self, noise: &Perlin, p: Vec3, voxel: IVec3) -> Option<MaterialId> {
        let mut found = None;
        for prim in &self.prims {
            let (lo, hi) = prim.bounds();
            if p.cmplt(lo).any() || p.cmpgt(hi).any() {
                continue;
            }
            if let Some(m) = prim.sample(noise, p, voxel) {
                if prim.is_wood() {
                    return Some(m);
                }
                found = found.or(Some(m));
            }
        }
        found
    }

    /// Sample for coarse levels: thin wood thickened to show at all, no
    /// gaps or noise.
    fn sample_coarse(&self, p: Vec3, size: f32) -> Option<MaterialId> {
        let mut found = None;
        for prim in &self.prims {
            match *prim {
                Prim::Wood { a, b, ra, rb, m } => {
                    let (d, t) = segment(p, a, b);
                    if d <= (ra + (rb - ra) * t).max(size * 0.5) {
                        return Some(m);
                    }
                }
                Prim::Blob { c, r, m, cap, .. } => {
                    if ((p - c) / r).length() <= 1.0 {
                        let top = ((p + Vec3::Y * size - c) / r).length() > 1.0;
                        found = found.or(Some(if top { cap.unwrap_or(m) } else { m }));
                    }
                }
            }
        }
        found
    }

    /// A brick cell's relation to the whole plant.
    fn relate(&self, p: Vec3, h: f32) -> Relation {
        let mut inside = None;
        for prim in &self.prims {
            match prim.relate(p, h) {
                Relation::Edge => return Relation::Edge,
                Relation::Inside(m) => {
                    if prim.is_wood() || inside.is_none() {
                        inside = Some(m);
                    }
                }
                Relation::Outside => {}
            }
        }
        inside.map_or(Relation::Outside, Relation::Inside)
    }
}

/// Deterministic random stream for building one plant.
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

    /// A unit direction in the horizontal plane.
    fn heading(&mut self) -> Vec3 {
        let a = self.next() * std::f32::consts::TAU;
        Vec3::new(a.cos(), 0.0, a.sin())
    }
}

fn leaves(c: Vec3, r: Vec3, m: MaterialId, salt: u32) -> Prim {
    Prim::Blob {
        c,
        r,
        m,
        fill: 0.62,
        cap: None,
        salt,
    }
}

fn wood(a: Vec3, b: Vec3, ra: f32, rb: f32, m: MaterialId) -> Prim {
    Prim::Wood { a, b, ra, rb, m }
}

/// A broadleaf tree: a leaning trunk flaring at the root, limbs spreading
/// from low on it, and a broad, uneven crown of leaf clusters at their ends,
/// over the top and in between. Tall (birch) trees keep a narrow crown.
fn broadleaf(root: Vec3, rng: &mut Rand, log: MaterialId, leaf: MaterialId, tall: bool) -> Plant {
    let height = if tall {
        rng.range(9.0, 14.0)
    } else {
        rng.range(7.0, 11.0)
    };
    let r = if tall {
        rng.range(0.16, 0.24)
    } else {
        rng.range(0.3, 0.5)
    };
    let lean = rng.heading() * rng.range(0.0, 0.07) * height;
    let mut prims = Vec::new();
    // Root flare and trunk in three bends.
    prims.push(wood(
        root - Vec3::Y * 0.4,
        root + Vec3::Y * 0.9,
        r * 1.7,
        r * 1.05,
        log,
    ));
    let top = root + Vec3::Y * height + lean;
    let mut prev = root + Vec3::Y * 0.9;
    for i in 1..=3 {
        let t = i as f32 / 3.0;
        let wobble = rng.heading() * rng.range(0.0, 0.3) * (1.0 - t);
        let next = root.lerp(top, t) + wobble;
        prims.push(wood(
            prev,
            next,
            r * (1.05 - 0.4 * (t - 1.0 / 3.0)),
            r * (1.05 - 0.4 * t),
            log,
        ));
        prev = next;
    }
    let salt = (rng.next() * 1e6) as u32;
    let crown_base = if tall {
        rng.range(0.5, 0.6)
    } else {
        rng.range(0.38, 0.48)
    };
    let limbs = if tall {
        3
    } else {
        4 + (rng.next() * 2.99) as u32
    };
    for i in 0..limbs {
        let at = root.lerp(top, rng.range(crown_base, crown_base + 0.3));
        let angle = (i as f32 + rng.range(0.0, 0.7)) / limbs as f32 * std::f32::consts::TAU;
        let out = Vec3::new(angle.cos(), 0.0, angle.sin());
        let reach = if tall {
            rng.range(1.0, 1.8)
        } else {
            rng.range(2.4, 4.2)
        };
        let end = at + out * reach + Vec3::Y * reach * rng.range(0.35, 0.8);
        let bend = at.lerp(end, 0.5) + Vec3::Y * rng.range(0.1, 0.5);
        prims.push(wood(at, bend, r * 0.55, r * 0.38, log));
        prims.push(wood(bend, end, r * 0.38, r * 0.2, log));
        let size = if tall {
            rng.range(1.1, 1.6)
        } else {
            rng.range(1.7, 2.6)
        };
        prims.push(leaves(
            end + Vec3::Y * 0.2,
            Vec3::new(size, size * 0.65, size * rng.range(0.85, 1.15)),
            leaf,
            salt + i,
        ));
    }
    // Over the top, and clusters scattered through the crown to fill it and
    // break its outline.
    let crown = if tall {
        rng.range(1.4, 2.0)
    } else {
        rng.range(2.4, 3.2)
    };
    prims.push(leaves(
        top + Vec3::Y * 0.2,
        Vec3::new(crown, crown * if tall { 1.3 } else { 0.62 }, crown),
        leaf,
        salt + 11,
    ));
    let extra = if tall {
        2
    } else {
        4 + (rng.next() * 2.99) as u32
    };
    let spread = if tall { 1.4 } else { 3.2 };
    for i in 0..extra {
        let at = root.lerp(top, rng.range(crown_base + 0.15, 0.95))
            + rng.heading() * rng.range(0.6, spread);
        let size = if tall {
            rng.range(1.0, 1.6)
        } else {
            rng.range(1.4, 2.2)
        };
        prims.push(leaves(
            at,
            Vec3::new(size, size * 0.7, size),
            leaf,
            salt + 21 + i,
        ));
    }
    Plant::new(prims)
}

/// A pine: a straight trunk and tiers of needles narrowing to a spike,
/// snow on every tier in the cold.
fn pine(root: Vec3, rng: &mut Rand, snowy: bool) -> Plant {
    let height = rng.range(10.0, 18.0);
    let r = rng.range(0.22, 0.36);
    let mut prims = vec![
        wood(
            root - Vec3::Y * 0.4,
            root + Vec3::Y * 0.7,
            r * 1.6,
            r,
            ids::PINE_LOG,
        ),
        wood(
            root + Vec3::Y * 0.7,
            root + Vec3::Y * height,
            r,
            r * 0.3,
            ids::PINE_LOG,
        ),
    ];
    let salt = (rng.next() * 1e6) as u32;
    let bottom = rng.range(0.22, 0.35) * height;
    let widest = rng.range(2.2, 3.2);
    let mut y = bottom;
    let mut i = 0;
    while y < height - 0.6 {
        let t = (y - bottom) / (height - bottom);
        let radius = widest * (1.0 - t).powf(0.9) + 0.35;
        prims.push(Prim::Blob {
            c: root + Vec3::Y * y,
            r: Vec3::new(radius, 0.55 + 0.2 * (1.0 - t), radius),
            m: ids::PINE_NEEDLES,
            fill: 0.7,
            cap: snowy.then_some(ids::SNOW),
            salt: salt + i,
        });
        y += rng.range(0.75, 1.05);
        i += 1;
    }
    prims.push(leaves(
        root + Vec3::Y * height,
        Vec3::new(0.35, 0.9, 0.35),
        ids::PINE_NEEDLES,
        salt + 99,
    ));
    Plant::new(prims)
}

fn bush(root: Vec3, rng: &mut Rand, leaf: MaterialId) -> Plant {
    let salt = (rng.next() * 1e6) as u32;
    let mut prims = Vec::new();
    for i in 0..(1 + (rng.next() * 2.99) as u32) {
        let at = root + rng.heading() * rng.range(0.0, 0.6) + Vec3::Y * rng.range(0.3, 0.6);
        let s = rng.range(0.55, 1.1);
        prims.push(Prim::Blob {
            c: at,
            r: Vec3::new(s, s * 0.8, s),
            m: leaf,
            fill: 0.75,
            cap: None,
            salt: salt + i,
        });
    }
    Plant::new(prims)
}

fn cactus(root: Vec3, rng: &mut Rand) -> Plant {
    let height = rng.range(1.5, 3.6);
    let r = rng.range(0.18, 0.27);
    let mut prims = vec![wood(
        root - Vec3::Y * 0.3,
        root + Vec3::Y * height,
        r,
        r * 0.9,
        ids::CACTUS,
    )];
    for _ in 0..(rng.next() * 2.4) as u32 {
        let at = root + Vec3::Y * rng.range(0.7, height * 0.7);
        let out = rng.heading();
        let elbow = at + out * rng.range(0.4, 0.7);
        prims.push(wood(at, elbow, r * 0.7, r * 0.7, ids::CACTUS));
        prims.push(wood(
            elbow,
            elbow + Vec3::Y * rng.range(0.5, 1.2),
            r * 0.7,
            r * 0.6,
            ids::CACTUS,
        ));
    }
    Plant::new(prims)
}

fn boulder(root: Vec3, rng: &mut Rand, mossy: bool) -> Plant {
    let salt = (rng.next() * 1e6) as u32;
    let rock = if rng.next() < 0.5 {
        ids::GRANITE
    } else {
        ids::GNEISS
    };
    let mut prims = Vec::new();
    for i in 0..(1 + (rng.next() * 2.5) as u32) {
        let s = rng.range(0.5, 1.6);
        prims.push(Prim::Blob {
            c: root + rng.heading() * rng.range(0.0, s * 0.6) + Vec3::Y * s * rng.range(-0.1, 0.3),
            r: Vec3::new(
                s * rng.range(0.9, 1.3),
                s * rng.range(0.6, 0.9),
                s * rng.range(0.9, 1.3),
            ),
            m: rock,
            fill: 1.0,
            cap: mossy.then_some(ids::MOSS),
            salt: salt + i,
        });
    }
    Plant::new(prims)
}

fn fallen_log(root: Vec3, rng: &mut Rand) -> Plant {
    let out = rng.heading();
    let len = rng.range(2.5, 5.0);
    let r = rng.range(0.22, 0.34);
    let a = root + Vec3::Y * (r * 0.7);
    let b = a + out * len;
    let salt = (rng.next() * 1e6) as u32;
    Plant::new(vec![
        wood(a, b, r, r * 0.85, ids::OAK_LOG),
        Prim::Blob {
            c: a.lerp(b, 0.5) + Vec3::Y * r * 0.9,
            r: Vec3::new(len * 0.3, 0.12, len * 0.3),
            m: ids::MOSS,
            fill: 0.5,
            cap: None,
            salt,
        },
    ])
}

/// The plant rooted in grid cell (i, k), if any, unless `taken` claims
/// its spot.
pub fn plant_in_cell(
    surface: &Surface,
    i: i32,
    k: i32,
    seed: u64,
    taken: &dyn Fn(f32, f32) -> bool,
) -> Option<Plant> {
    let h = hash3(i, 7, k, seed ^ 0x000f_107a);
    let x = (i as f32 + 0.1 + 0.8 * unit(h, 0)) * FLORA_CELL_M;
    let z = (k as f32 + 0.1 + 0.8 * unit(h, 17)) * FLORA_CELL_M;
    let s = surface.sample(x, z);
    let sea = surface.terrain.params.sea_level;
    if s.slope > 0.75 || s.river > 0.35 || s.height < sea + 0.4 || taken(x, z) {
        return None;
    }
    let biome = s.biome(sea);
    let roll = unit(h, 31);
    let root = Vec3::new(x, s.height, z);
    let mut rng = Rand(h ^ 0x5eed);
    let pick = unit(h, 47);
    let soil = s.soil_depth > 0.2;
    match biome {
        Biome::Forest if roll < 0.42 && soil => Some(if pick < 0.15 {
            broadleaf(root, &mut rng, ids::BIRCH_LOG, ids::BIRCH_LEAVES, true)
        } else if pick < 0.9 {
            broadleaf(root, &mut rng, ids::OAK_LOG, ids::LEAVES, false)
        } else {
            fallen_log(root, &mut rng)
        }),
        Biome::Forest if roll < 0.52 => Some(bush(root, &mut rng, ids::LEAVES)),
        Biome::BirchForest if roll < 0.4 && soil => Some(if pick < 0.75 {
            broadleaf(root, &mut rng, ids::BIRCH_LOG, ids::BIRCH_LEAVES, true)
        } else {
            broadleaf(root, &mut rng, ids::OAK_LOG, ids::LEAVES, false)
        }),
        Biome::BirchForest if roll < 0.48 => Some(bush(root, &mut rng, ids::BIRCH_LEAVES)),
        Biome::Taiga | Biome::SnowyTaiga if roll < 0.45 => {
            Some(pine(root, &mut rng, biome == Biome::SnowyTaiga))
        }
        Biome::Taiga if roll < 0.5 => Some(boulder(root, &mut rng, true)),
        Biome::Plains if roll < 0.035 && soil => {
            Some(broadleaf(root, &mut rng, ids::OAK_LOG, ids::LEAVES, false))
        }
        Biome::Plains if roll < 0.09 => Some(bush(root, &mut rng, ids::LEAVES)),
        Biome::Plains if roll < 0.1 => Some(boulder(root, &mut rng, false)),
        Biome::Desert if roll < 0.05 => Some(cactus(root, &mut rng)),
        Biome::Desert if roll < 0.06 => Some(boulder(root, &mut rng, false)),
        Biome::Alpine if roll < 0.06 => Some(boulder(root, &mut rng, false)),
        _ => None,
    }
}

/// Plants whose bounds may reach into the box `lo..hi` (metres), none
/// where `taken` says the ground is built on.
pub fn plants_near(
    surface: &Surface,
    lo: Vec3,
    hi: Vec3,
    seed: u64,
    taken: &dyn Fn(f32, f32) -> bool,
) -> Vec<Plant> {
    let c0 = ((lo.x - FLORA_REACH_M) / FLORA_CELL_M).floor() as i32;
    let c1 = ((hi.x + FLORA_REACH_M) / FLORA_CELL_M).floor() as i32;
    let k0 = ((lo.z - FLORA_REACH_M) / FLORA_CELL_M).floor() as i32;
    let k1 = ((hi.z + FLORA_REACH_M) / FLORA_CELL_M).floor() as i32;
    let mut out = Vec::new();
    for k in k0..=k1 {
        for i in c0..=c1 {
            if let Some(p) = plant_in_cell(surface, i, k, seed, taken)
                && p.hi.cmpge(lo).all()
                && p.lo.cmple(hi).all()
            {
                out.push(p);
            }
        }
    }
    out
}

/// What a plant may grow into: air, or ground cover it pushes aside.
fn replaceable(m: MaterialId) -> bool {
    m.is_air() || (m.get().kind == Kind::Foliage && m != ids::LEAVES && m != ids::PINE_NEEDLES)
}

/// Edge noise of a blob over one brick cell: sampled at the cell's
/// corners and interpolated inside, eight noise evaluations a cell instead
/// of one a voxel.
struct CellNoise {
    lo: Vec3,
    corners: [f32; 8],
}

impl CellNoise {
    fn new(noise: &Perlin, lo: Vec3, salt: u32) -> Self {
        let s = salt as f32 * 17.13;
        let mut corners = [0.0; 8];
        for (i, c) in corners.iter_mut().enumerate() {
            let p = lo + Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, (i >> 2) as f32) * 0.5;
            *c = noise.noise3(p.x * 0.9 + s, p.y * 0.9 - s, p.z * 0.9 + s * 0.5);
        }
        Self { lo, corners }
    }

    fn at(&self, p: Vec3) -> f32 {
        let f = ((p - self.lo) * 2.0).clamp(Vec3::ZERO, Vec3::ONE);
        let c = &self.corners;
        let x0 = c[0] + (c[1] - c[0]) * f.x;
        let x1 = c[2] + (c[3] - c[2]) * f.x;
        let x2 = c[4] + (c[5] - c[4]) * f.x;
        let x3 = c[6] + (c[7] - c[6]) * f.x;
        let y0 = x0 + (x1 - x0) * f.y;
        let y1 = x2 + (x3 - x2) * f.y;
        y0 + (y1 - y0) * f.z
    }
}

impl Prim {
    /// `sample` with the blob's edge noise already known for the cell.
    fn sample_in(
        &self,
        cell_noise: Option<&CellNoise>,
        p: Vec3,
        voxel: IVec3,
    ) -> Option<MaterialId> {
        match *self {
            Prim::Wood { a, b, ra, rb, m } => {
                let (d, t) = segment(p, a, b);
                (d <= ra + (rb - ra) * t).then_some(m)
            }
            Prim::Blob {
                c,
                r,
                m,
                fill,
                cap,
                salt,
            } => {
                let n = cell_noise.map_or(0.0, |cn| cn.at(p));
                let depth = ((p - c) / r).length() - n * 0.15;
                if depth > 1.0 {
                    return None;
                }
                if depth > 0.72 {
                    let h = hash3(voxel.x, voxel.y, voxel.z, salt as u64 ^ 0x1eaf);
                    if hash_unit(h) > fill {
                        return None;
                    }
                }
                if let Some(cap) = cap {
                    let above = p + Vec3::Y * 0.14;
                    let n_above = cell_noise.map_or(0.0, |cn| cn.at(above));
                    if ((above - c) / r).length() - n_above * 0.15 > 1.0 {
                        return Some(cap);
                    }
                }
                Some(m)
            }
        }
    }
}

/// Stamps plants into a chunk generated voxel by voxel.
pub fn stamp_full(tree: &mut ChunkTree, origin: IVec3, plants: &[Plant], noise: &Perlin) {
    let origin_m = origin.as_vec3() * VOXEL_M;
    for plant in plants {
        let lo = ((plant.lo - origin_m) * 2.0)
            .floor()
            .as_ivec3()
            .max(IVec3::ZERO);
        let hi = ((plant.hi - origin_m) * 2.0)
            .floor()
            .as_ivec3()
            .min(IVec3::splat(63));
        for cy in lo.y..=hi.y {
            for cz in lo.z..=hi.z {
                for cx in lo.x..=hi.x {
                    let cell = IVec3::new(cx, cy, cz);
                    let cell_lo = origin_m + cell.as_vec3() * 0.5;
                    let centre = cell_lo + 0.25;
                    match plant.relate(centre, 0.25) {
                        Relation::Outside => continue,
                        Relation::Inside(m) if tree.cell(cell) == Cell::Empty => {
                            tree.set_cell(cell, Cell::Uniform(m));
                            continue;
                        }
                        _ => {}
                    }
                    // Only the parts reaching this cell, wood first, each
                    // blob with its edge noise for the cell.
                    let cell_hi = cell_lo + 0.5;
                    let mut parts: Vec<(&Prim, Option<CellNoise>)> = plant
                        .prims
                        .iter()
                        .filter(|prim| {
                            let (a, b) = prim.bounds();
                            a.cmple(cell_hi).all() && b.cmpge(cell_lo).all()
                        })
                        .map(|prim| {
                            let cn = match *prim {
                                Prim::Blob { salt, .. } => {
                                    Some(CellNoise::new(noise, cell_lo, salt))
                                }
                                Prim::Wood { .. } => None,
                            };
                            (prim, cn)
                        })
                        .collect();
                    if parts.is_empty() {
                        continue;
                    }
                    parts.sort_by_key(|(prim, _)| !prim.is_wood());
                    let base = origin + cell * 8;
                    tree.edit_brick(cell, |brick| {
                        for i in 0..512 {
                            let l = IVec3::new(i & 7, (i >> 6) & 7, (i >> 3) & 7);
                            if !replaceable(brick.get(l)) {
                                continue;
                            }
                            let v = base + l;
                            let p = (v.as_vec3() + 0.5) * VOXEL_M;
                            for (prim, cn) in &parts {
                                if let Some(m) = prim.sample_in(cn.as_ref(), p, v) {
                                    brick.set(l, m);
                                    break;
                                }
                            }
                        }
                    });
                }
            }
        }
    }
}

/// Stamps plants into a chunk kept at one material per cell of `cells`
/// brick cells (1: 0.5 m cells, 4: 2 m nodes). At 2 m only crowns show.
pub fn stamp_coarse(tree: &mut ChunkTree, origin: IVec3, plants: &[Plant], cells: i32) {
    let origin_m = origin.as_vec3() * VOXEL_M;
    let size = cells as f32 * 0.5;
    let per_axis = 64 / cells;
    for plant in plants {
        let lo = ((plant.lo - origin_m) / size)
            .floor()
            .as_ivec3()
            .max(IVec3::ZERO);
        let hi = ((plant.hi - origin_m) / size)
            .floor()
            .as_ivec3()
            .min(IVec3::splat(per_axis - 1));
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                for x in lo.x..=hi.x {
                    let c = IVec3::new(x, y, z);
                    let centre = origin_m + (c.as_vec3() + 0.5) * size;
                    let Some(m) = plant.sample_coarse(centre, size) else {
                        continue;
                    };
                    if cells == 1 {
                        if tree.cell(c) == Cell::Empty {
                            tree.set_cell(c, Cell::Uniform(m));
                        }
                    } else if m.get().kind == Kind::Foliage || m == ids::SNOW {
                        tree.set_node(1, c, m);
                    }
                }
            }
        }
    }
}
