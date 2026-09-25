//! The sound of a place, measured by tracing it.
//!
//! Rays from the listener through the voxels, spread evenly over the sphere,
//! find how much of the sky is open, how far sound goes before meeting a
//! surface (the mean free path) and how much those surfaces swallow. Sabine's
//! formula turns that into a decay time; the nearest surfaces give the first
//! echoes. Between a source and the listener, the world's own material says
//! how much is lost and how dull what gets through becomes.

use glam::{DVec3, Vec3};
use mc2_voxel::march::raycast_filtered;
use mc2_voxel::material::{Kind, MaterialId};
use mc2_voxel::world::VoxelWorld;

/// Speed of sound, metres a second.
pub const SOUND: f32 = 343.0;
/// How far a probe ray looks for a surface, metres.
const REACH: f64 = 48.0;
/// Rays in a probe.
pub const RAYS: usize = 64;
/// Echoes kept from the nearest surfaces.
const TAPS: usize = 6;
/// Loss through matter: any at all costs this much, dB, and each metre
/// this much more, up to the most: a pane of glass takes about 22 dB, a
/// house wall 35 dB, a metre of rock all of it.
const DB_ANY: f32 = 18.0;
const DB_PER_M: f32 = 70.0;
const MOST_DB: f32 = 60.0;
/// Step along a source's path when weighing what is in the way, metres.
const STEP: f64 = 0.125;

/// Whether a material turns sound back: anything but air and flames.
fn reflects(m: MaterialId) -> bool {
    let mat = m.get();
    mat.kind != Kind::Air && mat.absorption > 0.0
}

/// An early echo: how long after the sound, how loud, from where.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tap {
    pub delay: f32,
    pub gain: f32,
    pub dir: Vec3,
}

/// What a place sounds like.
#[derive(Clone, Debug, PartialEq)]
pub struct Acoustics {
    /// Seconds for the reverb to fall 60 dB.
    pub rt60: f32,
    /// How much of the sound is reverb, 0..1.
    pub wet: f32,
    /// The reverb's size (1 for a four-metre mean free path).
    pub size: f32,
    /// The reverb's damping cutoff, Hz: soft surfaces dull it.
    pub damping: f32,
    /// Share of rays that met nothing: the open sky.
    pub open: f32,
    /// Mean distance between surfaces, metres.
    pub mean_free_path: f32,
    /// Mean absorption of the surfaces met.
    pub absorption: f32,
    pub taps: Vec<Tap>,
}

impl Default for Acoustics {
    /// Out in the open.
    fn default() -> Self {
        Acoustics {
            rt60: 0.2,
            wet: 0.0,
            size: 1.0,
            damping: 8000.0,
            open: 1.0,
            mean_free_path: REACH as f32,
            absorption: 0.0,
            taps: Vec::new(),
        }
    }
}

/// `n` directions spread evenly over the sphere.
pub fn sphere(n: usize) -> impl Iterator<Item = DVec3> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    (0..n).map(move |i| {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let r = (1.0 - y * y).sqrt();
        let a = golden * i as f64;
        DVec3::new(a.cos() * r, y, a.sin() * r)
    })
}

/// Measures the place around `at`.
pub fn measure(world: &VoxelWorld, at: DVec3) -> Acoustics {
    let mut hits: Vec<(f32, f32, Vec3)> = Vec::with_capacity(RAYS);
    for dir in sphere(RAYS) {
        if let Some(h) = raycast_filtered(world, at, dir, REACH, reflects) {
            hits.push((h.t as f32, h.material.get().absorption, dir.as_vec3()));
        }
    }
    let n = RAYS as f32;
    let open = (RAYS - hits.len()) as f32 / n;
    if hits.is_empty() {
        return Acoustics::default();
    }
    let mean_free_path = hits.iter().map(|h| h.0).sum::<f32>() / hits.len() as f32;
    let absorption = hits.iter().map(|h| h.1).sum::<f32>() / hits.len() as f32;
    // Sabine: RT60 = 0.161 V / (S a), with V / S a quarter of the mean
    // free path; open sky counts as a surface that swallows everything.
    let swallowed = (absorption * (1.0 - open) + open).max(0.01);
    let rt60 = (0.161 * mean_free_path / 4.0 / swallowed).clamp(0.1, 8.0);
    let mut near = hits.clone();
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    let taps = near
        .iter()
        .filter(|h| h.0 < 25.0)
        .take(TAPS)
        .map(|&(d, a, dir)| Tap {
            delay: 2.0 * d / SOUND,
            gain: 0.7 * (1.0 - a) / (1.0 + d),
            dir,
        })
        .collect();
    Acoustics {
        rt60,
        wet: 0.6 * (1.0 - open).powi(3),
        size: (mean_free_path / 4.0).clamp(0.3, 3.0),
        damping: 12_000.0 - 9_500.0 * absorption.clamp(0.0, 1.0),
        open,
        mean_free_path,
        absorption,
        taps,
    }
}

/// What reaches the listener from a source: how loud (0..1) and how much
/// of the highs survive (a low-pass cutoff, Hz).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Occlusion {
    pub gain: f32,
    pub cutoff: f32,
}

impl Occlusion {
    pub const CLEAR: Occlusion = Occlusion {
        gain: 1.0,
        cutoff: 16_000.0,
    };

    fn through(metres: f32) -> Occlusion {
        if metres <= 0.0 {
            return Occlusion::CLEAR;
        }
        let db = (DB_ANY + metres * DB_PER_M).min(MOST_DB);
        Occlusion {
            gain: 10f32.powf(-db / 20.0),
            cutoff: (16_000.0 * 0.5f32.powf(metres / 0.15)).clamp(250.0, 16_000.0),
        }
    }
}

/// Metres of matter that turns sound back along the segment `a..b`.
fn matter(world: &VoxelWorld, a: DVec3, b: DVec3) -> f32 {
    let len = a.distance(b);
    let steps = (len / STEP).ceil().max(1.0) as usize;
    let solid = (0..steps)
        .filter(|&i| {
            let p = a.lerp(b, (i as f64 + 0.5) / steps as f64);
            reflects(world.voxel((p * 16.0).floor().as_ivec3()))
        })
        .count();
    (solid as f64 * len / steps as f64) as f32
}

/// How a source at `source` is heard at `listener`: straight through what
/// lies between, or round it (over the top or past either side, a little
/// quieter for the longer way), whichever loses least.
pub fn occlusion(world: &VoxelWorld, source: DVec3, listener: DVec3) -> Occlusion {
    let direct = matter(world, source, listener);
    if direct == 0.0 {
        return Occlusion::CLEAR;
    }
    let mut best = Occlusion::through(direct);
    let mid = source.lerp(listener, 0.5);
    let along = (listener - source).normalize_or_zero();
    let side = along.cross(DVec3::Y).normalize_or_zero();
    for offset in [DVec3::Y * 3.0, side * 3.0, -side * 3.0, DVec3::Y * 6.0] {
        let via = mid + offset;
        let m = matter(world, source, via) + matter(world, via, listener);
        let mut o = Occlusion::through(m);
        // The way round is longer: a little quieter, and duller.
        o.gain *= 0.6;
        o.cutoff = o.cutoff.min(4_000.0);
        if o.gain > best.gain {
            best = o;
        }
    }
    best
}

/// Whether anything at all of a sound at `source` reaches `listener`
/// (for ears other than the player's).
pub fn audible(world: &VoxelWorld, source: DVec3, listener: DVec3, loudness: f32) -> bool {
    let d = source.distance(listener) as f32;
    let o = occlusion(world, source, listener);
    loudness * o.gain / (1.0 + d) > 0.01
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;
    use mc2_voxel::material::ids;

    /// A closed room of `wall` 8 m by 4 m by 8 m inside, its floor at 2 m,
    /// walls a quarter metre thick, on open ground.
    fn room(wall: MaterialId) -> VoxelWorld {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(511, 31, 511), ids::GRANITE);
        w.fill_box(IVec3::new(96, 28, 96), IVec3::new(235, 99, 235), wall);
        w.fill_box(IVec3::new(100, 32, 100), IVec3::new(231, 95, 231), ids::AIR);
        w
    }

    const CENTRE: DVec3 = DVec3::new(10.4, 3.7, 10.4);

    #[test]
    fn a_closed_stone_room_rings_and_a_woollen_one_does_not() {
        let stone = measure(&room(ids::GRANITE), CENTRE);
        assert!(stone.open < 0.02, "{stone:?}");
        assert!(stone.rt60 > 2.0, "stone rt60 {}", stone.rt60);
        assert!(stone.wet > 0.5);
        let wool = measure(&room(ids::WOOL), CENTRE);
        assert!(
            wool.rt60 < stone.rt60 * 0.25,
            "wool {} vs stone {}",
            wool.rt60,
            stone.rt60
        );
        assert!(wool.damping < stone.damping);
        // The first echoes come off the nearest walls, two metres away.
        let first = stone.taps.first().unwrap();
        assert!((first.delay - 2.0 * 1.7 / SOUND).abs() < 0.004, "{first:?}");
    }

    #[test]
    fn open_ground_is_dry() {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(511, 31, 511), ids::GRASS);
        let a = measure(&w, DVec3::new(16.0, 3.7, 16.0));
        assert!(a.open > 0.4, "{a:?}");
        assert!(a.wet < 0.15, "{a:?}");
        assert!(a.rt60 < 0.5, "{a:?}");
    }

    #[test]
    fn a_wall_muffles_and_a_way_round_leaks() {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(511, 31, 511), ids::GRANITE);
        let (s, l) = (DVec3::new(6.0, 3.0, 16.0), DVec3::new(14.0, 3.0, 16.0));
        assert_eq!(occlusion(&w, s, l), Occlusion::CLEAR);
        // A 1 m stone wall between, 2 m high: the sound goes over it.
        w.fill_box(
            IVec3::new(144, 32, 0),
            IVec3::new(159, 63, 511),
            ids::GRANITE,
        );
        let over = occlusion(&w, s, l);
        assert!(over.gain < 0.7 && over.gain > 0.3, "{over:?}");
        // Built up to 10 m, there is no way round: it is muffled hard.
        w.fill_box(
            IVec3::new(144, 32, 0),
            IVec3::new(159, 191, 511),
            ids::GRANITE,
        );
        let through = occlusion(&w, s, l);
        assert!(through.gain < 0.05, "{through:?}");
        assert!(through.cutoff <= 400.0, "{through:?}");
        assert!(!audible(&w, s, l, 0.2));
        assert!(audible(&w, s, l, 100.0));
    }
}
