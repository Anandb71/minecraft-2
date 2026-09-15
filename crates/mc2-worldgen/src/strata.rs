//! Stratigraphy: the rock column under every point of the world.
//!
//! A crystalline basement (gneiss with granite plutons) sits on bedrock. Above
//! it a sedimentary sequence repeats every period with per-cycle thickness
//! variation, the way transgression and regression cycles stack real basins.
//! Layers are dipped and gently folded by slow noise so a canyon wall shows
//! tilted bands, and volcanic provinces interleave basalt flows and rhyolite
//! tuff. The layer index is monotonic in elevation for a fixed column, which
//! lets chunk generation prove a node is one material by checking its top and
//! bottom.

use crate::noise::{Perlin, hash_unit, hash3};
use mc2_voxel::material::{MaterialId, ids};

const BEDROCK_M: f32 = 3.0;
const PERIOD_M: f32 = 72.0;

/// Sedimentary cycle, bottom to top, with nominal thickness fractions.
const SEQUENCE: &[(MaterialId, f32)] = &[
    (ids::CONGLOMERATE, 0.06),
    (ids::SANDSTONE, 0.18),
    (ids::SHALE, 0.10),
    (ids::LIMESTONE, 0.22),
    (ids::SANDSTONE, 0.12),
    (ids::CHALK, 0.08),
    (ids::SHALE, 0.08),
    (ids::LIMESTONE, 0.16),
];

pub struct Strata {
    fold: Perlin,
    region: Perlin,
    seed: u64,
    /// Cumulative normalised layer tops for the first cycles, so a lookup
    /// does no hashing. Deeper cycles fall back to computing them.
    cycles: Vec<[f32; 8]>,
}

const CACHED_CYCLES: i32 = 16;

fn cycle_bounds(seed: u64, cycle: i32) -> [f32; 8] {
    let mut weights = [0.0f32; 8];
    let mut total = 0.0;
    for (j, &(_, frac)) in SEQUENCE.iter().enumerate() {
        let jitter = 0.6 + 0.8 * hash_unit(hash3(cycle, j as i32, 0, seed));
        weights[j] = frac * jitter;
        total += weights[j];
    }
    let mut acc = 0.0;
    let mut out = [0.0f32; 8];
    for (o, w) in out.iter_mut().zip(weights) {
        acc += w / total;
        *o = acc;
    }
    out[7] = f32::INFINITY;
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerRef {
    /// Monotonic in elevation within a column.
    pub index: u32,
    pub material: MaterialId,
}

impl Strata {
    pub fn new(seed: u64) -> Self {
        Self {
            fold: Perlin::new(seed ^ 0x5717_a7a0),
            region: Perlin::new(seed ^ 0x09e6_1011),
            seed,
            cycles: (0..CACHED_CYCLES).map(|c| cycle_bounds(seed, c)).collect(),
        }
    }

    /// Vertical offset applied to elevation before layer lookup: a regional
    /// dip plus folds with a few hundred metres wavelength.
    pub fn warp(&self, x: f32, z: f32) -> f32 {
        let dip = self.fold.noise2(x / 6000.0, z / 6000.0) * 60.0;
        let folds = self.fold.fbm2(x / 900.0, z / 900.0, 3, 2.0, 0.5) * 14.0;
        dip + folds
    }

    /// Top of the crystalline basement in warped elevation.
    pub fn basement_top(&self, x: f32, z: f32) -> f32 {
        28.0 + self.region.noise2(x / 3000.0, z / 3000.0) * 20.0
    }

    /// 0 = sedimentary basin, 1 = volcanic province.
    pub fn volcanism(&self, x: f32, z: f32) -> f32 {
        (self
            .region
            .fbm2(x / 5000.0 + 7.3, z / 5000.0 - 2.1, 3, 2.0, 0.5)
            * 1.6
            + 0.3)
            .clamp(0.0, 1.0)
    }

    /// Layer containing warped elevation `yw` in column `(x, z)`.
    fn pluton(&self, x: f32, z: f32) -> bool {
        self.region.noise2(x / 400.0 + 91.0, z / 400.0 - 17.0) > 0.25
    }

    fn layer_warped(&self, x: f32, yw: f32, z: f32, volcanism: f32, basement: f32) -> LayerRef {
        self.layer_with(yw, volcanism, basement, || self.pluton(x, z))
    }

    fn layer_with(
        &self,
        yw: f32,
        volcanism: f32,
        basement: f32,
        pluton: impl Fn() -> bool,
    ) -> LayerRef {
        if yw < BEDROCK_M {
            return LayerRef {
                index: 0,
                material: ids::BEDROCK,
            };
        }
        if yw < basement {
            // Granite plutons intrude the gneiss. Constant along a column so
            // the monotonic index still identifies one material per column.
            return LayerRef {
                index: 1,
                material: if pluton() { ids::GRANITE } else { ids::GNEISS },
            };
        }
        let above = yw - basement;
        let cycle = (above / PERIOD_M).floor() as i32;
        let t = above / PERIOD_M - cycle as f32;
        // Per-cycle thickness variation, renormalised to the period.
        let computed;
        let bounds = match self.cycles.get(cycle as usize) {
            Some(b) if cycle >= 0 => b,
            _ => {
                computed = cycle_bounds(self.seed, cycle);
                &computed
            }
        };
        let j = bounds
            .iter()
            .position(|&top| t < top)
            .unwrap_or(SEQUENCE.len() - 1);
        let mut material = SEQUENCE[j].0;
        if volcanism > 0.55 {
            // Flows replace the limestones; tuffs replace the chalk.
            material = match material {
                m if m == ids::LIMESTONE => ids::BASALT,
                m if m == ids::CHALK => ids::RHYOLITE,
                m => m,
            };
        }
        LayerRef {
            index: 2 + cycle.max(0) as u32 * SEQUENCE.len() as u32 + j as u32,
            material,
        }
    }

    pub fn layer(&self, x: f32, y: f32, z: f32) -> LayerRef {
        let yw = y + self.warp(x, z);
        self.layer_warped(x, yw, z, self.volcanism(x, z), self.basement_top(x, z))
    }

    /// Column-constant parts of the lookup, for callers sampling many
    /// elevations in one column.
    pub fn column(&self, x: f32, z: f32) -> Column<'_> {
        Column {
            strata: self,
            warp: self.warp(x, z),
            volcanism: self.volcanism(x, z),
            basement: self.basement_top(x, z),
            pluton: self.pluton(x, z),
        }
    }

    pub fn hardness(&self, x: f32, y: f32, z: f32) -> f32 {
        self.layer(x, y, z).material.get().hardness
    }
}

pub struct Column<'a> {
    strata: &'a Strata,
    warp: f32,
    volcanism: f32,
    basement: f32,
    pluton: bool,
}

impl Column<'_> {
    pub fn layer(&self, y: f32) -> LayerRef {
        let pluton = self.pluton;
        self.strata
            .layer_with(y + self.warp, self.volcanism, self.basement, || pluton)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_index_is_monotonic_in_elevation() {
        let s = Strata::new(11);
        for (x, z) in [(100.0, 200.0), (5000.0, 9000.0), (12000.0, 300.0)] {
            let col = s.column(x, z);
            let mut prev = 0;
            for i in 0..5120 {
                let l = col.layer(i as f32 * 0.1);
                assert!(l.index >= prev, "index fell at y={}", i as f32 * 0.1);
                prev = l.index;
            }
            assert!(prev > 20, "a 512 m column crosses many layers");
        }
    }

    #[test]
    fn column_matches_point_lookup() {
        let s = Strata::new(5);
        let col = s.column(777.0, 4242.0);
        for y in [1.0, 20.0, 64.5, 200.0, 480.0] {
            assert_eq!(col.layer(y), s.layer(777.0, y, 4242.0));
        }
    }

    #[test]
    fn sequence_contains_several_rock_types() {
        let s = Strata::new(3);
        let col = s.column(1000.0, 1000.0);
        let mut seen = Vec::new();
        for i in 0..512 {
            let m = col.layer(i as f32).material;
            if !seen.contains(&m) {
                seen.push(m);
            }
        }
        assert!(seen.len() >= 5, "only {seen:?}");
    }
}
