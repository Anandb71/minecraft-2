//! Amplification: voxel-scale surface detail added under the constraint of
//! the eroded coarse terrain.
//!
//! The coarse field fixes the large forms and drainage. Detail is zero-mean
//! noise whose amplitude follows what the coarse field says about a place:
//! crags on steep eroded slopes, gentle swells on plains, nearly nothing on
//! valley floors where discharge is high, so detail never dams a river the
//! erosion carved. Surface materials follow slope, altitude, sediment and
//! water, then give way to the strata below.

use crate::flora::{Biome, Climate};
use crate::noise::Perlin;
use crate::strata::Strata;
use crate::terrain::CoarseTerrain;
use mc2_voxel::material::{MaterialId, ids};
use std::sync::Arc;

pub const SNOWLINE_M: f32 = 390.0;
pub const TREELINE_M: f32 = 330.0;

pub struct Surface {
    pub terrain: Arc<CoarseTerrain>,
    detail: Perlin,
    climate: Climate,
    flow_scale: f32,
}

/// Everything about a surface column that chunk generation needs.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceSample {
    pub height: f32,
    /// Rise over run of the coarse surface.
    pub slope: f32,
    /// 0 away from water, 1 in the strongest channels.
    pub river: f32,
    /// Loose material above the rock: soil, alluvium, beach sand.
    pub soil_depth: f32,
    /// Climate, each roughly 0..1: temperature (falling with altitude) and
    /// moisture. Both interpolate, so samples blend across a brick.
    pub temp: f32,
    pub wet: f32,
}

impl Surface {
    pub fn new(terrain: Arc<CoarseTerrain>) -> Self {
        let mut sorted: Vec<f32> = terrain.flow.data.clone();
        sorted.sort_by(f32::total_cmp);
        let flow_scale = sorted[sorted.len() * 99 / 100].max(1e-6);
        let detail = Perlin::new(terrain.seed ^ 0x00a3_f11d);
        let climate = Climate::new(terrain.seed);
        Self {
            terrain,
            detail,
            climate,
            flow_scale,
        }
    }

    pub fn sample(&self, x: f32, z: f32) -> SurfaceSample {
        let t = &self.terrain;
        let c = t.params.cell_m;
        let base = t.height_at(x, z);
        let gx = (t.height_at(x + c, z) - t.height_at(x - c, z)) / (2.0 * c);
        let gz = (t.height_at(x, z + c) - t.height_at(x, z - c)) / (2.0 * c);
        let slope = (gx * gx + gz * gz).sqrt();
        let river = (t.flow_at(x, z) / self.flow_scale).clamp(0.0, 1.0);
        let calm = 1.0 - river;

        let steep = (slope / 0.7).clamp(0.0, 1.0);
        let n = &self.detail;
        // Crags on steep ground, swells on plains, rubble at every scale.
        let crag = n.ridged2(x / 70.0, z / 70.0, 3, 2.1, 0.5) - 0.45;
        let swell = n.fbm2(x / 110.0, z / 110.0, 3, 2.0, 0.5);
        let rough = n.fbm2(x / 9.0, z / 9.0, 2, 2.0, 0.5);
        let height = base
            + calm * (steep * crag * 14.0 + (1.0 - steep) * swell * 2.5)
            + calm * rough * (0.25 + steep * 0.8);

        let sediment = t.sediment_at(x, z);
        let soil_depth = if slope > 1.0 {
            0.0
        } else {
            ((1.0 - slope) * 1.6 + sediment.min(6.0)) * calm.max(0.3)
        };
        let (temp, wet) = self.climate.at(x, z, height, t.params.sea_level, river);
        SurfaceSample {
            height,
            slope,
            river,
            soil_depth,
            temp,
            wet,
        }
    }

    /// Material of a voxel `depth` metres below the surface at elevation `y`.
    pub fn loose_material(&self, s: &SurfaceSample, depth: f32, y: f32) -> MaterialId {
        let sea = self.terrain.params.sea_level;
        if s.height < sea - 0.3 {
            // The sea floor.
            return if depth < 0.8 && s.slope < 0.4 {
                ids::SAND
            } else {
                ids::GRAVEL
            };
        }
        if s.height < sea + 1.5 && s.slope < 0.25 {
            return if depth < 0.9 { ids::SAND } else { ids::CLAY };
        }
        if s.river > 0.5 {
            return if depth < 0.6 { ids::GRAVEL } else { ids::SAND };
        }
        if y > SNOWLINE_M && depth < 0.5 {
            return ids::SNOW;
        }
        match s.biome(sea) {
            Biome::Desert => return ids::SAND,
            Biome::SnowyTaiga if depth < 0.12 && s.slope < 0.8 => return ids::SNOW,
            _ => {}
        }
        if depth < 0.2 && y < TREELINE_M && s.slope < 0.9 {
            return ids::GRASS;
        }
        if s.slope > 0.8 || y > TREELINE_M {
            return ids::GRAVEL;
        }
        ids::DIRT
    }
}

/// Material at a point given its column's surface and strata.
pub fn material_at(
    surface: &Surface,
    s: &SurfaceSample,
    column: &crate::strata::Column<'_>,
    y: f32,
) -> MaterialId {
    let depth = s.height - y;
    if depth < 0.0 {
        // Above the ground: the sea up to its level, then air.
        return if y < surface.terrain.params.sea_level {
            ids::WATER
        } else {
            MaterialId(0)
        };
    }
    if depth < s.soil_depth {
        return surface.loose_material(s, depth, y);
    }
    column.layer(y).material
}

/// Convenience for tests and tools: full lookup at a world point.
pub fn material_at_point(surface: &Surface, strata: &Strata, x: f32, y: f32, z: f32) -> MaterialId {
    let s = surface.sample(x, z);
    material_at(surface, &s, &strata.column(x, z), y)
}
