//! The coarse whole-world terrain: uplift-shaped base heights, then erosion.
//!
//! Erosion does not decompose into tiles, so it runs once over the entire
//! 16 km world at low resolution when the world is created and the result is
//! cached. Chunk generation later amplifies this field locally.

use crate::erosion::{ErosionParams, erode};
use crate::grid::Grid2;
use crate::noise::Perlin;

#[derive(Clone, Copy, Debug)]
pub struct TerrainParams {
    /// Cells per side of the coarse grid.
    pub size: usize,
    /// Metres per coarse cell.
    pub cell_m: f32,
    pub sea_level: f32,
    pub erosion: ErosionParams,
}

impl TerrainParams {
    /// The shipping world: 1024^2 cells of 16 m over 16 384 m.
    pub fn world() -> Self {
        Self {
            size: 1024,
            cell_m: 16.0,
            sea_level: 96.0,
            erosion: ErosionParams::default(),
        }
    }

    pub fn extent_m(&self) -> f32 {
        self.size as f32 * self.cell_m
    }
}

pub struct CoarseTerrain {
    pub seed: u64,
    pub params: TerrainParams,
    /// Surface elevation in metres.
    pub height: Grid2,
    /// Time-averaged water discharge from erosion; high along rivers.
    pub flow: Grid2,
    /// Thickness of deposited sediment in metres (valley fill, alluvium).
    pub sediment: Grid2,
}

/// Base elevation before erosion, metres, at world position `(x, z)`.
pub fn base_height(noise: &Perlin, seed: u64, p: &TerrainParams, x: f32, z: f32) -> f32 {
    let extent = p.extent_m();
    // A continent that falls to ocean within 2 km of the world edge.
    let edge = x.min(z).min(extent - x).min(extent - z);
    let edge_falloff = ((edge - 400.0) / 1800.0).clamp(0.0, 1.0);
    let edge_falloff = edge_falloff * edge_falloff * (3.0 - 2.0 * edge_falloff);
    let continent = noise.fbm2(x / 7000.0 + 3.1, z / 7000.0 - 1.7, 4, 2.0, 0.5) * 0.9 + 0.62;
    let land = (continent * edge_falloff).clamp(-0.3, 1.0);

    let mut h = p.sea_level - 45.0 + land * 120.0;

    let interior = ((land - 0.25) / 0.5).clamp(0.0, 1.0);
    h += crate::tectonics::uplift(noise, seed, x, z, interior);

    // Broad rolling hills on land; fine detail is added after erosion.
    let hills = noise.fbm2(x / 1400.0 + 12.0, z / 1400.0 - 5.0, 3, 2.0, 0.5);
    h += hills * 28.0 * interior.max(0.2);

    h.clamp(10.0, 500.0)
}

impl CoarseTerrain {
    pub fn generate(seed: u64, params: TerrainParams, progress: &mut dyn FnMut(&str, f32)) -> Self {
        let noise = Perlin::new(seed);
        progress("uplift", 0.0);
        let height = Grid2::from_fn(params.size, params.size, |i, j| {
            let x = (i as f32 + 0.5) * params.cell_m;
            let z = (j as f32 + 0.5) * params.cell_m;
            base_height(&noise, seed, &params, x, z)
        });
        let strata = crate::strata::Strata::new(seed);
        let out = erode(height, &strata, &params, progress);
        Self {
            seed,
            params,
            height: out.height,
            flow: out.flow,
            sediment: out.sediment,
        }
    }

    /// Bicubic elevation at a world position in metres.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let c = self.params.cell_m;
        self.height.bicubic(x / c - 0.5, z / c - 0.5)
    }

    pub fn flow_at(&self, x: f32, z: f32) -> f32 {
        let c = self.params.cell_m;
        self.flow.bilinear(x / c - 0.5, z / c - 0.5)
    }

    pub fn sediment_at(&self, x: f32, z: f32) -> f32 {
        let c = self.params.cell_m;
        self.sediment.bilinear(x / c - 0.5, z / c - 0.5)
    }
}
