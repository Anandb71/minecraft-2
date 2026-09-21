//! Karst caves: water dissolving carbonate rock along bedding and joints.
//!
//! Noise-worm tunnels were rejected in step 4. These passages follow two
//! vertical joint sets and a bedding plane, so they open in limestone, chalk
//! and marble, stay mostly horizontal, and meet in rooms. Lower passages
//! flood to a water table. Sinkholes punch down where a river sits over a
//! shaft.

use crate::amplify::Surface;
use crate::chunkgen::Lod;
use crate::noise::Perlin;
use crate::strata::Strata;
use glam::{IVec3, Vec3};
use mc2_voxel::coords::{CHUNK_VOXELS, ChunkPos};
use mc2_voxel::material::{MaterialId, ids};
use mc2_voxel::tree::{Cell, ChunkTree};

const VOXEL_M: f32 = 1.0 / 16.0;
/// Joint half-width in noise units. Wider is more Swiss cheese.
const JOINT: f32 = 0.13;
const OPEN_AT: f32 = 0.46;
const ROOF_M: f32 = 3.0;
const FLOOR_M: f32 = 88.0;

pub struct Karst {
    noise: Perlin,
}

pub fn soluble(m: MaterialId) -> bool {
    m == ids::LIMESTONE || m == ids::CHALK || m == ids::MARBLE
}

impl Karst {
    pub fn new(seed: u64) -> Self {
        Self {
            noise: Perlin::new(seed ^ 0x0ca7_e501),
        }
    }

    fn joints(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let n = &self.noise;
        let bed = n
            .noise3(x * (1.0 / 46.0), y * (1.0 / 6.2), z * (1.0 / 46.0))
            .abs();
        let jx = n
            .noise3(x * (1.0 / 8.4), y * (1.0 / 40.0), z * (1.0 / 54.0) + 19.0)
            .abs();
        let jz = n
            .noise3(x * (1.0 / 54.0) + 11.0, y * (1.0 / 40.0), z * (1.0 / 8.4))
            .abs();
        (bed, jx, jz)
    }

    /// 0 closed, 1 fully open. Independent of the host rock.
    pub fn openness(&self, x: f32, y: f32, z: f32, surface: f32) -> f32 {
        let depth = surface - y;
        if !(ROOF_M..=FLOOR_M).contains(&depth) || y < 8.0 {
            return 0.0;
        }
        let (bed, jx, jz) = self.joints(x, y, z);
        let corridor = (1.0 - (jx / JOINT).clamp(0.0, 1.0)) * (1.0 - (jz / JOINT).clamp(0.0, 1.0));
        let chamber = corridor * (1.0 - (bed / (JOINT + 0.05)).clamp(0.0, 1.0));
        let room = self
            .noise
            .fbm3(x / 72.0, y / 30.0, z / 72.0, 3, 2.0, 0.5)
            .max(0.0);
        (corridor * 0.9 + chamber * 0.5 + room * 0.18).min(1.0)
    }

    fn sinkhole_shaft(&self, p: Vec3, surface: f32, sea: f32, river: f32) -> bool {
        if river < 0.62 || surface <= sea + 4.0 {
            return false;
        }
        if p.y < 8.0 || p.y > surface || p.y < surface - 22.0 {
            return false;
        }
        let (_, jx, jz) = self.joints(p.x, surface - 8.0, p.z);
        jx <= 0.11 && jz <= 0.11
    }

    /// What a soluble voxel becomes, or `None` to leave it.
    pub fn dissolve(
        &self,
        host: MaterialId,
        p: Vec3,
        surface: f32,
        sea: f32,
        river: f32,
    ) -> Option<MaterialId> {
        if !soluble(host) {
            return None;
        }
        let joint = self.openness(p.x, p.y, p.z, surface) >= OPEN_AT;
        if !joint && !self.sinkhole_shaft(p, surface, sea, river) {
            return None;
        }
        let water_table = sea.max(surface - 26.0);
        Some(if p.y < water_table {
            ids::WATER
        } else {
            ids::AIR
        })
    }

    /// Carves passages into an already-generated tree. Node8 skips: caves
    /// thinner than 8 m would alias into Swiss cheese at that lod.
    pub fn carve(
        &self,
        tree: &mut ChunkTree,
        pos: ChunkPos,
        surface: &Surface,
        strata: &Strata,
        lod: Lod,
    ) {
        if lod == Lod::Node8 {
            return;
        }
        let step = match lod {
            Lod::Node2 => 64,
            _ => 32,
        };
        let origin = pos.origin();
        let sea = surface.terrain.params.sea_level;
        let n = CHUNK_VOXELS / step;
        for cz in 0..n {
            for cy in 0..n {
                for cx in 0..n {
                    let centre = origin + IVec3::new(cx, cy, cz) * step + step / 2;
                    let p = centre.as_vec3() * VOXEL_M;
                    let sample = surface.sample(p.x, p.z);
                    let host = strata.layer(p.x, p.y, p.z).material;
                    let open = self.openness(p.x, p.y, p.z, sample.height);
                    if open < OPEN_AT || !soluble(host) {
                        continue;
                    }
                    let radius_m = 1.15 + 2.6 * open;
                    if lod == Lod::Node2 && radius_m < 3.2 {
                        continue;
                    }
                    self.carve_ellipsoid(tree, origin, centre, radius_m, sample.height, sea);
                }
            }
        }
        self.carve_sinkholes(tree, origin, surface, sea, lod);
    }

    fn carve_sinkholes(
        &self,
        tree: &mut ChunkTree,
        origin: IVec3,
        surface: &Surface,
        sea: f32,
        lod: Lod,
    ) {
        if lod == Lod::Node2 {
            return;
        }
        let step = 32;
        let n = CHUNK_VOXELS / step;
        for cz in 0..n {
            for cx in 0..n {
                let x = (origin.x + cx * step + step / 2) as f32 * VOXEL_M;
                let z = (origin.z + cz * step + step / 2) as f32 * VOXEL_M;
                let sample = surface.sample(x, z);
                if sample.river < 0.62 || sample.height <= sea + 4.0 {
                    continue;
                }
                let (_, jx, jz) = self.joints(x, sample.height - 8.0, z);
                if jx > 0.11 || jz > 0.11 {
                    continue;
                }
                let top = (sample.height * 16.0) as i32;
                let bottom = ((sample.height - 22.0).max(8.0) * 16.0) as i32;
                for y in bottom..=top {
                    let centre = IVec3::new(
                        origin.x + cx * step + step / 2,
                        y,
                        origin.z + cz * step + step / 2,
                    );
                    self.carve_ellipsoid(tree, origin, centre, 1.35, sample.height, sea);
                }
            }
        }
    }

    fn carve_ellipsoid(
        &self,
        tree: &mut ChunkTree,
        origin: IVec3,
        centre: IVec3,
        radius_m: f32,
        surface: f32,
        sea: f32,
    ) {
        let rx = radius_m * 16.0;
        let ry = radius_m * 16.0 * 0.62;
        let rz = radius_m * 16.0;
        let lo: IVec3 =
            (centre - IVec3::new(rx as i32, ry as i32, rz as i32) - origin).max(IVec3::ZERO);
        let hi: IVec3 = (centre + IVec3::new(rx as i32, ry as i32, rz as i32) - origin)
            .min(IVec3::splat(CHUNK_VOXELS - 1));
        if lo.cmpgt(hi).any() {
            return;
        }
        let cell_lo: IVec3 = lo >> 3;
        let cell_hi: IVec3 = hi >> 3;
        for cz in cell_lo.z..=cell_hi.z {
            for cy in cell_lo.y..=cell_hi.y {
                for cx in cell_lo.x..=cell_hi.x {
                    let local = IVec3::new(cx, cy, cz) * 8 + 4;
                    let world = local + origin;
                    let d = (world - centre).as_vec3();
                    if (d.x / rx).powi(2) + (d.y / ry).powi(2) + (d.z / rz).powi(2) > 1.0 {
                        continue;
                    }
                    let cell = IVec3::new(cx, cy, cz);
                    // Palette bricks already have voxel-accurate walls from
                    // `brick_at`; smashing the cell would flatten them.
                    if matches!(tree.cell(cell), Cell::Brick(_)) {
                        continue;
                    }
                    let here = tree.voxel(local);
                    let p = world.as_vec3() * VOXEL_M;
                    let Some(fill) = self.dissolve(here, p, surface, sea, 0.0) else {
                        continue;
                    };
                    if fill.is_air() {
                        tree.set_cell(cell, Cell::Empty);
                    } else {
                        tree.set_cell(cell, Cell::Uniform(fill));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_carbonates_dissolve() {
        let k = Karst::new(1);
        assert!(
            k.dissolve(ids::GRANITE, Vec3::new(40.0, 50.0, 40.0), 90.0, 96.0, 0.0)
                .is_none()
        );
        assert!(soluble(ids::LIMESTONE) && soluble(ids::CHALK) && soluble(ids::MARBLE));
        assert!(!soluble(ids::SANDSTONE));
    }

    #[test]
    fn same_seed_same_openness() {
        let a = Karst::new(9);
        let b = Karst::new(9);
        let u = a.openness(120.0, 44.0, 80.0, 92.0);
        assert_eq!(u, b.openness(120.0, 44.0, 80.0, 92.0));
        assert_ne!(u, Karst::new(10).openness(120.0, 44.0, 80.0, 92.0));
        assert!((0.0..=1.0).contains(&u));
    }

    #[test]
    fn limestone_opens_along_joints_not_everywhere() {
        let k = Karst::new(7);
        let mut hits = 0;
        let mut n = 0;
        for x in (0..360).step_by(4) {
            for y in 24..70 {
                for z in (0..360).step_by(4) {
                    n += 1;
                    if k.dissolve(
                        ids::LIMESTONE,
                        Vec3::new(x as f32, y as f32, z as f32),
                        92.0,
                        96.0,
                        0.0,
                    )
                    .is_some()
                    {
                        hits += 1;
                    }
                }
            }
        }
        assert!(hits > 30, "karst never opened: {hits} / {n}");
        assert!(hits * 12 < n, "too much was dissolved: {hits} / {n}");
    }
}
