//! Chunk generation from the amplified surface and the strata below it.
//!
//! Generation works top-down through the tree. An 8 m node that lies wholly
//! below the loose surface layer and inside one stratum becomes a single
//! uniform node; otherwise its 2 m children are tried, then brick cells, and
//! only cells straddling the surface or a layer boundary are filled voxel by
//! voxel. Coarser levels of detail stop that descent early and give each
//! node or cell the material at its centre, which is how distant terrain is
//! represented without generating voxels nobody can resolve.

use crate::amplify::{Surface, SurfaceSample, material_at};
use crate::noise::{hash_unit, hash3};
use crate::strata::{Column, Strata};
use crate::terrain::CoarseTerrain;
use glam::IVec3;
use mc2_voxel::brick::Brick;
use mc2_voxel::coords::{CHUNK_VOXELS, ChunkPos};
use mc2_voxel::material::{MaterialId, ids};
use mc2_voxel::tree::{Cell, ChunkTree};
use std::sync::Arc;

const VOXEL_M: f32 = 1.0 / 16.0;
/// Upper bound on how far amplified detail rises above the coarse surface.
const DETAIL_MARGIN_M: f32 = 16.0;
/// Rock deeper than this below the surface is generated at cell resolution.
const BURIED_M: f32 = 2.0;

/// Level of detail, finest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lod {
    /// Every voxel.
    Full,
    /// One material per 0.5 m brick cell.
    Cell,
    /// One material per 2 m node.
    Node2,
    /// One material per 8 m node.
    Node8,
}

pub struct ChunkGenerator {
    pub surface: Surface,
    pub strata: Strata,
    seed: u64,
}

/// Per-chunk samples on the brick grid.
struct ChunkSamples<'a> {
    /// Surface samples at brick corners, 65 x 65.
    corners: Vec<SurfaceSample>,
    /// Strata columns at brick column centres, 64 x 64.
    columns: Vec<Column<'a>>,
    /// Minimum and maximum surface height per brick column, metres.
    hmin: Vec<f32>,
    hmax: Vec<f32>,
    /// Top of guaranteed rock per brick column: min height less soil.
    rock_top: Vec<f32>,
}

impl ChunkSamples<'_> {
    #[inline]
    fn corner(&self, x: usize, z: usize) -> &SurfaceSample {
        &self.corners[z * 65 + x]
    }

    fn region(&self, x0: usize, z0: usize, n: usize) -> (f32, f32, f32) {
        let (mut lo, mut hi, mut rock) = (f32::MAX, f32::MIN, f32::MAX);
        for z in z0..z0 + n {
            for x in x0..x0 + n {
                let i = z * 64 + x;
                lo = lo.min(self.hmin[i]);
                hi = hi.max(self.hmax[i]);
                rock = rock.min(self.rock_top[i]);
            }
        }
        (lo, hi, rock)
    }

    /// One stratum throughout `[y0, y1]` in every sampled column of a region?
    fn strata_uniform(
        &self,
        x0: usize,
        z0: usize,
        n: usize,
        y0: f32,
        y1: f32,
    ) -> Option<MaterialId> {
        let picks = [0, n / 2, n - 1];
        let first = self.columns[z0 * 64 + x0].layer(y0);
        for &dz in &picks {
            for &dx in &picks {
                let col = &self.columns[(z0 + dz) * 64 + x0 + dx];
                let (a, b) = (col.layer(y0), col.layer(y1));
                if a != first || b != first {
                    return None;
                }
            }
        }
        Some(first.material)
    }
}

impl ChunkGenerator {
    pub fn new(terrain: Arc<CoarseTerrain>) -> Self {
        let seed = terrain.seed;
        Self {
            surface: Surface::new(terrain),
            strata: Strata::new(seed),
            seed,
        }
    }

    pub fn terrain(&self) -> &CoarseTerrain {
        &self.surface.terrain
    }

    /// Conservative highest surface over a chunk's footprint, metres, from
    /// the coarse grid alone. Chunks entirely above it are air.
    pub fn max_height_over(&self, pos: ChunkPos) -> f32 {
        let t = self.terrain();
        let c = t.params.cell_m;
        let x0 = pos.0.x as f32 * CHUNK_VOXELS as f32 * VOXEL_M;
        let z0 = pos.0.z as f32 * CHUNK_VOXELS as f32 * VOXEL_M;
        let size = CHUNK_VOXELS as f32 * VOXEL_M;
        let (i0, i1) = (((x0 / c) as i64) - 2, (((x0 + size) / c) as i64) + 2);
        let (j0, j1) = (((z0 / c) as i64) - 2, (((z0 + size) / c) as i64) + 2);
        let mut hi = f32::MIN;
        for j in j0..=j1 {
            for i in i0..=i1 {
                hi = hi.max(t.height.at(i, j));
            }
        }
        // Bicubic overshoot plus amplified detail.
        hi + DETAIL_MARGIN_M + 8.0
    }

    /// True when the whole chunk lies below the lowest possible surface of
    /// its footprint: no ray reaches it until something is excavated.
    pub fn is_buried(&self, pos: ChunkPos) -> bool {
        let t = self.terrain();
        let c = t.params.cell_m;
        let x0 = pos.0.x as f32 * CHUNK_VOXELS as f32 * VOXEL_M;
        let z0 = pos.0.z as f32 * CHUNK_VOXELS as f32 * VOXEL_M;
        let size = CHUNK_VOXELS as f32 * VOXEL_M;
        let (i0, i1) = (((x0 / c) as i64) - 2, (((x0 + size) / c) as i64) + 2);
        let (j0, j1) = (((z0 / c) as i64) - 2, (((z0 + size) / c) as i64) + 2);
        let mut lo = f32::MAX;
        for j in j0..=j1 {
            for i in i0..=i1 {
                lo = lo.min(t.height.at(i, j));
            }
        }
        let top = (pos.origin().y + CHUNK_VOXELS) as f32 * VOXEL_M;
        top < lo - DETAIL_MARGIN_M - 8.0
    }

    fn samples(&self, pos: ChunkPos) -> ChunkSamples<'_> {
        let origin = pos.origin();
        let ox = origin.x as f32 * VOXEL_M;
        let oz = origin.z as f32 * VOXEL_M;
        let mut corners = Vec::with_capacity(65 * 65);
        for z in 0..65 {
            for x in 0..65 {
                corners.push(
                    self.surface
                        .sample(ox + x as f32 * 0.5, oz + z as f32 * 0.5),
                );
            }
        }
        let mut columns = Vec::with_capacity(64 * 64);
        let (mut hmin, mut hmax, mut rock_top) =
            (vec![0.0; 4096], vec![0.0; 4096], vec![0.0; 4096]);
        for z in 0..64 {
            for x in 0..64 {
                columns.push(
                    self.strata
                        .column(ox + x as f32 * 0.5 + 0.25, oz + z as f32 * 0.5 + 0.25),
                );
                let c = [
                    corners[z * 65 + x],
                    corners[z * 65 + x + 1],
                    corners[(z + 1) * 65 + x],
                    corners[(z + 1) * 65 + x + 1],
                ];
                let i = z * 64 + x;
                hmin[i] = c.iter().map(|s| s.height).fold(f32::MAX, f32::min);
                hmax[i] = c.iter().map(|s| s.height).fold(f32::MIN, f32::max);
                let soil = c.iter().map(|s| s.soil_depth).fold(0.0, f32::max);
                rock_top[i] = hmin[i] - soil;
            }
        }
        ChunkSamples {
            corners,
            columns,
            hmin,
            hmax,
            rock_top,
        }
    }

    /// Generates a chunk at the given level of detail.
    pub fn generate(&self, pos: ChunkPos, lod: Lod) -> ChunkTree {
        let mut tree = ChunkTree::new();
        let bottom_m = pos.origin().y as f32 * VOXEL_M;
        if bottom_m > self.max_height_over(pos) {
            return tree;
        }
        if lod != Lod::Full {
            return self.generate_coarse(pos, lod);
        }
        let s = self.samples(pos);
        let base_y = pos.origin().y;
        for l2 in 0..64u32 {
            let c2 = IVec3::new((l2 & 3) as i32, (l2 >> 4 & 3) as i32, (l2 >> 2 & 3) as i32);
            self.node(&mut tree, &s, 2, c2 * 16, 16, base_y);
        }
        if lod == Lod::Full {
            self.stamp_ores(&mut tree, pos);
        }
        tree
    }

    /// Coarse levels sample one surface column per node or cell and give
    /// every vertical cell the material at its centre. Cost scales with the
    /// number of columns: 4096 for `Cell`, 256 for `Node2`, 16 for `Node8`.
    fn generate_coarse(&self, pos: ChunkPos, lod: Lod) -> ChunkTree {
        let mut tree = ChunkTree::new();
        let (cells, level) = match lod {
            Lod::Cell => (1, 0),
            Lod::Node2 => (4, 1),
            _ => (16, 2),
        };
        let per_axis = 64 / cells;
        let origin = pos.origin();
        let size_m = cells as f32 * 0.5;
        for cz in 0..per_axis {
            for cx in 0..per_axis {
                let x = origin.x as f32 * VOXEL_M + (cx as f32 + 0.5) * size_m;
                let z = origin.z as f32 * VOXEL_M + (cz as f32 + 0.5) * size_m;
                let sample = self.surface.sample(x, z);
                let bottom = origin.y as f32 * VOXEL_M;
                if sample.height < bottom {
                    continue;
                }
                let column = self.strata.column(x, z);
                for cy in 0..per_axis {
                    let centre = bottom + (cy as f32 + 0.5) * size_m;
                    if centre > sample.height {
                        break;
                    }
                    let m = material_at(&self.surface, &sample, &column, centre);
                    if !m.is_air() {
                        self.put(&mut tree, level, IVec3::new(cx, cy, cz) * cells, m);
                    }
                }
            }
        }
        tree
    }

    /// Fills a node covering `cells` brick cells from `min_cell`.
    fn node(
        &self,
        tree: &mut ChunkTree,
        s: &ChunkSamples<'_>,

        level: u32,
        min_cell: IVec3,
        cells: i32,
        base_y: i32,
    ) {
        let (x0, z0, n) = (min_cell.x as usize, min_cell.z as usize, cells as usize);
        let y0 = (base_y + min_cell.y * 8) as f32 * VOXEL_M;
        let y1 = y0 + cells as f32 * 0.5;
        let (_, hmax, rock) = s.region(x0, z0, n);
        if y0 >= hmax {
            return;
        }
        // Wholly rock and one stratum: a single node or cell.
        if y1 <= rock
            && let Some(m) = s.strata_uniform(x0, z0, n, y0, y1 - 0.01)
        {
            self.put(tree, level, min_cell, m);
            return;
        }
        // Layer boundaries buried under metres of rock are invisible until
        // someone digs; keep them at brick-cell resolution until then
        // (`detail_brick` restores voxel detail on demand).
        if y1 <= rock - BURIED_M && cells == 1 {
            let cx = x0 + n / 2;
            let cz = z0 + n / 2;
            let centre = (y0 + y1) * 0.5;
            let sample = s.corner(cx.min(64), cz.min(64));
            let column = &s.columns[cz.min(63) * 64 + cx.min(63)];
            let m = material_at(&self.surface, sample, column, centre);
            if !m.is_air() {
                self.put(tree, level, min_cell, m);
            }
            return;
        }
        if cells == 1 {
            self.brick(tree, s, min_cell, base_y);
            return;
        }
        let child = cells / 4;
        for i in 0..64u32 {
            let c = IVec3::new((i & 3) as i32, (i >> 4 & 3) as i32, (i >> 2 & 3) as i32);
            self.node(
                tree,
                s,
                level.saturating_sub(1),
                min_cell + c * child,
                child,
                base_y,
            );
        }
    }

    /// Voxel-resolution contents of one brick cell as generation would
    /// produce at full detail, for edits into cells stored coarsely.
    pub fn detail_brick(&self, pos: ChunkPos, cell: IVec3) -> Brick {
        let s = self.samples(pos);
        let mut tree = ChunkTree::new();
        self.brick(&mut tree, &s, cell, pos.origin().y);
        let mut out = Brick::empty();
        for i in 0..512 {
            let l = IVec3::new(i & 7, (i >> 6) & 7, (i >> 3) & 7);
            let m = tree.voxel((cell << 3) + l);
            if !m.is_air() {
                out.set(l, m);
            }
        }
        out
    }

    fn put(&self, tree: &mut ChunkTree, level: u32, min_cell: IVec3, m: MaterialId) {
        match level {
            2 if min_cell.x % 16 == 0 && min_cell.y % 16 == 0 && min_cell.z % 16 == 0 => {
                tree.set_node(2, min_cell / 16, m);
            }
            1 if min_cell.x % 4 == 0 && min_cell.y % 4 == 0 && min_cell.z % 4 == 0 => {
                tree.set_node(1, min_cell / 4, m);
            }
            _ => tree.set_cell(min_cell, Cell::Uniform(m)),
        }
    }

    fn brick(&self, tree: &mut ChunkTree, s: &ChunkSamples<'_>, cell: IVec3, base_y: i32) {
        let (bx, bz) = (cell.x as usize, cell.z as usize);
        let c00 = s.corner(bx, bz);
        let c10 = s.corner(bx + 1, bz);
        let c01 = s.corner(bx, bz + 1);
        let c11 = s.corner(bx + 1, bz + 1);
        let column = &s.columns[bz * 64 + bx];
        let y_base = base_y + cell.y * 8;
        let mut brick = Brick::empty();
        for lz in 0..8 {
            for lx in 0..8 {
                let (tx, tz) = ((lx as f32 + 0.5) / 8.0, (lz as f32 + 0.5) / 8.0);
                let lerp = |a: f32, b: f32, c: f32, d: f32| {
                    let top = a + (b - a) * tx;
                    let bot = c + (d - c) * tx;
                    top + (bot - top) * tz
                };
                let sample = SurfaceSample {
                    height: lerp(c00.height, c10.height, c01.height, c11.height),
                    slope: lerp(c00.slope, c10.slope, c01.slope, c11.slope),
                    river: lerp(c00.river, c10.river, c01.river, c11.river),
                    soil_depth: lerp(
                        c00.soil_depth,
                        c10.soil_depth,
                        c01.soil_depth,
                        c11.soil_depth,
                    ),
                };
                for ly in 0..8 {
                    let y = (y_base + ly) as f32 * VOXEL_M + VOXEL_M * 0.5;
                    let m = material_at(&self.surface, &sample, column, y);
                    if !m.is_air() {
                        brick.set(IVec3::new(lx, ly, lz), m);
                    }
                }
            }
        }
        tree.set_brick(cell, brick);
    }

    /// Ore bodies placed by the geology they belong to: coal seams in shale,
    /// iron in sandstone, copper near the granite contact, rare gold in the
    /// basement. One candidate per 16 m cell, flattened along bedding.
    fn stamp_ores(&self, tree: &mut ChunkTree, pos: ChunkPos) {
        let origin = pos.origin();
        let lo = (origin - IVec3::splat(64)) >> 8i32;
        let hi = (origin + IVec3::splat(CHUNK_VOXELS + 64)) >> 8i32;
        for cz in lo.z..=hi.z {
            for cy in lo.y.max(0)..=hi.y {
                for cx in lo.x..=hi.x {
                    let h = hash3(cx, cy, cz, self.seed ^ 0x04e5);
                    let centre = IVec3::new(cx, cy, cz) * 256
                        + IVec3::new(
                            (hash_unit(h) * 256.0) as i32,
                            (hash_unit(h.rotate_left(17)) * 256.0) as i32,
                            (hash_unit(h.rotate_left(34)) * 256.0) as i32,
                        );
                    let cm = centre.as_vec3() * VOXEL_M;
                    if cm.y > self.max_height_over(pos) {
                        continue;
                    }
                    let host = self.strata.layer(cm.x, cm.y, cm.z).material;
                    let (ore, chance, radius) = match host {
                        m if m == ids::SHALE => (ids::COAL_ORE, 0.45, 2.5),
                        m if m == ids::SANDSTONE => (ids::IRON_ORE, 0.15, 1.8),
                        m if m == ids::GRANITE => (ids::COPPER_ORE, 0.12, 1.5),
                        m if m == ids::GNEISS => (ids::GOLD_ORE, 0.03, 0.8),
                        _ => continue,
                    };
                    if hash_unit(h.rotate_left(51)) > chance {
                        continue;
                    }
                    self.stamp_blob(tree, origin, centre, radius, host, ore);
                }
            }
        }
    }

    fn stamp_blob(
        &self,
        tree: &mut ChunkTree,
        origin: IVec3,
        centre: IVec3,
        radius_m: f32,
        host: MaterialId,
        ore: MaterialId,
    ) {
        let r = radius_m * 16.0;
        // Seams are three times wider than they are thick.
        let (rx, ry) = (r, r / 3.0);
        let lo = (centre - IVec3::new(rx as i32, ry as i32, rx as i32) - origin).max(IVec3::ZERO);
        let hi = (centre + IVec3::new(rx as i32, ry as i32, rx as i32) - origin)
            .min(IVec3::splat(CHUNK_VOXELS - 1));
        if lo.cmpgt(hi).any() {
            return;
        }
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let d = (IVec3::new(x, y, z) + origin - centre).as_vec3();
                    if (d.x / rx).powi(2) + (d.y / ry).powi(2) + (d.z / rx).powi(2) > 1.0 {
                        continue;
                    }
                    let v = IVec3::new(x, y, z);
                    if tree.voxel(v) == host {
                        tree.set_voxel(v, ore);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erosion::ErosionParams;
    use crate::terrain::TerrainParams;
    use mc2_voxel::coords::CHUNK_SHIFT;

    fn generator() -> ChunkGenerator {
        let params = TerrainParams {
            size: 128,
            cell_m: 128.0,
            sea_level: 96.0,
            erosion: ErosionParams {
                iterations: 60,
                ..Default::default()
            },
        };
        ChunkGenerator::new(Arc::new(CoarseTerrain::generate(7, params, &mut |_, _| {})))
    }

    /// A chunk column position on land somewhere near the middle of the map.
    fn land_chunk(g: &ChunkGenerator) -> ChunkPos {
        let t = g.terrain();
        for r in 0..200 {
            let x = 256 + (r * 37) % 64;
            let z = 256 + (r * 53) % 64;
            let h = t.height_at(
                (x << CHUNK_SHIFT >> 4) as f32,
                (z << CHUNK_SHIFT >> 4) as f32,
            );
            if h > t.params.sea_level + 20.0 && h < 400.0 {
                return ChunkPos(IVec3::new(x, (h / 32.0) as i32, z));
            }
        }
        panic!("no land found");
    }

    fn top_solid(tree: &ChunkTree, x: i32, z: i32) -> Option<i32> {
        (0..512)
            .rev()
            .find(|&y| !tree.voxel(IVec3::new(x, y, z)).is_air())
    }

    #[test]
    fn generation_is_deterministic() {
        let g = generator();
        let pos = land_chunk(&g);
        let a = g.generate(pos, Lod::Full);
        let b = g.generate(pos, Lod::Full);
        for v in [
            IVec3::new(5, 100, 9),
            IVec3::new(300, 250, 400),
            IVec3::new(511, 17, 0),
        ] {
            assert_eq!(a.voxel(v), b.voxel(v));
        }
        assert_eq!(a.cells().count(), b.cells().count());
    }

    #[test]
    fn adjacent_chunks_agree_at_the_seam() {
        let g = generator();
        let pos = land_chunk(&g);
        let east = ChunkPos(pos.0 + IVec3::X);
        let (a, b) = (g.generate(pos, Lod::Full), g.generate(east, Lod::Full));
        let mut compared = 0;
        for z in (0..512).step_by(7) {
            if let (Some(ha), Some(hb)) = (top_solid(&a, 511, z), top_solid(&b, 0, z)) {
                assert!((ha - hb).abs() <= 3, "seam step at z={z}: {ha} vs {hb}");
                compared += 1;
            }
        }
        assert!(compared > 20, "surface did not cross the chunk: {compared}");
    }

    #[test]
    fn deep_rock_is_cheap() {
        let g = generator();
        let pos = ChunkPos(IVec3::new(256, 1, 256));
        assert!(g.is_buried(pos));
        let placeholder = g.generate(pos, Lod::Node8);
        assert!(
            placeholder.memory_bytes() < 8 * 1024,
            "{} bytes",
            placeholder.memory_bytes()
        );
        let full = g.generate(pos, Lod::Full);
        assert!(
            full.memory_bytes() < 1024 * 1024,
            "{} bytes at full detail",
            full.memory_bytes()
        );
        assert!(!full.uniform_nodes().is_empty());
        assert!(!g.is_buried(land_chunk(&g)));
    }

    #[test]
    fn coarser_lods_cost_less() {
        let g = generator();
        let pos = land_chunk(&g);
        let sizes: Vec<usize> = [Lod::Full, Lod::Cell, Lod::Node2, Lod::Node8]
            .iter()
            .map(|&l| g.generate(pos, l).memory_bytes())
            .collect();
        assert!(sizes.windows(2).all(|w| w[0] >= w[1]), "{sizes:?}");
        assert!(sizes[0] > sizes[3] * 4, "{sizes:?}");
    }

    #[test]
    fn sky_chunks_are_empty() {
        let g = generator();
        assert!(
            g.generate(ChunkPos(IVec3::new(256, 15, 256)), Lod::Full)
                .is_empty()
        );
    }
}
