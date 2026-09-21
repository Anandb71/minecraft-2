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
use crate::flora::{self, FLORA_HEIGHT_M, GROUND_COVER_M};
use crate::karst::Karst;
use crate::noise::Perlin;
use crate::noise::{hash_unit, hash3};
use crate::settlement::Settlements;
use crate::strata::{Column, Strata};
use crate::terrain::CoarseTerrain;
use glam::IVec3;
use mc2_voxel::brick::Brick;
use mc2_voxel::coords::{CHUNK_VOXELS, ChunkPos};
use mc2_voxel::material::{MaterialId, ids};
use mc2_voxel::tree::ChunkTree;
use rayon::prelude::*;
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
    /// Edges of leaf clusters and stones.
    flora_noise: Perlin,
    /// Grow plants and build villages (off for terrain-only tests and
    /// measurements).
    pub flora: bool,
    pub settlements: Settlements,
    karst: Karst,
    /// Plants by chunk column: every chunk stacked in a column shares them.
    plant_cache: std::sync::Mutex<PlantCache>,
}

type PlantCache = mc2_core::FxHashMap<(i32, i32), Arc<Vec<flora::Plant>>>;

/// Dense generation target turned into a tree in one pass.
struct Builder {
    /// Material per brick cell, 0 for air, `x + z*64 + y*4096`.
    cells: Vec<u16>,
    bricks: Vec<(u32, Brick)>,
}

impl Builder {
    fn new() -> Self {
        Self {
            cells: vec![0; 64 * 64 * 64],
            bricks: Vec::new(),
        }
    }

    fn index(c: IVec3) -> u32 {
        (c.x + c.z * 64 + c.y * 4096) as u32
    }

    /// Fills a node (`level` 2: 16 cells, 1: 4 cells, 0: one cell).
    fn fill(&mut self, level: u32, min_cell: IVec3, m: MaterialId) {
        let n = 1 << (2 * level);
        for y in min_cell.y..min_cell.y + n {
            for z in min_cell.z..min_cell.z + n {
                let row = (y * 4096 + z * 64) as usize;
                self.cells[row + min_cell.x as usize..row + (min_cell.x + n) as usize].fill(m.0);
            }
        }
    }
}

/// Per-chunk samples on the brick grid.
struct ChunkSamples<'a> {
    /// The chunk's minimum voxel.
    origin: IVec3,
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
            flora_noise: Perlin::new(seed ^ 0x1eaf_b10b),
            flora: true,
            settlements: Settlements::new(seed),
            karst: Karst::new(seed),
            plant_cache: Default::default(),
        }
    }

    pub fn terrain(&self) -> &CoarseTerrain {
        &self.surface.terrain
    }

    /// Conservative highest surface over a chunk's footprint, metres, from
    /// the coarse grid alone. Chunks entirely above it are air.
    pub fn max_height_over(&self, pos: ChunkPos) -> f32 {
        self.column_range(pos.0.x, pos.0.z).1
    }

    /// Conservative (lowest, highest) surface elevation over a chunk
    /// column's footprint, metres, from the coarse grid plus the largest
    /// possible amplified detail. Independent of the chunk's height, so
    /// streaming evaluates it once per column.
    pub fn column_range(&self, cx: i32, cz: i32) -> (f32, f32) {
        let t = self.terrain();
        let c = t.params.cell_m;
        let size = CHUNK_VOXELS as f32 * VOXEL_M;
        let x0 = cx as f32 * size;
        let z0 = cz as f32 * size;
        let (i0, i1) = (((x0 / c) as i64) - 2, (((x0 + size) / c) as i64) + 2);
        let (j0, j1) = (((z0 / c) as i64) - 2, (((z0 + size) / c) as i64) + 2);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for j in j0..=j1 {
            for i in i0..=i1 {
                let h = t.height.at(i, j);
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        // Bicubic overshoot plus amplified detail; above that the tallest
        // tree, and never below the sea's surface.
        let sea = t.params.sea_level;
        let mut top = (hi + DETAIL_MARGIN_M + 8.0 + FLORA_HEIGHT_M).max(sea + 1.0);
        // Towers stand far above any tree.
        if self.flora {
            let lo_m = glam::Vec3::new(x0, -1e4, z0);
            let hi_m = glam::Vec3::new(x0 + size, 1e4, z0 + size);
            for v in self.settlements.near(&self.surface, lo_m, hi_m) {
                top = top.max(v.hi.y + 1.0);
            }
        }
        (lo - DETAIL_MARGIN_M - 8.0, top)
    }

    /// True when the whole chunk lies below the lowest possible surface of
    /// its footprint: no ray reaches it until something is excavated.
    pub fn is_buried(&self, pos: ChunkPos) -> bool {
        let top = (pos.origin().y + CHUNK_VOXELS) as f32 * VOXEL_M;
        top < self.column_range(pos.0.x, pos.0.z).0
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
            origin,
            corners,
            columns,
            hmin,
            hmax,
            rock_top,
        }
    }

    /// Generates a chunk at the given level of detail.
    pub fn generate(&self, pos: ChunkPos, lod: Lod) -> ChunkTree {
        let bottom_m = pos.origin().y as f32 * VOXEL_M;
        if bottom_m > self.max_height_over(pos) {
            return ChunkTree::new();
        }
        if matches!(lod, Lod::Node2 | Lod::Node8) {
            return self.generate_nodes(pos, lod);
        }
        let mut b = Builder::new();
        if lod == Lod::Cell {
            self.generate_coarse(&mut b, pos, lod);
            let mut tree = ChunkTree::from_dense(&b.cells, b.bricks);
            self.karst
                .carve(&mut tree, pos, &self.surface, &self.strata, lod);
            flora::stamp_coarse(&mut tree, pos.origin(), &self.plants_near(pos), 1);
            for v in self.villages_near(pos) {
                v.stamp_coarse(&mut tree, pos.origin(), 1);
            }
            return tree;
        }
        let s = self.samples(pos);
        let base_y = pos.origin().y;
        for l2 in 0..64u32 {
            let c2 = IVec3::new((l2 & 3) as i32, (l2 >> 4 & 3) as i32, (l2 >> 2 & 3) as i32);
            self.node(&mut b, &s, 2, c2 * 16, 16, base_y);
        }
        let mut tree = ChunkTree::from_dense(&b.cells, b.bricks);
        self.karst
            .carve(&mut tree, pos, &self.surface, &self.strata, lod);
        self.stamp_ores(&mut tree, pos);
        flora::stamp_full(
            &mut tree,
            pos.origin(),
            &self.plants_near(pos),
            &self.flora_noise,
        );
        for v in self.villages_near(pos) {
            v.stamp_full(&mut tree, pos.origin());
        }
        tree
    }

    /// Villages that may reach into a chunk.
    fn villages_near(&self, pos: ChunkPos) -> Vec<std::sync::Arc<crate::settlement::Village>> {
        if !self.flora {
            return Vec::new();
        }
        let lo = pos.origin().as_vec3() * VOXEL_M;
        let hi = lo + glam::Vec3::splat(CHUNK_VOXELS as f32 * VOXEL_M);
        self.settlements.near(&self.surface, lo, hi)
    }

    /// Plants that may reach into a chunk; none for chunks far from the
    /// surface.
    fn plants_near(&self, pos: ChunkPos) -> Vec<flora::Plant> {
        let lo = pos.origin().as_vec3() * VOXEL_M;
        let hi = lo + glam::Vec3::splat(CHUNK_VOXELS as f32 * VOXEL_M);
        let (ground_lo, _) = self.column_range(pos.0.x, pos.0.z);
        if !self.flora || hi.y < ground_lo {
            return Vec::new();
        }
        self.column_plants(pos.0.x, pos.0.z)
            .iter()
            .filter(|p| p.hi.cmpge(lo).all() && p.lo.cmple(hi).all())
            .cloned()
            .collect()
    }

    /// Every plant reaching into a chunk column, built once and cached.
    fn column_plants(&self, cx: i32, cz: i32) -> Arc<Vec<flora::Plant>> {
        if let Some(p) = self.plant_cache.lock().expect("plant cache").get(&(cx, cz)) {
            return p.clone();
        }
        let size = CHUNK_VOXELS as f32 * VOXEL_M;
        let lo = glam::Vec3::new(cx as f32 * size, -1e4, cz as f32 * size);
        let hi = lo + glam::Vec3::new(size, 2e4, size);
        let villages = self.settlements.near(&self.surface, lo, hi);
        let taken = |x: f32, z: f32| villages.iter().any(|v| v.claims(x, z));
        let plants = Arc::new(flora::plants_near(&self.surface, lo, hi, self.seed, &taken));
        let mut cache = self.plant_cache.lock().expect("plant cache");
        if cache.len() > 8192 {
            cache.clear();
        }
        cache.insert((cx, cz), plants.clone());
        plants
    }

    /// Coarse levels sample one surface column per node or cell and give
    /// every vertical cell the material at its centre. Cost scales with the
    /// number of columns: 4096 for `Cell`, 256 for `Node2`, 16 for `Node8`.
    fn generate_coarse(&self, b: &mut Builder, pos: ChunkPos, lod: Lod) {
        self.coarse_columns(pos, lod, &mut |level, min_cell, m| {
            b.fill(level, min_cell, m)
        });
    }

    fn coarse_columns(
        &self,
        pos: ChunkPos,
        lod: Lod,
        emit: &mut dyn FnMut(u32, IVec3, MaterialId),
    ) {
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
                let sea = self.surface.terrain.params.sea_level;
                if sample.height < bottom && sea < bottom {
                    continue;
                }
                let column = self.strata.column(x, z);
                for cy in 0..per_axis {
                    let centre = bottom + (cy as f32 + 0.5) * size_m;
                    if centre > sample.height {
                        if centre < sea {
                            emit(level, IVec3::new(cx, cy, cz) * cells, ids::WATER);
                            continue;
                        }
                        break;
                    }
                    // The topmost cell shows the surface material, not
                    // whatever lies half a cell down.
                    let top_of_column = centre + size_m > sample.height;
                    let probe = if top_of_column {
                        sample.height - 0.01
                    } else {
                        centre
                    };
                    let m = material_at(&self.surface, &sample, &column, probe);
                    if !m.is_air() {
                        emit(level, IVec3::new(cx, cy, cz) * cells, m);
                    }
                }
            }
        }
    }

    /// 2 m and 8 m levels touch at most 256 columns: set nodes directly
    /// rather than paying for a dense 64^3 scan.
    fn generate_nodes(&self, pos: ChunkPos, lod: Lod) -> ChunkTree {
        let mut tree = ChunkTree::new();
        let cells = if lod == Lod::Node2 { 4 } else { 16 };
        self.coarse_columns(pos, lod, &mut |level, min_cell, m| {
            tree.set_node(level, min_cell / cells, m);
        });
        self.karst
            .carve(&mut tree, pos, &self.surface, &self.strata, lod);
        if lod == Lod::Node2 {
            flora::stamp_coarse(&mut tree, pos.origin(), &self.plants_near(pos), 4);
            for v in self.villages_near(pos) {
                v.stamp_coarse(&mut tree, pos.origin(), 4);
            }
        }
        tree
    }

    /// Fills a node covering `cells` brick cells from `min_cell`.
    fn node(
        &self,
        b: &mut Builder,
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
        let sea = self.surface.terrain.params.sea_level;
        // Above the ground and its cover: open water below the sea's
        // surface, air above it.
        if y0 >= hmax + GROUND_COVER_M {
            if y1 <= sea {
                b.fill(level, min_cell, ids::WATER);
                return;
            }
            if y0 >= sea {
                return;
            }
        }
        // Wholly rock and one stratum: a single node or cell.
        if y1 <= rock
            && let Some(m) = s.strata_uniform(x0, z0, n, y0, y1 - 0.01)
        {
            b.fill(level, min_cell, m);
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
                b.fill(level, min_cell, m);
            }
            return;
        }
        if cells == 1 {
            let brick = self.brick(s, min_cell, base_y);
            b.bricks.push((Builder::index(min_cell), brick));
            return;
        }
        let child = cells / 4;
        for i in 0..64u32 {
            let c = IVec3::new((i & 3) as i32, (i >> 4 & 3) as i32, (i >> 2 & 3) as i32);
            self.node(
                b,
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
        self.brick(&s, cell, pos.origin().y)
    }

    /// `detail_brick` for many cells of one chunk.
    pub fn detail_bricks(&self, pos: ChunkPos, cells: &[IVec3]) -> Vec<Brick> {
        // Sample only the brick columns the cells stand in, once per column
        // and in parallel: an edit touches a few hundred of a chunk's 4096.
        let origin = pos.origin();
        let (ox, oz) = (origin.x as f32 * VOXEL_M, origin.z as f32 * VOXEL_M);
        let mut columns: mc2_core::FxHashMap<(i32, i32), Vec<usize>> = Default::default();
        for (i, c) in cells.iter().enumerate() {
            columns.entry((c.x, c.z)).or_default().push(i);
        }
        let columns: Vec<((i32, i32), Vec<usize>)> = columns.into_iter().collect();
        let mut made: Vec<(usize, Brick)> = columns
            .par_iter()
            .flat_map_iter(|&((x, z), ref idx)| {
                let corner = |x: i32, z: i32| {
                    self.surface
                        .sample(ox + x as f32 * 0.5, oz + z as f32 * 0.5)
                };
                let corners = [
                    corner(x, z),
                    corner(x + 1, z),
                    corner(x, z + 1),
                    corner(x + 1, z + 1),
                ];
                let column = self
                    .strata
                    .column(ox + x as f32 * 0.5 + 0.25, oz + z as f32 * 0.5 + 0.25);
                idx.iter()
                    .map(|&i| (i, self.brick_at(&corners, &column, cells[i], origin)))
                    .collect::<Vec<_>>()
            })
            .collect();
        made.sort_unstable_by_key(|(i, _)| *i);
        made.into_iter().map(|(_, b)| b).collect()
    }

    fn brick(&self, s: &ChunkSamples<'_>, cell: IVec3, base_y: i32) -> Brick {
        let origin = IVec3::new(s.origin.x, base_y, s.origin.z);
        let (bx, bz) = (cell.x as usize, cell.z as usize);
        let corners = [
            *s.corner(bx, bz),
            *s.corner(bx + 1, bz),
            *s.corner(bx, bz + 1),
            *s.corner(bx + 1, bz + 1),
        ];
        self.brick_at(&corners, &s.columns[bz * 64 + bx], cell, origin)
    }

    /// A brick from its column's four corner samples (x-z order: 00, 10,
    /// 01, 11) and strata column.
    fn brick_at(
        &self,
        corners: &[SurfaceSample; 4],
        column: &Column<'_>,
        cell: IVec3,
        origin: IVec3,
    ) -> Brick {
        let [c00, c10, c01, c11] = corners;
        let y_base = origin.y + cell.y * 8;
        // World voxel column of the brick's first voxel; brick cells of a
        // chunk share its origin's x and z.
        let (base_x, base_z) = (origin.x + cell.x * 8, origin.z + cell.z * 8);
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
                    temp: lerp(c00.temp, c10.temp, c01.temp, c11.temp),
                    wet: lerp(c00.wet, c10.wet, c01.wet, c11.wet),
                };
                let sea = self.surface.terrain.params.sea_level;
                // Grass and flowers grow on grassy ground above the sea.
                let grassy = sample.height > sea + 0.3
                    && sample.soil_depth > 0.1
                    && self.surface.loose_material(&sample, 0.05, sample.height) == ids::GRASS;
                let (vx, vz) = (base_x + lx, base_z + lz);
                let mut column_m = [MaterialId(0); 8];
                for ly in 0..8 {
                    let y = (y_base + ly) as f32 * VOXEL_M + VOXEL_M * 0.5;
                    let mut m = material_at(&self.surface, &sample, column, y);
                    let above = y - sample.height;
                    if m.is_air() && grassy && self.flora && above < GROUND_COVER_M {
                        m = flora::ground_cover(&sample, sea, vx, vz, above, self.seed);
                    }
                    let wx = vx as f32 * VOXEL_M;
                    let wz = vz as f32 * VOXEL_M;
                    if let Some(fill) = self.karst.dissolve(
                        m,
                        glam::Vec3::new(wx, y, wz),
                        sample.height,
                        sea,
                        sample.river,
                        sample.slope,
                    ) {
                        m = fill;
                    }
                    column_m[ly as usize] = m;
                }
                self.karst
                    .dress_column(&mut column_m, y_base, vx, vz, sample.height, sea);
                for (ly, m) in column_m.into_iter().enumerate() {
                    if !m.is_air() {
                        brick.set(IVec3::new(lx, ly as i32, lz), m);
                    }
                }
            }
        }
        brick
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

    #[test]
    fn detail_bricks_match_whole_chunk_sampling() {
        let g = generator();
        let pos = land_chunk(&g);
        // Cells in a few columns through the chunk's height, including
        // column 63 at its edge.
        let mut cells = Vec::new();
        for z in 20..24 {
            for x in (24..28).chain([63]) {
                for y in (0..64).step_by(9) {
                    cells.push(IVec3::new(x, y, z));
                }
            }
        }
        let fast = g.detail_bricks(pos, &cells);
        for (c, brick) in cells.iter().zip(&fast) {
            let reference = g.detail_brick(pos, *c);
            for i in 0..512 {
                let l = IVec3::new(i & 7, (i >> 6) & 7, (i >> 3) & 7);
                assert_eq!(brick.get(l), reference.get(l), "cell {c} voxel {l}");
            }
        }
    }

    /// Top of the ground in a column, not counting plants or water.
    fn top_solid(tree: &ChunkTree, x: i32, z: i32) -> Option<i32> {
        let plant = [ids::OAK_LOG, ids::PINE_LOG, ids::BIRCH_LOG, ids::CACTUS];
        (0..512).rev().find(|&y| {
            let m = tree.voxel(IVec3::new(x, y, z));
            m.is_solid() && !plant.contains(&m)
        })
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
        let mut g = generator();
        g.flora = false;
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
