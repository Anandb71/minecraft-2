//! Rigid bodies on the GPU.
//!
//! Each body shape is uploaded once into a pooled word buffer: a coarse
//! occupancy bitmap over 4^3 blocks, then 16-bit materials, two to a word.
//! Every frame the visible bodies are written to a table (grid frame in
//! camera-relative voxels, this frame's and last frame's) and binned into a
//! uniform culling grid laid over their combined bounds, so a ray that
//! misses the debris costs one slab test. Layouts match `bodies.wgsl`.

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, IVec3, Quat, Vec3};
use mc2_physics::BodyShape;
use std::sync::Arc;

/// Culling grid cells per axis, at most.
pub const GRID_DIM: i32 = 32;
/// Cell array plus body index lists, words.
const GRID_WORDS: usize = (GRID_DIM * GRID_DIM * GRID_DIM) as usize + (1 << 17);
/// Cells pack a list offset (20 bits) and a count (12 bits).
const CELL_COUNT_BITS: u32 = 12;

/// A body to draw: its shape and the pose of its voxel grid (the corner of
/// voxel (0, 0, 0) and the grid-to-world rotation).
#[derive(Clone)]
pub struct BodyInstance {
    /// Stable identity across frames.
    pub key: u64,
    pub shape: Arc<BodyShape>,
    pub grid_origin: DVec3,
    pub grid_rotation: Quat,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuBody {
    /// Grid origin, camera-relative voxels.
    pub origin: [f32; 3],
    /// Low 11 bits of the body key, written into visibility ids.
    pub tag: u32,
    /// Grid axes in world space.
    pub axis_x: [f32; 3],
    /// Offset of the material words in the shape pool.
    pub voxels: u32,
    pub axis_y: [f32; 3],
    /// Offset of the coarse occupancy words.
    pub coarse: u32,
    pub axis_z: [f32; 3],
    pub _pad0: u32,
    pub size: [i32; 3],
    pub _pad1: u32,
    /// Last frame's pose, relative to this frame's camera voxel.
    pub prev_origin: [f32; 3],
    pub _pad2: u32,
    pub prev_axis_x: [f32; 3],
    pub _pad3: u32,
    pub prev_axis_y: [f32; 3],
    pub _pad4: u32,
    pub prev_axis_z: [f32; 3],
    pub _pad5: u32,
}

/// Words of a shape in the pool: coarse bitmap, then materials.
pub fn encode_shape(shape: &BodyShape) -> (Vec<u32>, u32) {
    let size = shape.size;
    let blocks = (size + 3) / 4;
    let bits = (blocks.x * blocks.y * blocks.z) as usize;
    let coarse_words = bits.div_ceil(32);
    let n = (size.x * size.y * size.z) as usize;
    let mut words = vec![0u32; coarse_words + n.div_ceil(2)];
    for z in 0..size.z {
        for y in 0..size.y {
            for x in 0..size.x {
                let i = (x + size.x * (y + size.y * z)) as usize;
                let m = shape.voxels[i];
                if !m.is_solid() {
                    continue;
                }
                words[coarse_words + i / 2] |= u32::from(m.0) << (16 * (i % 2));
                let b = IVec3::new(x, y, z) / 4;
                let bit = (b.x + blocks.x * (b.y + blocks.y * b.z)) as usize;
                words[bit / 32] |= 1 << (bit % 32);
            }
        }
    }
    (words, coarse_words as u32)
}

/// Header and cells of the culling grid, before upload.
#[derive(Debug, PartialEq)]
pub struct CullGrid {
    pub min: Vec3,
    pub cell: f32,
    pub dims: IVec3,
    /// Cells, then the body index lists they point into.
    pub words: Vec<u32>,
}

/// Bins camera-relative voxel-space boxes into a grid over their union.
/// Coarsens the grid until the index lists fit.
pub fn build_grid(boxes: &[(Vec3, Vec3)]) -> CullGrid {
    if boxes.is_empty() {
        return CullGrid {
            min: Vec3::ZERO,
            cell: 1.0,
            dims: IVec3::ZERO,
            words: Vec::new(),
        };
    }
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for (a, b) in boxes {
        lo = lo.min(*a);
        hi = hi.max(*b);
    }
    let extent = (hi - lo).max(Vec3::ONE);
    // Cells no smaller than a metre.
    let mut cell = (extent.max_element() / GRID_DIM as f32).max(16.0);
    loop {
        let dims = (extent / cell)
            .ceil()
            .as_ivec3()
            .clamp(IVec3::ONE, IVec3::splat(GRID_DIM));
        let cells = (dims.x * dims.y * dims.z) as usize;
        let cell_of = |p: Vec3| {
            ((p - lo) / cell)
                .floor()
                .as_ivec3()
                .clamp(IVec3::ZERO, dims - 1)
        };
        let mut counts = vec![0u32; cells];
        let mut total = 0usize;
        for (a, b) in boxes {
            let (c0, c1) = (cell_of(*a), cell_of(*b));
            for z in c0.z..=c1.z {
                for y in c0.y..=c1.y {
                    for x in c0.x..=c1.x {
                        counts[(x + dims.x * (y + dims.y * z)) as usize] += 1;
                        total += 1;
                    }
                }
            }
        }
        let too_many = counts.iter().any(|&c| c >= 1 << CELL_COUNT_BITS);
        if cells + total > GRID_WORDS || too_many {
            cell *= 2.0;
            continue;
        }
        let mut words = vec![0u32; cells + total];
        let mut next = cells as u32;
        let mut fill = vec![0u32; cells];
        for (i, &c) in counts.iter().enumerate() {
            words[i] = next << CELL_COUNT_BITS;
            fill[i] = next;
            next += c;
        }
        for (k, (a, b)) in boxes.iter().enumerate() {
            let (c0, c1) = (cell_of(*a), cell_of(*b));
            for z in c0.z..=c1.z {
                for y in c0.y..=c1.y {
                    for x in c0.x..=c1.x {
                        let i = (x + dims.x * (y + dims.y * z)) as usize;
                        words[fill[i] as usize] = k as u32;
                        fill[i] += 1;
                        words[i] += 1;
                    }
                }
            }
        }
        return CullGrid {
            min: lo,
            cell,
            dims,
            words,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::{MaterialId, ids};

    #[test]
    fn gpu_body_matches_the_wgsl_layout() {
        assert_eq!(std::mem::size_of::<GpuBody>(), 144);
    }

    #[test]
    fn shapes_encode_occupancy_and_materials() {
        let size = IVec3::new(5, 2, 3);
        let mut voxels = vec![MaterialId(0); 30];
        voxels[0] = ids::GRANITE;
        // x 4, y 1, z 2: in coarse block (1, 0, 0).
        let far = 4 + 5 * (1 + 2 * 2);
        voxels[far] = ids::PLANKS;
        let shape = BodyShape::from_voxels(size, voxels).unwrap();
        let (words, coarse) = encode_shape(&shape);
        // Blocks 2 x 1 x 1: one bitmap word with both blocks set.
        assert_eq!(coarse, 1);
        assert_eq!(words[0], 0b11);
        assert_eq!(words.len(), 1 + 15);
        let get = |i: usize| (words[1 + i / 2] >> (16 * (i % 2))) & 0xffff;
        assert_eq!(get(0), u32::from(ids::GRANITE.0));
        assert_eq!(get(far), u32::from(ids::PLANKS.0));
        assert_eq!(get(1), 0);
    }

    #[test]
    fn grid_lists_every_box_in_the_cells_it_touches() {
        let boxes = [
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(8.0, 8.0, 8.0)),
            (Vec3::new(100.0, 0.0, 0.0), Vec3::new(140.0, 20.0, 8.0)),
        ];
        let g = build_grid(&boxes);
        assert_eq!(g.min, Vec3::ZERO);
        assert_eq!(g.cell, 16.0);
        assert_eq!(g.dims, IVec3::new(9, 2, 1));
        let cell = |x: i32, y: i32, z: i32| {
            let w = g.words[(x + g.dims.x * (y + g.dims.y * z)) as usize];
            let (off, n) = ((w >> CELL_COUNT_BITS) as usize, (w & 0xfff) as usize);
            g.words[off..off + n].to_vec()
        };
        assert_eq!(cell(0, 0, 0), vec![0]);
        assert_eq!(cell(3, 0, 0), Vec::<u32>::new());
        assert_eq!(cell(6, 0, 0), vec![1]);
        assert_eq!(cell(8, 1, 0), vec![1]);
    }

    #[test]
    fn crowded_grids_coarsen_until_the_lists_fit() {
        // Thousands of large overlapping boxes spread over a wide area.
        let boxes: Vec<(Vec3, Vec3)> = (0..4000)
            .map(|i| {
                let x = (i % 64) as f32 * 40.0;
                let z = (i / 64) as f32 * 40.0;
                (
                    Vec3::new(x, 0.0, z),
                    Vec3::new(x + 1000.0, 1000.0, z + 1000.0),
                )
            })
            .collect();
        let g = build_grid(&boxes);
        assert!(g.words.len() <= GRID_WORDS);
        assert!(g.dims.max_element() <= GRID_DIM);
    }
}
