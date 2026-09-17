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
use mc2_core::FxHashMap;
use mc2_gpu::alloc::{Range, RangeAllocator};
use mc2_physics::BodyShape;
use std::sync::Arc;

/// Bodies drawn per frame; farther ones are dropped first.
pub const MAX_BODIES: usize = 1024;
/// Bodies farther than this from the camera are not drawn, metres.
pub const DRAW_DISTANCE_M: f64 = 384.0;
/// Shape pool, words (16 MB).
const VOXEL_WORDS: u32 = 1 << 22;
/// Culling grid cells per axis, at most.
pub const GRID_DIM: i32 = 32;
/// Header words before the cell array.
const GRID_HEADER_WORDS: usize = 8;
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

struct ShapeSlot {
    range: Range,
    coarse: u32,
    shape: Arc<BodyShape>,
    seen: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BodyGpuStats {
    pub drawn: usize,
    pub pool_words: u64,
    pub list_words: usize,
}

pub struct BodyGpu {
    pub table: wgpu::Buffer,
    pub voxels: wgpu::Buffer,
    pub grid: wgpu::Buffer,
    alloc: RangeAllocator,
    shapes: FxHashMap<u64, ShapeSlot>,
    /// Pose drawn last frame, world metres.
    prev: FxHashMap<u64, (DVec3, Quat)>,
    frame: u64,
    pub stats: BodyGpuStats,
}

impl BodyGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            table: buffer(
                "body table",
                (MAX_BODIES * std::mem::size_of::<GpuBody>()) as u64,
            ),
            voxels: buffer("body voxels", u64::from(VOXEL_WORDS) * 4),
            grid: buffer("body grid", ((GRID_HEADER_WORDS + GRID_WORDS) * 4) as u64),
            alloc: RangeAllocator::new(VOXEL_WORDS, &[]),
            shapes: FxHashMap::default(),
            prev: FxHashMap::default(),
            frame: 0,
            stats: BodyGpuStats::default(),
        }
    }

    /// Uploads new shapes, frees shapes no longer drawn, and writes this
    /// frame's table and culling grid.
    pub fn update(&mut self, queue: &wgpu::Queue, bodies: &[BodyInstance], camera_m: DVec3) {
        mc2_core::scope!("bodies.update");
        self.frame += 1;
        let camera_voxel = (camera_m * 16.0).floor();
        let mut visible: Vec<&BodyInstance> = bodies
            .iter()
            .filter(|b| b.grid_origin.distance(camera_m) < DRAW_DISTANCE_M)
            .collect();
        if visible.len() > MAX_BODIES {
            visible.sort_by(|a, b| {
                a.grid_origin
                    .distance_squared(camera_m)
                    .total_cmp(&b.grid_origin.distance_squared(camera_m))
            });
            visible.truncate(MAX_BODIES);
        }
        let mut table = Vec::with_capacity(visible.len());
        let mut boxes = Vec::with_capacity(visible.len());
        let mut next_prev = FxHashMap::default();
        for b in visible {
            let Some((voxels, coarse)) = self.slot(queue, b) else {
                continue;
            };
            let rel = |p: DVec3| (p * 16.0 - camera_voxel).as_vec3();
            let axes = |q: Quat| [q * Vec3::X, q * Vec3::Y, q * Vec3::Z];
            let (prev_origin, prev_rot) = self
                .prev
                .get(&b.key)
                .copied()
                .unwrap_or((b.grid_origin, b.grid_rotation));
            let a = axes(b.grid_rotation);
            let pa = axes(prev_rot);
            let origin = rel(b.grid_origin);
            let size = b.shape.size;
            table.push(GpuBody {
                origin: origin.to_array(),
                tag: (b.key & 0x7ff) as u32,
                axis_x: a[0].to_array(),
                voxels,
                axis_y: a[1].to_array(),
                coarse,
                axis_z: a[2].to_array(),
                size: size.to_array(),
                prev_origin: rel(prev_origin).to_array(),
                prev_axis_x: pa[0].to_array(),
                prev_axis_y: pa[1].to_array(),
                prev_axis_z: pa[2].to_array(),
                ..Default::default()
            });
            // Camera-relative bounds of the rotated grid box.
            let half = size.as_vec3() * 0.5;
            let centre = origin + a[0] * half.x + a[1] * half.y + a[2] * half.z;
            let reach = a[0].abs() * half.x + a[1].abs() * half.y + a[2].abs() * half.z;
            boxes.push((centre - reach, centre + reach));
            next_prev.insert(b.key, (b.grid_origin, b.grid_rotation));
        }
        self.prev = next_prev;
        // Shapes of bodies not drawn for a second are released.
        let frame = self.frame;
        let alloc = &mut self.alloc;
        self.shapes.retain(|_, s| {
            let keep = frame - s.seen < 60;
            if !keep {
                alloc.free(s.range);
            }
            keep
        });

        let grid = build_grid(&boxes);
        let mut header = [0u32; GRID_HEADER_WORDS];
        header[0..3].copy_from_slice(bytemuck::cast_slice(&grid.min.to_array()));
        header[3] = grid.cell.to_bits();
        header[4..7].copy_from_slice(bytemuck::cast_slice(&grid.dims.to_array()));
        header[7] = table.len() as u32;
        queue.write_buffer(&self.grid, 0, bytemuck::cast_slice(&header));
        if !grid.words.is_empty() {
            queue.write_buffer(
                &self.grid,
                (GRID_HEADER_WORDS * 4) as u64,
                bytemuck::cast_slice(&grid.words),
            );
        }
        if !table.is_empty() {
            queue.write_buffer(&self.table, 0, bytemuck::cast_slice(&table));
        }
        self.stats = BodyGpuStats {
            drawn: table.len(),
            pool_words: self.alloc.used_words(),
            list_words: grid.words.len(),
        };
    }

    /// The pool offsets of a body's shape, uploading it on first sight.
    fn slot(&mut self, queue: &wgpu::Queue, b: &BodyInstance) -> Option<(u32, u32)> {
        if let Some(s) = self.shapes.get_mut(&b.key) {
            if Arc::ptr_eq(&s.shape, &b.shape) {
                s.seen = self.frame;
                return Some((s.range.offset + s.coarse, s.range.offset));
            }
            // The body's shape changed (it broke): upload the new one.
            let old = self.shapes.remove(&b.key).expect("slot present");
            self.alloc.free(old.range);
        }
        let (words, coarse) = encode_shape(&b.shape);
        let Some(range) = self.alloc.alloc(words.len() as u32) else {
            log::warn!("body voxel pool full; body {} not drawn", b.key);
            return None;
        };
        queue.write_buffer(
            &self.voxels,
            u64::from(range.offset) * 4,
            bytemuck::cast_slice(&words),
        );
        self.shapes.insert(
            b.key,
            ShapeSlot {
                range,
                coarse,
                shape: b.shape.clone(),
                seen: self.frame,
            },
        );
        Some((range.offset + coarse, range.offset))
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
