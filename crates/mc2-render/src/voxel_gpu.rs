//! GPU residency of the voxel world.
//!
//! Two fixed storage buffers hold the tree words and the brick words (see
//! `mc2_voxel::gpu_layout`). Chunk structure (a few kilobytes of nodes and
//! leaf words) is uploaded whenever a chunk changes. Bricks are uploaded on
//! demand: a leaf word starts out `NOT_RESIDENT` carrying the brick's
//! dominant material, the marcher shades it as a 0.5 m LOD cell and writes
//! the leaf word's offset into a feedback table, and the next frame uploads
//! that brick within a byte budget. Bricks near the camera are also pushed
//! proactively so the player's surroundings never show LOD.
//!
//! Above chunks, each 512 m sector gets a small block: its root node, its
//! 128 m nodes, and a copy of every chunk root, so children stay contiguous.

use crate::camera::VOXELS_PER_METRE;
use glam::{DVec3, IVec3, Vec3};
use mc2_core::{FxHashMap, FxHashSet};
use mc2_gpu::alloc::{Range, RangeAllocator};
use mc2_gpu::{bind, bind_group, layout};
use mc2_voxel::coords::{self, CHUNK_SHIFT, ChunkPos, SECTOR_SHIFT};
use mc2_voxel::gpu_layout::{self as gl, Lod};
use mc2_voxel::material::{Kind, MATERIALS, MaterialId};
use mc2_voxel::tree::ChunkTree;
use mc2_voxel::world::VoxelWorld;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

const FEEDBACK_SLOTS: u64 = 4096;
const SECTORS: usize = (coords::WORLD_SECTORS_XZ * coords::WORLD_SECTORS_XZ) as usize;
const CHUNKS_PER_SECTOR: i32 = 1 << (SECTOR_SHIFT - CHUNK_SHIFT);

#[derive(Clone, Copy, Debug)]
pub struct GpuWorldConfig {
    pub tree_words: u32,
    pub voxel_words: u32,
    /// Brick bytes uploaded per frame.
    pub upload_budget: usize,
    /// Bricks within this many metres of the camera are uploaded unasked.
    pub proximity_m: f64,
    /// Main-thread time for chunk structure uploads per frame.
    pub structure_budget_ms: f32,
}

impl Default for GpuWorldConfig {
    fn default() -> Self {
        Self {
            tree_words: 48 << 20,
            voxel_words: 96 << 20,
            upload_budget: 24 << 20,
            proximity_m: 12.0,
            structure_budget_ms: 4.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMaterial {
    albedo: [f32; 3],
    roughness: f32,
    emission: [f32; 3],
    metallic: f32,
    ior: f32,
    kind: u32,
    flags: u32,
    _pad: f32,
}

struct BrickAlloc {
    range: Range,
    state: Option<Range>,
}

struct GpuChunk {
    range: Range,
    bricks: FxHashMap<u32, BrickAlloc>,
    /// `(relative leaf word, slab slot)`, sorted by word.
    leaves: Vec<(u32, u32)>,
    /// Slab slot to relative leaf word.
    slot_leaf: FxHashMap<u32, u32>,
    /// Chunk-local brick cell of every brick, for proximity uploads.
    brick_cells: Vec<(IVec3, u32)>,
    root: [u32; 4],
    lod: Lod,
    fully_resident: bool,
    tree_id: u64,
}

#[derive(Default)]
struct SectorBlock {
    range: Option<Range>,
    /// Chunks of this sector currently on the GPU.
    members: FxHashSet<ChunkPos>,
    /// Absolute offset of each chunk's root copy in the sector block.
    root_copies: FxHashMap<ChunkPos, u32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuWorldStats {
    pub chunks: usize,
    pub pending_chunks: usize,
    pub bricks_resident: usize,
    pub bricks_uploaded_last_frame: usize,
    pub bytes_uploaded_last_frame: usize,
    pub feedback_requests_last_frame: usize,
    pub tree_mb: f32,
    pub voxel_mb: f32,
}

struct Staging {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
}

pub struct GpuWorld {
    pub tree: wgpu::Buffer,
    pub voxels: wgpu::Buffer,
    sectors: wgpu::Buffer,
    materials: wgpu::Buffer,
    feedback: wgpu::Buffer,
    staging: Vec<Staging>,
    pending_copy: Option<usize>,
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    /// Rigid bodies, bound alongside the world (bindings 5-7).
    pub bodies: crate::bodies::BodyGpu,
    tree_alloc: RangeAllocator,
    voxel_alloc: RangeAllocator,
    chunks: FxHashMap<ChunkPos, GpuChunk>,
    by_offset: BTreeMap<u32, ChunkPos>,
    sector_blocks: Vec<SectorBlock>,
    dirty_sectors: FxHashSet<usize>,
    queue: Vec<(ChunkPos, u32)>,
    config: GpuWorldConfig,
    tree_pool_warned: bool,
    /// Chunks changed in the world but not yet re-uploaded.
    pending: FxHashSet<ChunkPos>,
    /// Camera brick cell at the last proximity scan.
    proximity_at: Option<IVec3>,
    /// Chunks whose content reached the GPU (or left it) since `take_changed`.
    changed: Vec<ChunkPos>,
    pub stats: GpuWorldStats,
}

fn sector_index(pos: ChunkPos) -> Option<usize> {
    if !pos.in_world() {
        return None;
    }
    let s = pos.0 >> (SECTOR_SHIFT - CHUNK_SHIFT);
    Some((s.x + s.z * coords::WORLD_SECTORS_XZ) as usize)
}

const FREE: u8 = 0;
const PENDING: u8 = 1;
const MAPPED: u8 = 2;

impl GpuWorld {
    pub fn new(device: &wgpu::Device, config: GpuWorldConfig) -> Self {
        let storage = |label: &str, words: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: words * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
                mapped_at_creation: false,
            })
        };
        let tree = storage(
            "voxel tree",
            u64::from(config.tree_words),
            wgpu::BufferUsages::empty(),
        );
        let voxels = storage(
            "voxel bricks",
            u64::from(config.voxel_words),
            wgpu::BufferUsages::empty(),
        );
        let sectors = storage("voxel sectors", SECTORS as u64, wgpu::BufferUsages::empty());
        let feedback = storage(
            "voxel feedback",
            FEEDBACK_SLOTS,
            wgpu::BufferUsages::COPY_SRC,
        );
        let mats: Vec<GpuMaterial> = MATERIALS
            .iter()
            .map(|m| GpuMaterial {
                albedo: m.albedo,
                roughness: m.roughness,
                emission: m.emission,
                metallic: m.metallic,
                ior: m.ior,
                kind: match m.kind {
                    Kind::Air => 0,
                    Kind::Solid => 1,
                    Kind::Foliage => 2,
                    Kind::Transparent => 3,
                    Kind::Liquid => 4,
                },
                flags: 0,
                _pad: 0.0,
            })
            .collect();
        let materials = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel materials"),
            size: std::mem::size_of_val(mats.as_slice()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        materials
            .get_mapped_range_mut(..)
            .expect("materials mapped at creation")
            .copy_from_slice(bytemuck::cast_slice(&mats));
        materials.unmap();

        let staging = (0..3)
            .map(|i| Staging {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("feedback staging {i}")),
                    size: FEEDBACK_SLOTS * 4,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(FREE)),
            })
            .collect();

        let visibility = wgpu::ShaderStages::COMPUTE;
        let layout = layout(
            device,
            "voxel world",
            visibility,
            &[
                bind::storage(true),
                bind::storage(true),
                bind::storage(true),
                bind::storage(true),
                bind::storage(false),
                bind::storage(true),
                bind::storage(true),
                bind::storage(true),
            ],
        );
        let bodies = crate::bodies::BodyGpu::new(device);
        let bind_group = bind_group(
            device,
            "voxel world",
            &layout,
            &[
                tree.as_entire_binding(),
                voxels.as_entire_binding(),
                sectors.as_entire_binding(),
                materials.as_entire_binding(),
                feedback.as_entire_binding(),
                bodies.table.as_entire_binding(),
                bodies.voxels.as_entire_binding(),
                bodies.grid.as_entire_binding(),
            ],
        );
        Self {
            tree,
            voxels,
            sectors,
            materials,
            feedback,
            staging,
            pending_copy: None,
            layout,
            bind_group,
            bodies,
            tree_alloc: RangeAllocator::new(config.tree_words, &[4]),
            voxel_alloc: RangeAllocator::new(
                config.voxel_words,
                &[
                    gl::NARROW_BRICK_WORDS as u32,
                    gl::WIDE_BRICK_WORDS as u32,
                    gl::STATE_WORDS as u32,
                ],
            ),
            chunks: FxHashMap::default(),
            by_offset: BTreeMap::new(),
            sector_blocks: (0..SECTORS).map(|_| SectorBlock::default()).collect(),
            dirty_sectors: FxHashSet::default(),
            queue: Vec::new(),
            config,
            tree_pool_warned: false,
            pending: FxHashSet::default(),
            proximity_at: None,
            changed: Vec::new(),
            stats: GpuWorldStats::default(),
        }
    }

    pub fn materials_buffer(&self) -> &wgpu::Buffer {
        &self.materials
    }

    fn write_words(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u32, words: &[u32]) {
        queue.write_buffer(buffer, u64::from(offset) * 4, bytemuck::cast_slice(words));
    }

    fn free_chunk(&mut self, pos: ChunkPos) {
        if let Some(c) = self.chunks.remove(&pos) {
            self.tree_alloc.free(c.range);
            self.by_offset.remove(&c.range.offset);
            for b in c.bricks.into_values() {
                self.voxel_alloc.free(b.range);
                if let Some(s) = b.state {
                    self.voxel_alloc.free(s);
                }
            }
        }
        if let Some(s) = sector_index(pos) {
            self.sector_blocks[s].members.remove(&pos);
            self.dirty_sectors.insert(s);
        }
    }

    /// Encodes and writes one brick, reusing its allocation when the size
    /// class is unchanged. Returns bytes written.
    fn upload_brick(
        &mut self,
        queue: &wgpu::Queue,
        pos: ChunkPos,
        tree: &ChunkTree,
        slot: u32,
    ) -> Option<usize> {
        let brick = tree.brick(slot);
        let words = gl::brick_words(brick) as u32;
        let chunk = self.chunks.get_mut(&pos)?;
        let leaf_word = *chunk.slot_leaf.get(&slot)?;
        let old = chunk.bricks.remove(&slot);
        let (range, mut state) = match old {
            Some(a) if a.range.words == words => (a.range, a.state),
            Some(a) => {
                self.voxel_alloc.free(a.range);
                (self.voxel_alloc.alloc(words)?, a.state)
            }
            None => (self.voxel_alloc.alloc(words)?, None),
        };
        let mut state_words = Vec::new();
        let has_state = gl::encode_state(brick, &mut state_words);
        match (has_state, state) {
            (true, None) => state = self.voxel_alloc.alloc(gl::STATE_WORDS as u32),
            (false, Some(s)) => {
                self.voxel_alloc.free(s);
                state = None;
            }
            _ => {}
        }
        let mut out = Vec::with_capacity(words as usize);
        gl::encode_brick(brick, state.map_or(0, |s| s.offset), &mut out);
        out.resize(range.words as usize, 0);
        Self::write_words(queue, &self.voxels, range.offset, &out);
        let mut bytes = out.len() * 4;
        if let Some(s) = state {
            Self::write_words(queue, &self.voxels, s.offset, &state_words);
            bytes += state_words.len() * 4;
        }
        let chunk = self.chunks.get_mut(&pos).expect("chunk checked above");
        chunk.bricks.insert(slot, BrickAlloc { range, state });
        let wide = if brick.is_wide() { gl::WIDE } else { 0 };
        let at = chunk.range.offset + leaf_word;
        Self::write_words(queue, &self.tree, at, &[range.offset | wide]);
        Some(bytes)
    }

    fn upload_structure(
        &mut self,
        queue: &wgpu::Queue,
        pos: ChunkPos,
        tree: &ChunkTree,
        prepared: Option<gl::FlatChunk>,
    ) {
        // A different tree (regenerated at another level of detail) shares no
        // brick slots with the old one: drop its bricks before flattening.
        if self
            .chunks
            .get(&pos)
            .is_some_and(|c| c.tree_id != tree.id())
        {
            let c = self.chunks.get_mut(&pos).expect("checked");
            let bricks: Vec<BrickAlloc> = c.bricks.drain().map(|(_, b)| b).collect();
            c.tree_id = tree.id();
            for b in bricks {
                self.voxel_alloc.free(b.range);
                if let Some(s) = b.state {
                    self.voxel_alloc.free(s);
                }
            }
        }
        let existing = self.chunks.get(&pos);
        let residency = |slot: u32| {
            existing.and_then(|c| {
                c.bricks
                    .get(&slot)
                    .map(|b| (b.range.offset, b.range.words == gl::WIDE_BRICK_WORDS as u32))
            })
        };
        // A worker-prepared flattening assumes no resident bricks.
        let mut flat = match prepared {
            Some(f) if existing.is_none_or(|c| c.bricks.is_empty()) => f,
            _ => gl::flatten_chunk(tree, &residency),
        };
        let live: FxHashSet<u32> = flat.brick_leaves.iter().map(|&(_, s)| s).collect();

        let mut chunk = match self.chunks.remove(&pos) {
            Some(c) => {
                self.by_offset.remove(&c.range.offset);
                c
            }
            None => GpuChunk {
                range: Range {
                    offset: 0,
                    words: 0,
                },
                bricks: FxHashMap::default(),
                leaves: Vec::new(),
                slot_leaf: FxHashMap::default(),
                brick_cells: Vec::new(),
                root: [0; 4],
                lod: flat.lod,
                fully_resident: false,
                tree_id: tree.id(),
            },
        };
        chunk.bricks.retain(|slot, alloc| {
            let keep = live.contains(slot);
            if !keep {
                self.voxel_alloc.free(alloc.range);
                if let Some(s) = alloc.state {
                    self.voxel_alloc.free(s);
                }
            }
            keep
        });
        let words = flat.words.len() as u32;
        if chunk.range.words < words
            || chunk.range.words > words.saturating_mul(4)
            || chunk.range.words == 0
        {
            if chunk.range.words != 0 {
                self.tree_alloc.free(chunk.range);
            }
            match self.tree_alloc.alloc(words) {
                Some(r) => chunk.range = r,
                None => {
                    if !self.tree_pool_warned {
                        log::error!(
                            "voxel tree pool exhausted; chunks beyond this are not uploaded"
                        );
                        self.tree_pool_warned = true;
                    }
                    return;
                }
            }
        }
        flat.relocate(chunk.range.offset);
        Self::write_words(queue, &self.tree, chunk.range.offset, &flat.words);
        chunk.root = [flat.words[0], flat.words[1], flat.words[2], flat.words[3]];
        chunk.lod = flat.lod;
        chunk.leaves = flat.brick_leaves;
        chunk.leaves.sort_unstable();
        chunk.slot_leaf = chunk.leaves.iter().map(|&(w, s)| (s, w)).collect();
        chunk.brick_cells = tree
            .cells()
            .filter_map(|(c, cell)| match cell {
                mc2_voxel::tree::Cell::Brick(slot) => Some((c, slot)),
                _ => None,
            })
            .collect();
        chunk.fully_resident = chunk
            .leaves
            .iter()
            .all(|(_, s)| chunk.bricks.contains_key(s));
        self.by_offset.insert(chunk.range.offset, pos);

        let sector = sector_index(pos);
        let copy = sector.and_then(|s| self.sector_blocks[s].root_copies.get(&pos).copied());
        match copy {
            Some(offset) => Self::write_words(queue, &self.tree, offset, &chunk.root),
            None => {
                if let Some(s) = sector {
                    self.sector_blocks[s].members.insert(pos);
                    self.dirty_sectors.insert(s);
                }
            }
        }
        self.chunks.insert(pos, chunk);
    }

    fn rebuild_sector(&mut self, queue: &wgpu::Queue, s: usize) {
        let sx = s as i32 % coords::WORLD_SECTORS_XZ;
        let sz = s as i32 / coords::WORLD_SECTORS_XZ;
        let origin = IVec3::new(sx, 0, sz) * CHUNKS_PER_SECTOR;
        // Group chunks by their 128 m node.
        let mut l4: BTreeMap<u32, Vec<(u32, ChunkPos)>> = BTreeMap::new();
        for &pos in &self.sector_blocks[s].members {
            if !self.chunks.contains_key(&pos) {
                continue;
            }
            let local = pos.0 - origin;
            let a = mc2_voxel::tree::child_index(local >> 2);
            let c = mc2_voxel::tree::child_index(local & 3);
            l4.entry(a).or_default().push((c, pos));
        }
        let block = &mut self.sector_blocks[s];
        if let Some(r) = block.range.take() {
            self.tree_alloc.free(r);
        }
        block.root_copies.clear();
        if l4.is_empty() {
            Self::write_words(queue, &self.sectors, s as u32, &[0]);
            return;
        }
        let n4 = l4.len();
        let total_chunks: usize = l4.values().map(Vec::len).sum();
        let words = 4 + 4 * n4 + 4 * total_chunks;
        let Some(range) = self.tree_alloc.alloc(words as u32) else {
            log::error!("voxel tree pool exhausted; sector {s} not linked");
            return;
        };
        let base = range.offset;
        let mut out = vec![0u32; range.words as usize];
        let mut root_mask = 0u64;
        let mut root_lods = Vec::new();
        let mut next = 4 + 4 * n4;
        for (k, (a, list)) in l4.iter_mut().enumerate() {
            list.sort_by_key(|(c, _)| *c);
            let mut mask = 0u64;
            let mut lods = Vec::new();
            let child_block = next;
            for (j, (c, pos)) in list.iter().enumerate() {
                mask |= 1 << c;
                let chunk = &self.chunks[pos];
                let at = child_block + j * 4;
                out[at..at + 4].copy_from_slice(&chunk.root);
                lods.push(chunk.lod);
                self.sector_blocks[s]
                    .root_copies
                    .insert(*pos, base + at as u32);
            }
            next += list.len() * 4;
            let lod = combine_lods(&lods);
            let at = 4 + k * 4;
            out[at..at + 4].copy_from_slice(&[
                base + child_block as u32,
                mask as u32,
                (mask >> 32) as u32,
                gl::pack_lod(&lod),
            ]);
            root_mask |= 1 << a;
            root_lods.push(lod);
        }
        let lod = combine_lods(&root_lods);
        out[..4].copy_from_slice(&[
            base + 4,
            root_mask as u32,
            (root_mask >> 32) as u32,
            gl::pack_lod(&lod),
        ]);
        Self::write_words(queue, &self.tree, base, &out);
        Self::write_words(queue, &self.sectors, s as u32, &[base]);
        self.sector_blocks[s].range = Some(range);
    }

    /// Queues up to `limit` non-resident bricks within the proximity radius.
    fn queue_proximity(&mut self, camera_m: DVec3, limit: usize) {
        if self.config.proximity_m <= 0.0 {
            return;
        }
        let cam_v = camera_m * VOXELS_PER_METRE;
        let cell = (cam_v / 8.0).floor().as_ivec3();
        if self.proximity_at == Some(cell) {
            return;
        }
        let r_v = self.config.proximity_m * VOXELS_PER_METRE;
        let mut found = Vec::new();
        for (pos, chunk) in &self.chunks {
            if chunk.fully_resident {
                continue;
            }
            let min = pos.origin().as_dvec3();
            let max = min + f64::from(coords::CHUNK_VOXELS);
            if cam_v.clamp(min, max).distance(cam_v) > r_v {
                continue;
            }
            for &(c, slot) in &chunk.brick_cells {
                if chunk.bricks.contains_key(&slot) {
                    continue;
                }
                let centre = min + (c * 8).as_dvec3() + 4.0;
                if centre.distance(cam_v) <= r_v {
                    found.push((*pos, slot));
                }
            }
            if found.len() >= limit {
                break;
            }
        }
        // Only a scan that found everything may be skipped next frame.
        if found.len() < limit {
            self.proximity_at = Some(cell);
        }
        self.queue.extend(found);
    }

    /// Streams structure, bricks and sector links for this frame.
    pub fn update(&mut self, queue: &wgpu::Queue, world: &mut VoxelWorld, camera_m: DVec3) {
        mc2_core::scope!("voxel_gpu.update");
        let mut uploaded = 0usize;
        let mut bytes = 0usize;

        let dirty_scope = mc2_core::profiler::ScopeGuard::new("voxel_gpu.dirty");
        self.pending.extend(world.take_dirty());
        let mut order: Vec<ChunkPos> = self.pending.iter().copied().collect();
        let cam_v = camera_m * VOXELS_PER_METRE;
        order.sort_by_key(|p| {
            let c = p.origin().as_dvec3() + f64::from(coords::CHUNK_VOXELS / 2);
            c.distance_squared(cam_v) as u64
        });
        let started = std::time::Instant::now();
        for pos in order {
            if started.elapsed().as_secs_f32() * 1000.0 > self.config.structure_budget_ms {
                break;
            }
            self.pending.remove(&pos);
            self.changed.push(pos);
            let Some(dirty) = world.take_chunk_dirty(pos) else {
                self.free_chunk(pos);
                continue;
            };
            let (structure, bricks) = dirty;
            let prepared = world.chunk_untracked(pos).and_then(ChunkTree::take_flat);
            let tree = world.chunk(pos).expect("chunk has dirty state");
            if structure
                || !self.chunks.contains_key(&pos)
                || self.chunks[&pos].tree_id != tree.id()
            {
                self.upload_structure(queue, pos, tree, prepared);
            }
            for slot in bricks {
                let resident = self
                    .chunks
                    .get(&pos)
                    .is_some_and(|c| c.bricks.contains_key(&slot));
                if resident && let Some(b) = self.upload_brick(queue, pos, tree, slot) {
                    bytes += b;
                    uploaded += 1;
                }
            }
        }
        drop(dirty_scope);
        let requests = self.harvest_feedback();
        self.stats.feedback_requests_last_frame = requests.len();
        for word in requests {
            let Some((&base, &pos)) = self.by_offset.range(..=word).next_back() else {
                continue;
            };
            let Some(chunk) = self.chunks.get(&pos) else {
                continue;
            };
            let rel = word - base;
            if rel >= chunk.range.words {
                continue;
            }
            if let Ok(i) = chunk.leaves.binary_search_by_key(&rel, |&(w, _)| w) {
                self.queue.push((pos, chunk.leaves[i].1));
            }
        }
        let prox_scope = mc2_core::profiler::ScopeGuard::new("voxel_gpu.proximity");
        if self.queue.len() < 4096 {
            self.queue_proximity(camera_m, 4096);
        }

        drop(prox_scope);
        let bricks_scope = mc2_core::profiler::ScopeGuard::new("voxel_gpu.bricks");
        // Nearest requests first.
        let cam_v = camera_m * VOXELS_PER_METRE;
        self.queue.sort_by_key(|(pos, _)| {
            let c = pos.origin().as_dvec3() + f64::from(coords::CHUNK_VOXELS / 2);
            (c.distance_squared(cam_v) as u64, pos.0.x, pos.0.y, pos.0.z)
        });
        self.queue.dedup();
        let mut remaining = Vec::new();
        for (pos, slot) in std::mem::take(&mut self.queue) {
            if bytes >= self.config.upload_budget {
                remaining.push((pos, slot));
                continue;
            }
            let Some(tree) = world.chunk(pos) else {
                continue;
            };
            let resident = self
                .chunks
                .get(&pos)
                .is_none_or(|c| c.bricks.contains_key(&slot));
            if resident {
                continue;
            }
            match self.upload_brick(queue, pos, tree, slot) {
                Some(b) => {
                    bytes += b;
                    uploaded += 1;
                }
                None => {
                    if self.voxel_alloc.high_water() + gl::WIDE_BRICK_WORDS as u32
                        > self.voxel_alloc.capacity()
                    {
                        log::warn!("voxel brick pool exhausted");
                        break;
                    }
                }
            }
        }
        self.queue = remaining;
        drop(bricks_scope);
        let resident_scope = mc2_core::profiler::ScopeGuard::new("voxel_gpu.resident");
        for chunk in self.chunks.values_mut() {
            if !chunk.fully_resident {
                chunk.fully_resident = chunk
                    .leaves
                    .iter()
                    .all(|(_, s)| chunk.bricks.contains_key(s));
            }
        }

        drop(resident_scope);
        let _sector_scope = mc2_core::profiler::ScopeGuard::new("voxel_gpu.sectors");
        for s in std::mem::take(&mut self.dirty_sectors) {
            self.rebuild_sector(queue, s);
        }

        self.stats.chunks = self.chunks.len();
        self.stats.pending_chunks = self.pending.len();
        self.stats.bricks_resident = self.chunks.values().map(|c| c.bricks.len()).sum();
        self.stats.bricks_uploaded_last_frame = uploaded;
        self.stats.bytes_uploaded_last_frame = bytes;
        self.stats.tree_mb = self.tree_alloc.used_words() as f32 * 4.0 / 1.0e6;
        self.stats.voxel_mb = self.voxel_alloc.used_words() as f32 * 4.0 / 1.0e6;
    }

    /// Chunks uploaded or removed since the last call.
    pub fn take_changed(&mut self) -> Vec<ChunkPos> {
        std::mem::take(&mut self.changed)
    }

    /// Clears the GPU feedback table; call before dispatching marchers.
    pub fn begin_frame(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.feedback, 0, &[0u8; (FEEDBACK_SLOTS * 4) as usize]);
    }

    /// Copies the feedback table for readback; call after the marchers.
    pub fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.pending_copy = None;
        if let Some(i) = self
            .staging
            .iter()
            .position(|s| s.state.load(Ordering::Acquire) == FREE)
        {
            encoder.copy_buffer_to_buffer(
                &self.feedback,
                0,
                &self.staging[i].buffer,
                0,
                FEEDBACK_SLOTS * 4,
            );
            self.pending_copy = Some(i);
        }
    }

    pub fn after_submit(&mut self) {
        if let Some(i) = self.pending_copy.take() {
            let st = self.staging[i].state.clone();
            st.store(PENDING, Ordering::Release);
            self.staging[i]
                .buffer
                .map_async(wgpu::MapMode::Read, .., move |r| {
                    st.store(if r.is_ok() { MAPPED } else { FREE }, Ordering::Release);
                });
        }
    }

    fn harvest_feedback(&mut self) -> Vec<u32> {
        let mut out = Vec::new();
        for s in &self.staging {
            if s.state.load(Ordering::Acquire) != MAPPED {
                continue;
            }
            if let Ok(view) = s.buffer.get_mapped_range(..) {
                out.extend(
                    view.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| u32::from_le_bytes(*b))
                        .filter(|&w| w != 0)
                        .map(|w| w - 1),
                );
            }
            s.buffer.unmap();
            s.state.store(FREE, Ordering::Release);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn is_brick_resident(&self, pos: ChunkPos, slot: u32) -> bool {
        self.chunks
            .get(&pos)
            .is_some_and(|c| c.bricks.contains_key(&slot))
    }
}

fn combine_lods(lods: &[Lod]) -> Lod {
    let mut hist: Vec<(MaterialId, f32)> = Vec::new();
    let mut normal = Vec3::ZERO;
    let mut coverage = 0.0;
    let mut spread = 0.0;
    for l in lods {
        coverage += l.coverage;
        normal += l.normal * l.coverage * (1.0 - l.spread * 0.5);
        spread += l.spread * l.coverage;
        match hist.iter_mut().find(|(m, _)| *m == l.material) {
            Some((_, w)) => *w += l.coverage,
            None => hist.push((l.material, l.coverage)),
        }
    }
    let material = hist
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(MaterialId(0), |h| h.0);
    let n = normal.normalize_or(Vec3::Y);
    Lod {
        material,
        normal: n,
        spread: if coverage > 0.0 {
            (spread / coverage).clamp(0.0, 1.0)
        } else {
            1.0
        },
        coverage: coverage / 64.0,
    }
}
