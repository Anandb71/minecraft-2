//! GPU word layout of the voxel structure, and chunk flattening into it.
//!
//! Everything the marcher reads lives in two `array<u32>` storage buffers.
//!
//! **Tree buffer.** A node is four words:
//!
//! | word | bits   | meaning                                               |
//! |------|--------|-------------------------------------------------------|
//! | 0    | 31     | `LEAF_PARENT`: children are brick-cell leaf words     |
//! | 0    | 30     | `NOT_RESIDENT`: children are not on the GPU           |
//! | 0    | 29     | `UNIFORM_NODE`: one material throughout (bits 0..16)  |
//! | 0    | 0..29  | absolute word offset of the compact child block       |
//! | 1, 2 |        | child mask, low and high halves                       |
//! | 3    | 0..16  | LOD material                                          |
//! | 3    | 16..28 | LOD normal, octahedral 6+6 bits                       |
//! | 3    | 28..32 | normal spread 0 (flat) ..15 (no dominant direction)   |
//!
//! A leaf word (child of a `LEAF_PARENT` node) is one of:
//! - `UNIFORM` (bit 31) with the material in bits 0..16,
//! - `NOT_RESIDENT` (bit 30) with the brick's dominant material in 0..16,
//! - a brick: word offset into the voxel buffer in bits 0..29, `WIDE` bit 29.
//!
//! **Voxel buffer.** A narrow brick is 89 words: header (state offset, 0 for
//! none), 16 occupancy words (eight `u64`), 8 palette words (sixteen `u16`),
//! 64 index words (eight nibbles each, voxel `i` at word `i/8` bit `4*(i%8)`).
//! A wide brick is 273 words: header, occupancy, 128 palette words, 128
//! index words (four bytes each). A state block is 128 words, four state
//! bytes per word.

use crate::brick::{Brick, BrickRaw};
use crate::material::MaterialId;
use crate::tree::{Cell, ChunkTree, L1, L2, child_coord};
use glam::Vec3;

pub const NODE_WORDS: usize = 4;
pub const LEAF_PARENT: u32 = 1 << 31;
pub const NOT_RESIDENT: u32 = 1 << 30;
pub const UNIFORM_NODE: u32 = 1 << 29;
pub const PTR_MASK: u32 = (1 << 29) - 1;
pub const UNIFORM: u32 = 1 << 31;
pub const WIDE: u32 = 1 << 29;
pub const BRICK_PTR_MASK: u32 = (1 << 29) - 1;

pub const NARROW_BRICK_WORDS: usize = 89;
pub const WIDE_BRICK_WORDS: usize = 273;
pub const STATE_WORDS: usize = 128;

/// Filtered appearance of a subtree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lod {
    pub material: MaterialId,
    pub normal: Vec3,
    /// 0 = every surface faces `normal`, 1 = no preferred direction.
    pub spread: f32,
    /// Solid fraction of the node's volume.
    pub coverage: f32,
}

pub fn oct_encode(n: Vec3) -> (f32, f32) {
    let l1 = n.x.abs() + n.y.abs() + n.z.abs();
    let mut p = (n.x / l1, n.z / l1);
    if n.y < 0.0 {
        p = (
            (1.0 - p.1.abs()) * p.0.signum(),
            (1.0 - p.0.abs()) * p.1.signum(),
        );
    }
    (p.0 * 0.5 + 0.5, p.1 * 0.5 + 0.5)
}

pub fn pack_lod(lod: &Lod) -> u32 {
    let (u, v) = oct_encode(lod.normal);
    let q = |x: f32| ((x * 63.0).round() as u32).min(63);
    let spread = ((lod.spread * 15.0).round() as u32).min(15);
    u32::from(lod.material.0) | q(u) << 16 | q(v) << 22 | spread << 28
}

/// Material histogram small enough to live on the stack for most nodes.
#[derive(Default)]
struct Histogram(Vec<(MaterialId, u32)>);

impl Histogram {
    fn add(&mut self, m: MaterialId, n: u32) {
        if m.is_air() || n == 0 {
            return;
        }
        match self.0.iter_mut().find(|(k, _)| *k == m) {
            Some((_, c)) => *c += n,
            None => self.0.push((m, n)),
        }
    }

    fn merge(&mut self, other: &Histogram) {
        for &(m, n) in &other.0 {
            self.add(m, n);
        }
    }

    /// Dominant entry by voxel coverage, never by averaging colours.
    fn dominant(&self) -> MaterialId {
        self.0
            .iter()
            .max_by_key(|&&(m, n)| (n, std::cmp::Reverse(m.0)))
            .map_or(MaterialId(0), |&(m, _)| m)
    }
}

struct Summary {
    hist: Histogram,
    coverage: f32,
    lod: Lod,
}

fn summarise(children: impl Iterator<Item = (u32, f32, Histogram)>) -> Summary {
    let mut hist = Histogram::default();
    let mut gradient = Vec3::ZERO;
    let mut total = 0.0;
    let mut per_child_max = 0.0;
    for (i, coverage, h) in children {
        let offset = child_coord(i).as_vec3() - Vec3::splat(1.5);
        gradient += offset * coverage;
        per_child_max += offset.length();
        total += coverage;
        hist.merge(&h);
    }
    let coverage = total / 64.0;
    // Solid mass sits opposite the surface normal.
    let g = -gradient;
    let len = g.length();
    let (normal, spread) = if len > 1e-4 {
        (
            g / len,
            1.0 - (len / per_child_max.max(1e-4)).clamp(0.0, 1.0),
        )
    } else {
        (Vec3::Y, 1.0)
    };
    let material = hist.dominant();
    Summary {
        hist,
        coverage,
        lod: Lod {
            material,
            normal,
            spread,
            coverage,
        },
    }
}

fn cell_summary(tree: &ChunkTree, cell: Cell) -> (f32, Histogram) {
    let mut h = Histogram::default();
    match cell {
        Cell::Empty => (0.0, h),
        Cell::Uniform(m) => {
            h.add(m, 512);
            (1.0, h)
        }
        Cell::Brick(i) => {
            let b = tree.brick(i);
            for (m, n) in b.materials() {
                h.add(m, u32::from(n));
            }
            (b.solid_count() as f32 / 512.0, h)
        }
    }
}

/// How a brick cell's leaf word should refer to its brick.
pub trait BrickResidency {
    /// Voxel-buffer word offset of the brick in slab slot `slot`, if uploaded.
    fn brick_offset(&self, slot: u32) -> Option<(u32, bool)>;
}

impl<F: Fn(u32) -> Option<(u32, bool)>> BrickResidency for F {
    fn brick_offset(&self, slot: u32) -> Option<(u32, bool)> {
        self(slot)
    }
}

/// A chunk flattened into tree words with pointers relative to the block start.
#[derive(Clone, Debug)]
pub struct FlatChunk {
    pub words: Vec<u32>,
    /// `(word index, slab slot)` of every leaf word that refers to a brick.
    pub brick_leaves: Vec<(u32, u32)>,
    pub lod: Lod,
}

impl FlatChunk {
    /// Rebases child pointers by `base`. Leaf words hold voxel buffer offsets
    /// and material ids, which are not rebased.
    pub fn relocate(&mut self, base: u32) {
        let mut stack = vec![0usize];
        while let Some(n) = stack.pop() {
            let w0 = self.words[n];
            if w0 & (NOT_RESIDENT | UNIFORM_NODE) != 0 {
                continue;
            }
            let rel = (w0 & PTR_MASK) as usize;
            self.words[n] = (w0 & !PTR_MASK) | (rel as u32 + base);
            if w0 & LEAF_PARENT == 0 {
                let count = (u64::from(self.words[n + 1]) | u64::from(self.words[n + 2]) << 32)
                    .count_ones() as usize;
                stack.extend((0..count).map(|k| rel + k * NODE_WORDS));
            }
        }
    }
}

fn node_words(child_ptr: u32, flags: u32, mask: u64, lod: &Lod) -> [u32; 4] {
    [
        child_ptr | flags,
        mask as u32,
        (mask >> 32) as u32,
        pack_lod(lod),
    ]
}

/// Flattens a chunk: root node at word 0, then each level's child blocks.
pub fn flatten_chunk(tree: &ChunkTree, bricks: &impl BrickResidency) -> FlatChunk {
    let mut words = vec![0u32; NODE_WORDS];
    let mut brick_leaves = Vec::new();

    // Children blocks are appended breadth first so a node's block is contiguous.
    let l2_block = words.len();
    words.resize(l2_block + tree.root.items.len() * NODE_WORDS, 0);
    let mut l2_summaries = Vec::with_capacity(tree.root.items.len());
    for (k, (i2, l2)) in tree.root.iter().enumerate() {
        let (words_l2, cov, hist) = flatten_l2(tree, l2, &mut words, &mut brick_leaves, bricks);
        words[l2_block + k * NODE_WORDS..l2_block + (k + 1) * NODE_WORDS]
            .copy_from_slice(&words_l2);
        l2_summaries.push((i2, cov, hist));
    }
    let s = summarise(l2_summaries.into_iter());
    let root = node_words(l2_block as u32, 0, tree.root.mask, &s.lod);
    words[..NODE_WORDS].copy_from_slice(&root);
    FlatChunk {
        words,
        brick_leaves,
        lod: s.lod,
    }
}

fn uniform_node(material: MaterialId, voxels: u32) -> ([u32; 4], f32, Histogram) {
    let lod = Lod {
        material,
        normal: Vec3::Y,
        spread: 1.0,
        coverage: 1.0,
    };
    let mut hist = Histogram::default();
    hist.add(material, voxels);
    (
        [
            UNIFORM_NODE | u32::from(material.0),
            u32::MAX,
            u32::MAX,
            pack_lod(&lod),
        ],
        1.0,
        hist,
    )
}

fn flatten_l2(
    tree: &ChunkTree,
    l2: &L2,
    words: &mut Vec<u32>,
    leaves: &mut Vec<(u32, u32)>,
    bricks: &impl BrickResidency,
) -> ([u32; 4], f32, Histogram) {
    let nodes = match l2 {
        L2::Uniform(m) => return uniform_node(*m, 128 * 128 * 128),
        L2::Nodes(nodes) => nodes,
    };
    let block = words.len();
    words.resize(block + nodes.items.len() * NODE_WORDS, 0);
    let mut summaries = Vec::with_capacity(nodes.items.len());
    for (k, (i1, l1)) in nodes.iter().enumerate() {
        let (w, cov, hist) = flatten_l1(tree, l1, words, leaves, bricks);
        words[block + k * NODE_WORDS..block + (k + 1) * NODE_WORDS].copy_from_slice(&w);
        summaries.push((i1, cov, hist));
    }
    let s = summarise(summaries.into_iter());
    (
        node_words(block as u32, 0, nodes.mask, &s.lod),
        s.coverage,
        s.hist,
    )
}

fn flatten_l1(
    tree: &ChunkTree,
    l1: &L1,
    words: &mut Vec<u32>,
    leaves: &mut Vec<(u32, u32)>,
    bricks: &impl BrickResidency,
) -> ([u32; 4], f32, Histogram) {
    let cells = match l1 {
        L1::Uniform(m) => return uniform_node(*m, 32 * 32 * 32),
        L1::Cells(cells) => cells,
    };
    let block = words.len();
    let mut summaries = Vec::with_capacity(cells.items.len());
    for (i, &cell) in cells.iter() {
        let word = match cell {
            Cell::Empty => unreachable!("empty cells are absent from the mask"),
            Cell::Uniform(m) => UNIFORM | u32::from(m.0),
            Cell::Brick(slot) => {
                leaves.push((words.len() as u32, slot));
                match bricks.brick_offset(slot) {
                    Some((offset, wide)) => offset | if wide { WIDE } else { 0 },
                    None => {
                        let dominant = tree.brick(slot).dominant().map_or(0, |(m, _)| m.0);
                        NOT_RESIDENT | u32::from(dominant)
                    }
                }
            }
        };
        words.push(word);
        let (cov, hist) = cell_summary(tree, cell);
        summaries.push((i, cov, hist));
    }
    let s = summarise(summaries.into_iter());
    (
        node_words(block as u32, LEAF_PARENT, cells.mask, &s.lod),
        s.coverage,
        s.hist,
    )
}

/// Encodes a brick into voxel-buffer words. `state_offset` is written to the
/// header; pass 0 when the brick has no state block.
pub fn encode_brick(b: &Brick, state_offset: u32, out: &mut Vec<u32>) {
    out.push(state_offset);
    for m in b.occupancy() {
        out.push(*m as u32);
        out.push((*m >> 32) as u32);
    }
    match b.raw() {
        BrickRaw::Narrow { palette, nibbles } => {
            for pair in palette.as_chunks::<2>().0 {
                out.push(u32::from(pair[0].0) | u32::from(pair[1].0) << 16);
            }
            for bytes in nibbles.as_chunks::<4>().0 {
                out.push(u32::from_le_bytes(*bytes));
            }
        }
        BrickRaw::Wide { palette, indices } => {
            let mut full = [0u16; 256];
            for (dst, m) in full.iter_mut().zip(palette) {
                *dst = m.0;
            }
            for pair in full.as_chunks::<2>().0 {
                out.push(u32::from(pair[0]) | u32::from(pair[1]) << 16);
            }
            for bytes in indices.as_chunks::<4>().0 {
                out.push(u32::from_le_bytes(*bytes));
            }
        }
    }
}

pub fn encode_state(b: &Brick, out: &mut Vec<u32>) -> bool {
    let Some(state) = b.raw_state() else {
        return false;
    };
    for bytes in state.as_chunks::<4>().0 {
        out.push(u32::from_le_bytes([
            bytes[0].0, bytes[1].0, bytes[2].0, bytes[3].0,
        ]));
    }
    true
}

pub fn brick_words(b: &Brick) -> usize {
    if b.is_wide() {
        WIDE_BRICK_WORDS
    } else {
        NARROW_BRICK_WORDS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::ids;
    use glam::IVec3;

    /// Reads a voxel back out of flattened words the way the shader does.
    fn gpu_voxel(tree: &[u32], voxels: &[u32], root: usize, v: IVec3) -> u16 {
        let mut node = root;
        let mut shift: i32 = 7; // chunk root children are 128 voxels
        loop {
            let c = (v >> shift) & 3;
            let idx = (c.x + (c.z << 2) + (c.y << 4)) as u32;
            let mask = u64::from(tree[node + 1]) | u64::from(tree[node + 2]) << 32;
            if mask >> idx & 1 == 0 {
                return 0;
            }
            let slot = (mask & ((1u64 << idx) - 1)).count_ones() as usize;
            if tree[node] & UNIFORM_NODE != 0 {
                return tree[node] as u16;
            }
            let ptr = (tree[node] & PTR_MASK) as usize;
            if tree[node] & LEAF_PARENT != 0 {
                let leaf = tree[ptr + slot];
                if leaf & UNIFORM != 0 || leaf & NOT_RESIDENT != 0 {
                    return leaf as u16;
                }
                let base = (leaf & BRICK_PTR_MASK) as usize;
                let l = v & 7;
                let i = (l.x + l.z * 8 + l.y * 64) as usize;
                if leaf & WIDE != 0 {
                    let p = (voxels[base + 145 + i / 4] >> (8 * (i % 4))) & 0xff;
                    return (voxels[base + 17 + p as usize / 2] >> (16 * (p & 1))) as u16;
                }
                let p = (voxels[base + 25 + i / 8] >> (4 * (i % 8))) & 0xf;
                return (voxels[base + 17 + p as usize / 2] >> (16 * (p & 1))) as u16;
            }
            node = ptr + slot * NODE_WORDS;
            if tree[node] & UNIFORM_NODE != 0 {
                return tree[node] as u16;
            }
            shift -= 2;
        }
    }

    #[test]
    fn flattened_words_decode_to_the_same_voxels() {
        let mut t = ChunkTree::new();
        let mut rng = crate::samples::Rng(7);
        for _ in 0..4000 {
            let v = IVec3::new(
                (rng.next_f64() * 512.0) as i32,
                (rng.next_f64() * 64.0) as i32,
                (rng.next_f64() * 512.0) as i32,
            );
            t.set_voxel(v, MaterialId(1 + (rng.next_f64() * 40.0) as u16));
        }
        t.set_cell(IVec3::new(3, 3, 3), Cell::Uniform(ids::BASALT));
        t.set_node(2, IVec3::new(3, 3, 3), ids::GRANITE);
        t.set_node(1, IVec3::new(0, 12, 0), ids::SAND);
        // A wide brick.
        for i in 0..512 {
            let l = IVec3::new(i & 7, (i >> 6) & 7, (i >> 3) & 7);
            t.set_voxel(IVec3::new(80, 8, 8) + l, MaterialId(1 + (i % 30) as u16));
        }

        let mut voxels = Vec::new();
        let mut offsets = std::collections::HashMap::new();
        for (_, cell) in t.cells() {
            if let Cell::Brick(slot) = cell {
                offsets.insert(slot, (voxels.len() as u32, t.brick(slot).is_wide()));
                encode_brick(t.brick(slot), 0, &mut voxels);
            }
        }
        let mut flat = flatten_chunk(&t, &|slot| offsets.get(&slot).copied());
        let base = 1000;
        flat.relocate(base);
        let mut tree = vec![0u32; base as usize];
        tree.extend_from_slice(&flat.words);

        let uniform_probes = t
            .uniform_nodes()
            .into_iter()
            .map(|n| n.min_cell + IVec3::splat(n.cells - 1));
        for b in t.cells().map(|(b, _)| b).chain(uniform_probes) {
            for probe in [IVec3::ZERO, IVec3::splat(7), IVec3::new(3, 1, 6)] {
                let v = (b << 3) + probe;
                assert_eq!(
                    gpu_voxel(&tree, &voxels, base as usize, v),
                    t.voxel(v).0,
                    "voxel {v}"
                );
            }
        }
        let v = IVec3::new(80, 8, 8) + IVec3::new(5, 2, 7);
        assert_eq!(gpu_voxel(&tree, &voxels, base as usize, v), t.voxel(v).0);
    }

    #[test]
    fn lod_prefers_coverage_and_points_away_from_mass() {
        let mut t = ChunkTree::new();
        // Solid floor of granite with a thin sand crust on top.
        for x in 0..64 {
            for z in 0..64 {
                for y in 0..30 {
                    t.set_cell(IVec3::new(x, y, z), Cell::Uniform(ids::GRANITE));
                }
                t.set_cell(IVec3::new(x, 30, z), Cell::Uniform(ids::SAND));
            }
        }
        let flat = flatten_chunk(&t, &|_| None);
        assert_eq!(flat.lod.material, ids::GRANITE);
        assert!(flat.lod.normal.y > 0.95, "normal {}", flat.lod.normal);
        assert!(flat.lod.spread < 0.5);
        let word = pack_lod(&flat.lod);
        assert_eq!(word & 0xffff, u32::from(ids::GRANITE.0));
    }

    #[test]
    fn unresident_bricks_carry_dominant_material() {
        let mut t = ChunkTree::new();
        for i in 0..40 {
            t.set_voxel(IVec3::new(i % 8, 0, 0), ids::IRON_ORE);
        }
        t.set_voxel(IVec3::new(0, 1, 0), ids::SAND);
        let flat = flatten_chunk(&t, &|_| None);
        let (idx, _) = flat.brick_leaves[0];
        let leaf = flat.words[idx as usize];
        assert!(leaf & NOT_RESIDENT != 0);
        assert_eq!(leaf & 0xffff, u32::from(ids::IRON_ORE.0));
    }
}
