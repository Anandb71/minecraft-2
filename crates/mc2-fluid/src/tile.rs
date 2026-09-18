//! Sparse storage: 8^3 tiles of cells, each with its populations, and the
//! lookups that cross tile edges.

use crate::lattice::Q;
use glam::{IVec3, Vec3};

pub const TILE: i32 = 8;
pub const TILE_CELLS: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    Gas = 0,
    Interface = 1,
    Liquid = 2,
    Solid = 3,
}

#[derive(Clone)]
pub(crate) struct Tile {
    pub pos: IVec3,
    pub f: Vec<f32>,
    pub kind: Vec<Kind>,
    pub mass: Vec<f32>,
    pub rho: Vec<f32>,
    pub u: Vec<Vec3>,
    /// Neighbouring tiles, 27 slots (dx, dy, dz in -1..=1), or u32::MAX.
    pub near: [u32; 27],
    pub awake: bool,
    pub still_steps: u32,
}

impl Tile {
    pub fn new(pos: IVec3, solid: impl Fn(IVec3) -> bool) -> Self {
        let mut kind = vec![Kind::Gas; TILE_CELLS];
        for (i, k) in kind.iter_mut().enumerate() {
            if solid(pos * TILE + local_of(i)) {
                *k = Kind::Solid;
            }
        }
        Self {
            pos,
            f: vec![0.0; TILE_CELLS * Q],
            kind,
            mass: vec![0.0; TILE_CELLS],
            rho: vec![1.0; TILE_CELLS],
            u: vec![Vec3::ZERO; TILE_CELLS],
            near: [u32::MAX; 27],
            awake: true,
            still_steps: 0,
        }
    }
}

#[inline]
pub(crate) fn local_of(i: usize) -> IVec3 {
    let i = i as i32;
    IVec3::new(i & 7, (i >> 3) & 7, i >> 6)
}

#[inline]
pub(crate) fn index_of(l: IVec3) -> usize {
    (l.x + 8 * (l.y + 8 * l.z)) as usize
}

#[inline]
pub(crate) fn near_slot(d: IVec3) -> usize {
    ((d.x + 1) + 3 * ((d.y + 1) + 3 * (d.z + 1))) as usize
}

/// The tile and cell a neighbour of cell `l` in tile `t` lies in.
#[inline]
pub(crate) fn neighbour(tiles: &[Tile], t: usize, l: IVec3, d: IVec3) -> Option<(usize, usize)> {
    let n = l + d;
    let step = IVec3::new(
        (n.x >> 3).clamp(-1, 1),
        (n.y >> 3).clamp(-1, 1),
        (n.z >> 3).clamp(-1, 1),
    );
    let tile = if step == IVec3::ZERO {
        t
    } else {
        let s = tiles[t].near[near_slot(step)];
        if s == u32::MAX {
            return None;
        }
        s as usize
    };
    Some((tile, index_of(n & 7)))
}
