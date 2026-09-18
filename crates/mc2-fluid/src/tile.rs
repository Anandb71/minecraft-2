//! Sparse storage: 8^3 tiles of cells, each with its populations, and the
//! lookups that cross tile edges.

use crate::lattice::{C, MIRROR, Q};
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

/// Where a flat wall reflects population `q` of cell `l` in tile `t` from,
/// when the cell behind it along `q` is solid: the cell one step back along
/// the axis that runs parallel to the wall, and the population mirrored
/// across the wall. None for face populations and in corners, which bounce
/// straight back, and when that cell holds no water.
pub(crate) fn specular(
    tiles: &[Tile],
    t: usize,
    l: IVec3,
    q: usize,
) -> Option<(usize, usize, usize)> {
    // Faces come first in `C`; only the 12 edges can slide along a wall.
    if q < 7 {
        return None;
    }
    let c = IVec3::from_array(C[q]);
    let (a1, a2) = match (c.x != 0, c.y != 0) {
        (true, true) => (0, 1),
        (true, false) => (0, 2),
        _ => (1, 2),
    };
    let back = |a: usize| {
        let mut d = IVec3::ZERO;
        d[a] = -c[a];
        d
    };
    let solid = |d: IVec3| {
        neighbour(tiles, t, l, d).is_some_and(|(st, si)| tiles[st].kind[si] == Kind::Solid)
    };
    let (d, normal) = match (solid(back(a1)), solid(back(a2))) {
        (false, true) => (back(a1), a2),
        (true, false) => (back(a2), a1),
        _ => return None,
    };
    let (mt, mi) = neighbour(tiles, t, l, d)?;
    matches!(tiles[mt].kind[mi], Kind::Liquid | Kind::Interface).then_some((
        mt,
        mi,
        MIRROR[normal][q],
    ))
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
