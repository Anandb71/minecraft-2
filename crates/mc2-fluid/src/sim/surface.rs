//! The free surface moving. After streaming (which leaves in `conv` what
//! each interface cell wants to become) four gather passes run, each reading
//! of its neighbours only what an earlier pass wrote, so the GPU can run
//! them as they are:
//!
//! 1. Flag: interface cells that overfilled become liquid, gas next to them
//!    becomes interface, and interface cells that emptied become gas unless
//!    they touch new liquid.
//! 2. Apply: new interface cells start at the mean state of the water
//!    around them, liquid next to new gas becomes interface, and each
//!    converting cell sets aside the mass it gains or loses by converting.
//! 3. Share: that excess is split among the cell's interface neighbours.
//! 4. Gather: interface cells collect the shares around them.
//!
//! Terrain edits reopen or close space between steps.

use super::{FluidWorld, Terrain};
use crate::lattice::{C, Q, W, equilibrium};
use crate::tile::{
    FROM_GAS, Kind, TILE, TILE_CELLS, TO_GAS, TO_LIQUID, Tile, kind_of, local_of, need_of,
    neighbour, pack, transition_of,
};
use glam::{IVec3, Vec3};
use rayon::prelude::*;

/// The 18 neighbours of cell `l` in tile `t` that are allocated.
fn around(tiles: &[Tile], t: usize, l: IVec3) -> impl Iterator<Item = (usize, usize)> + '_ {
    C[1..]
        .iter()
        .filter_map(move |d| neighbour(tiles, t, l, IVec3::from_array(*d)))
}

/// What pass 4 leaves in one tile: masses, the liquid cells whose
/// populations take a share, and the tiles its water needs.
struct Gathered {
    mass: Vec<f32>,
    liquid: Vec<(usize, f32)>,
    need: u32,
}

/// What pass 2 leaves in one tile.
struct Applied {
    kind: Vec<Kind>,
    mass: Vec<f32>,
    rho: Vec<f32>,
    u: Vec<Vec3>,
    excess: Vec<f32>,
    /// Cells that became interface from gas, to start at equilibrium.
    fresh: Vec<usize>,
    moving: bool,
}

impl FluidWorld {
    /// Pass 1: each cell's transition, from what streaming asked for.
    pub(super) fn flag(&mut self) {
        let tiles = &self.tiles;
        let out: Vec<Option<Vec<u8>>> = tiles
            .par_iter()
            .enumerate()
            .map(|(t, tile)| {
                if !tile.awake {
                    return None;
                }
                let mut next = vec![0u8; TILE_CELLS];
                for (i, n) in next.iter_mut().enumerate() {
                    let k = tile.kind[i];
                    let filling_near = || {
                        around(tiles, t, local_of(i))
                            .any(|(nt, ni)| tiles[nt].conv[ni] == TO_LIQUID)
                    };
                    let transition = match (k, tile.conv[i]) {
                        (Kind::Gas, _) if filling_near() => FROM_GAS,
                        (Kind::Interface, TO_LIQUID) => TO_LIQUID,
                        (Kind::Interface, TO_GAS) if !filling_near() => TO_GAS,
                        _ => 0,
                    };
                    *n = pack(k, transition);
                }
                Some(next)
            })
            .collect();
        for (tile, next) in self.tiles.iter_mut().zip(out) {
            if let Some(next) = next {
                tile.next = next;
            }
        }
    }

    /// Pass 2: conversions take effect.
    pub(super) fn apply(&mut self) {
        let tiles = &self.tiles;
        let out: Vec<Option<Applied>> = tiles
            .par_iter()
            .enumerate()
            .map(|(t, tile)| {
                if !tile.awake {
                    return None;
                }
                let mut a = Applied {
                    kind: tile.kind.clone(),
                    mass: tile.mass.clone(),
                    rho: tile.rho.clone(),
                    u: tile.u.clone(),
                    excess: vec![0.0; TILE_CELLS],
                    fresh: Vec::new(),
                    moving: false,
                };
                for i in 0..TILE_CELLS {
                    let next = tile.next[i];
                    let l = local_of(i);
                    match transition_of(next) {
                        FROM_GAS => {
                            // The mean of the water around it, not counting
                            // cells that are emptying or new themselves.
                            let (mut rho, mut u, mut n) = (0.0f32, Vec3::ZERO, 0.0f32);
                            for (nt, ni) in around(tiles, t, l) {
                                let o = tiles[nt].next[ni];
                                if matches!(kind_of(o), Kind::Liquid | Kind::Interface)
                                    && transition_of(o) != TO_GAS
                                {
                                    rho += tiles[nt].rho[ni];
                                    u += tiles[nt].u[ni];
                                    n += 1.0;
                                }
                            }
                            let (rho, u) = if n > 0.0 {
                                (rho / n, u / n)
                            } else {
                                (1.0, Vec3::ZERO)
                            };
                            a.kind[i] = Kind::Interface;
                            a.mass[i] = 0.0;
                            a.rho[i] = rho;
                            a.u[i] = u;
                            a.fresh.push(i);
                            a.moving = true;
                        }
                        TO_LIQUID => {
                            a.excess[i] = tile.mass[i] - tile.rho[i];
                            a.kind[i] = Kind::Liquid;
                            a.mass[i] = tile.rho[i];
                            a.moving = true;
                        }
                        TO_GAS => {
                            a.excess[i] = tile.mass[i];
                            a.kind[i] = Kind::Gas;
                            a.mass[i] = 0.0;
                            a.moving = true;
                        }
                        _ => {
                            let emptying_near = || {
                                around(tiles, t, l)
                                    .any(|(nt, ni)| transition_of(tiles[nt].next[ni]) == TO_GAS)
                            };
                            if kind_of(next) == Kind::Liquid && emptying_near() {
                                a.kind[i] = Kind::Interface;
                                a.mass[i] = tile.rho[i];
                                a.moving = true;
                            }
                        }
                    }
                }
                Some(a)
            })
            .collect();
        for (tile, a) in self.tiles.iter_mut().zip(out) {
            let Some(a) = a else { continue };
            for &i in &a.fresh {
                for q in 0..Q {
                    tile.f[i * Q + q] = equilibrium(q, a.rho[i], a.u[i]);
                }
            }
            tile.kind = a.kind;
            tile.mass = a.mass;
            tile.rho = a.rho;
            tile.u = a.u;
            tile.excess = a.excess;
            tile.moving |= a.moving;
        }
    }

    /// Pass 3: each converting cell splits its excess among the water
    /// around it (interface and liquid cells); with none, the mass is lost
    /// (and counted).
    pub(super) fn share(&mut self) {
        let tiles = &self.tiles;
        let out: Vec<Option<(Vec<f32>, f64)>> = tiles
            .par_iter()
            .enumerate()
            .map(|(t, tile)| {
                if !tile.awake {
                    return None;
                }
                let mut massex = vec![0.0; TILE_CELLS];
                let mut lost = 0.0f64;
                for (i, m) in massex.iter_mut().enumerate() {
                    if !matches!(transition_of(tile.next[i]), TO_LIQUID | TO_GAS) {
                        continue;
                    }
                    let targets = around(tiles, t, local_of(i))
                        .filter(|&(nt, ni)| {
                            matches!(tiles[nt].kind[ni], Kind::Interface | Kind::Liquid)
                        })
                        .count();
                    if targets == 0 {
                        lost += f64::from(tile.excess[i]);
                    } else {
                        *m = tile.excess[i] / targets as f32;
                    }
                }
                Some((massex, lost))
            })
            .collect();
        for (tile, o) in self.tiles.iter_mut().zip(out) {
            if let Some((massex, lost)) = o {
                tile.massex = massex;
                self.stats.lost_mass += lost;
            }
        }
    }

    /// Pass 4: water collects its neighbours' shares (an interface cell in
    /// its mass; a liquid cell, whose mass is its density, in its
    /// populations by weight, which leaves its momentum alone); every tile
    /// notes which tiles around it its water needs.
    pub(super) fn gather(&mut self) {
        let tiles = &self.tiles;
        let out: Vec<Option<Gathered>> = tiles
            .par_iter()
            .enumerate()
            .map(|(t, tile)| {
                if !tile.awake {
                    return None;
                }
                let mut g = Gathered {
                    mass: tile.mass.clone(),
                    liquid: Vec::new(),
                    need: 0,
                };
                for (i, m) in g.mass.iter_mut().enumerate() {
                    let l = local_of(i);
                    let k = tile.kind[i];
                    if !matches!(k, Kind::Interface | Kind::Liquid) {
                        continue;
                    }
                    g.need |= need_of(l);
                    let share: f32 = around(tiles, t, l)
                        .map(|(nt, ni)| tiles[nt].massex[ni])
                        .sum();
                    if share == 0.0 {
                        continue;
                    }
                    *m += share;
                    if k == Kind::Liquid {
                        g.liquid.push((i, share));
                    }
                }
                Some(g)
            })
            .collect();
        for (tile, g) in self.tiles.iter_mut().zip(out) {
            let Some(g) = g else { continue };
            for (i, share) in g.liquid {
                tile.rho[i] += share;
                for q in 0..Q {
                    tile.f[i * Q + q] += W[q] * share;
                }
            }
            tile.mass = g.mass;
            tile.need = g.need;
        }
    }

    /// Terrain changed in the cells `lo..=hi`: refresh solidity there and
    /// wake the tiles around it.
    pub fn terrain_changed(&mut self, lo: IVec3, hi: IVec3, terrain: &dyn Terrain) {
        let (tlo, thi) = (
            (lo - 1).div_euclid(IVec3::splat(TILE)),
            (hi + 1).div_euclid(IVec3::splat(TILE)),
        );
        for z in tlo.z..=thi.z {
            for y in tlo.y..=thi.y {
                for x in tlo.x..=thi.x {
                    let Some(&t) = self.index.get(&IVec3::new(x, y, z)) else {
                        continue;
                    };
                    let tile = &mut self.tiles[t];
                    for i in 0..TILE_CELLS {
                        let c = tile.pos * TILE + local_of(i);
                        let solid = terrain.solid(c);
                        match (tile.kind[i], solid) {
                            (Kind::Solid, false) => tile.kind[i] = Kind::Gas,
                            (Kind::Gas, true) => tile.kind[i] = Kind::Solid,
                            _ => {}
                        }
                        tile.next[i] = pack(tile.kind[i], 0);
                    }
                    tile.awake = true;
                    tile.still_steps = 0;
                }
            }
        }
        // Liquid that now faces open space is surface: it must track its
        // mass, or it would stream into the gas without accounting.
        self.close_surface();
        // Water next to newly opened space has somewhere to go.
        self.allocate_margins(terrain, true);
    }

    /// Liquid cells touching gas (or unallocated space) become full
    /// interface cells, so the interface layer stays closed.
    pub(super) fn close_surface(&mut self) {
        let mut exposed = Vec::new();
        for (t, tile) in self.tiles.iter().enumerate() {
            for i in 0..TILE_CELLS {
                if tile.kind[i] != Kind::Liquid {
                    continue;
                }
                let c = tile.pos * TILE + local_of(i);
                if C.iter()
                    .skip(1)
                    .any(|d| self.kind(c + IVec3::from_array(*d)) == Kind::Gas)
                {
                    exposed.push((t, i));
                }
            }
        }
        for (t, i) in exposed {
            let tile = &mut self.tiles[t];
            tile.kind[i] = Kind::Interface;
            tile.next[i] = pack(Kind::Interface, 0);
            tile.mass[i] = tile.rho[i];
            tile.awake = true;
            tile.still_steps = 0;
        }
    }
}
