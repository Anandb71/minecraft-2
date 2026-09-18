//! Free-surface lattice Boltzmann over sparse tiles: the CPU reference.
//!
//! Cells are 0.5 m (one brick cell) and live in 8^3 tiles allocated where
//! water is. Liquid and interface cells carry D3Q19 populations; gas cells
//! carry nothing. Each step (Körner et al. 2005, with the "only missing"
//! reconstruction recommended by Schwarzmeier et al. 2022):
//!
//! 1. Stream by pulling: a population arrives from the neighbour behind it;
//!    one arriving from solid mostly slides along the wall (the rest
//!    bounces back), one that would arrive from gas is rebuilt from the gas
//!    pressure and the cell's velocity.
//! 2. Interface cells exchange mass along each link with what streams across
//!    it, weighted by the mean fill of the two cells when both are
//!    interface.
//! 3. Collide with BGK plus a Smagorinsky subgrid term and Guo gravity.
//! 4. Interface cells that overfill become liquid (gas neighbours become
//!    interface, initialised from their neighbours' average), cells that
//!    empty become gas (liquid neighbours become interface), and the excess
//!    mass of each conversion is shared among the interface cells around it.
//!
//! Tiles whose cells have been still for a second go to sleep and cost
//! nothing; motion at a tile's face wakes its neighbour.

use crate::lattice::{C, Q, W, equilibrium, guo, les_tau, moments, opp};
use crate::tile::{
    Kind, TILE, TILE_CELLS, Tile, index_of, local_of, near_slot, neighbour, specular,
};
use glam::{IVec3, Vec3};
use mc2_core::FxHashMap;
use rayon::prelude::*;

mod activity;
mod surface;
#[cfg(test)]
mod tests;

/// Cell edge, metres: one brick cell.
pub const CELL_M: f64 = 0.5;
/// Simulated seconds per step.
pub const STEP_S: f64 = 1.0 / 240.0;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Base relaxation time; the subgrid model raises it where it shears.
    pub tau: f32,
    pub smagorinsky: f32,
    /// Gravity in lattice units.
    pub gravity: Vec3,
    /// Gas density, the pressure the free surface sees.
    pub rho_gas: f32,
    /// Fill beyond [0, 1] by this much converts an interface cell.
    pub fill_slack: f32,
    /// How much of what reaches a flat wall slides along it (reflected)
    /// rather than bouncing straight back: 0 is no-slip, 1 free slip.
    pub wall_slip: f32,
}

impl Default for Params {
    fn default() -> Self {
        let g = 9.81 * STEP_S * STEP_S / CELL_M;
        Self {
            tau: 0.51,
            smagorinsky: 0.127,
            gravity: Vec3::new(0.0, -g as f32, 0.0),
            rho_gas: 1.0,
            fill_slack: 1e-2,
            wall_slip: 0.995,
        }
    }
}

/// Which cells of the world are solid.
pub trait Terrain: Sync {
    fn solid(&self, cell: IVec3) -> bool;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FluidStats {
    pub tiles: usize,
    pub awake: usize,
    pub liquid: usize,
    pub interface: usize,
    /// Excess mass that found no interface cell to go to, lattice units.
    pub lost_mass: f64,
    pub step_ms: f32,
}

/// A tile's new populations, mass, density and velocity.
type TileState = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec3>);

pub struct FluidWorld {
    tiles: Vec<Tile>,
    index: FxHashMap<IVec3, usize>,
    pub params: Params,
    pub stats: FluidStats,
}

impl Default for FluidWorld {
    fn default() -> Self {
        Self::new(Params::default())
    }
}

impl FluidWorld {
    pub fn new(params: Params) -> Self {
        Self {
            tiles: Vec::new(),
            index: FxHashMap::default(),
            params,
            stats: FluidStats::default(),
        }
    }

    fn tile_of(cell: IVec3) -> (IVec3, usize) {
        (
            cell.div_euclid(IVec3::splat(TILE)),
            index_of(cell.rem_euclid(IVec3::splat(TILE))),
        )
    }

    /// The tile holding `pos`, allocating it (all gas and solid) if needed.
    fn ensure_tile(&mut self, pos: IVec3, terrain: &dyn Terrain) -> usize {
        if let Some(&t) = self.index.get(&pos) {
            return t;
        }
        let t = self.tiles.len();
        self.tiles.push(Tile::new(pos, |c| terrain.solid(c)));
        self.index.insert(pos, t);
        // Link it with its neighbours both ways.
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let d = IVec3::new(dx, dy, dz);
                    if let Some(&o) = self.index.get(&(pos + d)) {
                        self.tiles[t].near[near_slot(d)] = o as u32;
                        self.tiles[o].near[near_slot(-d)] = t as u32;
                    }
                }
            }
        }
        t
    }

    pub fn kind(&self, cell: IVec3) -> Kind {
        let (tp, i) = Self::tile_of(cell);
        self.index
            .get(&tp)
            .map_or(Kind::Gas, |&t| self.tiles[t].kind[i])
    }

    /// Fill of a cell: 1 for liquid, the interface fill, 0 otherwise.
    pub fn fill(&self, cell: IVec3) -> f32 {
        let (tp, i) = Self::tile_of(cell);
        let Some(&t) = self.index.get(&tp) else {
            return 0.0;
        };
        let tile = &self.tiles[t];
        match tile.kind[i] {
            Kind::Liquid => 1.0,
            Kind::Interface => (tile.mass[i] / tile.rho[i].max(1e-6)).clamp(0.0, 1.0),
            _ => 0.0,
        }
    }

    pub fn velocity(&self, cell: IVec3) -> Vec3 {
        let (tp, i) = Self::tile_of(cell);
        self.index
            .get(&tp)
            .map_or(Vec3::ZERO, |&t| self.tiles[t].u[i])
    }

    /// Fills cells with still water at hydrostatic pressure. Cells next to
    /// gas become interface cells, full.
    pub fn add_water(&mut self, cells: impl IntoIterator<Item = IVec3>, terrain: &dyn Terrain) {
        let mut added = Vec::new();
        for c in cells {
            let (tp, i) = Self::tile_of(c);
            let t = self.ensure_tile(tp, terrain);
            let tile = &mut self.tiles[t];
            if tile.kind[i] == Kind::Solid {
                continue;
            }
            tile.kind[i] = Kind::Liquid;
            tile.rho[i] = 1.0;
            tile.mass[i] = 1.0;
            tile.u[i] = Vec3::ZERO;
            tile.awake = true;
            tile.still_steps = 0;
            added.push(c);
        }
        // Close the surface: liquid touching gas is an interface.
        for &c in &added {
            for d in C.iter().skip(1) {
                let n = c + IVec3::from_array(*d);
                if self.kind(n) == Kind::Gas {
                    let (tp, i) = Self::tile_of(c);
                    let t = self.index[&tp];
                    self.tiles[t].kind[i] = Kind::Interface;
                    break;
                }
            }
        }
        // Start at hydrostatic pressure, the weight of the water above
        // (see the surface pressure in `step`), or the body rings.
        let g = self.params.gravity.length();
        for &c in &added {
            let mut above = 0.0;
            let mut n = c + IVec3::Y;
            while matches!(self.kind(n), Kind::Liquid | Kind::Interface) {
                above += self.fill(n);
                n += IVec3::Y;
            }
            let rho = self.params.rho_gas + 3.0 * g * (above + 0.5);
            let (tp, i) = Self::tile_of(c);
            let tile = &mut self.tiles[self.index[&tp]];
            tile.rho[i] = rho;
            tile.mass[i] = rho;
            for (f, w) in tile.f[i * Q..(i + 1) * Q].iter_mut().zip(W) {
                *f = w * rho;
            }
        }
        for c in self.gas_neighbours_of_interfaces() {
            let (tp, _) = Self::tile_of(c);
            self.ensure_tile(tp, terrain);
        }
    }

    fn gas_neighbours_of_interfaces(&self) -> Vec<IVec3> {
        let mut out = Vec::new();
        for tile in &self.tiles {
            for i in 0..TILE_CELLS {
                if tile.kind[i] != Kind::Interface {
                    continue;
                }
                let c = tile.pos * TILE + local_of(i);
                for d in C.iter().skip(1) {
                    let n = c + IVec3::from_array(*d);
                    let (tp, _) = Self::tile_of(n);
                    if !self.index.contains_key(&tp) {
                        out.push(n);
                    }
                }
            }
        }
        out
    }

    /// Liquid mass, lattice units (cells of rest density).
    pub fn mass(&self) -> f64 {
        let mut m = 0.0f64;
        for tile in &self.tiles {
            for i in 0..TILE_CELLS {
                m += match tile.kind[i] {
                    Kind::Liquid => f64::from(tile.rho[i]),
                    Kind::Interface => f64::from(tile.mass[i]),
                    _ => 0.0,
                };
            }
        }
        m
    }

    /// Water volume, cubic metres at rest density.
    pub fn volume_m3(&self) -> f64 {
        self.mass() * CELL_M * CELL_M * CELL_M
    }

    /// Advances one lattice step (`STEP_S` seconds).
    pub fn step(&mut self, terrain: &dyn Terrain) {
        let start = std::time::Instant::now();
        let p = self.params;
        let tiles = &self.tiles;
        // 1-3. Stream, exchange mass, collide: every awake tile reads the
        // old state of all tiles and writes its own new state.
        let updated: Vec<Option<TileState>> = tiles
            .par_iter()
            .enumerate()
            .map(|(t, tile)| {
                if !tile.awake {
                    return None;
                }
                let mut f = tile.f.clone();
                let mut mass = tile.mass.clone();
                let mut rho = tile.rho.clone();
                let mut u = tile.u.clone();
                for i in 0..TILE_CELLS {
                    let k = tile.kind[i];
                    if k != Kind::Liquid && k != Kind::Interface {
                        continue;
                    }
                    let l = local_of(i);
                    let fill_here = if k == Kind::Interface {
                        (tile.mass[i] / tile.rho[i].max(1e-6)).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    let mut fi = [0.0f32; Q];
                    let mut dm = 0.0f32;
                    fi[0] = tile.f[i * Q];
                    // The pressure the free surface presses on the cell
                    // centre: the gas, plus the weight of whatever part of
                    // the cell's water stands above its centre. Without it
                    // a half-full cell and a nearly full one push the same
                    // and sub-cell bumps never level out.
                    let rho_gas = p.rho_gas + 3.0 * p.gravity.length() * (fill_here - 0.5);
                    for q in 1..Q {
                        let d = IVec3::from_array(C[q]);
                        let out_q = tile.f[i * Q + opp(q)];
                        match neighbour(tiles, t, l, -d) {
                            Some((st, si)) => {
                                let src = &tiles[st];
                                match src.kind[si] {
                                    Kind::Solid => match specular(tiles, t, l, q) {
                                        Some((mt, mi, mq)) => {
                                            let m = &tiles[mt];
                                            let slide = m.f[mi * Q + mq];
                                            fi[q] =
                                                p.wall_slip * slide + (1.0 - p.wall_slip) * out_q;
                                            if k == Kind::Interface {
                                                let exchange = p.wall_slip * (slide - out_q);
                                                dm += if m.kind[mi] == Kind::Liquid {
                                                    exchange
                                                } else {
                                                    let fill_there = (m.mass[mi]
                                                        / m.rho[mi].max(1e-6))
                                                    .clamp(0.0, 1.0);
                                                    0.5 * (fill_here + fill_there) * exchange
                                                };
                                            }
                                        }
                                        None => fi[q] = out_q,
                                    },
                                    Kind::Gas => {
                                        fi[q] = equilibrium(q, rho_gas, tile.u[i])
                                            + equilibrium(opp(q), rho_gas, tile.u[i])
                                            - out_q;
                                    }
                                    nk => {
                                        let incoming = src.f[si * Q + q];
                                        fi[q] = incoming;
                                        if k == Kind::Interface {
                                            let exchange = incoming - out_q;
                                            dm += if nk == Kind::Liquid {
                                                exchange
                                            } else {
                                                let fill_there = (src.mass[si]
                                                    / src.rho[si].max(1e-6))
                                                .clamp(0.0, 1.0);
                                                0.5 * (fill_here + fill_there) * exchange
                                            };
                                        }
                                    }
                                }
                            }
                            // Beyond the allocated tiles lies gas.
                            None => {
                                fi[q] = equilibrium(q, rho_gas, tile.u[i])
                                    + equilibrium(opp(q), rho_gas, tile.u[i])
                                    - out_q;
                            }
                        }
                    }
                    let force = p.gravity;
                    let (r, v) = moments(&fi, force * rho_guess(&fi));
                    let force = force * r;
                    let tau = les_tau(&fi, r, v, p.tau, p.smagorinsky);
                    for q in 0..Q {
                        let feq = equilibrium(q, r, v);
                        f[i * Q + q] = fi[q] - (fi[q] - feq) / tau + guo(q, tau, v, force);
                    }
                    rho[i] = r;
                    u[i] = v;
                    if k == Kind::Interface {
                        mass[i] += dm;
                    } else {
                        mass[i] = r;
                    }
                }
                Some((f, mass, rho, u))
            })
            .collect();
        for (tile, new) in self.tiles.iter_mut().zip(updated) {
            if let Some((f, mass, rho, u)) = new {
                tile.f = f;
                tile.mass = mass;
                tile.rho = rho;
                tile.u = u;
            }
        }

        // 4. Conversions.
        let converted = self.convert(terrain);
        self.update_activity(&converted);
        self.stats.step_ms = start.elapsed().as_secs_f32() * 1000.0;
        self.refresh_stats();
    }
}

/// Density estimate for the force term before the moments are known.
#[inline]
fn rho_guess(f: &[f32; Q]) -> f32 {
    f.iter().sum()
}
