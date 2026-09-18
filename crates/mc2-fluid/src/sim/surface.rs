//! The free surface moving: interface cells that fill become liquid and
//! open the gas beyond them, cells that empty become gas and expose the
//! liquid behind them, and terrain edits reopen or close space.

use super::{FluidWorld, Terrain};
use crate::lattice::{C, Q, equilibrium};
use crate::tile::{Kind, TILE, TILE_CELLS, local_of};
use glam::{IVec3, Vec3};
use mc2_core::FxHashMap;

impl FluidWorld {
    /// Interface cells that filled or emptied change kind; returns the tiles
    /// where anything converted.
    pub(super) fn convert(&mut self, terrain: &dyn Terrain) -> Vec<usize> {
        let eps = self.params.fill_slack;
        let mut to_liquid = Vec::new();
        let mut to_gas = Vec::new();
        for (t, tile) in self.tiles.iter().enumerate() {
            if !tile.awake {
                continue;
            }
            for i in 0..TILE_CELLS {
                if tile.kind[i] != Kind::Interface {
                    continue;
                }
                let rho = tile.rho[i];
                let m = tile.mass[i];
                let c = tile.pos * TILE + local_of(i);
                let mut gas_near = false;
                let mut fluid_near = false;
                for d in C.iter().skip(1) {
                    match self.kind(c + IVec3::from_array(*d)) {
                        Kind::Gas => gas_near = true,
                        Kind::Liquid => fluid_near = true,
                        _ => {}
                    }
                }
                if m > (1.0 + eps) * rho || !gas_near {
                    to_liquid.push((t, i));
                } else if m < -eps * rho || (!fluid_near && m < 0.1 * rho) {
                    to_gas.push((t, i));
                }
            }
        }
        let mut touched: Vec<usize> = Vec::new();
        let mut excess: Vec<(IVec3, f32)> = Vec::new();
        let mut filled: FxHashMap<IVec3, ()> = FxHashMap::default();
        // Filling first: a cell next to new liquid must not empty this step.
        for &(t, i) in &to_liquid {
            let c = self.tiles[t].pos * TILE + local_of(i);
            let tile = &mut self.tiles[t];
            excess.push((c, tile.mass[i] - tile.rho[i]));
            tile.kind[i] = Kind::Liquid;
            tile.mass[i] = tile.rho[i];
            filled.insert(c, ());
            touched.push(t);
        }
        for &(t, i) in &to_liquid {
            let c = self.tiles[t].pos * TILE + local_of(i);
            for d in C.iter().skip(1) {
                let n = c + IVec3::from_array(*d);
                if self.kind(n) == Kind::Gas {
                    self.gas_to_interface(n, terrain);
                    touched.push(self.index[&Self::tile_of(n).0]);
                }
            }
        }
        for &(t, i) in &to_gas {
            let c = self.tiles[t].pos * TILE + local_of(i);
            if self.tiles[t].kind[i] != Kind::Interface {
                continue;
            }
            let next_to_new_liquid = C
                .iter()
                .skip(1)
                .any(|d| filled.contains_key(&(c + IVec3::from_array(*d))));
            if next_to_new_liquid {
                continue;
            }
            let tile = &mut self.tiles[t];
            excess.push((c, tile.mass[i]));
            tile.kind[i] = Kind::Gas;
            tile.mass[i] = 0.0;
            touched.push(t);
            for d in C.iter().skip(1) {
                let n = c + IVec3::from_array(*d);
                let (tp, ni) = Self::tile_of(n);
                if let Some(&nt) = self.index.get(&tp)
                    && self.tiles[nt].kind[ni] == Kind::Liquid
                {
                    let tile = &mut self.tiles[nt];
                    tile.kind[ni] = Kind::Interface;
                    tile.mass[ni] = tile.rho[ni];
                }
            }
        }
        // Share each conversion's excess among the interface cells around it.
        for (c, m) in excess {
            let mut targets = Vec::new();
            for d in C.iter().skip(1) {
                let n = c + IVec3::from_array(*d);
                let (tp, ni) = Self::tile_of(n);
                if let Some(&nt) = self.index.get(&tp)
                    && self.tiles[nt].kind[ni] == Kind::Interface
                {
                    targets.push((nt, ni));
                }
            }
            if targets.is_empty() {
                self.stats.lost_mass += f64::from(m);
                continue;
            }
            let share = m / targets.len() as f32;
            for (nt, ni) in targets {
                self.tiles[nt].mass[ni] += share;
            }
        }
        touched.sort_unstable();
        touched.dedup();
        touched
    }

    /// A gas cell next to new liquid becomes interface, empty, with the
    /// equilibrium of its liquid and interface neighbours' mean state.
    fn gas_to_interface(&mut self, c: IVec3, terrain: &dyn Terrain) {
        let (tp, i) = Self::tile_of(c);
        let t = self.ensure_tile(tp, terrain);
        if self.tiles[t].kind[i] != Kind::Gas {
            return;
        }
        let (mut rho, mut u, mut n) = (0.0f32, Vec3::ZERO, 0.0f32);
        for d in C.iter().skip(1) {
            let nb = c + IVec3::from_array(*d);
            let (ntp, ni) = Self::tile_of(nb);
            if let Some(&nt) = self.index.get(&ntp) {
                let tile = &self.tiles[nt];
                if matches!(tile.kind[ni], Kind::Liquid | Kind::Interface) {
                    rho += tile.rho[ni];
                    u += tile.u[ni];
                    n += 1.0;
                }
            }
        }
        let (rho, u) = if n > 0.0 {
            (rho / n, u / n)
        } else {
            (1.0, Vec3::ZERO)
        };
        let tile = &mut self.tiles[t];
        tile.kind[i] = Kind::Interface;
        tile.mass[i] = 0.0;
        tile.rho[i] = rho;
        tile.u[i] = u;
        for q in 0..Q {
            tile.f[i * Q + q] = equilibrium(q, rho, u);
        }
        tile.awake = true;
        tile.still_steps = 0;
    }
}
