//! Sleep: tiles whose water has been still for a second stop stepping;
//! motion at a tile's face wakes the neighbour across it.

use super::FluidWorld;
use crate::tile::Kind;

/// Steps a tile must stay still before it sleeps.
const SLEEP_STEPS: u32 = 240;
/// Below this speed (lattice units, about 1 cm/s) a cell counts as still.
const STILL_SPEED: f32 = 1e-4;

impl FluidWorld {
    /// Still tiles sleep; motion at a face wakes the neighbour across it.
    pub(super) fn update_activity(&mut self, converted: &[usize]) {
        let mut wake: Vec<usize> = Vec::new();
        for (t, tile) in self.tiles.iter_mut().enumerate() {
            if !tile.awake {
                continue;
            }
            let moving = tile.u.iter().zip(&tile.kind).any(|(v, k)| {
                matches!(k, Kind::Liquid | Kind::Interface) && v.length() > STILL_SPEED
            });
            if moving || converted.contains(&t) {
                tile.still_steps = 0;
            } else {
                tile.still_steps += 1;
            }
            if tile.still_steps > SLEEP_STEPS {
                tile.awake = false;
                continue;
            }
            if moving {
                for &n in &tile.near {
                    if n != u32::MAX {
                        wake.push(n as usize);
                    }
                }
            }
        }
        for t in wake {
            let tile = &mut self.tiles[t];
            if !tile.awake {
                tile.awake = true;
                tile.still_steps = 0;
            }
        }
    }

    pub(super) fn refresh_stats(&mut self) {
        let (mut liquid, mut interface) = (0, 0);
        for tile in &self.tiles {
            for k in &tile.kind {
                match k {
                    Kind::Liquid => liquid += 1,
                    Kind::Interface => interface += 1,
                    _ => {}
                }
            }
        }
        self.stats.tiles = self.tiles.len();
        self.stats.awake = self.tiles.iter().filter(|t| t.awake).count();
        self.stats.liquid = liquid;
        self.stats.interface = interface;
    }
}
