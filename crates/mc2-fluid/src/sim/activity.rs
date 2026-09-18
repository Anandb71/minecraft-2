//! Sleep: tiles whose water has been still for a second stop stepping;
//! motion or conversion in a tile keeps it and the tiles around it awake.

use super::FluidWorld;
use crate::tile::Kind;

/// Steps a tile must stay still before it sleeps.
const SLEEP_STEPS: u32 = 240;
/// Below this speed (lattice units, about 1 cm/s) a cell counts as still.
pub(super) const STILL_SPEED: f32 = 1e-4;

impl FluidWorld {
    /// Every tile (asleep or not) counts the steps since anything moved in
    /// it or around it; after a second of stillness it sleeps.
    pub(super) fn update_activity(&mut self) {
        let moving: Vec<bool> = self.tiles.iter().map(|t| t.moving).collect();
        for tile in &mut self.tiles {
            // `near` includes the tile itself.
            let stirred = tile
                .near
                .iter()
                .any(|&n| n != u32::MAX && moving[n as usize]);
            tile.still_steps = if stirred {
                0
            } else {
                tile.still_steps.saturating_add(1)
            };
            tile.awake = tile.still_steps <= SLEEP_STEPS;
            tile.moving = false;
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
