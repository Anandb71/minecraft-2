//! Lists the villages and towns within a few kilometres of the map centre,
//! nearest first, with their centres: places to point a camera at.
//!
//! cargo run -p mc2-worldgen --release --example settlements

use mc2_worldgen::amplify::Surface;
use mc2_worldgen::settlement::{Settlements, VILLAGE_CELL_M};
use mc2_worldgen::terrain::TerrainParams;
use std::sync::Arc;

fn main() {
    let path = std::path::Path::new("worlds/default/terrain_42.bin");
    let terrain =
        mc2_worldgen::cache::load_or_generate(path, 42, TerrainParams::world(), &mut |_, _| {})
            .expect("terrain");
    let terrain = Arc::new(terrain);
    let centre = terrain.params.extent_m() * 0.5;
    let surface = Surface::new(terrain.clone());
    let settlements = Settlements::new(terrain.seed);
    let c = (centre / VILLAGE_CELL_M).floor() as i32;
    let mut found = Vec::new();
    for k in c - 6..=c + 6 {
        for i in c - 6..=c + 6 {
            if let Some(v) = settlements.village(&surface, i, k) {
                let d = ((v.centre.x - centre).powi(2) + (v.centre.z - centre).powi(2)).sqrt();
                found.push((d, v));
            }
        }
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (d, v) in found.iter().take(20) {
        println!(
            "{:>5} {:>6.0} m away  centre {:.0} {:.0} {:.0}  top {:.0} m",
            if v.town { "town" } else { "village" },
            d,
            v.centre.x,
            v.centre.y,
            v.centre.z,
            v.hi.y
        );
    }
}
