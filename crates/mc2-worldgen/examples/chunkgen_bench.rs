//! Chunk generation cost and memory per level of detail on the real world.
//! Uses the cached terrain from `worlds/default` (generated on first run).

use glam::IVec3;
use mc2_voxel::coords::ChunkPos;
use mc2_worldgen::chunkgen::{ChunkGenerator, Lod};
use mc2_worldgen::terrain::TerrainParams;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let path = std::path::Path::new("worlds/default/terrain_42.bin");
    let terrain =
        mc2_worldgen::cache::load_or_generate(path, 42, TerrainParams::world(), &mut |_, _| {})
            .expect("terrain");
    let mut generator = ChunkGenerator::new(Arc::new(terrain));
    // Plants and villages on unless NO_FLORA is set; the patch centred on
    // chunk column CX, CZ (default the map centre).
    generator.flora = std::env::var("NO_FLORA").is_err();
    let env = |k: &str, d: i32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (cx, cz) = (env("CX", 256), env("CZ", 256));
    // A 6x6 patch of columns, every non-air height.
    let mut positions = Vec::new();
    for z in cz - 3..cz + 3 {
        for x in cx - 3..cx + 3 {
            let (lo, hi) = generator.column_range(x, z);
            for y in 0..16 {
                let bottom = (y * 32) as f32;
                if bottom > hi {
                    break;
                }
                if bottom + 32.0 >= lo {
                    positions.push(ChunkPos(IVec3::new(x, y, z)));
                }
            }
        }
    }
    println!("{} surface chunks", positions.len());
    for lod in [Lod::Full, Lod::Cell, Lod::Node2, Lod::Node8] {
        let t = Instant::now();
        let mut bytes = 0;
        let mut bricks = 0;
        let mut words = 0;
        let mut flatten_s = 0.0;
        for &p in &positions {
            let tree = generator.generate(p, lod);
            bytes += tree.memory_bytes();
            bricks += tree.brick_count();
            let f = Instant::now();
            words += mc2_voxel::gpu_layout::flatten_chunk(&tree, &|_| None)
                .words
                .len();
            flatten_s += f.elapsed().as_secs_f64();
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / positions.len() as f64;
        println!(
            "  flatten alone: {:.2} ms/chunk",
            flatten_s * 1000.0 / positions.len() as f64
        );
        println!(
            "{lod:?}: {ms:.2} ms/chunk single-threaded, {:.1} KB CPU, {:.1} KB tree words, {:.0} bricks per chunk",
            bytes as f64 / positions.len() as f64 / 1024.0,
            words as f64 * 4.0 / positions.len() as f64 / 1024.0,
            bricks as f64 / positions.len() as f64
        );
    }
}
