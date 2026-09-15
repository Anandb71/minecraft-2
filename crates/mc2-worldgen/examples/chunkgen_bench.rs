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
    let generator = ChunkGenerator::new(Arc::new(terrain));
    // A 6x6 patch of columns around the map centre, every non-air height.
    let mut positions = Vec::new();
    for z in 253..259 {
        for x in 253..259 {
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
        for &p in &positions {
            let tree = generator.generate(p, lod);
            bytes += tree.memory_bytes();
            bricks += tree.brick_count();
            words += mc2_voxel::gpu_layout::flatten_chunk(&tree, &|_| None)
                .words
                .len();
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / positions.len() as f64;
        println!(
            "{lod:?}: {ms:.2} ms/chunk single-threaded, {:.1} KB CPU, {:.1} KB tree words, {:.0} bricks per chunk",
            bytes as f64 / positions.len() as f64 / 1024.0,
            words as f64 * 4.0 / positions.len() as f64 / 1024.0,
            bricks as f64 / positions.len() as f64
        );
    }
}
