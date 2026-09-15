//! Rays per second of the hierarchical CPU marcher against the brute force walk.

use glam::DVec3;
use mc2_voxel::march::{raycast, raycast_brute};
use mc2_voxel::samples::{Rng, rolling_terrain};
use std::time::Instant;

fn main() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let t = Instant::now();
    let world = rolling_terrain(&mut rng);
    println!(
        "built {} chunks, {:.1} MB, in {:.2}s",
        world.chunk_count(),
        world.memory_bytes() as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
    let rays: Vec<(DVec3, DVec3)> = (0..200_000)
        .map(|_| {
            let o = DVec3::new(
                rng.next_f64() * 64.0,
                6.0 + rng.next_f64() * 16.0,
                rng.next_f64() * 37.5,
            );
            let d = DVec3::new(
                rng.next_f64() - 0.5,
                rng.next_f64() - 0.5,
                rng.next_f64() - 0.5,
            );
            (o, d)
        })
        .collect();
    type Cast =
        fn(&mc2_voxel::world::VoxelWorld, DVec3, DVec3, f64) -> Option<mc2_voxel::march::RayHit>;
    let casts: [(&str, Cast); 2] = [("hierarchical", raycast), ("brute force", raycast_brute)];
    for (name, f) in casts {
        let t = Instant::now();
        let (mut hits, mut iters) = (0u64, 0u64);
        for &(o, d) in &rays {
            if let Some(h) = f(&world, o, d, 40.0) {
                hits += 1;
                iters += u64::from(h.iterations);
            }
        }
        let secs = t.elapsed().as_secs_f64();
        println!(
            "{name:>12}: {:.2} M rays/s, {:.1} ns/ray, {:.1} iterations/hit, {hits} hits",
            rays.len() as f64 / secs / 1e6,
            secs * 1e9 / rays.len() as f64,
            iters as f64 / hits.max(1) as f64
        );
    }
}
