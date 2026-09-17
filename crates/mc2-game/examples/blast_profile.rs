//! Cost of one TNT blast and of the physics steps that follow, on flat
//! granite under a metre of dirt.
//! `cargo run --release -p mc2-game --example blast_profile`

use glam::{DVec3, IVec3};
use mc2_physics::PhysicsWorld;
use mc2_physics::explode::explode;
use mc2_voxel::material::ids;
use mc2_voxel::world::VoxelWorld;
use std::time::Instant;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let mut w = VoxelWorld::new();
    w.fill_box(IVec3::ZERO, IVec3::new(1023, 127, 1023), ids::GRANITE);
    w.fill_box(
        IVec3::new(0, 128, 0),
        IVec3::new(1023, 143, 1023),
        ids::DIRT,
    );
    let mut p = PhysicsWorld::new();
    let mut blasts = Vec::new();
    for (i, x) in [30.0, 34.0, 38.0].into_iter().enumerate() {
        let t = Instant::now();
        let r = explode(&mut w, &mut p, DVec3::new(x, 8.5, 32.0), 3.0, 48, i as u64);
        blasts.push(ms(t));
        println!(
            "blast {i}: {:.1} ms, {} voxels, {} debris",
            blasts[i], r.removed_voxels, r.debris
        );
    }
    let mut steps = Vec::new();
    let mut pair_ms = 0.0;
    for _ in 0..1200 {
        let t = Instant::now();
        p.step(1.0 / 120.0, &w);
        steps.push(ms(t));
        pair_ms += f64::from(p.stats.pair_ms);
    }
    let first: f64 = steps[..120].iter().sum::<f64>() / 120.0;
    let mean: f64 = steps.iter().sum::<f64>() / steps.len() as f64;
    steps.sort_by(f64::total_cmp);
    println!(
        "{} bodies: step mean {mean:.2} ms (first second {first:.2}, pairs {:.2}), p99 {:.2} ms, max {:.2} ms; {} awake after 10 s",
        p.bodies.len(),
        pair_ms / 1200.0,
        steps[steps.len() * 99 / 100],
        steps[steps.len() - 1],
        p.stats.awake,
    );
}
