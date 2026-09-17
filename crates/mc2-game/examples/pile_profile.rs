//! Step cost against the size of a debris pile: 2 m pieces dropped into a
//! heap, as a collapsing tower makes.
//! `cargo run --release -p mc2-game --example pile_profile`

use glam::{DVec3, IVec3, Quat, Vec3};
use mc2_physics::{BodyShape, PhysicsWorld};
use mc2_voxel::material::ids;
use mc2_voxel::world::VoxelWorld;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let mut w = VoxelWorld::new();
    w.fill_box(IVec3::ZERO, IVec3::new(1023, 63, 1023), ids::BEDROCK);
    let shape = Arc::new(
        BodyShape::from_voxels(IVec3::splat(32), vec![ids::STONE_BRICK; 32 * 32 * 32]).unwrap(),
    );
    for &n in &[50usize, 150, 300, 600] {
        let mut p = PhysicsWorld::new();
        let side = (n as f64).cbrt().ceil() as i32;
        for i in 0..n as i32 {
            let (x, y, z) = (i % side, i / (side * side), (i / side) % side);
            let pos = DVec3::new(
                16.0 + f64::from(x) * 2.1,
                4.2 + f64::from(y) * 2.1,
                16.0 + f64::from(z) * 2.1,
            );
            let id = p.spawn(shape.clone(), pos, Quat::IDENTITY);
            p.body_mut(id).unwrap().vel = Vec3::new(0.0, -1.0, 0.0);
        }
        let mut steps = Vec::new();
        for _ in 0..600 {
            let t = Instant::now();
            p.step(1.0 / 120.0, &w);
            steps.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mean = steps.iter().sum::<f64>() / steps.len() as f64;
        steps.sort_by(f64::total_cmp);
        println!(
            "{n} bodies: step mean {mean:.2} ms, p99 {:.2} ms, max {:.2} ms; {} awake, {} pairs",
            steps[steps.len() * 99 / 100],
            steps[steps.len() - 1],
            p.stats.awake,
            p.stats.pairs,
        );
    }
}
