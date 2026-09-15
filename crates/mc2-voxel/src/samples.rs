//! Deterministic scenes for tests, benchmarks and golden images.

use crate::material::ids;
use crate::world::VoxelWorld;
use glam::{DVec3, IVec3};

/// Xorshift64: reproducible across platforms, no dependency.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// 64 m x 37.5 m of rolling terrain with grass over granite and sandstone,
/// a basalt floor, carved spherical caves and scattered ore. Spans two
/// chunk boundaries along x.
pub fn rolling_terrain(rng: &mut Rng) -> VoxelWorld {
    let mut w = VoxelWorld::new();
    for x in 0..1024 {
        for z in 0..600 {
            let h =
                200.0 + 40.0 * (f64::from(x) * 0.013).sin() + 25.0 * (f64::from(z) * 0.021).cos();
            let top = h as i32;
            let mat = if (x / 16 + z / 16) % 5 == 0 {
                ids::SANDSTONE
            } else {
                ids::GRANITE
            };
            for y in (top - 6).max(150)..=top {
                let m = if y > top - 2 { ids::GRASS } else { mat };
                w.set_voxel(IVec3::new(x, y, z), m);
            }
        }
    }
    w.fill_box(
        IVec3::new(0, 100, 0),
        IVec3::new(1023, 149, 599),
        ids::BASALT,
    );
    for _ in 0..40 {
        let c = DVec3::new(
            rng.next_f64() * 1024.0,
            120.0 + rng.next_f64() * 120.0,
            rng.next_f64() * 600.0,
        );
        w.fill_sphere(c, 4.0 + rng.next_f64() * 20.0, ids::AIR);
    }
    for _ in 0..200 {
        let v = IVec3::new(
            (rng.next_f64() * 1024.0) as i32,
            240 + (rng.next_f64() * 60.0) as i32,
            (rng.next_f64() * 600.0) as i32,
        );
        w.set_voxel(v, ids::IRON_ORE);
    }
    w
}
