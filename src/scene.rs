//! Startup scene. Replaced by generated terrain once worldgen lands.

use glam::DVec3;
use mc2_render::camera::Camera;
use mc2_voxel::samples::{Rng, rolling_terrain};
use mc2_voxel::world::VoxelWorld;

pub fn demo() -> (VoxelWorld, Camera) {
    let world = rolling_terrain(&mut Rng(0x9e37_79b9_7f4a_7c15));
    let mut camera = Camera {
        position: DVec3::new(6.0, 19.0, 4.0),
        ..Default::default()
    };
    camera.look_at(DVec3::new(34.0, 12.0, 26.0));
    (world, camera)
}
