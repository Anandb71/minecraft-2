//! Game layer: player, interaction, blocks, on a standalone bevy_ecs world.

pub mod collide;
pub mod input;
pub mod player;

use bevy_ecs::prelude::*;
use mc2_voxel::world::VoxelWorld;
use mc2_worldgen::stream::ChunkStreamer;

/// The loaded voxel world.
#[derive(Resource, Default)]
pub struct Voxels(pub VoxelWorld);

/// Chunk streaming, absent until terrain has loaded.
#[derive(Resource, Default)]
pub struct Streaming(pub Option<ChunkStreamer>);
