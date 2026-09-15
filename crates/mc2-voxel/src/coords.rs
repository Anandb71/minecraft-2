//! Spatial units and conversions.
//!
//! The finest unit is the voxel, 1/16 m. Everything above it is a power of
//! two so conversions are shifts:
//!
//! | unit    | voxels | metres | notes                                  |
//! |---------|--------|--------|----------------------------------------|
//! | voxel   | 1      | 0.0625 | material + state                       |
//! | subblk  | 4      | 0.25   | one 64-bit occupancy mask              |
//! | brick   | 8      | 0.5    | 4-bit palette storage, streaming unit  |
//! | block   | 16     | 1      | macro layer: gameplay, inventory, save |
//! | L1 node | 32     | 2      | 4^3 bricks                             |
//! | L2 node | 128    | 8      |                                        |
//! | chunk   | 512    | 32     | L3 node, CPU edit and upload unit      |
//! | L4 node | 2048   | 128    |                                        |
//! | sector  | 8192   | 512    | L5 tree root, streamed region          |
//!
//! The world is 32 x 1 x 32 sectors: 16384 m square, 512 m tall.

use glam::IVec3;

pub const VOXEL_SIZE_M: f32 = 1.0 / 16.0;
pub const VOXELS_PER_BLOCK: i32 = 16;
pub const BRICK_SHIFT: i32 = 3;
pub const BRICK_VOXELS: i32 = 1 << BRICK_SHIFT;
pub const BLOCK_SHIFT: i32 = 4;
pub const CHUNK_SHIFT: i32 = 9;
pub const CHUNK_VOXELS: i32 = 1 << CHUNK_SHIFT;
pub const CHUNK_BLOCKS: i32 = CHUNK_VOXELS / VOXELS_PER_BLOCK;
pub const CHUNK_BRICKS: i32 = CHUNK_VOXELS / BRICK_VOXELS;
pub const SECTOR_SHIFT: i32 = 13;
pub const SECTOR_VOXELS: i32 = 1 << SECTOR_SHIFT;

pub const WORLD_SECTORS_XZ: i32 = 32;
pub const WORLD_SECTORS_Y: i32 = 1;
pub const WORLD_BLOCKS_XZ: i32 = WORLD_SECTORS_XZ * (SECTOR_VOXELS / VOXELS_PER_BLOCK);
pub const WORLD_BLOCKS_Y: i32 = WORLD_SECTORS_Y * (SECTOR_VOXELS / VOXELS_PER_BLOCK);

/// World voxel coordinate.
pub type VoxelPos = IVec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkPos(pub IVec3);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockPos(pub IVec3);

impl ChunkPos {
    pub fn of_voxel(v: VoxelPos) -> Self {
        Self(v >> CHUNK_SHIFT)
    }

    pub fn of_block(b: BlockPos) -> Self {
        Self(b.0 >> (CHUNK_SHIFT - BLOCK_SHIFT))
    }

    /// Voxel coordinate of the chunk's minimum corner.
    pub fn origin(self) -> VoxelPos {
        self.0 << CHUNK_SHIFT
    }

    pub fn in_world(self) -> bool {
        let per_sector = 1 << (SECTOR_SHIFT - CHUNK_SHIFT);
        let (xz, y) = (WORLD_SECTORS_XZ * per_sector, WORLD_SECTORS_Y * per_sector);
        self.0.x >= 0
            && self.0.z >= 0
            && self.0.y >= 0
            && self.0.x < xz
            && self.0.z < xz
            && self.0.y < y
    }
}

impl BlockPos {
    pub fn of_voxel(v: VoxelPos) -> Self {
        Self(v >> BLOCK_SHIFT)
    }

    pub fn origin(self) -> VoxelPos {
        self.0 << BLOCK_SHIFT
    }
}

/// Voxel position within its chunk, each component in `0..512`.
pub fn chunk_local(v: VoxelPos) -> IVec3 {
    v & (CHUNK_VOXELS - 1)
}

pub fn in_world_voxel(v: VoxelPos) -> bool {
    v.cmpge(IVec3::ZERO).all()
        && v.x < WORLD_SECTORS_XZ * SECTOR_VOXELS
        && v.z < WORLD_SECTORS_XZ * SECTOR_VOXELS
        && v.y < WORLD_SECTORS_Y * SECTOR_VOXELS
}

pub fn metres_to_voxel(p: glam::DVec3) -> VoxelPos {
    (p * f64::from(VOXELS_PER_BLOCK)).floor().as_ivec3()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_coordinates_floor_correctly() {
        assert_eq!(
            ChunkPos::of_voxel(IVec3::new(-1, 0, 511)).0,
            IVec3::new(-1, 0, 0)
        );
        assert_eq!(chunk_local(IVec3::new(-1, 512, 3)), IVec3::new(511, 0, 3));
        assert_eq!(
            BlockPos::of_voxel(IVec3::new(-17, 15, 16)).0,
            IVec3::new(-2, 0, 1)
        );
    }

    #[test]
    fn world_extents_match_brief() {
        assert_eq!(WORLD_BLOCKS_XZ, 16384);
        assert_eq!(WORLD_BLOCKS_Y, 512);
        assert_eq!(CHUNK_BLOCKS, 32);
        assert_eq!(CHUNK_BRICKS, 64);
        assert!(ChunkPos(IVec3::new(511, 15, 0)).in_world());
        assert!(!ChunkPos(IVec3::new(512, 0, 0)).in_world());
    }
}
