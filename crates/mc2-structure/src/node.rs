//! What the structural solver knows about one 1 m block: its mass, what it
//! is mostly made of, and how much solid material each face offers the
//! neighbour on that side.

use glam::IVec3;
use mc2_physics::collision::{Occ, world_occ};
use mc2_voxel::coords::{BRICK_SHIFT, BlockPos, CHUNK_SHIFT, ChunkPos, VOXEL_SIZE_M};
use mc2_voxel::material::{MaterialId, ids};
use mc2_voxel::tree::Cell;
use mc2_voxel::world::VoxelWorld;

/// Blocks with fewer solid voxels are not structure (a pebble, a tuft).
pub const MIN_NODE_VOXELS: u32 = 16;

/// Face directions, in the order of `NodeInfo::faces`.
pub const FACES: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

/// Index of the face pointing the other way.
pub fn opposite(face: usize) -> usize {
    face ^ 1
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeInfo {
    pub solid: u32,
    /// Kilograms.
    pub mass: f32,
    /// The solid material with the most voxels.
    pub material: MaterialId,
    /// Solid voxels on each face (0 to 256), in `FACES` order.
    pub faces: [u16; 6],
}

impl NodeInfo {
    pub const EMPTY: Self = Self {
        solid: 0,
        mass: 0.0,
        material: MaterialId(0),
        faces: [0; 6],
    };

    pub fn is_node(&self) -> bool {
        self.solid >= MIN_NODE_VOXELS
    }

    pub fn is_bedrock(&self) -> bool {
        self.material == ids::BEDROCK
    }
}

/// Summarises a block from its eight brick cells. Uniform cells cost one
/// lookup; bricks use their palette counts for mass and their solid
/// occupancy for faces.
pub fn block_info(world: &VoxelWorld, block: BlockPos) -> NodeInfo {
    let origin = block.origin();
    // A block lies wholly inside one chunk.
    let pos = ChunkPos::of_voxel(origin);
    let Some(tree) = world.chunk(pos) else {
        return NodeInfo::EMPTY;
    };
    let voxel_volume = VOXEL_SIZE_M * VOXEL_SIZE_M * VOXEL_SIZE_M;
    let mut info = NodeInfo::EMPTY;
    let mut counts: Vec<(MaterialId, u32)> = Vec::new();
    let mut add = |m: MaterialId, n: u32, info: &mut NodeInfo| {
        if !m.is_solid() || n == 0 {
            return;
        }
        info.solid += n;
        info.mass += n as f32 * m.get().density * voxel_volume;
        match counts.iter_mut().find(|(k, _)| *k == m) {
            Some((_, c)) => *c += n,
            None => counts.push((m, n)),
        }
    };
    for i in 0..8 {
        let offset = IVec3::new(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        let cell = (origin >> BRICK_SHIFT) + offset;
        let local = cell - (pos.0 << (CHUNK_SHIFT - BRICK_SHIFT));
        let occ = match tree.cell(local) {
            Cell::Empty => continue,
            Cell::Uniform(m) => {
                add(m, 512, &mut info);
                if m.is_solid() { Occ::Full } else { Occ::Empty }
            }
            Cell::Brick(b) => {
                for (m, n) in tree.brick(b).materials() {
                    add(m, u32::from(n), &mut info);
                }
                world_occ(world, cell)
            }
        };
        if occ.is_empty() {
            continue;
        }
        // Faces of the block this cell lies on.
        for (f, dir) in FACES.iter().enumerate() {
            let axis = dir.abs().max_position();
            let on_face = if dir[axis] > 0 {
                offset[axis] == 1
            } else {
                offset[axis] == 0
            };
            if !on_face {
                continue;
            }
            let fixed = if dir[axis] > 0 { 7 } else { 0 };
            let n = match &occ {
                Occ::Full => 64,
                Occ::Empty => 0,
                Occ::Bits(_) => {
                    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                    let mut n = 0;
                    for a in 0..8 {
                        for b in 0..8 {
                            let mut l = IVec3::ZERO;
                            l[axis] = fixed;
                            l[u] = a;
                            l[v] = b;
                            n += u16::from(occ.solid(l));
                        }
                    }
                    n
                }
            };
            info.faces[f] += n;
        }
    }
    if let Some(&(m, _)) = counts
        .iter()
        .max_by_key(|(m, n)| (*n, std::cmp::Reverse(m.0)))
    {
        info.material = m;
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_and_carved_blocks_report_mass_and_faces() {
        let mut w = VoxelWorld::new();
        let b = BlockPos(IVec3::new(3, 2, 5));
        let o = b.origin();
        w.fill_box(o, o + 15, ids::GRANITE);
        let full = block_info(&w, b);
        assert_eq!(full.solid, 4096);
        assert!(
            (full.mass - 4096.0 * 2700.0 / 4096.0).abs() < 0.5,
            "{}",
            full.mass
        );
        assert_eq!(full.material, ids::GRANITE);
        assert_eq!(full.faces, [256; 6]);
        assert!(full.is_node());

        // Carve the top half and a 4x4 hole through the bottom face; plank
        // the rest of the top layer of the bottom half.
        w.fill_box(o + IVec3::new(0, 8, 0), o + 15, ids::AIR);
        w.fill_box(o, o + IVec3::new(3, 7, 3), ids::AIR);
        w.fill_box(
            o + IVec3::new(0, 7, 0),
            o + IVec3::new(15, 7, 15),
            ids::PLANKS,
        );
        let carved = block_info(&w, b);
        assert_eq!(carved.faces[2], 0, "top face is air");
        assert_eq!(carved.faces[3], 256 - 16, "bottom face has a hole");
        assert_eq!(carved.faces[0], 16 * 8);
        assert_eq!(carved.faces[1], 16 * 8 - 4 * 7);
        assert_eq!(carved.solid, 16 * 16 * 8 - 16 * 7);
        // 7 layers of granite minus the hole, one layer of planks.
        let expect = ((16 * 16 * 7 - 16 * 7) as f32 * 2700.0 + 256.0 * 600.0) / 4096.0;
        assert!(
            (carved.mass - expect).abs() < 0.5,
            "{} vs {expect}",
            carved.mass
        );
        assert_eq!(carved.material, ids::GRANITE);

        let air = block_info(&w, BlockPos(IVec3::new(3, 3, 5)));
        assert!(!air.is_node());
    }
}
