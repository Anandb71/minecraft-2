//! The loaded voxel world: chunk trees keyed by chunk position.

use crate::coords::{self, BRICK_SHIFT, CHUNK_SHIFT, ChunkPos, VoxelPos};
use crate::material::MaterialId;
use crate::tree::{Cell, ChunkTree};
use glam::IVec3;
use mc2_core::{FxHashMap, FxHashSet};

#[derive(Default)]
pub struct VoxelWorld {
    chunks: FxHashMap<ChunkPos, ChunkTree>,
    dirty: FxHashSet<ChunkPos>,
}

impl VoxelWorld {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_chunk(&mut self, pos: ChunkPos, tree: ChunkTree) {
        self.chunks.insert(pos, tree);
        self.dirty.insert(pos);
    }

    pub fn remove_chunk(&mut self, pos: ChunkPos) -> Option<ChunkTree> {
        self.dirty.insert(pos);
        self.chunks.remove(&pos)
    }

    pub fn chunk(&self, pos: ChunkPos) -> Option<&ChunkTree> {
        self.chunks.get(&pos)
    }

    pub fn chunk_mut(&mut self, pos: ChunkPos) -> Option<&mut ChunkTree> {
        self.dirty.insert(pos);
        self.chunks.get_mut(&pos)
    }

    pub fn chunks(&self) -> impl Iterator<Item = (&ChunkPos, &ChunkTree)> {
        self.chunks.iter()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn voxel(&self, v: VoxelPos) -> MaterialId {
        self.chunks
            .get(&ChunkPos::of_voxel(v))
            .map_or(MaterialId(0), |t| t.voxel(coords::chunk_local(v)))
    }

    /// Sets a voxel, creating the chunk if needed. Out-of-world writes are
    /// ignored and report air.
    pub fn set_voxel(&mut self, v: VoxelPos, m: MaterialId) -> MaterialId {
        if !coords::in_world_voxel(v) {
            return MaterialId(0);
        }
        let pos = ChunkPos::of_voxel(v);
        let tree = self.chunks.entry(pos).or_default();
        let old = tree.set_voxel(coords::chunk_local(v), m);
        if old != m {
            self.dirty.insert(pos);
        }
        old
    }

    /// Fills an inclusive voxel box. Whole brick cells become uniform cells
    /// directly; only the boundary bricks are edited voxel by voxel.
    pub fn fill_box(&mut self, min: VoxelPos, max: VoxelPos, m: MaterialId) {
        let lo = min.max(IVec3::ZERO);
        let hi = max.min(IVec3::new(
            coords::WORLD_SECTORS_XZ * coords::SECTOR_VOXELS - 1,
            coords::WORLD_SECTORS_Y * coords::SECTOR_VOXELS - 1,
            coords::WORLD_SECTORS_XZ * coords::SECTOR_VOXELS - 1,
        ));
        if lo.cmpgt(hi).any() {
            return;
        }
        let blo = lo >> BRICK_SHIFT;
        let bhi = hi >> BRICK_SHIFT;
        for bz in blo.z..=bhi.z {
            for by in blo.y..=bhi.y {
                for bx in blo.x..=bhi.x {
                    let b = IVec3::new(bx, by, bz);
                    let bmin = b << BRICK_SHIFT;
                    let bmax = bmin + IVec3::splat(7);
                    let chunk = ChunkPos(b >> (CHUNK_SHIFT - BRICK_SHIFT));
                    let local_brick = b - (chunk.0 << (CHUNK_SHIFT - BRICK_SHIFT));
                    let tree = self.chunks.entry(chunk).or_default();
                    self.dirty.insert(chunk);
                    if bmin.cmpge(lo).all() && bmax.cmple(hi).all() {
                        let cell = if m.is_air() {
                            Cell::Empty
                        } else {
                            Cell::Uniform(m)
                        };
                        tree.set_cell(local_brick, cell);
                        continue;
                    }
                    let a = lo.max(bmin) - bmin;
                    let z = hi.min(bmax) - bmin;
                    tree.edit_brick(local_brick, |brick| {
                        for y in a.y..=z.y {
                            for zz in a.z..=z.z {
                                for x in a.x..=z.x {
                                    brick.set(IVec3::new(x, y, zz), m);
                                }
                            }
                        }
                    });
                }
            }
        }
        self.chunks.retain(|_, t| !t.is_empty());
    }

    /// Sets every voxel whose centre lies within `radius` voxels of `center`.
    /// Returns the voxels that changed with their previous materials.
    pub fn fill_sphere(
        &mut self,
        center: glam::DVec3,
        radius: f64,
        m: MaterialId,
    ) -> Vec<(VoxelPos, MaterialId)> {
        let lo = (center - radius).floor().as_ivec3();
        let hi = (center + radius).ceil().as_ivec3();
        let r2 = radius * radius;
        let mut changed = Vec::new();
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let v = IVec3::new(x, y, z);
                    if (v.as_dvec3() + 0.5 - center).length_squared() > r2 {
                        continue;
                    }
                    let old = self.set_voxel(v, m);
                    if old != m {
                        changed.push((v, old));
                    }
                }
            }
        }
        changed
    }

    /// Chunks touched since the last call, in no particular order.
    pub fn take_dirty(&mut self) -> Vec<ChunkPos> {
        self.dirty.drain().collect()
    }

    pub fn memory_bytes(&self) -> usize {
        self.chunks.values().map(ChunkTree::memory_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::ids;

    #[test]
    fn voxels_cross_chunk_boundaries() {
        let mut w = VoxelWorld::new();
        let a = IVec3::new(511, 100, 511);
        let b = IVec3::new(512, 100, 512);
        w.set_voxel(a, ids::GRANITE);
        w.set_voxel(b, ids::SAND);
        assert_eq!(w.voxel(a), ids::GRANITE);
        assert_eq!(w.voxel(b), ids::SAND);
        assert_eq!(w.chunk_count(), 2);
        assert_eq!(w.set_voxel(IVec3::new(-1, 0, 0), ids::SAND), ids::AIR);
    }

    #[test]
    fn fill_box_uses_uniform_cells_inside() {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::new(3, 0, 0), IVec3::new(100, 50, 60), ids::BASALT);
        assert_eq!(w.voxel(IVec3::new(3, 0, 0)), ids::BASALT);
        assert_eq!(w.voxel(IVec3::new(100, 50, 60)), ids::BASALT);
        assert_eq!(w.voxel(IVec3::new(2, 0, 0)), ids::AIR);
        assert_eq!(w.voxel(IVec3::new(101, 50, 60)), ids::AIR);
        let tree = w.chunk(ChunkPos(IVec3::ZERO)).unwrap();
        // Only boundary bricks need palette storage.
        let cells = tree.cells().count();
        assert!(
            tree.brick_count() < cells / 2,
            "{} bricks of {cells}",
            tree.brick_count()
        );
        w.fill_box(IVec3::ZERO, IVec3::splat(200), ids::AIR);
        assert_eq!(w.chunk_count(), 0);
    }

    #[test]
    fn sphere_reports_changes() {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::splat(63), ids::GRANITE);
        let changed = w.fill_sphere(glam::DVec3::splat(32.0), 8.0, ids::AIR);
        assert!(
            changed.len() > 2000 && changed.len() < 2300,
            "{}",
            changed.len()
        );
        assert!(changed.iter().all(|(_, m)| *m == ids::GRANITE));
        assert_eq!(w.voxel(IVec3::splat(32)), ids::AIR);
    }
}
