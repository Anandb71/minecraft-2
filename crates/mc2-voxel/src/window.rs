//! A read-only window of brick cells cut out of the world, for many voxel
//! lookups in a small region such as a body's collision samples: one hash
//! lookup and tree descent per brick cell when the window is built, then
//! an array index per voxel.

use crate::brick::Brick;
use crate::coords::{BRICK_SHIFT, CHUNK_SHIFT, ChunkPos, VoxelPos};
use crate::material::MaterialId;
use crate::tree::Cell;
use crate::world::VoxelWorld;
use glam::IVec3;

/// Windows larger than this many brick cells fall back to world lookups.
pub const MAX_WINDOW_CELLS: usize = 1 << 15;

#[derive(Clone, Copy)]
enum CellRef<'a> {
    Uniform(MaterialId),
    Brick(&'a Brick),
}

pub struct VoxelWindow<'a> {
    world: &'a VoxelWorld,
    min_cell: IVec3,
    dims: IVec3,
    cells: Vec<CellRef<'a>>,
    solid: bool,
}

impl<'a> VoxelWindow<'a> {
    /// Covers the inclusive voxel box `min..=max`.
    pub fn new(world: &'a VoxelWorld, min: VoxelPos, max: VoxelPos) -> Self {
        let min_cell = min >> BRICK_SHIFT;
        let dims = (max >> BRICK_SHIFT) - min_cell + 1;
        let count = dims.max(IVec3::ZERO).element_product() as usize;
        if count == 0 || count > MAX_WINDOW_CELLS {
            return Self {
                world,
                min_cell,
                dims: IVec3::ZERO,
                cells: Vec::new(),
                solid: true,
            };
        }
        let mut cells = Vec::with_capacity(count);
        let mut solid = false;
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let c = min_cell + IVec3::new(x, y, z);
                    let pos = ChunkPos::of_voxel(c << BRICK_SHIFT);
                    let local = c - (pos.0 << (CHUNK_SHIFT - BRICK_SHIFT));
                    let r = match world.chunk(pos).map(|t| (t, t.cell(local))) {
                        None | Some((_, Cell::Empty)) => CellRef::Uniform(MaterialId(0)),
                        Some((_, Cell::Uniform(m))) => CellRef::Uniform(m),
                        Some((t, Cell::Brick(i))) => CellRef::Brick(t.brick(i)),
                    };
                    solid |= !matches!(r, CellRef::Uniform(MaterialId(0)));
                    cells.push(r);
                }
            }
        }
        Self {
            world,
            min_cell,
            dims,
            cells,
            solid,
        }
    }

    /// False when every cell of the window is air: nothing inside can touch
    /// the world. Oversized windows always report true.
    pub fn any_matter(&self) -> bool {
        self.solid
    }

    #[inline]
    pub fn voxel(&self, v: VoxelPos) -> MaterialId {
        let c = (v >> BRICK_SHIFT) - self.min_cell;
        if c.cmplt(IVec3::ZERO).any() || c.cmpge(self.dims).any() {
            return self.world.voxel(v);
        }
        let i = (c.x + self.dims.x * (c.y + self.dims.y * c.z)) as usize;
        match self.cells[i] {
            CellRef::Uniform(m) => m,
            CellRef::Brick(b) => b.get(v & ((1 << BRICK_SHIFT) - 1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::ids;

    #[test]
    fn window_agrees_with_the_world_inside_and_out() {
        let mut w = VoxelWorld::new();
        w.fill_box(
            IVec3::new(500, 0, 500),
            IVec3::new(530, 20, 530),
            ids::GRANITE,
        );
        w.set_voxel(IVec3::new(510, 21, 511), ids::PLANKS);
        w.set_voxel(IVec3::new(513, 22, 509), ids::DIRT);
        let win = VoxelWindow::new(&w, IVec3::new(505, 15, 505), IVec3::new(520, 30, 516));
        assert!(win.any_matter());
        for z in 495..540 {
            for y in 10..35 {
                for x in 495..540 {
                    let v = IVec3::new(x, y, z);
                    assert_eq!(win.voxel(v), w.voxel(v), "{v}");
                }
            }
        }
        let sky = VoxelWindow::new(&w, IVec3::new(505, 40, 505), IVec3::new(520, 60, 520));
        assert!(!sky.any_matter());
    }
}
