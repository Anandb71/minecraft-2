//! What the solver collides with: which voxels are solid, a brick cell at a
//! time. The voxel world answers directly; the physics thread answers from
//! a mirror of just the cells near its bodies, fed by the thread that owns
//! the world.

use glam::IVec3;
use mc2_core::FxHashMap;
use mc2_voxel::brick::subblock_bit;
use mc2_voxel::coords::{BRICK_SHIFT, CHUNK_SHIFT, ChunkPos};
use mc2_voxel::tree::Cell;
use mc2_voxel::world::VoxelWorld;
use std::borrow::Cow;

/// Solid voxels of one 8^3 brick cell, in brick occupancy layout (one
/// 64-bit mask per 4^3 subblock).
#[derive(Clone, Debug, PartialEq)]
pub enum Occ<'a> {
    Empty,
    Full,
    Bits(Cow<'a, [u64; 8]>),
}

impl Occ<'_> {
    #[inline]
    pub fn solid(&self, local: IVec3) -> bool {
        match self {
            Occ::Empty => false,
            Occ::Full => true,
            Occ::Bits(bits) => {
                let (s, b) = subblock_bit(local);
                (bits[s] >> b) & 1 != 0
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Occ::Empty => true,
            Occ::Full => false,
            Occ::Bits(bits) => bits.iter().all(|&w| w == 0),
        }
    }

    pub fn into_owned(self) -> Occ<'static> {
        match self {
            Occ::Empty => Occ::Empty,
            Occ::Full => Occ::Full,
            Occ::Bits(b) => Occ::Bits(Cow::Owned(b.into_owned())),
        }
    }
}

/// Anything that can say which voxels of a brick cell are solid.
pub trait SolidCells: Sync {
    /// Occupancy of a brick cell, or `None` when this source does not know
    /// it yet.
    fn occ(&self, cell: IVec3) -> Option<Occ<'_>>;
}

/// Solid occupancy of a brick cell of the world. Bricks whose materials are
/// all solid (or air) reuse their occupancy masks; others are tested voxel
/// by voxel.
pub fn world_occ(world: &VoxelWorld, cell: IVec3) -> Occ<'_> {
    let pos = ChunkPos::of_voxel(cell * 8);
    let local = cell - (pos.0 << (CHUNK_SHIFT - BRICK_SHIFT));
    let Some(tree) = world.chunk(pos) else {
        return Occ::Empty;
    };
    match tree.cell(local) {
        Cell::Empty => Occ::Empty,
        Cell::Uniform(m) if m.is_solid() => Occ::Full,
        Cell::Uniform(_) => Occ::Empty,
        Cell::Brick(i) => {
            let brick = tree.brick(i);
            if brick.materials().all(|(m, _)| m.is_air() || m.is_solid()) {
                return Occ::Bits(Cow::Borrowed(brick.occupancy()));
            }
            let mut bits = [0u64; 8];
            for i in 0..512 {
                let l = IVec3::new(i & 7, (i >> 6) & 7, (i >> 3) & 7);
                if brick.get(l).is_solid() {
                    let (s, b) = subblock_bit(l);
                    bits[s] |= 1 << b;
                }
            }
            Occ::Bits(Cow::Owned(bits))
        }
    }
}

impl SolidCells for VoxelWorld {
    fn occ(&self, cell: IVec3) -> Option<Occ<'_>> {
        Some(world_occ(self, cell))
    }
}

/// Cells mirrored from the world, owned by the physics thread.
#[derive(Default)]
pub struct CollisionMap {
    cells: FxHashMap<IVec3, Occ<'static>>,
}

impl CollisionMap {
    pub fn insert(&mut self, cell: IVec3, occ: Occ<'static>) {
        self.cells.insert(cell, occ);
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Drops cells for which `keep` is false; returns them.
    pub fn evict(&mut self, mut keep: impl FnMut(IVec3) -> bool) -> Vec<IVec3> {
        let mut gone = Vec::new();
        self.cells.retain(|&c, _| {
            let k = keep(c);
            if !k {
                gone.push(c);
            }
            k
        });
        gone
    }
}

impl SolidCells for CollisionMap {
    fn occ(&self, cell: IVec3) -> Option<Occ<'_>> {
        self.cells.get(&cell).map(|o| match o {
            Occ::Empty => Occ::Empty,
            Occ::Full => Occ::Full,
            Occ::Bits(b) => Occ::Bits(Cow::Borrowed(b.as_ref())),
        })
    }
}

/// Brick cells covering a voxel box, resolved once for many lookups.
pub struct CollisionWindow<'a, S: SolidCells> {
    source: &'a S,
    min_cell: IVec3,
    dims: IVec3,
    cells: Vec<Occ<'a>>,
    any_solid: bool,
    /// Cells the source could not answer; they read as empty.
    pub missing: Vec<IVec3>,
}

/// Windows larger than this many cells read the source cell by cell.
const MAX_WINDOW_CELLS: usize = 1 << 15;

impl<'a, S: SolidCells> CollisionWindow<'a, S> {
    /// Covers the inclusive voxel box `min..=max`.
    pub fn new(source: &'a S, min: IVec3, max: IVec3) -> Self {
        let min_cell = min >> BRICK_SHIFT;
        let dims = (max >> BRICK_SHIFT) - min_cell + 1;
        let count = dims.max(IVec3::ZERO).element_product() as usize;
        let mut w = Self {
            source,
            min_cell,
            dims: IVec3::ZERO,
            cells: Vec::new(),
            any_solid: true,
            missing: Vec::new(),
        };
        if count == 0 || count > MAX_WINDOW_CELLS {
            return w;
        }
        w.dims = dims;
        w.cells.reserve(count);
        w.any_solid = false;
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let c = min_cell + IVec3::new(x, y, z);
                    let occ = match source.occ(c) {
                        Some(o) => o,
                        None => {
                            w.missing.push(c);
                            Occ::Empty
                        }
                    };
                    w.any_solid |= !occ.is_empty();
                    w.cells.push(occ);
                }
            }
        }
        w
    }

    /// False when nothing in the window is solid. Oversized windows always
    /// report true.
    pub fn any_solid(&self) -> bool {
        self.any_solid
    }

    #[inline]
    pub fn solid(&self, v: IVec3) -> bool {
        let cell = v >> BRICK_SHIFT;
        let c = cell - self.min_cell;
        let local = v & ((1 << BRICK_SHIFT) - 1);
        if c.cmplt(IVec3::ZERO).any() || c.cmpge(self.dims).any() {
            return self.source.occ(cell).is_some_and(|o| o.solid(local));
        }
        let i = (c.x + self.dims.x * (c.y + self.dims.y * c.z)) as usize;
        self.cells[i].solid(local)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;

    fn world() -> VoxelWorld {
        let mut w = VoxelWorld::new();
        w.fill_box(
            IVec3::new(500, 0, 500),
            IVec3::new(530, 20, 530),
            ids::GRANITE,
        );
        w.set_voxel(IVec3::new(510, 21, 511), ids::PLANKS);
        // Water and leaves are not solid, even inside a brick of stone.
        w.set_voxel(IVec3::new(513, 22, 509), ids::WATER);
        w.set_voxel(IVec3::new(514, 20, 509), ids::WATER);
        w.set_voxel(IVec3::new(515, 21, 509), ids::LEAVES);
        w
    }

    #[test]
    fn windows_agree_with_world_solidity() {
        let w = world();
        let win = CollisionWindow::new(&w, IVec3::new(505, 15, 505), IVec3::new(520, 30, 516));
        assert!(win.any_solid());
        assert!(win.missing.is_empty());
        for z in 495..540 {
            for y in 10..35 {
                for x in 495..540 {
                    let v = IVec3::new(x, y, z);
                    assert_eq!(win.solid(v), w.voxel(v).is_solid(), "{v}");
                }
            }
        }
        let sky = CollisionWindow::new(&w, IVec3::new(505, 40, 505), IVec3::new(520, 60, 520));
        assert!(!sky.any_solid());
    }

    #[test]
    fn a_mirror_answers_what_it_was_given_and_reports_the_rest() {
        let w = world();
        let mut map = CollisionMap::default();
        for z in 62..=66 {
            for y in 1..=3 {
                for x in 62..=66 {
                    let c = IVec3::new(x, y, z);
                    map.insert(c, world_occ(&w, c).into_owned());
                }
            }
        }
        let win = CollisionWindow::new(&map, IVec3::new(500, 10, 500), IVec3::new(540, 30, 540));
        assert!(!win.missing.is_empty());
        let inside = CollisionWindow::new(&map, IVec3::new(500, 10, 500), IVec3::new(530, 30, 530));
        assert!(inside.missing.is_empty());
        for z in 500..=530 {
            for y in 10..=30 {
                for x in 500..=530 {
                    let v = IVec3::new(x, y, z);
                    assert_eq!(inside.solid(v), w.voxel(v).is_solid(), "{v}");
                }
            }
        }
        let gone = map.evict(|c| c.x < 64);
        assert_eq!(gone.len(), 3 * 3 * 5);
        assert_eq!(map.len(), 2 * 3 * 5);
    }
}
