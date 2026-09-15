//! Sparse 64-tree for one 32 m chunk.
//!
//! Three levels of 4^3 fan-out sit above the bricks: the chunk root (children
//! are 8 m L2 nodes), L2 (children are 2 m L1 nodes) and L1 (children are
//! 0.5 m brick cells). Every node stores a 64-bit child mask and a compact
//! child array: child `i` lives at `popcount(mask & ((1 << i) - 1))`. A brick
//! cell is empty (absent from the mask), uniform (one material reference, no
//! brick allocated) or a full palette brick. An intact block of stone is
//! eight uniform cells: a few bytes.
//!
//! Child index convention (shared with the GPU): `x + z*4 + y*16`.

use crate::brick::Brick;
use crate::material::MaterialId;
use glam::IVec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Empty,
    Uniform(MaterialId),
    /// Index into the chunk's brick slab.
    Brick(u32),
}

#[inline]
pub fn child_index(c: IVec3) -> u32 {
    (c.x + (c.z << 2) + (c.y << 4)) as u32
}

#[inline]
pub fn child_coord(i: u32) -> IVec3 {
    IVec3::new((i & 3) as i32, (i >> 4 & 3) as i32, (i >> 2 & 3) as i32)
}

#[inline]
fn slot(mask: u64, i: u32) -> usize {
    (mask & ((1u64 << i) - 1)).count_ones() as usize
}

/// Compact 64-way child array.
#[derive(Clone, Debug)]
pub struct Sparse64<T> {
    pub mask: u64,
    pub items: Vec<T>,
}

impl<T> Default for Sparse64<T> {
    fn default() -> Self {
        Self {
            mask: 0,
            items: Vec::new(),
        }
    }
}

impl<T> Sparse64<T> {
    pub fn get(&self, i: u32) -> Option<&T> {
        (self.mask >> i & 1 != 0).then(|| &self.items[slot(self.mask, i)])
    }

    pub fn get_mut(&mut self, i: u32) -> Option<&mut T> {
        if self.mask >> i & 1 == 0 {
            return None;
        }
        let s = slot(self.mask, i);
        Some(&mut self.items[s])
    }

    pub fn get_or_insert_with(&mut self, i: u32, f: impl FnOnce() -> T) -> &mut T {
        let s = slot(self.mask, i);
        if self.mask >> i & 1 == 0 {
            self.items.insert(s, f());
            self.mask |= 1 << i;
        }
        &mut self.items[s]
    }

    pub fn remove(&mut self, i: u32) -> Option<T> {
        if self.mask >> i & 1 == 0 {
            return None;
        }
        self.mask &= !(1 << i);
        Some(self.items.remove(slot(self.mask, i)))
    }

    pub fn is_empty(&self) -> bool {
        self.mask == 0
    }

    /// `(child index, item)` in ascending index order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        let mut m = self.mask;
        let mut k = 0;
        std::iter::from_fn(move || {
            if m == 0 {
                return None;
            }
            let i = m.trailing_zeros();
            m &= m - 1;
            k += 1;
            Some((i, &self.items[k - 1]))
        })
    }
}

pub type L1 = Sparse64<Cell>;
pub type L2 = Sparse64<L1>;

#[derive(Clone, Debug, Default)]
pub struct ChunkTree {
    pub root: Sparse64<L2>,
    bricks: Vec<Brick>,
    free: Vec<u32>,
    /// Slab indices whose contents changed since the last `take_dirty`.
    dirty_bricks: Vec<u32>,
    /// Set when any node or cell kind changed.
    structure_dirty: bool,
}

/// Brick cell coordinate in `0..64` split into its three child indices.
#[inline]
fn path(b: IVec3) -> (u32, u32, u32) {
    debug_assert!(b.cmpge(IVec3::ZERO).all() && b.cmplt(IVec3::splat(64)).all());
    (
        child_index(b >> 4),
        child_index((b >> 2) & 3),
        child_index(b & 3),
    )
}

impl ChunkTree {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_empty()
    }

    pub fn brick(&self, idx: u32) -> &Brick {
        &self.bricks[idx as usize]
    }

    pub fn brick_count(&self) -> usize {
        self.bricks.len() - self.free.len()
    }

    pub fn cell(&self, b: IVec3) -> Cell {
        let (a, m, c) = path(b);
        self.root
            .get(a)
            .and_then(|l2| l2.get(m))
            .and_then(|l1| l1.get(c))
            .copied()
            .unwrap_or(Cell::Empty)
    }

    fn alloc_brick(&mut self, brick: Brick) -> u32 {
        match self.free.pop() {
            Some(i) => {
                self.bricks[i as usize] = brick;
                i
            }
            None => {
                self.bricks.push(brick);
                self.bricks.len() as u32 - 1
            }
        }
    }

    fn release(&mut self, cell: Cell) {
        if let Cell::Brick(i) = cell {
            self.bricks[i as usize] = Brick::empty();
            self.free.push(i);
        }
    }

    /// Replaces a brick cell. Bricks passed in are normalised: an empty brick
    /// becomes `Empty`, a single-material brick becomes `Uniform`.
    pub fn set_cell(&mut self, b: IVec3, cell: Cell) {
        let old = self.cell(b);
        if old == cell {
            return;
        }
        let (a, m, c) = path(b);
        self.structure_dirty = true;
        match cell {
            Cell::Empty => {
                if let Some(l2) = self.root.get_mut(a) {
                    if let Some(l1) = l2.get_mut(m) {
                        l1.remove(c);
                        if l1.is_empty() {
                            l2.remove(m);
                        }
                    }
                    if l2.is_empty() {
                        self.root.remove(a);
                    }
                }
            }
            _ => {
                let l1 = self
                    .root
                    .get_or_insert_with(a, Sparse64::default)
                    .get_or_insert_with(m, Sparse64::default);
                *l1.get_or_insert_with(c, || Cell::Empty) = cell;
            }
        }
        self.release(old);
    }

    /// Stores a brick, collapsing it to `Empty` or `Uniform` when possible.
    pub fn set_brick(&mut self, b: IVec3, brick: Brick) {
        if brick.is_empty() {
            self.set_cell(b, Cell::Empty);
        } else if let Some(m) = brick.uniform() {
            self.set_cell(b, Cell::Uniform(m));
        } else if let Cell::Brick(i) = self.cell(b) {
            self.bricks[i as usize] = brick;
            self.dirty_bricks.push(i);
        } else {
            let i = self.alloc_brick(brick);
            self.set_cell(b, Cell::Brick(i));
            self.dirty_bricks.push(i);
        }
    }

    /// Voxel lookup, `v` in `0..512` chunk-local voxels.
    pub fn voxel(&self, v: IVec3) -> MaterialId {
        match self.cell(v >> 3) {
            Cell::Empty => MaterialId(0),
            Cell::Uniform(m) => m,
            Cell::Brick(i) => self.bricks[i as usize].get(v & 7),
        }
    }

    /// Sets a voxel; returns the previous material.
    pub fn set_voxel(&mut self, v: IVec3, mat: MaterialId) -> MaterialId {
        let b = v >> 3;
        let l = v & 7;
        match self.cell(b) {
            Cell::Empty if mat.is_air() => mat,
            Cell::Uniform(u) if u == mat => mat,
            Cell::Brick(i) => {
                let brick = &mut self.bricks[i as usize];
                let old = brick.set(l, mat);
                if old != mat {
                    if brick.is_empty() {
                        self.set_cell(b, Cell::Empty);
                    } else if let Some(u) = brick.uniform() {
                        self.set_cell(b, Cell::Uniform(u));
                    } else {
                        self.dirty_bricks.push(i);
                    }
                }
                old
            }
            other => {
                let (mut brick, old) = match other {
                    Cell::Uniform(u) => (Brick::filled(u), u),
                    _ => (Brick::empty(), MaterialId(0)),
                };
                brick.set(l, mat);
                let i = self.alloc_brick(brick);
                self.set_cell(b, Cell::Brick(i));
                self.dirty_bricks.push(i);
                old
            }
        }
    }

    /// Mutable access to a brick cell as a full brick, materialising uniform
    /// or empty cells, then renormalising after `f` runs.
    pub fn edit_brick<R>(&mut self, b: IVec3, f: impl FnOnce(&mut Brick) -> R) -> R {
        match self.cell(b) {
            Cell::Brick(i) => {
                let brick = &mut self.bricks[i as usize];
                let r = f(brick);
                brick.compact_state();
                if brick.is_empty() {
                    self.set_cell(b, Cell::Empty);
                } else if let Some(u) = brick.uniform() {
                    self.set_cell(b, Cell::Uniform(u));
                } else {
                    self.dirty_bricks.push(i);
                }
                r
            }
            other => {
                let mut brick = match other {
                    Cell::Uniform(u) => Brick::filled(u),
                    _ => Brick::empty(),
                };
                let r = f(&mut brick);
                brick.compact_state();
                self.set_brick(b, brick);
                r
            }
        }
    }

    /// All non-empty brick cells as `(brick coordinate, cell)`.
    pub fn cells(&self) -> impl Iterator<Item = (IVec3, Cell)> + '_ {
        self.root.iter().flat_map(|(a, l2)| {
            l2.iter().flat_map(move |(m, l1)| {
                l1.iter().map(move |(c, &cell)| {
                    (
                        (child_coord(a) << 4) + (child_coord(m) << 2) + child_coord(c),
                        cell,
                    )
                })
            })
        })
    }

    /// Takes the dirty state: (structure changed, changed brick slab indices).
    pub fn take_dirty(&mut self) -> (bool, Vec<u32>) {
        let mut bricks = std::mem::take(&mut self.dirty_bricks);
        bricks.sort_unstable();
        bricks.dedup();
        // Freed slots hold an empty brick; their removal is a structure change.
        bricks.retain(|&i| !self.bricks[i as usize].is_empty());
        (std::mem::replace(&mut self.structure_dirty, false), bricks)
    }

    pub fn memory_bytes(&self) -> usize {
        let mut n = std::mem::size_of::<Self>();
        n += self.root.items.capacity() * std::mem::size_of::<L2>();
        for (_, l2) in self.root.iter() {
            n += l2.items.capacity() * std::mem::size_of::<L1>();
            for (_, l1) in l2.iter() {
                n += l1.items.capacity() * std::mem::size_of::<Cell>();
            }
        }
        n + self.bricks.iter().map(Brick::memory_bytes).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::ids;

    #[test]
    fn sparse64_keeps_items_in_index_order() {
        let mut s: Sparse64<u32> = Sparse64::default();
        for i in [40, 3, 63, 0, 17] {
            *s.get_or_insert_with(i, || 0) = i * 10;
        }
        let order: Vec<u32> = s
            .iter()
            .map(|(i, v)| {
                assert_eq!(*v, i * 10);
                i
            })
            .collect();
        assert_eq!(order, vec![0, 3, 17, 40, 63]);
        assert_eq!(s.remove(17), Some(170));
        assert_eq!(s.get(40), Some(&400));
        assert_eq!(s.get(17), None);
    }

    #[test]
    fn voxel_edits_materialise_and_collapse_bricks() {
        let mut t = ChunkTree::new();
        let v = IVec3::new(300, 20, 511);
        t.set_voxel(v, ids::GRANITE);
        assert_eq!(t.voxel(v), ids::GRANITE);
        assert_eq!(t.brick_count(), 1);
        t.set_voxel(v, ids::AIR);
        assert!(t.is_empty(), "clearing the only voxel prunes every level");
        assert_eq!(t.brick_count(), 0);

        let b = IVec3::new(10, 2, 60);
        t.set_cell(b, Cell::Uniform(ids::BASALT));
        let inside = (b << 3) + IVec3::new(1, 2, 3);
        assert_eq!(t.voxel(inside), ids::BASALT);
        t.set_voxel(inside, ids::AIR);
        assert!(matches!(t.cell(b), Cell::Brick(_)));
        t.set_voxel(inside, ids::BASALT);
        assert_eq!(t.cell(b), Cell::Uniform(ids::BASALT));
        assert_eq!(t.brick_count(), 0);
    }

    #[test]
    fn cells_iterates_coordinates_back() {
        let mut t = ChunkTree::new();
        let coords = [
            IVec3::new(0, 0, 0),
            IVec3::new(63, 63, 63),
            IVec3::new(5, 40, 22),
        ];
        for c in coords {
            t.set_cell(c, Cell::Uniform(ids::DIRT));
        }
        let mut seen: Vec<IVec3> = t.cells().map(|(c, _)| c).collect();
        seen.sort_by_key(|c| (c.x, c.y, c.z));
        let mut want = coords.to_vec();
        want.sort_by_key(|c| (c.x, c.y, c.z));
        assert_eq!(seen, want);
    }

    #[test]
    fn dirty_tracking_reports_edits_once() {
        let mut t = ChunkTree::new();
        t.set_voxel(IVec3::new(1, 1, 1), ids::SAND);
        t.set_voxel(IVec3::new(2, 1, 1), ids::SAND);
        let (structure, bricks) = t.take_dirty();
        assert!(structure);
        assert_eq!(bricks.len(), 1);
        let (structure, bricks) = t.take_dirty();
        assert!(!structure && bricks.is_empty());
    }
}
