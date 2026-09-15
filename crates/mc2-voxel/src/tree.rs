//! Sparse 64-tree for one 32 m chunk.
//!
//! Three levels of 4^3 fan-out sit above the bricks: the chunk root (children
//! are 8 m L2 nodes), L2 (children are 2 m L1 nodes) and L1 (children are
//! 0.5 m brick cells). Every node stores a 64-bit child mask and a compact
//! child array: child `i` lives at `popcount(mask & ((1 << i) - 1))`.
//!
//! Solid regions collapse at every level. A brick cell is empty (absent from
//! the mask), uniform (one material reference, no brick allocated) or a
//! palette brick. An L1 or L2 node whose every child is the same uniform
//! material is itself stored as a single uniform node, so a chunk of solid
//! rock deep underground is 64 material references rather than a quarter of
//! a million cells.
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
    /// All 64 children set to clones of `item`.
    pub fn full(item: T) -> Self
    where
        T: Clone,
    {
        Self {
            mask: u64::MAX,
            items: vec![item; 64],
        }
    }

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

    pub fn is_full(&self) -> bool {
        self.mask == u64::MAX
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

/// 2 m node: brick cells, or one material throughout.
#[derive(Clone, Debug)]
pub enum L1 {
    Uniform(MaterialId),
    Cells(Sparse64<Cell>),
}

/// 8 m node: 2 m nodes, or one material throughout.
#[derive(Clone, Debug)]
pub enum L2 {
    Uniform(MaterialId),
    Nodes(Sparse64<L1>),
}

impl L1 {
    fn uniform(&self) -> Option<MaterialId> {
        match self {
            L1::Uniform(m) => Some(*m),
            L1::Cells(_) => None,
        }
    }
}

/// Material shared by every item of a full child array, if any.
fn common_uniform<T>(
    s: &Sparse64<T>,
    uniform: impl Fn(&T) -> Option<MaterialId>,
) -> Option<MaterialId> {
    if !s.is_full() {
        return None;
    }
    let first = uniform(&s.items[0])?;
    s.items
        .iter()
        .all(|it| uniform(it) == Some(first))
        .then_some(first)
}

#[derive(Clone, Debug)]
pub struct ChunkTree {
    /// Unique per constructed tree, so GPU residency can tell a regenerated
    /// chunk from an edited one (brick slab slots are not comparable).
    id: u64,
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

/// A region of one material stored as a single node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformNode {
    /// Minimum corner in brick cells.
    pub min_cell: IVec3,
    /// Edge length in brick cells: 4 for a 2 m node, 16 for an 8 m node.
    pub cells: i32,
    pub material: MaterialId,
}

impl Default for ChunkTree {
    fn default() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            root: Sparse64::default(),
            bricks: Vec::new(),
            free: Vec::new(),
            dirty_bricks: Vec::new(),
            structure_dirty: false,
        }
    }
}

impl ChunkTree {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn id(&self) -> u64 {
        self.id
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
        match self.root.get(a) {
            None => Cell::Empty,
            Some(L2::Uniform(u)) => Cell::Uniform(*u),
            Some(L2::Nodes(n)) => match n.get(m) {
                None => Cell::Empty,
                Some(L1::Uniform(u)) => Cell::Uniform(*u),
                Some(L1::Cells(cells)) => cells.get(c).copied().unwrap_or(Cell::Empty),
            },
        }
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

    fn release_l1(&mut self, l1: &L1) {
        if let L1::Cells(cells) = l1 {
            for &c in &cells.items {
                self.release(c);
            }
        }
    }

    fn release_l2(&mut self, l2: &L2) {
        if let L2::Nodes(nodes) = l2 {
            for n in &nodes.items {
                self.release_l1(n);
            }
        }
    }

    /// Replaces a brick cell. Uniform nodes on the path split as needed and
    /// the path re-collapses afterwards.
    pub fn set_cell(&mut self, b: IVec3, cell: Cell) {
        let old = self.cell(b);
        if old == cell {
            return;
        }
        let (a, m, c) = path(b);
        self.structure_dirty = true;

        let l2 = self
            .root
            .get_or_insert_with(a, || L2::Nodes(Sparse64::default()));
        if let L2::Uniform(u) = *l2 {
            *l2 = L2::Nodes(Sparse64::full(L1::Uniform(u)));
        }
        let L2::Nodes(nodes) = l2 else {
            unreachable!("split above");
        };
        let l1 = nodes.get_or_insert_with(m, || L1::Cells(Sparse64::default()));
        if let L1::Uniform(u) = *l1 {
            *l1 = L1::Cells(Sparse64::full(Cell::Uniform(u)));
        }
        let L1::Cells(cells) = l1 else {
            unreachable!("split above");
        };
        match cell {
            Cell::Empty => {
                cells.remove(c);
            }
            _ => *cells.get_or_insert_with(c, || Cell::Empty) = cell,
        }

        // Collapse upward.
        if cells.is_empty() {
            nodes.remove(m);
        } else if let Some(u) = common_uniform(cells, |c| match c {
            Cell::Uniform(u) => Some(*u),
            _ => None,
        }) {
            *l1 = L1::Uniform(u);
        }
        if nodes.is_empty() {
            self.root.remove(a);
        } else if let Some(u) = common_uniform(nodes, L1::uniform) {
            *l2 = L2::Uniform(u);
        }
        self.release(old);
    }

    /// Replaces a whole 2 m node (`level` 1, `node` in `0..16`) or 8 m node
    /// (`level` 2, `node` in `0..4`) with one material, or clears it.
    pub fn set_node(&mut self, level: u32, node: IVec3, material: MaterialId) {
        self.structure_dirty = true;
        match level {
            2 => {
                let a = child_index(node);
                if let Some(old) = self.root.remove(a) {
                    self.release_l2(&old);
                }
                if !material.is_air() {
                    *self.root.get_or_insert_with(a, || L2::Uniform(material)) =
                        L2::Uniform(material);
                }
            }
            1 => {
                let a = child_index(node >> 2);
                let m = child_index(node & 3);
                let mut removed = None;
                if let Some(l2) = self.root.get_mut(a) {
                    if let L2::Uniform(u) = *l2 {
                        if u == material {
                            return;
                        }
                        *l2 = L2::Nodes(Sparse64::full(L1::Uniform(u)));
                    }
                    if let L2::Nodes(nodes) = l2 {
                        removed = nodes.remove(m);
                    }
                }
                if let Some(old) = &removed {
                    self.release_l1(old);
                }
                if !material.is_air() {
                    let l2 = self
                        .root
                        .get_or_insert_with(a, || L2::Nodes(Sparse64::default()));
                    if let L2::Nodes(nodes) = l2 {
                        *nodes.get_or_insert_with(m, || L1::Uniform(material)) =
                            L1::Uniform(material);
                    }
                }
                let collapse = match self.root.get(a) {
                    Some(L2::Nodes(nodes)) if nodes.is_empty() => Some(None),
                    Some(L2::Nodes(nodes)) => common_uniform(nodes, L1::uniform).map(Some),
                    _ => None,
                };
                match collapse {
                    Some(None) => {
                        self.root.remove(a);
                    }
                    Some(Some(u)) => {
                        *self.root.get_mut(a).expect("present") = L2::Uniform(u);
                    }
                    None => {}
                }
            }
            _ => panic!("set_node supports levels 1 and 2, got {level}"),
        }
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

    /// Explicit brick cells as `(brick coordinate, cell)`. Regions stored as
    /// uniform nodes are reported by `uniform_nodes` instead.
    pub fn cells(&self) -> impl Iterator<Item = (IVec3, Cell)> + '_ {
        self.root.iter().flat_map(|(a, l2)| {
            let nodes = match l2 {
                L2::Nodes(n) => Some(n),
                L2::Uniform(_) => None,
            };
            nodes.into_iter().flat_map(move |nodes| {
                nodes.iter().flat_map(move |(m, l1)| {
                    let cells = match l1 {
                        L1::Cells(c) => Some(c),
                        L1::Uniform(_) => None,
                    };
                    cells.into_iter().flat_map(move |cells| {
                        cells.iter().map(move |(c, &cell)| {
                            (
                                (child_coord(a) << 4) + (child_coord(m) << 2) + child_coord(c),
                                cell,
                            )
                        })
                    })
                })
            })
        })
    }

    /// Every uniform 2 m and 8 m node.
    pub fn uniform_nodes(&self) -> Vec<UniformNode> {
        let mut out = Vec::new();
        for (a, l2) in self.root.iter() {
            match l2 {
                L2::Uniform(m) => out.push(UniformNode {
                    min_cell: child_coord(a) << 4,
                    cells: 16,
                    material: *m,
                }),
                L2::Nodes(nodes) => {
                    for (i, l1) in nodes.iter() {
                        if let L1::Uniform(m) = l1 {
                            out.push(UniformNode {
                                min_cell: (child_coord(a) << 4) + (child_coord(i) << 2),
                                cells: 4,
                                material: *m,
                            });
                        }
                    }
                }
            }
        }
        out
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
            if let L2::Nodes(nodes) = l2 {
                n += nodes.items.capacity() * std::mem::size_of::<L1>();
                for (_, l1) in nodes.iter() {
                    if let L1::Cells(c) = l1 {
                        n += c.items.capacity() * std::mem::size_of::<Cell>();
                    }
                }
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

    #[test]
    fn filling_a_node_cell_by_cell_collapses_it() {
        let mut t = ChunkTree::new();
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    t.set_cell(
                        IVec3::new(x, y, z) + IVec3::new(8, 0, 4),
                        Cell::Uniform(ids::GRANITE),
                    );
                }
            }
        }
        let nodes = t.uniform_nodes();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].cells, 4);
        assert_eq!(nodes[0].min_cell, IVec3::new(8, 0, 4));
        assert_eq!(t.cells().count(), 0);
        assert_eq!(t.voxel(IVec3::new(70, 3, 40)), ids::GRANITE);
    }

    #[test]
    fn uniform_nodes_split_on_edit_and_recollapse() {
        let mut t = ChunkTree::new();
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    t.set_node(2, IVec3::new(x, y, z), ids::BASALT);
                }
            }
        }
        assert_eq!(t.root.items.len(), 64);
        assert!(
            t.memory_bytes() < 4096,
            "solid chunk costs {} bytes",
            t.memory_bytes()
        );
        let v = IVec3::new(200, 100, 300);
        assert_eq!(t.set_voxel(v, ids::AIR), ids::BASALT);
        assert_eq!(t.voxel(v), ids::AIR);
        assert_eq!(t.voxel(v + IVec3::X), ids::BASALT);
        assert_eq!(t.brick_count(), 1);
        t.set_voxel(v, ids::BASALT);
        assert_eq!(t.brick_count(), 0);
        assert_eq!(
            t.uniform_nodes().iter().filter(|n| n.cells == 16).count(),
            64
        );
    }

    #[test]
    fn set_node_level_one_inside_uniform_parent() {
        let mut t = ChunkTree::new();
        t.set_node(2, IVec3::ZERO, ids::GRANITE);
        t.set_node(1, IVec3::new(1, 2, 3), ids::SAND);
        assert_eq!(t.voxel(IVec3::new(32, 64, 96)), ids::SAND);
        assert_eq!(t.voxel(IVec3::new(0, 0, 0)), ids::GRANITE);
        t.set_node(1, IVec3::new(1, 2, 3), ids::GRANITE);
        assert_eq!(t.uniform_nodes().len(), 1, "parent collapses back");
        t.set_node(1, IVec3::new(0, 0, 0), ids::AIR);
        assert_eq!(t.voxel(IVec3::new(5, 5, 5)), ids::AIR);
        assert_eq!(t.voxel(IVec3::new(40, 5, 5)), ids::GRANITE);
    }
}
