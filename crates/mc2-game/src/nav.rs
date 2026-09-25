//! Finding a way on foot across the voxels.
//!
//! The ground is a grid of half-metre columns. A column is standable where
//! it has a solid top with head room over it; a walker may move to a
//! neighbouring column if that column's top is within a step up or a
//! drop down of its own and nothing stands in the way between them (a
//! house wall is thinner than a column). A* over those columns, eight
//! ways, never cutting a corner, gives a path of column centres at their
//! ground height.

use crate::collide::SolidField;
use glam::{DVec3, IVec2, IVec3};
use mc2_core::{FxHashMap, FxHashSet};
use mc2_voxel::world::VoxelWorld;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Column width, metres.
pub const CELL: f64 = 0.5;
/// Highest step up and furthest step down between columns, metres.
pub const STEP: f64 = 0.55;
pub const DROP: f64 = 1.1;
/// Room a walker needs above the ground, metres.
pub const HEAD: f64 = 1.9;
/// Columns a search may open before it gives up.
const BUDGET: usize = 2500;

/// The ground a walker would stand on in the column at (x, z) coming from
/// height `y`: the highest solid top with head room, from a step above `y`
/// down to a drop below it, metres.
pub fn stand(world: &VoxelWorld, x: f64, z: f64, y: f64) -> Option<f64> {
    let vx = (x * 16.0).floor() as i32;
    let vz = (z * 16.0).floor() as i32;
    let top = ((y + STEP) * 16.0).floor() as i32;
    let bottom = ((y - DROP) * 16.0).floor() as i32;
    let head = (HEAD * 16.0).ceil() as i32;
    let solid = |vy: i32| world.solid(IVec3::new(vx, vy, vz));
    let mut vy = top;
    while vy >= bottom {
        if solid(vy) && !solid(vy + 1) && (vy + 2..=vy + head).all(|h| !solid(h)) {
            return Some(f64::from(vy + 1) / 16.0);
        }
        vy -= 1;
    }
    None
}

/// Whether a walker can pass from `a` to `b` (feet, a column apart)
/// without walking into anything: three points along the way, each at
/// knee, waist and head height over the higher of the two.
fn clear(world: &VoxelWorld, a: DVec3, b: DVec3) -> bool {
    let base = a.y.max(b.y);
    [0.25, 0.5, 0.75].iter().all(|&t| {
        let p = a.lerp(b, t);
        [0.3, 0.9, 1.5].iter().all(|&h| {
            let v = (DVec3::new(p.x, base + h, p.z) * 16.0).floor().as_ivec3();
            !world.solid(v)
        })
    })
}

fn cell_of(p: DVec3) -> IVec2 {
    IVec2::new((p.x / CELL).floor() as i32, (p.z / CELL).floor() as i32)
}

fn centre(c: IVec2) -> (f64, f64) {
    ((f64::from(c.x) + 0.5) * CELL, (f64::from(c.y) + 0.5) * CELL)
}

#[derive(Debug, PartialEq)]
struct Open {
    f: f64,
    cell: IVec2,
}

impl Eq for Open {}

impl Ord for Open {
    fn cmp(&self, other: &Self) -> Ordering {
        // A max-heap on the negated estimate: the cheapest comes out first.
        other.f.total_cmp(&self.f)
    }
}

impl PartialOrd for Open {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// How a search is getting on.
#[derive(Debug)]
pub enum Progress {
    Found(Vec<DVec3>),
    NoWay,
    Searching,
}

/// An A* search for a way on foot, run a few columns at a time so a long
/// one never holds up a frame.
#[derive(Debug)]
pub struct Search {
    start: IVec2,
    goal: IVec2,
    to: DVec3,
    /// Per column: its ground, cost so far, and where it was reached from.
    seen: FxHashMap<IVec2, (f64, f64, IVec2)>,
    open: BinaryHeap<Open>,
    done: FxHashSet<IVec2>,
    /// A column is reached from several neighbours at much the same
    /// height: its ground is read once per height.
    ground: FxHashMap<(IVec2, i32), Option<f64>>,
}

impl Search {
    /// A search from `from` (feet) to `to`, or None if there is nowhere to
    /// stand at `to`, so no way there.
    pub fn new(world: &VoxelWorld, from: DVec3, to: DVec3) -> Option<Search> {
        let start = cell_of(from);
        let goal = cell_of(to);
        let (gx, gz) = centre(goal);
        stand(world, gx, gz, to.y)?;
        let (sx, sz) = centre(start);
        let start_y = stand(world, sx, sz, from.y).unwrap_or(from.y);
        let mut search = Search {
            start,
            goal,
            to,
            seen: FxHashMap::default(),
            open: BinaryHeap::new(),
            done: FxHashSet::default(),
            ground: FxHashMap::default(),
        };
        search.seen.insert(start, (start_y, 0.0, start));
        let f = search.estimate(start);
        search.open.push(Open { f, cell: start });
        Some(search)
    }

    fn estimate(&self, c: IVec2) -> f64 {
        let (x, z) = centre(c);
        (x - self.to.x).hypot(z - self.to.z)
    }

    /// Opens up to `columns` more columns; returns how it is getting on and
    /// how many it opened.
    pub fn advance(&mut self, world: &VoxelWorld, columns: usize) -> (Progress, usize) {
        mc2_core::scope!("nav.search");
        let mut opened = 0;
        while opened < columns {
            let Some(Open { cell, .. }) = self.open.pop() else {
                return (Progress::NoWay, opened);
            };
            if !self.done.insert(cell) {
                continue;
            }
            opened += 1;
            if cell == self.goal {
                return (Progress::Found(self.path_to(cell)), opened);
            }
            if self.done.len() > BUDGET {
                return (Progress::NoWay, opened);
            }
            self.expand(world, cell);
        }
        (Progress::Searching, opened)
    }

    fn path_to(&self, cell: IVec2) -> Vec<DVec3> {
        let mut path = Vec::new();
        let mut c = cell;
        loop {
            let (y, _, from) = self.seen[&c];
            let (x, z) = centre(c);
            path.push(DVec3::new(x, y, z));
            if c == self.start {
                break;
            }
            c = from;
        }
        path.reverse();
        path
    }

    fn expand(&mut self, world: &VoxelWorld, cell: IVec2) {
        let (y, g, _) = self.seen[&cell];
        let (cx, cz) = centre(cell);
        let here = DVec3::new(cx, y, cz);
        let ground = &mut self.ground;
        let mut step = |d: IVec2| -> Option<f64> {
            let (x, z) = centre(cell + d);
            let y2 = *ground
                .entry((cell + d, (y * 16.0).round() as i32))
                .or_insert_with(|| stand(world, x, z, y));
            let y2 = y2?;
            clear(world, here, DVec3::new(x, y2, z)).then_some(y2)
        };
        let straight = [IVec2::X, IVec2::NEG_X, IVec2::Y, IVec2::NEG_Y];
        let mut ok = [None; 4];
        for (i, d) in straight.iter().enumerate() {
            ok[i] = step(*d);
        }
        let mut next: Vec<(IVec2, f64, f64)> = straight
            .iter()
            .zip(ok)
            .filter_map(|(d, y2)| y2.map(|y2| (*d, y2, CELL)))
            .collect();
        // Diagonals only where both sides are open: no cutting corners.
        for (a, b) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
            if ok[a].is_some() && ok[b].is_some() {
                let d = straight[a] + straight[b];
                if let Some(y2) = step(d) {
                    next.push((d, y2, CELL * std::f64::consts::SQRT_2));
                }
            }
        }
        for (d, y2, len) in next {
            let n = cell + d;
            // Climbing costs a little more than walking.
            let cost = g + len + (y2 - y).max(0.0) * 2.0;
            if self.seen.get(&n).is_none_or(|&(_, old, _)| cost < old) {
                self.seen.insert(n, (y2, cost, cell));
                let f = cost + self.estimate(n);
                self.open.push(Open { f, cell: n });
            }
        }
    }
}

/// A walkable path from `from` (feet) to near `to`, as column centres at
/// their ground heights ending on `to`'s column, or None if there is none
/// within the search budget: the whole search at once.
pub fn find(world: &VoxelWorld, from: DVec3, to: DVec3) -> Option<Vec<DVec3>> {
    match Search::new(world, from, to)?.advance(world, usize::MAX).0 {
        Progress::Found(path) => Some(path),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;

    /// A 16 m floor with its top at 2 m.
    fn floor() -> VoxelWorld {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(255, 31, 255), ids::GRANITE);
        w
    }

    #[test]
    fn standing_needs_ground_and_head_room() {
        let mut w = floor();
        assert_eq!(stand(&w, 4.0, 4.0, 2.0), Some(2.0));
        // A low roof over the column leaves no room to stand.
        w.fill_box(IVec3::new(60, 50, 60), IVec3::new(70, 52, 70), ids::GRANITE);
        assert_eq!(stand(&w, 4.1, 4.1, 2.0), None);
        // A step of 0.5 m can be climbed; a wall of 1 m cannot.
        w.fill_box(
            IVec3::new(96, 32, 0),
            IVec3::new(111, 39, 255),
            ids::GRANITE,
        );
        assert_eq!(stand(&w, 6.5, 8.0, 2.0), Some(2.5));
        w.fill_box(
            IVec3::new(128, 32, 0),
            IVec3::new(143, 47, 255),
            ids::GRANITE,
        );
        assert_eq!(stand(&w, 8.5, 8.0, 2.0), None);
    }

    #[test]
    fn a_path_goes_round_a_wall() {
        let mut w = floor();
        // A wall from z = 2 m to z = 12 m at x = 8 m, 2 m high.
        w.fill_box(
            IVec3::new(128, 32, 32),
            IVec3::new(131, 63, 191),
            ids::GRANITE,
        );
        let from = DVec3::new(4.0, 2.0, 7.0);
        let to = DVec3::new(12.0, 2.0, 7.0);
        let path = find(&w, from, to).expect("a way round");
        // It ends where it was sent and never passes through the wall.
        let last = *path.last().unwrap();
        assert!((last.x - to.x).abs() < CELL && (last.z - to.z).abs() < CELL);
        for w in path.windows(2) {
            // No step crosses the wall's plane where the wall is.
            let crosses = (w[0].x < 8.0) != (w[1].x < 8.0);
            let z = (w[0].z + w[1].z) * 0.5;
            assert!(!(crosses && z > 2.0 && z < 12.0), "{} -> {}", w[0], w[1]);
            assert!((w[1].y - 2.0).abs() < 1e-9);
        }
        // Round the end of the wall, so longer than straight across.
        let len: f64 = path.windows(2).map(|w| w[0].distance(w[1])).sum();
        assert!(len > 10.0, "{len}");
    }

    #[test]
    fn a_thin_wall_is_not_walked_through() {
        let mut w = floor();
        // A house wall a quarter metre thick, 2 m high, right across.
        w.fill_box(
            IVec3::new(132, 32, 0),
            IVec3::new(135, 63, 255),
            ids::GRANITE,
        );
        assert!(find(&w, DVec3::new(6.0, 2.0, 8.0), DVec3::new(10.0, 2.0, 8.0)).is_none());
        // With a doorway in it, the way goes through the door.
        w.fill_box(IVec3::new(132, 32, 64), IVec3::new(135, 63, 79), ids::AIR);
        let path = find(&w, DVec3::new(6.0, 2.0, 8.0), DVec3::new(10.0, 2.0, 8.0)).expect("door");
        let through = path
            .windows(2)
            .find(|w| (w[0].x < 8.3) != (w[1].x < 8.3))
            .unwrap();
        let z = (through[0].z + through[1].z) * 0.5;
        assert!(z > 4.0 && z < 5.0, "crossed at z = {z}");
    }

    #[test]
    fn a_search_in_slices_finds_the_same_way() {
        let mut w = floor();
        w.fill_box(
            IVec3::new(128, 32, 32),
            IVec3::new(131, 63, 191),
            ids::GRANITE,
        );
        let (from, to) = (DVec3::new(4.0, 2.0, 7.0), DVec3::new(12.0, 2.0, 7.0));
        let whole = find(&w, from, to).unwrap();
        let mut search = Search::new(&w, from, to).unwrap();
        let mut slices = 0;
        let sliced = loop {
            slices += 1;
            match search.advance(&w, 10) {
                (Progress::Found(p), n) => {
                    assert!(n <= 10);
                    break p;
                }
                (Progress::Searching, n) => assert_eq!(n, 10),
                (Progress::NoWay, _) => panic!("lost the way"),
            }
        };
        assert!(slices > 3, "{slices}");
        assert_eq!(sliced, whole);
        // Nowhere to stand at the goal: no search at all.
        assert!(Search::new(&w, from, DVec3::new(20.0, 2.0, 7.0)).is_none());
    }

    #[test]
    fn no_path_into_a_closed_box() {
        let mut w = floor();
        // A walled pen 3 m high round (8, 8).
        w.fill_box(
            IVec3::new(112, 32, 112),
            IVec3::new(143, 79, 143),
            ids::GRANITE,
        );
        w.fill_box(IVec3::new(116, 32, 116), IVec3::new(139, 79, 139), ids::AIR);
        assert!(find(&w, DVec3::new(3.0, 2.0, 3.0), DVec3::new(8.0, 2.0, 8.0)).is_none());
    }
}
