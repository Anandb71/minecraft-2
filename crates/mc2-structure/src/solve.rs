//! The load graph.
//!
//! 1. Distance to support: Dijkstra from the anchors, where resting on a
//!    node below costs 1, leaning on a node beside costs 1.5 and hanging
//!    from a node above costs 4, so load prefers to go down.
//! 2. Nodes the search never reaches are islands: nothing holds them up.
//! 3. Load and bending moment are gathered from the farthest node inward.
//!    Each node hands what it carries to its neighbours nearer to support,
//!    in proportion to shared face area (a node resting on something hands
//!    most of it down). A sideways hand-off adds the moment of the carried
//!    load about the 1 m lever. Moments are horizontal vectors, so the
//!    opposite sides of a symmetric roof cancel instead of adding.
//! 4. Every hand-off is checked against the weaker of the two materials:
//!    compression over the shared area, bending over the section modulus of
//!    that area, shear for sideways support, tension for hanging support.
//!    The node with the worst ratio above 1 fails first.

use crate::node::{FACES, NodeInfo, opposite};
use glam::{IVec3, Vec3};
use mc2_core::FxHashMap;
use mc2_voxel::coords::BlockPos;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const G: f32 = 9.81;
const COST_DOWN: f32 = 1.0;
const COST_SIDE: f32 = 1.5;
const COST_UP: f32 = 4.0;

#[derive(Clone, Copy, Debug)]
pub struct RegionNode {
    pub pos: BlockPos,
    pub info: NodeInfo,
    /// Untouched generated terrain: monolithic rock mass, never crushed.
    pub natural: bool,
    pub anchor: bool,
    /// On the open top of the region: what lies above was not examined.
    pub open: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Strength of a joint between placed blocks, as a fraction of the
    /// weaker material's: mortar and nails, not solid stone.
    pub joint: f32,
    /// Strength of natural rock mass relative to an intact sample: joints,
    /// bedding planes and fractures.
    pub rock_mass: f32,
    /// Failures reported per solve; the rest wait for the re-solve their
    /// removal triggers.
    pub max_failures: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            joint: 0.25,
            rock_mass: 0.3,
            max_failures: 4,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Outcome {
    /// Failing blocks, worst first, with their stress ratio.
    pub failed: Vec<(BlockPos, f32)>,
    /// Connected groups that reach no anchor.
    pub islands: Vec<Vec<BlockPos>>,
    /// Unsupported groups that touch the open top of the region: they may
    /// hang from what lies above, so they need a taller region.
    pub open_islands: usize,
    /// Highest stress ratio anywhere.
    pub worst: f32,
    pub nodes: usize,
}

#[derive(PartialEq)]
struct Visit(f32, usize);

impl Eq for Visit {}

impl Ord for Visit {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0).then(other.1.cmp(&self.1))
    }
}

impl PartialOrd for Visit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct Edge {
    to: usize,
    face: usize,
    /// Shared solid area, square metres.
    area: f32,
}

/// Stress over strength of one hand-off from node `a` (carrying `load` kg
/// and `moment` kg m) to its support `b` across `face`.
fn stress_ratio(
    a: &RegionNode,
    b: &RegionNode,
    face: usize,
    area: f32,
    load: f32,
    moment: Vec3,
    p: &Params,
) -> f32 {
    let (ma, mb) = (a.info.material.get(), b.info.material.get());
    let natural = a.natural && b.natural;
    let factor = if natural { p.rock_mass } else { p.joint };
    let tensile = ma.tensile.min(mb.tensile) * factor * 1e6;
    let compressive = ma.compressive.min(mb.compressive) * 1e6;
    let side = area.sqrt();
    // Section modulus of a square section with the shared area.
    let modulus = side * side * side / 6.0;
    let over = |stress: f32, strength: f32| {
        if stress <= 0.0 {
            0.0
        } else if strength <= 0.0 {
            f32::INFINITY
        } else {
            stress / strength
        }
    };
    let axial = load * G / area;
    let bending = moment.length() * G / modulus;
    match FACES[face].y {
        // Resting on b.
        -1 => {
            let crush = if natural {
                0.0
            } else {
                over(axial, compressive)
            };
            crush.max(over(bending - axial, tensile))
        }
        // Hanging from b.
        1 => over(axial + bending, tensile),
        // Leaning on b: bending plus shear across the joint.
        _ => over(bending, tensile).max(over(axial, tensile)),
    }
}

pub fn solve(nodes: &[RegionNode], p: &Params) -> Outcome {
    let n = nodes.len();
    let index: FxHashMap<IVec3, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, r)| (r.pos.0, i))
        .collect();
    let edges: Vec<Vec<Edge>> = nodes
        .iter()
        .map(|r| {
            FACES
                .iter()
                .enumerate()
                .filter_map(|(f, d)| {
                    let &to = index.get(&(r.pos.0 + *d))?;
                    let shared = r.info.faces[f].min(nodes[to].info.faces[opposite(f)]);
                    (shared > 0).then(|| Edge {
                        to,
                        face: f,
                        area: f32::from(shared) / 256.0,
                    })
                })
                .collect()
        })
        .collect();

    // 1. Distance to support.
    let mut dist = vec![f32::INFINITY; n];
    let mut heap = BinaryHeap::new();
    for (i, r) in nodes.iter().enumerate() {
        if r.anchor || r.info.is_bedrock() {
            dist[i] = 0.0;
            heap.push(Visit(0.0, i));
        }
    }
    while let Some(Visit(d, u)) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        for e in &edges[u] {
            // e points from u to v; v is supported by u across the opposite face.
            let cost = match FACES[e.face].y {
                1 => COST_DOWN,
                -1 => COST_UP,
                _ => COST_SIDE,
            };
            let nd = d + cost;
            if nd < dist[e.to] {
                dist[e.to] = nd;
                heap.push(Visit(nd, e.to));
            }
        }
    }

    // 2. Islands.
    let mut out = Outcome {
        nodes: n,
        ..Default::default()
    };
    let mut seen = vec![false; n];
    for start in 0..n {
        if seen[start] || dist[start].is_finite() {
            continue;
        }
        let mut group = Vec::new();
        let mut open = false;
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(u) = stack.pop() {
            group.push(nodes[u].pos);
            open |= nodes[u].open;
            for e in &edges[u] {
                if !seen[e.to] && !dist[e.to].is_finite() {
                    seen[e.to] = true;
                    stack.push(e.to);
                }
            }
        }
        if open {
            out.open_islands += 1;
        } else {
            out.islands.push(group);
        }
    }

    // 3 and 4. Loads, from the farthest node inward.
    let mut order: Vec<usize> = (0..n)
        .filter(|&i| dist[i].is_finite() && dist[i] > 0.0)
        .collect();
    order.sort_by(|&a, &b| dist[b].total_cmp(&dist[a]));
    let mut load: Vec<f32> = nodes.iter().map(|r| r.info.mass).collect();
    let mut moment = vec![Vec3::ZERO; n];
    let mut ratio = vec![0.0f32; n];
    for &i in &order {
        let supports: Vec<&Edge> = edges[i].iter().filter(|e| dist[e.to] < dist[i]).collect();
        let rests = supports.iter().any(|e| FACES[e.face].y == -1);
        let weight = |e: &Edge| {
            let w = if FACES[e.face].y == -1 || !rests {
                1.0
            } else {
                0.25
            };
            w * e.area
        };
        let total: f32 = supports.iter().map(|e| weight(e)).sum();
        if total <= 0.0 {
            continue;
        }
        for e in &supports {
            let f = weight(e) / total;
            let carried = f * load[i];
            // Moment of the carried weight about the 1 m lever to a
            // support beside this node: r x F, horizontal.
            let d = FACES[e.face].as_vec3();
            let lever = Vec3::new(d.z * carried, 0.0, -d.x * carried);
            let m = f * moment[i] + lever;
            let r = stress_ratio(&nodes[i], &nodes[e.to], e.face, e.area, carried, m, p);
            ratio[i] = ratio[i].max(r);
            load[e.to] += carried;
            moment[e.to] += m;
        }
    }
    out.worst = ratio.iter().copied().fold(0.0, f32::max);

    // Worst failures, spread apart so one break is not reported many times.
    let mut failing: Vec<usize> = (0..n).filter(|&i| ratio[i] > 1.0).collect();
    failing.sort_by(|&a, &b| ratio[b].total_cmp(&ratio[a]));
    for i in failing {
        if out.failed.len() >= p.max_failures {
            break;
        }
        let near = out
            .failed
            .iter()
            .any(|(q, _)| (q.0 - nodes[i].pos.0).abs().max_element() <= 2);
        if !near {
            out.failed.push((nodes[i].pos, ratio[i]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::{MaterialId, ids};

    fn full(m: MaterialId) -> NodeInfo {
        NodeInfo {
            solid: 4096,
            mass: m.get().density,
            material: m,
            faces: [256; 6],
        }
    }

    fn node(p: IVec3, m: MaterialId, natural: bool, anchor: bool) -> RegionNode {
        RegionNode {
            pos: BlockPos(p),
            info: full(m),
            natural,
            anchor,
            open: false,
        }
    }

    /// A 1 m column `height` blocks tall on an anchor of the same material
    /// (so the footing is no weaker than the tower), with a cantilever of
    /// `length` blocks from its top.
    fn tower_and_arm(m: MaterialId, height: i32, length: i32) -> Vec<RegionNode> {
        let mut v = vec![node(IVec3::ZERO, m, false, true)];
        for y in 1..=height {
            v.push(node(IVec3::new(0, y, 0), m, false, false));
        }
        for x in 1..=length {
            v.push(node(IVec3::new(x, height, 0), m, false, false));
        }
        v
    }

    #[test]
    fn short_arms_hold_and_long_arms_fail_at_the_root() {
        let p = Params::default();
        // Stone brick: tensile 2.5 MPa at a quarter strength. A 1 m square
        // arm breaks at its root once 3 w L^2 exceeds that: about 3 m.
        let short = solve(&tower_and_arm(ids::STONE_BRICK, 4, 2), &p);
        assert!(short.failed.is_empty(), "{short:?}");
        let long = solve(&tower_and_arm(ids::STONE_BRICK, 4, 4), &p);
        assert_eq!(long.failed[0].0, BlockPos(IVec3::new(1, 4, 0)), "{long:?}");
        // Planks hold 20 m and break at 30.
        let planks = solve(&tower_and_arm(ids::PLANKS, 4, 20), &p);
        assert!(planks.failed.is_empty(), "worst {}", planks.worst);
        let thirty = solve(&tower_and_arm(ids::PLANKS, 4, 30), &p);
        assert_eq!(thirty.failed[0].0, BlockPos(IVec3::new(1, 4, 0)));
    }

    #[test]
    fn a_symmetric_roof_does_not_snap_its_column() {
        // A 3 x 3 cap on a 1 m column. Each arm is a 1 m cantilever, but
        // opposite arms pull opposite ways, so the column below carries
        // their weight and no net bending.
        let mut v = vec![node(IVec3::ZERO, ids::STONE_BRICK, false, true)];
        for y in 1..=6 {
            v.push(node(IVec3::new(0, y, 0), ids::STONE_BRICK, false, false));
        }
        for x in -1..=1 {
            for z in -1..=1 {
                if x != 0 || z != 0 {
                    v.push(node(IVec3::new(x, 7, z), ids::STONE_BRICK, false, false));
                }
            }
        }
        v.push(node(IVec3::new(0, 7, 0), ids::STONE_BRICK, false, false));
        let out = solve(&v, &Params::default());
        assert!(out.failed.is_empty(), "{out:?}");
        assert!(out.worst < 1.0, "worst {}", out.worst);
        // Knock the cap off and reach the same distance to one side only:
        // now nothing balances the moment and the root gives way.
        v.retain(|r| r.pos.0.y != 7);
        for x in 1..=4 {
            v.push(node(IVec3::new(x, 6, 0), ids::STONE_BRICK, false, false));
        }
        let lopsided = solve(&v, &Params::default());
        assert_eq!(lopsided.failed[0].0, BlockPos(IVec3::new(1, 6, 0)));
    }

    #[test]
    fn a_cut_column_leaves_the_tower_as_an_island() {
        // A 2x2 tower 10 m tall, one column cut from 3 m up.
        let mut v = vec![];
        for x in 0..2 {
            for z in 0..2 {
                v.push(node(IVec3::new(x, 0, z), ids::GRANITE, true, true));
                for y in 1..10 {
                    if y < 3 && !(x == 0 && z == 0) {
                        // Only one leg reaches the ground.
                        continue;
                    }
                    v.push(node(IVec3::new(x, y, z), ids::STONE_BRICK, false, false));
                }
            }
        }
        let p = Params::default();
        let standing = solve(&v, &p);
        assert!(standing.islands.is_empty());
        // Cut the leg.
        v.retain(|r| r.pos.0 != IVec3::new(0, 2, 0));
        let cut = solve(&v, &p);
        assert_eq!(cut.islands.len(), 1);
        assert_eq!(cut.islands[0].len(), 4 * 7);
    }

    #[test]
    fn natural_ground_is_not_crushed_but_sand_cannot_overhang() {
        let p = Params::default();
        // 40 m of sand on an anchor: fine.
        let mut v = vec![node(IVec3::ZERO, ids::GRANITE, true, true)];
        for y in 1..=40 {
            v.push(node(IVec3::new(0, y, 0), ids::SAND, true, false));
        }
        assert!(solve(&v, &p).failed.is_empty());
        // One sand block reaching out from it: fails at once.
        v.push(node(IVec3::new(1, 20, 0), ids::SAND, true, false));
        let out = solve(&v, &p);
        assert_eq!(out.failed[0].0, BlockPos(IVec3::new(1, 20, 0)));
        // The same overhang in granite holds.
        for r in &mut v {
            if !r.anchor {
                r.info = full(ids::GRANITE);
            }
        }
        assert!(solve(&v, &p).failed.is_empty());
    }

    #[test]
    fn a_placed_brick_column_crushes_under_enough_weight() {
        let p = Params::default();
        // 80 MPa / (2300 kg m^-3 g) is 3.5 km: a real column never crushes.
        let tall = solve(&tower_and_arm(ids::STONE_BRICK, 200, 0), &p);
        assert!(tall.failed.is_empty(), "worst {}", tall.worst);
        // Snow compressive 0.05 MPa: 17 m.
        let snow = solve(&tower_and_arm(ids::SNOW, 30, 0), &p);
        assert!(!snow.failed.is_empty());
        let low = solve(&tower_and_arm(ids::SNOW, 10, 0), &p);
        assert!(low.failed.is_empty(), "worst {}", low.worst);
    }

    #[test]
    fn islands_open_to_the_top_are_left_alone() {
        let mut v = vec![node(IVec3::ZERO, ids::GRANITE, true, true)];
        v.push(node(IVec3::new(0, 5, 0), ids::GRANITE, true, false));
        let mut top = node(IVec3::new(0, 6, 0), ids::GRANITE, true, false);
        top.open = true;
        v.push(top);
        let out = solve(&v, &Params::default());
        assert!(out.islands.is_empty());
        assert_eq!(out.open_islands, 1);
    }
}
