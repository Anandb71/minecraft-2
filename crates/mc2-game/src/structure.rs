//! Structural integrity in the game.
//!
//! Every edit marks its blocks dirty. A region grows from the dirty blocks:
//! untouched terrain within `REGION_RADIUS` blocks, and every placed block
//! connected to it however far it reaches. Terrain outside that box
//! anchors the region, except above it, where the box is open. Gathering
//! reads the world a budgeted slice per frame (block summaries are cached
//! until edited), the solve runs on a worker thread, and its verdict is
//! applied when it arrives: failed blocks crumble into half-metre pieces,
//! unsupported islands fall as 2 m pieces a few per frame, lowest first,
//! and islands open to the top are re-examined in a taller region.

use crate::physics::Physics;
use crate::{Streaming, Voxels};
use bevy_ecs::prelude::*;
use glam::{IVec3, Vec3};
use mc2_core::{FxHashMap, FxHashSet};
use mc2_structure::node::FACES;
use mc2_structure::{NodeInfo, Outcome, Params, RegionNode, block_info, cut, solve};
use mc2_voxel::coords::{BlockPos, ChunkPos, VOXELS_PER_BLOCK};
use mc2_voxel::world::VoxelWorld;
use mc2_worldgen::stream::ChunkStreamer;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::time::Instant;

/// Natural terrain examined around an edit, blocks.
pub const REGION_RADIUS: i32 = 8;
/// Regions bigger than this are not solved (a mountain is not a building).
pub const MAX_REGION_NODES: usize = 60_000;
/// How much taller a region may grow for islands open to its top, blocks.
pub const MAX_OPEN_GROWTH: i32 = 64;
/// Islands bigger than this stay put.
pub const MAX_ISLAND_BLOCKS: usize = 20_000;
/// Per-frame time for reading the world into a region, milliseconds.
pub const GATHER_BUDGET_MS: f32 = 1.0;
/// Per-frame time for cutting falling pieces out of the world.
pub const CUT_BUDGET_MS: f32 = 1.5;
/// Failed blocks crumble into pieces this size, voxels.
pub const CRUMB_VOXELS: i32 = 8;
/// Falling islands break into pieces this size, voxels.
pub const PIECE_VOXELS: i32 = 32;

#[derive(Clone, Copy, Debug, Default)]
pub struct StructureStats {
    pub regions: u64,
    pub last_nodes: usize,
    pub last_solve_ms: f32,
    pub last_gather_ms: f32,
    pub worst_ratio: f32,
    pub failures: u64,
    pub islands: u64,
    pub pieces: u64,
    pub dirty: usize,
    pub falling: usize,
    pub skipped_regions: u64,
}

struct Gather {
    seeds: Vec<BlockPos>,
    /// Natural terrain box, blocks, inclusive.
    lo: IVec3,
    hi: IVec3,
    frontier: Vec<BlockPos>,
    visited: FxHashSet<IVec3>,
    nodes: Vec<RegionNode>,
    growth: i32,
    spent_ms: f32,
}

struct Job {
    nodes: Vec<RegionNode>,
    params: Params,
}

struct Verdict {
    outcome: Outcome,
    /// The region's nodes as gathered, to check nothing changed since.
    nodes: FxHashMap<IVec3, NodeInfo>,
    seeds: Vec<BlockPos>,
    growth: i32,
    solve_ms: f32,
}

struct Solver {
    jobs: Sender<(Job, Verdict)>,
    verdicts: Mutex<Receiver<Verdict>>,
    busy: bool,
}

impl Solver {
    fn start() -> Self {
        let (job_tx, job_rx) = channel::<(Job, Verdict)>();
        let (out_tx, out_rx) = channel::<Verdict>();
        std::thread::Builder::new()
            .name("structure".into())
            .spawn(move || {
                while let Ok((job, mut verdict)) = job_rx.recv() {
                    let start = Instant::now();
                    verdict.outcome = solve(&job.nodes, &job.params);
                    verdict.solve_ms = start.elapsed().as_secs_f32() * 1000.0;
                    if out_tx.send(verdict).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn structure thread");
        Self {
            jobs: job_tx,
            verdicts: Mutex::new(out_rx),
            busy: false,
        }
    }
}

#[derive(Resource)]
pub struct Structure {
    /// Blocks holding placed material.
    pub built: FxHashSet<IVec3>,
    cache: FxHashMap<IVec3, NodeInfo>,
    dirty: FxHashSet<IVec3>,
    gather: Option<Gather>,
    solver: Solver,
    /// Groups of island blocks waiting to be cut loose, one 2 m cell each.
    falling: VecDeque<Vec<BlockPos>>,
    /// Regions to re-examine taller, because something in them may hang
    /// from what was above the region.
    pending_growth: Vec<(Vec<BlockPos>, i32)>,
    pub params: Params,
    pub stats: StructureStats,
    /// Off in tests of other systems that do not want terrain to move.
    pub enabled: bool,
}

impl Default for Structure {
    fn default() -> Self {
        Self {
            built: FxHashSet::default(),
            cache: FxHashMap::default(),
            dirty: FxHashSet::default(),
            gather: None,
            solver: Solver::start(),
            falling: VecDeque::new(),
            pending_growth: Vec::new(),
            params: Params::default(),
            stats: StructureStats::default(),
            enabled: true,
        }
    }
}

fn blocks_in(lo: IVec3, hi: IVec3) -> impl Iterator<Item = IVec3> {
    let (a, b) = (
        lo.div_euclid(IVec3::splat(VOXELS_PER_BLOCK)),
        hi.div_euclid(IVec3::splat(VOXELS_PER_BLOCK)),
    );
    (a.z..=b.z).flat_map(move |z| {
        (a.y..=b.y).flat_map(move |y| (a.x..=b.x).map(move |x| IVec3::new(x, y, z)))
    })
}

impl Structure {
    /// Voxels in `lo..=hi` changed.
    pub fn edited(&mut self, lo: IVec3, hi: IVec3) {
        for b in blocks_in(lo, hi) {
            self.cache.remove(&b);
            self.dirty.insert(b);
        }
    }

    /// Voxels in `lo..=hi` now hold placed material.
    pub fn mark_built(&mut self, lo: IVec3, hi: IVec3) {
        for b in blocks_in(lo, hi) {
            self.built.insert(b);
        }
    }

    fn info(&mut self, world: &VoxelWorld, p: IVec3) -> NodeInfo {
        *self
            .cache
            .entry(p)
            .or_insert_with(|| block_info(world, BlockPos(p)))
    }

    fn start_gather(&mut self, seeds: Vec<BlockPos>, growth: i32) {
        let (mut lo, mut hi) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN));
        let mut frontier = Vec::new();
        for s in &seeds {
            lo = lo.min(s.0 - REGION_RADIUS);
            hi = hi.max(s.0 + REGION_RADIUS);
            frontier.push(*s);
            for d in FACES {
                frontier.push(BlockPos(s.0 + d));
            }
        }
        hi.y += growth;
        self.gather = Some(Gather {
            seeds,
            lo,
            hi,
            frontier,
            visited: FxHashSet::default(),
            nodes: Vec::new(),
            growth,
            spent_ms: 0.0,
        });
    }

    /// Advances gathering for up to the frame budget. Returns true when the
    /// region is complete.
    fn gather_step(&mut self, world: &VoxelWorld, streamer: Option<&ChunkStreamer>) -> bool {
        let start = Instant::now();
        let Some(mut g) = self.gather.take() else {
            return false;
        };
        let mut steps = 0u32;
        while let Some(p) = g.frontier.pop() {
            if !g.visited.insert(p.0) {
                continue;
            }
            steps += 1;
            if steps.is_multiple_of(8) && start.elapsed().as_secs_f32() * 1000.0 > GATHER_BUDGET_MS
            {
                g.frontier.push(p);
                g.visited.remove(&p.0);
                break;
            }
            // Terrain that is not loaded is assumed to hold.
            let loaded = streamer.is_none_or(|s| s.lod_of(ChunkPos::of_block(p)).is_some());
            let info = self.info(world, p.0);
            let natural = !self.built.contains(&p.0);
            let inside = p.0.cmpge(g.lo).all() && p.0.cmple(g.hi).all();
            if !loaded {
                g.nodes.push(RegionNode {
                    pos: p,
                    info: NodeInfo {
                        solid: 4096,
                        faces: [256; 6],
                        ..info
                    },
                    natural: true,
                    anchor: true,
                    open: false,
                });
                continue;
            }
            if !info.is_node() {
                continue;
            }
            let mut node = RegionNode {
                pos: p,
                info,
                natural,
                anchor: false,
                open: false,
            };
            let expand = if natural && !inside {
                if p.0.y > g.hi.y {
                    node.open = true;
                } else {
                    node.anchor = true;
                }
                false
            } else {
                true
            };
            g.nodes.push(node);
            if g.nodes.len() > MAX_REGION_NODES {
                log::warn!(
                    "structural region around {:?} exceeds {MAX_REGION_NODES} nodes; skipped",
                    g.seeds.first()
                );
                self.stats.skipped_regions += 1;
                return false;
            }
            if expand {
                for d in FACES {
                    let q = p.0 + d;
                    if !g.visited.contains(&q) {
                        g.frontier.push(BlockPos(q));
                    }
                }
            }
        }
        g.spent_ms += start.elapsed().as_secs_f32() * 1000.0;
        let done = g.frontier.is_empty();
        self.gather = Some(g);
        done
    }

    fn send(&mut self) {
        let Some(g) = self.gather.take() else {
            return;
        };
        self.stats.last_gather_ms = g.spent_ms;
        let verdict = Verdict {
            outcome: Outcome::default(),
            nodes: g.nodes.iter().map(|n| (n.pos.0, n.info)).collect(),
            seeds: g.seeds,
            growth: g.growth,
            solve_ms: 0.0,
        };
        let job = Job {
            nodes: g.nodes,
            params: self.params,
        };
        self.solver
            .jobs
            .send((job, verdict))
            .expect("structure thread gone");
        self.solver.busy = true;
        self.stats.regions += 1;
    }

    /// Applies a verdict whose blocks still look as they did.
    fn apply(&mut self, v: Verdict, world: &mut VoxelWorld, physics: &mut Physics) {
        self.stats.last_nodes = v.outcome.nodes;
        self.stats.last_solve_ms = v.solve_ms;
        self.stats.worst_ratio = v.outcome.worst;
        for (block, _) in &v.outcome.failed {
            if !self.unchanged(&v.nodes, block.0) {
                continue;
            }
            let o = block.origin();
            let crumbs = cut(world, &[*block], CRUMB_VOXELS);
            for mut d in crumbs {
                // Crumbs part slightly as they give way.
                let away = (d.pos - (o.as_dvec3() + 8.0) / 16.0).as_vec3();
                d.vel = away.normalize_or_zero() * 0.3 + Vec3::NEG_Y * 0.2;
                physics.host.spawn_debris(&d);
                self.stats.pieces += 1;
            }
            physics.world_edited(o, o + 15);
            self.stats.failures += 1;
        }
        for island in v.outcome.islands {
            if island.len() > MAX_ISLAND_BLOCKS {
                log::warn!("island of {} blocks left standing", island.len());
                continue;
            }
            if !island.iter().all(|b| self.unchanged(&v.nodes, b.0)) {
                continue;
            }
            let mut groups: FxHashMap<IVec3, Vec<BlockPos>> = FxHashMap::default();
            for b in island {
                let key = (b.0 * VOXELS_PER_BLOCK).div_euclid(IVec3::splat(PIECE_VOXELS));
                groups.entry(key).or_default().push(b);
            }
            let mut keys: Vec<IVec3> = groups.keys().copied().collect();
            keys.sort_unstable_by_key(|k| (k.y, k.x, k.z));
            for k in keys {
                if let Some(g) = groups.remove(&k) {
                    self.falling.push_back(g);
                }
            }
            self.stats.islands += 1;
        }
        if v.outcome.open_islands > 0 && v.growth < MAX_OPEN_GROWTH {
            // Look higher before letting anything fall that may hang.
            self.pending_growth
                .push((v.seeds, v.growth + 2 * REGION_RADIUS));
        }
    }

    /// Whether a block still looks as it did when the region was gathered.
    /// Edits drop blocks from the cache, so a cached summary that still
    /// matches the region's means nothing has touched it since. A block
    /// that changed makes its region's verdict stale, so it is marked
    /// dirty and examined again.
    fn unchanged(&mut self, nodes: &FxHashMap<IVec3, NodeInfo>, p: IVec3) -> bool {
        let then = nodes.get(&p);
        if then.is_some() && self.cache.get(&p) == then {
            return true;
        }
        self.dirty.insert(p);
        false
    }

    /// Cuts queued falling pieces within the frame budget.
    fn drop_pieces(&mut self, world: &mut VoxelWorld, physics: &mut Physics) {
        let start = Instant::now();
        while start.elapsed().as_secs_f32() * 1000.0 < CUT_BUDGET_MS {
            let Some(group) = self.falling.pop_front() else {
                break;
            };
            for b in &group {
                self.cache.remove(&b.0);
            }
            for d in cut(world, &group, PIECE_VOXELS) {
                physics.host.spawn_debris(&d);
                self.stats.pieces += 1;
            }
            let (lo, hi) = group.iter().fold(
                (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN)),
                |(lo, hi), b| (lo.min(b.origin()), hi.max(b.origin() + 15)),
            );
            physics.world_edited(lo, hi);
        }
    }
}

/// One frame of structural work: hear about edits, take a verdict, drop
/// what is falling, and gather or send the next region.
pub fn update_structure(
    mut voxels: ResMut<Voxels>,
    streaming: Res<Streaming>,
    mut physics: ResMut<Physics>,
    mut structure: ResMut<Structure>,
    mut interaction: ResMut<crate::interact::Interaction>,
) {
    mc2_core::scope!("structure.update");
    let s = &mut *structure;
    for (lo, hi) in interaction.built.drain(..) {
        s.mark_built(lo, hi);
    }
    for (lo, hi) in std::mem::take(&mut physics.baked_out) {
        s.mark_built(lo, hi);
    }
    for (lo, hi) in std::mem::take(&mut physics.edited_out) {
        s.edited(lo, hi);
    }
    if !s.enabled {
        s.dirty.clear();
        return;
    }
    let world = &mut voxels.0;
    if s.solver.busy {
        let received = s
            .solver
            .verdicts
            .get_mut()
            .expect("verdict channel lock")
            .try_recv();
        match received {
            Ok(v) => {
                mc2_core::scope!("structure.apply");
                s.solver.busy = false;
                s.apply(v, world, &mut physics);
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => panic!("structure thread died"),
        }
    }
    {
        mc2_core::scope!("structure.cut");
        s.drop_pieces(world, &mut physics);
    }
    if s.gather.is_none() {
        if let Some((seeds, growth)) = s.pending_growth.pop() {
            s.start_gather(seeds, growth);
        } else if let Some(&first) = s.dirty.iter().next() {
            // One region at a time: the dirty blocks near the first one.
            let reach = 2 * REGION_RADIUS;
            let seeds: Vec<BlockPos> = s
                .dirty
                .iter()
                .filter(|p| (**p - first).abs().max_element() <= reach)
                .map(|p| BlockPos(*p))
                .collect();
            for b in &seeds {
                s.dirty.remove(&b.0);
            }
            s.start_gather(seeds, 0);
        }
    }
    if s.gather.is_some() && !s.solver.busy {
        mc2_core::scope!("structure.gather");
        if s.gather_step(world, streaming.0.as_ref()) {
            s.send();
        }
    }
    s.stats.dirty = s.dirty.len();
    s.stats.falling = s.falling.len();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Game;
    use mc2_voxel::material::ids;

    /// Bedrock under 3 m of granite, out to 40 m each way.
    fn ground(game: &mut Game) {
        let v = &mut game.world.resource_mut::<Voxels>().0;
        v.fill_box(IVec3::ZERO, IVec3::new(639, 15, 639), ids::BEDROCK);
        v.fill_box(
            IVec3::new(0, 16, 0),
            IVec3::new(639, 16 * 4 - 1, 639),
            ids::GRANITE,
        );
    }

    /// Fills blocks `lo..=hi` with placed material.
    fn build(game: &mut Game, lo: IVec3, hi: IVec3, m: mc2_voxel::material::MaterialId) {
        let (a, b) = (BlockPos(lo).origin(), BlockPos(hi).origin() + 15);
        game.world.resource_mut::<Voxels>().0.fill_box(a, b, m);
        let s = &mut *game.world.resource_mut::<Structure>();
        s.mark_built(a, b);
        s.edited(a, b);
    }

    /// Runs until `done`, or gives up after 20 simulated seconds.
    fn run_until(game: &mut Game, mut done: impl FnMut(&mut Game) -> bool) -> bool {
        for _ in 0..1200 {
            game.update(1.0 / 60.0);
            if done(game) {
                return true;
            }
        }
        false
    }

    fn bodies(game: &Game) -> usize {
        game.world.resource::<Physics>().host.body_count()
    }

    #[test]
    fn cutting_a_column_brings_the_tower_down() {
        let mut game = Game::new();
        ground(&mut game);
        // A 1 m column 10 m tall carrying a 3 x 3 cap.
        build(
            &mut game,
            IVec3::new(20, 4, 20),
            IVec3::new(20, 13, 20),
            ids::STONE_BRICK,
        );
        build(
            &mut game,
            IVec3::new(19, 14, 19),
            IVec3::new(21, 14, 21),
            ids::STONE_BRICK,
        );
        // It stands.
        for _ in 0..300 {
            game.update(1.0 / 60.0);
        }
        assert_eq!(bodies(&game), 0, "the tower fell on its own");
        let stats = game.world.resource::<Structure>().stats;
        assert!(stats.regions > 0, "no region was solved");

        // Cut the column at 6 m.
        {
            let cut = BlockPos(IVec3::new(20, 6, 20));
            let (lo, hi) = (cut.origin(), cut.origin() + 15);
            game.world
                .resource_mut::<Voxels>()
                .0
                .fill_box(lo, hi, ids::AIR);
            game.world.resource_mut::<Structure>().edited(lo, hi);
        }
        assert!(
            run_until(&mut game, |g| bodies(g) > 0),
            "nothing fell after the column was cut"
        );
        // Everything above the cut leaves the world, a few pieces a frame.
        let gone = |g: &mut Game, b: IVec3| {
            g.world
                .resource::<Voxels>()
                .0
                .voxel(BlockPos(b).origin() + 8)
                .is_air()
        };
        assert!(
            run_until(&mut game, |g| gone(g, IVec3::new(20, 9, 20))
                && gone(g, IVec3::new(20, 14, 20))),
            "the tower is still up there"
        );
        // The stump below the cut stays.
        assert!(!gone(&mut game, IVec3::new(20, 5, 20)));
    }

    #[test]
    fn a_long_cantilever_breaks_off_at_its_root() {
        let mut game = Game::new();
        ground(&mut game);
        build(
            &mut game,
            IVec3::new(30, 4, 30),
            IVec3::new(30, 9, 30),
            ids::STONE_BRICK,
        );
        // Eight metres of stone reaching out of the tower top.
        build(
            &mut game,
            IVec3::new(31, 9, 30),
            IVec3::new(38, 9, 30),
            ids::STONE_BRICK,
        );
        assert!(
            run_until(&mut game, |g| bodies(g) > 0),
            "the arm held: worst ratio {}",
            game.world.resource::<Structure>().stats.worst_ratio
        );
        let s = game.world.resource::<Structure>().stats;
        assert!(s.failures > 0 || s.islands > 0, "{s:?}");
        // The far end of the arm is no longer part of the world.
        assert!(
            run_until(&mut game, |g| {
                let v = &g.world.resource::<Voxels>().0;
                v.voxel(BlockPos(IVec3::new(38, 9, 30)).origin() + 8)
                    .is_air()
            }),
            "the arm is still standing"
        );
    }

    #[test]
    fn short_arms_and_plain_ground_are_left_alone() {
        let mut game = Game::new();
        ground(&mut game);
        build(
            &mut game,
            IVec3::new(30, 4, 30),
            IVec3::new(30, 9, 30),
            ids::STONE_BRICK,
        );
        build(
            &mut game,
            IVec3::new(31, 9, 30),
            IVec3::new(32, 9, 30),
            ids::STONE_BRICK,
        );
        for _ in 0..600 {
            game.update(1.0 / 60.0);
        }
        assert_eq!(bodies(&game), 0);
        let v = &game.world.resource::<Voxels>().0;
        assert!(
            !v.voxel(BlockPos(IVec3::new(32, 9, 30)).origin() + 8)
                .is_air()
        );
    }
}
