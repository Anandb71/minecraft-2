//! Chunk streaming: which chunks exist around the camera, at what detail.
//!
//! Every frame the streamer derives a desired level of detail for each chunk
//! column within the outermost radius, nearest first. Generation runs on the
//! rayon pool and results come back over a channel; the main thread inserts
//! a bounded number per frame so a burst of finished chunks never becomes a
//! frame spike. Chunks refine as the camera approaches and coarsen or unload
//! as it leaves, with hysteresis so a camera sitting on a boundary does not
//! regenerate forever. Chunks the player has changed are pinned at full
//! detail and never regenerated.

use crate::chunkgen::{ChunkGenerator, Lod};
use glam::{DVec3, IVec3};
use mc2_core::{FxHashMap, FxHashSet};
use mc2_voxel::coords::{self, CHUNK_VOXELS, ChunkPos};
use mc2_voxel::tree::ChunkTree;
use mc2_voxel::world::VoxelWorld;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct StreamConfig {
    /// Horizontal radius, metres, inside which each level applies.
    pub full_m: f32,
    pub cell_m: f32,
    pub node2_m: f32,
    pub node8_m: f32,
    /// Extra distance before a chunk coarsens or unloads.
    pub hysteresis_m: f32,
    pub max_inflight: usize,
    /// Main-thread time spent inserting finished chunks per frame.
    pub insert_budget_ms: f32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            full_m: 64.0,
            cell_m: 192.0,
            node2_m: 512.0,
            node8_m: 1536.0,
            hysteresis_m: 48.0,
            max_inflight: 64,
            insert_budget_ms: 3.0,
        }
    }
}

impl StreamConfig {
    fn lod_for(&self, d: f32, slack: f32) -> Option<Lod> {
        if d <= self.full_m + slack {
            Some(Lod::Full)
        } else if d <= self.cell_m + slack {
            Some(Lod::Cell)
        } else if d <= self.node2_m + slack {
            Some(Lod::Node2)
        } else if d <= self.node8_m + slack {
            Some(Lod::Node8)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StreamStats {
    pub loaded: usize,
    pub inflight: usize,
    pub queued: usize,
    pub inserted_last_frame: usize,
    pub generated_total: u64,
    pub full: usize,
}

struct Done {
    pos: ChunkPos,
    lod: Lod,
    tree: ChunkTree,
}

pub struct ChunkStreamer {
    generator: Arc<ChunkGenerator>,
    pub config: StreamConfig,
    loaded: FxHashMap<ChunkPos, Lod>,
    pinned: FxHashSet<ChunkPos>,
    inflight: FxHashMap<ChunkPos, Lod>,
    tx: Sender<Done>,
    /// Behind a mutex only so the streamer is `Sync` and can live in an ECS
    /// resource; it is only ever touched from `update`.
    rx: std::sync::Mutex<Receiver<Done>>,
    finished: Vec<Done>,
    queue: Vec<(u64, ChunkPos, Lod)>,
    targets: FxHashMap<ChunkPos, Lod>,
    unload: Vec<ChunkPos>,
    last_centre: Option<IVec3>,
    pub stats: StreamStats,
}

impl ChunkStreamer {
    pub fn new(generator: Arc<ChunkGenerator>, config: StreamConfig) -> Self {
        let (tx, rx) = channel();
        Self {
            generator,
            config,
            loaded: FxHashMap::default(),
            pinned: FxHashSet::default(),
            inflight: FxHashMap::default(),
            tx,
            rx: std::sync::Mutex::new(rx),
            finished: Vec::new(),
            queue: Vec::new(),
            targets: FxHashMap::default(),
            unload: Vec::new(),
            last_centre: None,
            stats: StreamStats::default(),
        }
    }

    pub fn generator(&self) -> &Arc<ChunkGenerator> {
        &self.generator
    }

    /// Marks a chunk as changed by play: it stays at full detail forever.
    pub fn pin(&mut self, pos: ChunkPos) {
        self.pinned.insert(pos);
    }

    pub fn lod_of(&self, pos: ChunkPos) -> Option<Lod> {
        self.loaded.get(&pos).copied()
    }

    /// Desired chunks and their detail, recomputed when the camera crosses a
    /// chunk boundary.
    fn rebuild_targets(&mut self, camera: DVec3) {
        let cam_xz = glam::Vec2::new(camera.x as f32, camera.z as f32);
        let chunk_m = CHUNK_VOXELS as f32 / 16.0;
        let slack = self.config.hysteresis_m;
        let radius_chunks = ((self.config.node8_m + slack) / chunk_m).ceil() as i32 + 1;
        let centre = ChunkPos::of_voxel(coords::metres_to_voxel(camera)).0;
        let max_y = (coords::WORLD_SECTORS_Y * coords::SECTOR_VOXELS) / CHUNK_VOXELS;
        self.targets.clear();
        self.queue.clear();
        for dz in -radius_chunks..=radius_chunks {
            for dx in -radius_chunks..=radius_chunks {
                let (cx, cz) = (centre.x + dx, centre.z + dz);
                let mid = glam::Vec2::new((cx as f32 + 0.5) * chunk_m, (cz as f32 + 0.5) * chunk_m);
                let d = ((mid - cam_xz).length() - chunk_m * 0.71).max(0.0);
                let ideal = self.config.lod_for(d, 0.0);
                let Some(relaxed) = self.config.lod_for(d, slack) else {
                    continue;
                };
                let (column_lo, column_hi) = self.generator.column_range(cx, cz);
                for cy in 0..max_y {
                    let pos = ChunkPos(IVec3::new(cx, cy, cz));
                    if !pos.in_world() {
                        continue;
                    }
                    let bottom = (cy * CHUNK_VOXELS) as f32 / 16.0;
                    if bottom > column_hi {
                        break;
                    }
                    let current = self.loaded.get(&pos).copied();
                    // Keep what is loaded while it stays within the slack;
                    // refine immediately when the ideal is finer.
                    let lod = match (ideal, current) {
                        (Some(i), Some(c)) if c <= relaxed && c < i => c,
                        (Some(i), _) => i,
                        (None, Some(c)) => c.max(relaxed),
                        (None, None) => continue,
                    };
                    // Buried chunks are invisible; keep only a thin solid layer
                    // under a nearby camera for digging and physics.
                    let pinned = self.pinned.contains(&pos);
                    let top = bottom + chunk_m;
                    if !pinned && top < column_lo {
                        let below_camera = camera.y as f32 - bottom;
                        if lod != Lod::Full || below_camera > 64.0 {
                            continue;
                        }
                    }
                    let lod = if pinned { Lod::Full } else { lod };
                    self.targets.insert(pos, lod);
                    if current == Some(lod) || self.inflight.get(&pos) == Some(&lod) {
                        continue;
                    }
                    let key = u64::from(lod as u8) << 40 | d as u64;
                    self.queue.push((key, pos, lod));
                }
            }
        }
        let targets = &self.targets;
        self.unload.extend(
            self.loaded
                .keys()
                .filter(|p| !targets.contains_key(p))
                .copied(),
        );
        for p in &self.unload {
            self.loaded.remove(p);
        }
        // Finest and nearest last, so dispatch pops them first.
        self.queue.sort_unstable_by_key(|q| std::cmp::Reverse(q.0));
    }

    /// Advances streaming by one frame.
    pub fn update(&mut self, world: &mut VoxelWorld, camera: DVec3) {
        mc2_core::scope!("stream.update");
        let centre = ChunkPos::of_voxel(coords::metres_to_voxel(camera)).0;
        if self.last_centre != Some(centre) {
            self.last_centre = Some(centre);
            self.rebuild_targets(camera);
        }
        for pos in self.unload.drain(..) {
            world.remove_chunk(pos);
        }

        // Dispatch, nearest and finest first (the queue is sorted descending
        // so the best candidate pops off the end).
        while self.inflight.len() < self.config.max_inflight {
            let Some((_, pos, lod)) = self.queue.pop() else {
                break;
            };
            if self.inflight.contains_key(&pos) {
                continue;
            }
            self.inflight.insert(pos, lod);
            let generator = self.generator.clone();
            let tx = self.tx.clone();
            rayon::spawn(move || {
                let mut tree = generator.generate(pos, lod);
                tree.prepare_flat();
                let _ = tx.send(Done { pos, lod, tree });
            });
        }

        // Insert within the time budget.
        let start = Instant::now();
        if let Ok(rx) = self.rx.lock() {
            self.finished.extend(rx.try_iter());
        }
        let mut inserted = 0;
        while let Some(done) = self.finished.pop() {
            self.inflight.remove(&done.pos);
            self.stats.generated_total += 1;
            if self.targets.get(&done.pos) == Some(&done.lod) {
                if done.tree.is_empty() {
                    world.remove_chunk(done.pos);
                } else {
                    world.insert_chunk(done.pos, done.tree);
                }
                self.loaded.insert(done.pos, done.lod);
                inserted += 1;
            }
            if start.elapsed().as_secs_f32() * 1000.0 > self.config.insert_budget_ms {
                break;
            }
        }
        self.stats.loaded = self.loaded.len();
        self.stats.inflight = self.inflight.len();
        self.stats.queued = self.queue.len();
        self.stats.inserted_last_frame = inserted;
        self.stats.full = self.loaded.values().filter(|l| **l == Lod::Full).count();
    }

    /// True once nothing near the camera is waiting for generation.
    pub fn settled(&self) -> bool {
        self.queue.is_empty() && self.inflight.is_empty() && self.finished.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erosion::ErosionParams;
    use crate::terrain::{CoarseTerrain, TerrainParams};

    fn streamer(config: StreamConfig) -> ChunkStreamer {
        let params = TerrainParams {
            size: 128,
            cell_m: 128.0,
            sea_level: 96.0,
            erosion: ErosionParams {
                iterations: 30,
                ..Default::default()
            },
        };
        let terrain = Arc::new(CoarseTerrain::generate(5, params, &mut |_, _| {}));
        ChunkStreamer::new(Arc::new(ChunkGenerator::new(terrain)), config)
    }

    fn settle(s: &mut ChunkStreamer, world: &mut VoxelWorld, camera: DVec3) {
        for _ in 0..20_000 {
            s.update(world, camera);
            if s.settled() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("streaming never settled");
    }

    fn small() -> StreamConfig {
        StreamConfig {
            full_m: 20.0,
            cell_m: 60.0,
            node2_m: 120.0,
            node8_m: 200.0,
            hysteresis_m: 16.0,
            max_inflight: 16,
            insert_budget_ms: 1000.0,
        }
    }

    #[test]
    fn detail_falls_off_with_distance_and_follows_the_camera() {
        let mut s = streamer(small());
        let mut world = VoxelWorld::new();
        settle(&mut s, &mut world, DVec3::new(8000.0, 200.0, 8000.0));
        let lods: Vec<Lod> = s.loaded.values().copied().collect();
        assert!(
            lods.contains(&Lod::Full) && lods.contains(&Lod::Node8),
            "{lods:?}"
        );

        // Move 400 m: the old neighbourhood unloads, the new one loads.
        settle(&mut s, &mut world, DVec3::new(8400.0, 200.0, 8000.0));
        let near_old = |p: &ChunkPos| {
            ((f64::from(p.0.x) + 0.5) * 32.0 - 8000.0).abs() < 32.0
                && ((f64::from(p.0.z) + 0.5) * 32.0 - 8000.0).abs() < 32.0
        };
        assert_eq!(s.loaded.keys().filter(|p| near_old(p)).count(), 0);
        assert!(world.chunks().all(|(p, _)| s.loaded.contains_key(p)));
        assert!(s.loaded.values().any(|l| *l == Lod::Full));
    }

    #[test]
    fn pinned_chunks_stay_full() {
        let mut s = streamer(small());
        let mut world = VoxelWorld::new();
        settle(&mut s, &mut world, DVec3::new(8000.0, 200.0, 8000.0));
        let pos = *s
            .loaded
            .iter()
            .find(|(_, l)| **l == Lod::Full)
            .expect("a full chunk")
            .0;
        s.pin(pos);
        settle(&mut s, &mut world, DVec3::new(8150.0, 200.0, 8000.0));
        assert_eq!(s.lod_of(pos), Some(Lod::Full));
    }
}
