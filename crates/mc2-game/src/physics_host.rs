//! The rigid body world behind a message boundary.
//!
//! The game thread owns the voxel world. In threaded mode the physics world
//! lives on its own thread and sees the voxel world only as the collision
//! cells the game thread has sent it: each step reports the cells its
//! bodies needed and lacked (those bodies wait), and the next job carries
//! them, along with fresh copies of mirrored cells that were edited. The
//! game thread never waits: it sends a job when the physics thread is idle
//! and reads the newest snapshot of the bodies when one has arrived, so a
//! slow step costs physics time, not frames.
//!
//! Inline mode runs each job at once against the voxel world itself, for
//! tests and scripted captures that need determinism.

use glam::{DVec3, IVec3, Quat, Vec3};
use mc2_core::{FxHashMap, FxHashSet};
use mc2_physics::collision::{CollisionMap, Occ, world_occ};
use mc2_physics::explode::Debris;
use mc2_physics::{Body, BodyId, BodyShape, Obstacle, PhysicsStats, PhysicsWorld};
use mc2_voxel::world::VoxelWorld;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Steps one job may run when physics has fallen behind; the rest of the
/// backlog is dropped (the world runs slow rather than hitching).
const MAX_STEPS_PER_JOB: u32 = 4;
/// Bodies that fall this far below the world are removed, metres.
const KILL_DEPTH_M: f64 = -64.0;
/// Mirrored cells farther than this from every body are evicted, metres.
const MIRROR_MARGIN_M: f64 = 4.0;

pub enum Command {
    Spawn {
        id: BodyId,
        shape: Arc<BodyShape>,
        pos: DVec3,
        rot: Quat,
        vel: Vec3,
        ang_vel: Vec3,
    },
    Remove(BodyId),
    Blast {
        centre: DVec3,
        radius: f32,
    },
    Wake {
        min: DVec3,
        max: DVec3,
    },
    /// Joins two bodies as they lie then, the way `hang` says (its parent
    /// index unused). Both join `group`.
    Joint {
        a: BodyId,
        b: BodyId,
        hang: Hang,
        group: u32,
    },
}

/// One part of a ragdoll: its shape and pose (centre of mass and principal
/// frame), and how it hangs from an earlier part.
#[derive(Clone)]
pub struct RagdollPart {
    pub shape: Arc<BodyShape>,
    pub pos: DVec3,
    pub rot: Quat,
    /// Metres a second.
    pub vel: Vec3,
    pub hang: Option<Hang>,
}

/// How a part hangs from an earlier one: the joint's world point, the
/// parent's axis and this part's (world), and either how far this part's
/// may swing from the parent's, or a hinge (its world axis and the bend
/// allowed about it, radians).
#[derive(Clone, Copy, Debug)]
pub struct Hang {
    pub parent: usize,
    pub at: DVec3,
    pub cone: Vec3,
    pub axis: Vec3,
    pub swing: f32,
    pub hinge: Option<(Vec3, f32, f32)>,
}

pub struct Job {
    pub steps: u32,
    pub dt: f32,
    pub commands: Vec<Command>,
    pub cells: Vec<(IVec3, Occ<'static>)>,
    pub obstacles: Vec<Obstacle>,
}

#[derive(Clone, Default)]
pub struct Frame {
    pub bodies: Vec<Body>,
    pub stats: PhysicsStats,
    /// Cells the last step lacked.
    pub missing: Vec<IVec3>,
    /// Cells the physics thread dropped from its mirror.
    pub evicted: Vec<IVec3>,
    /// Bodies that fell out of the world.
    pub lost: Vec<BodyId>,
    pub steps: u32,
    /// Time the whole job took, milliseconds.
    pub job_ms: f32,
    pub mirrored_cells: usize,
}

#[derive(Default)]
struct Sim {
    world: PhysicsWorld,
    map: CollisionMap,
}

impl Sim {
    /// Runs a job against the voxel world itself (inline) or the mirror.
    fn run(&mut self, job: Job, voxels: Option<&VoxelWorld>) -> Frame {
        let start = std::time::Instant::now();
        for c in job.commands {
            match c {
                Command::Spawn {
                    id,
                    shape,
                    pos,
                    rot,
                    vel,
                    ang_vel,
                } => {
                    self.world.spawn_debris(
                        id,
                        &Debris {
                            shape,
                            pos,
                            rot,
                            vel,
                            ang_vel,
                        },
                    );
                }
                Command::Remove(id) => {
                    self.world.remove(id);
                }
                Command::Blast { centre, radius } => {
                    self.world.blast_impulse(centre, radius);
                }
                Command::Wake { min, max } => self.world.wake_region(min, max),
                Command::Joint { a, b, hang, group } => {
                    if let (Some(ba), Some(bb)) = (self.world.body(a), self.world.body(b)) {
                        let h = hang;
                        let j = match h.hinge {
                            Some((axis, min, max)) => mc2_physics::Joint::hinged(
                                ba, bb, h.at, h.cone, h.axis, axis, min, max,
                            ),
                            None => mc2_physics::Joint::new(ba, bb, h.at, h.cone, h.axis, h.swing),
                        };
                        self.world.add_joint(j, group);
                    }
                }
            }
        }
        for (cell, occ) in job.cells {
            self.map.insert(cell, occ);
        }
        self.world.obstacles = job.obstacles;
        for _ in 0..job.steps {
            match voxels {
                Some(v) => self.world.step(job.dt, v),
                None => self.world.step(job.dt, &self.map),
            }
        }
        let mut lost = Vec::new();
        self.world.bodies.retain(|b| {
            let keep = b.pos.y > KILL_DEPTH_M;
            if !keep {
                lost.push(b.id);
            }
            keep
        });
        let evicted = if voxels.is_some() {
            Vec::new()
        } else {
            let spheres: Vec<(DVec3, f64)> = self
                .world
                .bodies
                .iter()
                .map(|b| (b.pos, f64::from(b.shape.radius) + MIRROR_MARGIN_M))
                .collect();
            self.map.evict(|c| {
                let centre = (c.as_dvec3() + 0.5) * 0.5;
                spheres
                    .iter()
                    .any(|(p, r)| p.distance_squared(centre) <= r * r)
            })
        };
        Frame {
            bodies: self.world.bodies.clone(),
            stats: self.world.stats,
            missing: std::mem::take(&mut self.world.missing_cells),
            evicted,
            lost,
            steps: job.steps,
            job_ms: start.elapsed().as_secs_f32() * 1000.0,
            mirrored_cells: self.map.len(),
        }
    }
}

enum Mode {
    Inline(Box<Sim>),
    Threaded {
        jobs: Option<Sender<Job>>,
        // Only the game thread reads it; the lock makes the host Sync.
        frames: Mutex<Receiver<Frame>>,
        busy: bool,
        thread: Option<JoinHandle<()>>,
    },
}

pub struct PhysicsHost {
    mode: Mode,
    next_id: u64,
    next_group: u32,
    commands: Vec<Command>,
    pending_steps: u32,
    obstacles: Vec<Obstacle>,
    frame: Frame,
    index: FxHashMap<BodyId, usize>,
    /// Bodies spawned or removed since the snapshot was taken, applied to
    /// what the game thread sees until a newer snapshot includes them.
    spawned: Vec<Body>,
    removed: FxHashSet<BodyId>,
    mirrored: FxHashSet<IVec3>,
    requested: Vec<IVec3>,
    edited: FxHashSet<IVec3>,
    /// Bodies that fell out of the world since last asked.
    lost: Vec<BodyId>,
}

impl Default for PhysicsHost {
    fn default() -> Self {
        Self::inline()
    }
}

impl PhysicsHost {
    fn with_mode(mode: Mode) -> Self {
        Self {
            mode,
            next_id: 1,
            next_group: 1,
            commands: Vec::new(),
            pending_steps: 0,
            obstacles: Vec::new(),
            frame: Frame::default(),
            index: FxHashMap::default(),
            spawned: Vec::new(),
            removed: FxHashSet::default(),
            mirrored: FxHashSet::default(),
            requested: Vec::new(),
            edited: FxHashSet::default(),
            lost: Vec::new(),
        }
    }

    pub fn inline() -> Self {
        Self::with_mode(Mode::Inline(Box::default()))
    }

    /// Physics on its own thread.
    pub fn threaded() -> Self {
        let (job_tx, job_rx) = channel::<Job>();
        let (frame_tx, frame_rx) = channel::<Frame>();
        let thread = std::thread::Builder::new()
            .name("physics".into())
            .spawn(move || {
                let mut sim = Sim::default();
                while let Ok(job) = job_rx.recv() {
                    mc2_core::scope!("physics.job");
                    let frame = sim.run(job, None);
                    if frame_tx.send(frame).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn physics thread");
        Self::with_mode(Mode::Threaded {
            jobs: Some(job_tx),
            frames: Mutex::new(frame_rx),
            busy: false,
            thread: Some(thread),
        })
    }

    pub fn is_threaded(&self) -> bool {
        matches!(self.mode, Mode::Threaded { .. })
    }

    fn allocate(&mut self) -> BodyId {
        let id = BodyId(self.next_id);
        self.next_id += 1;
        id
    }

    pub fn spawn(&mut self, shape: Arc<BodyShape>, pos: DVec3, rot: Quat, vel: Vec3) -> BodyId {
        self.spawn_debris(&Debris {
            shape,
            pos,
            rot,
            vel,
            ang_vel: Vec3::ZERO,
        })
    }

    pub fn spawn_debris(&mut self, d: &Debris) -> BodyId {
        let id = self.allocate();
        let mut body = Body::new(id, d.shape.clone(), d.pos, d.rot);
        body.vel = d.vel;
        body.ang_vel = d.ang_vel;
        self.spawned.push(body);
        self.commands.push(Command::Spawn {
            id,
            shape: d.shape.clone(),
            pos: d.pos,
            rot: d.rot,
            vel: d.vel,
            ang_vel: d.ang_vel,
        });
        id
    }

    /// Spawns jointed parts, none colliding with another; returns their
    /// ids in order.
    pub fn spawn_ragdoll(&mut self, parts: &[RagdollPart]) -> Vec<BodyId> {
        let group = self.next_group;
        self.next_group += 1;
        let ids: Vec<BodyId> = parts
            .iter()
            .map(|p| {
                self.spawn_debris(&Debris {
                    shape: p.shape.clone(),
                    pos: p.pos,
                    rot: p.rot,
                    vel: p.vel,
                    ang_vel: Vec3::ZERO,
                })
            })
            .collect();
        for (i, p) in parts.iter().enumerate() {
            if let Some(hang) = p.hang {
                self.commands.push(Command::Joint {
                    a: ids[hang.parent],
                    b: ids[i],
                    hang,
                    group,
                });
            }
        }
        ids
    }

    /// Removes a body; returns its last known state.
    pub fn remove(&mut self, id: BodyId) -> Option<Body> {
        let body = self.body(id).cloned()?;
        self.removed.insert(id);
        self.spawned.retain(|b| b.id != id);
        self.commands.push(Command::Remove(id));
        Some(body)
    }

    pub fn blast_impulse(&mut self, centre: DVec3, radius: f32) {
        self.commands.push(Command::Blast { centre, radius });
    }

    /// Voxels in `min..=max` changed: resend mirrored cells there and wake
    /// the bodies resting on them.
    pub fn edited(&mut self, min: IVec3, max: IVec3) {
        for z in (min.z >> 3)..=(max.z >> 3) {
            for y in (min.y >> 3)..=(max.y >> 3) {
                for x in (min.x >> 3)..=(max.x >> 3) {
                    let c = IVec3::new(x, y, z);
                    if self.mirrored.contains(&c) {
                        self.edited.insert(c);
                    }
                }
            }
        }
        self.commands.push(Command::Wake {
            min: min.as_dvec3() / 16.0,
            max: (max + 1).as_dvec3() / 16.0,
        });
    }

    pub fn set_obstacles(&mut self, obstacles: Vec<Obstacle>) {
        self.obstacles = obstacles;
    }

    /// Bodies as the game thread knows them: the newest snapshot plus
    /// spawns and removals it does not include yet.
    pub fn bodies(&self) -> impl Iterator<Item = &Body> {
        self.frame
            .bodies
            .iter()
            .filter(|b| !self.removed.contains(&b.id))
            .chain(self.spawned.iter())
    }

    pub fn body(&self, id: BodyId) -> Option<&Body> {
        if self.removed.contains(&id) {
            return None;
        }
        self.index
            .get(&id)
            .map(|&i| &self.frame.bodies[i])
            .or_else(|| self.spawned.iter().find(|b| b.id == id))
    }

    pub fn body_count(&self) -> usize {
        self.bodies().count()
    }

    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// Bodies lost below the world since the last call.
    pub fn take_lost(&mut self) -> Vec<BodyId> {
        std::mem::take(&mut self.lost)
    }

    /// One fixed tick of `dt` seconds: take any finished frame, then give
    /// the physics world its next job if it is free.
    pub fn tick(&mut self, voxels: &VoxelWorld, dt: f32) {
        self.pending_steps += 1;
        let mut arrived = None;
        if let Mode::Threaded { frames, busy, .. } = &mut self.mode {
            let received = frames.get_mut().expect("frame channel lock").try_recv();
            match received {
                Ok(frame) => {
                    *busy = false;
                    arrived = Some(frame);
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => panic!("physics thread died"),
            }
        }
        if let Some(frame) = arrived {
            self.accept(frame);
        }
        if matches!(self.mode, Mode::Threaded { busy: true, .. }) {
            return;
        }
        let steps = self.pending_steps.min(MAX_STEPS_PER_JOB);
        self.pending_steps = 0;
        let job = self.build_job(voxels, steps, dt);
        match &mut self.mode {
            Mode::Inline(sim) => {
                let frame = sim.run(job, Some(voxels));
                self.accept(frame);
            }
            Mode::Threaded { jobs, busy, .. } => {
                if let Some(tx) = jobs {
                    tx.send(job).expect("physics thread gone");
                    *busy = true;
                }
            }
        }
    }

    fn build_job(&mut self, voxels: &VoxelWorld, steps: u32, dt: f32) -> Job {
        let mut cells = Vec::new();
        if self.is_threaded() {
            let mut want: Vec<IVec3> = std::mem::take(&mut self.requested);
            want.extend(self.edited.drain());
            want.sort_unstable_by_key(|c| (c.x, c.y, c.z));
            want.dedup();
            for c in want {
                cells.push((c, world_occ(voxels, c).into_owned()));
                self.mirrored.insert(c);
            }
        }
        Job {
            steps,
            dt,
            commands: std::mem::take(&mut self.commands),
            cells,
            obstacles: self.obstacles.clone(),
        }
    }

    fn accept(&mut self, frame: Frame) {
        for c in &frame.evicted {
            self.mirrored.remove(c);
        }
        self.requested.extend(frame.missing.iter().copied());
        self.lost.extend(frame.lost.iter().copied());
        // A frame reflects every command sent before it: spawns sent are in
        // the snapshot (or already lost), removals sent are gone from it.
        let unsent: FxHashSet<BodyId> = self
            .commands
            .iter()
            .filter_map(|c| match c {
                Command::Spawn { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        self.spawned.retain(|b| unsent.contains(&b.id));
        let ids: FxHashSet<BodyId> = frame.bodies.iter().map(|b| b.id).collect();
        self.removed.retain(|id| ids.contains(id));
        self.index = frame
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| (b.id, i))
            .collect();
        self.frame = frame;
    }
}

impl Drop for PhysicsHost {
    fn drop(&mut self) {
        if let Mode::Threaded { jobs, thread, .. } = &mut self.mode {
            jobs.take();
            if let Some(t) = thread.take() {
                let _ = t.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;

    fn floor() -> VoxelWorld {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(255, 31, 255), ids::GRANITE);
        w
    }

    fn cube() -> Arc<BodyShape> {
        Arc::new(BodyShape::from_voxels(IVec3::splat(8), vec![ids::GRANITE; 512]).unwrap())
    }

    fn settle(host: &mut PhysicsHost, w: &VoxelWorld, seconds: f64) {
        let start = std::time::Instant::now();
        let mut sim_time = 0.0;
        while sim_time < seconds {
            host.tick(w, 1.0 / 120.0);
            sim_time += 1.0 / 120.0;
            if host.is_threaded() {
                // Give the physics thread real time to keep up.
                std::thread::sleep(std::time::Duration::from_micros(1500));
            }
            assert!(start.elapsed().as_secs() < 30, "physics never caught up");
        }
    }

    #[test]
    fn threaded_and_inline_bodies_both_come_to_rest() {
        let w = floor();
        for mut host in [PhysicsHost::inline(), PhysicsHost::threaded()] {
            let shape = cube();
            let rot = shape.principal;
            let id = host.spawn(shape, DVec3::new(8.0, 3.0, 8.0), rot, Vec3::ZERO);
            // Visible to the game at once, before physics has seen it.
            assert!(host.body(id).is_some());
            settle(&mut host, &w, 3.0);
            let b = host.body(id).expect("body");
            assert!(
                (b.pos.y - 2.25).abs() < 0.05,
                "{} rests at {}",
                host.is_threaded(),
                b.pos.y
            );
            let removed = host.remove(id).expect("removed");
            assert_eq!(removed.id, id);
            assert!(host.body(id).is_none());
            settle(&mut host, &w, 0.2);
            assert!(host.body(id).is_none());
            assert_eq!(host.body_count(), 0);
        }
    }

    #[test]
    fn edits_reach_the_physics_thread() {
        let mut w = floor();
        let mut host = PhysicsHost::threaded();
        let shape = cube();
        let rot = shape.principal;
        let id = host.spawn(shape, DVec3::new(8.0, 2.3, 8.0), rot, Vec3::ZERO);
        settle(&mut host, &w, 2.0);
        assert!((host.body(id).unwrap().pos.y - 2.25).abs() < 0.05);
        // Dig the floor out from under it.
        let (lo, hi) = (IVec3::new(112, 0, 112), IVec3::new(143, 31, 143));
        w.fill_box(lo, hi, ids::AIR);
        host.edited(lo, hi);
        settle(&mut host, &w, 2.0);
        let y = host.body(id).unwrap().pos.y;
        assert!(y < 0.5, "still hovering at {y}");
    }

    #[test]
    fn bodies_that_fall_out_of_the_world_are_lost() {
        let w = VoxelWorld::new();
        let mut host = PhysicsHost::inline();
        let shape = cube();
        let rot = shape.principal;
        let id = host.spawn(shape, DVec3::new(8.0, -60.0, 8.0), rot, Vec3::ZERO);
        settle(&mut host, &w, 2.0);
        assert!(host.body(id).is_none());
        assert_eq!(host.take_lost(), vec![id]);
    }
}
