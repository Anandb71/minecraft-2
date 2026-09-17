//! Terrain loading in the background, chunk streaming into the game world,
//! and choosing where the player arrives.

use glam::DVec3;
use mc2_game::{Game, Streaming, Voxels};
use mc2_render::camera::{Camera, Celestial};
use mc2_worldgen::chunkgen::ChunkGenerator;
use mc2_worldgen::stream::{ChunkStreamer, StreamConfig, StreamStats};
use mc2_worldgen::terrain::{CoarseTerrain, TerrainParams};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

type LoadResult = Result<Arc<CoarseTerrain>, String>;

pub struct TerrainLoader {
    loading: Option<JoinHandle<LoadResult>>,
    progress: Arc<Mutex<(String, f32)>>,
    stream_config: StreamConfig,
}

impl TerrainLoader {
    /// Starts loading (or generating and caching) the world's terrain.
    pub fn start(seed: u64, dir: PathBuf, stream_config: StreamConfig) -> Self {
        let progress = Arc::new(Mutex::new(("loading terrain".to_owned(), 0.0)));
        let report = progress.clone();
        let loading = std::thread::Builder::new()
            .name("terrain".into())
            .spawn(move || {
                let path = dir.join(format!("terrain_{seed}.bin"));
                mc2_worldgen::cache::load_or_generate(
                    &path,
                    seed,
                    TerrainParams::world(),
                    &mut |stage, f| {
                        if let Ok(mut p) = report.lock() {
                            *p = (stage.to_owned(), f);
                        }
                    },
                )
                .map(Arc::new)
                .map_err(|e| format!("terrain cache {}: {e}", path.display()))
            })
            .expect("spawn terrain thread");
        Self {
            loading: Some(loading),
            progress,
            stream_config,
        }
    }

    /// Once loading finishes, installs streaming into the game and returns
    /// the terrain (exactly once).
    pub fn poll(&mut self, game: &mut Game) -> Result<Option<Arc<CoarseTerrain>>, String> {
        if !self.loading.as_ref().is_some_and(JoinHandle::is_finished) {
            return Ok(None);
        }
        let handle = self.loading.take().expect("checked above");
        let terrain = handle
            .join()
            .map_err(|_| "terrain thread panicked".to_owned())??;
        let generator = Arc::new(ChunkGenerator::new(terrain.clone()));
        game.world
            .insert_resource(Streaming(Some(ChunkStreamer::new(
                generator,
                self.stream_config,
            ))));
        Ok(Some(terrain))
    }

    /// Blocks until terrain is ready.
    pub fn wait(&mut self, game: &mut Game) -> Result<Arc<CoarseTerrain>, String> {
        loop {
            if let Some(t) = self.poll(game)? {
                return Ok(t);
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    pub fn loading_text(&self) -> Option<String> {
        self.loading.as_ref()?;
        let p = self.progress.lock().ok()?;
        Some(format!("generating world: {} {:.0}%", p.0, p.1 * 100.0))
    }
}

/// Streams chunks around `camera` into the game's voxel world.
/// The renderer's view of the sky from the world clock.
pub fn celestial(game: &Game) -> Celestial {
    let clock = game.world.resource::<mc2_game::clock::WorldClock>();
    let sky = clock.sky();
    Celestial {
        sun_dir: sky.sun_dir,
        sun_illuminance: glam::Vec3::splat(sky.sun_illuminance),
        moon_dir: sky.moon_dir,
        moon_illuminance: sky.moon_illuminance,
        moon_phase: sky.moon_phase,
        star_rotation: sky.sidereal_angle,
        latitude: clock.latitude.to_radians() as f32,
    }
}

pub fn stream(game: &mut Game, camera: DVec3) {
    game.world
        .resource_scope(|world, mut streaming: bevy_ecs::prelude::Mut<Streaming>| {
            if let Some(s) = streaming.0.as_mut() {
                let mut voxels = world.resource_mut::<Voxels>();
                s.update(&mut voxels.0, camera);
            }
        });
}

pub fn stream_stats(game: &Game) -> Option<StreamStats> {
    game.world
        .resource::<Streaming>()
        .0
        .as_ref()
        .map(|s| s.stats)
}

pub fn stream_settled(game: &Game) -> bool {
    game.world
        .resource::<Streaming>()
        .0
        .as_ref()
        .is_some_and(ChunkStreamer::settled)
}

/// A camera standing on open land near the middle of the map, looking at
/// the highest ground within a few kilometres.
pub fn spawn_camera(terrain: &CoarseTerrain) -> Camera {
    let extent = terrain.params.extent_m();
    let sea = terrain.params.sea_level;
    let centre = extent * 0.5;
    let mut best = None;
    'search: for ring in 0..200 {
        let r = ring as f32 * 64.0;
        let steps = (ring * 8).max(1);
        for k in 0..steps {
            let a = k as f32 / steps as f32 * std::f32::consts::TAU;
            let (x, z) = (centre + r * a.cos(), centre + r * a.sin());
            let h = terrain.height_at(x, z);
            let slope =
                (terrain.height_at(x + 16.0, z) - terrain.height_at(x - 16.0, z)).abs() / 32.0;
            if h > sea + 12.0 && h < 300.0 && slope < 0.25 {
                best = Some((x, z, h));
                break 'search;
            }
        }
    }
    let (x, z, h) = best.unwrap_or((centre, centre, terrain.height_at(centre, centre)));
    let mut view = (x + 1.0, z, h);
    for k in 0..64 {
        let a = k as f32 / 64.0 * std::f32::consts::TAU;
        for d in [800.0, 1600.0, 2400.0] {
            let (vx, vz) = (x + d * a.cos(), z + d * a.sin());
            let vh = terrain.height_at(vx, vz);
            if vh > view.2 {
                view = (vx, vz, vh);
            }
        }
    }
    let mut camera = Camera {
        position: DVec3::new(f64::from(x), f64::from(h) + 6.0, f64::from(z)),
        ..Default::default()
    };
    camera.look_at(DVec3::new(
        f64::from(view.0),
        f64::from(h) + 40.0,
        f64::from(view.1),
    ));
    camera
}

/// Hands the game's rigid bodies to the renderer, posed between fixed ticks.
pub fn pose_bodies(game: &Game, renderer: &mut mc2_render::Renderer) {
    let alpha = game.world.resource::<mc2_game::input::Time>().alpha;
    let physics = game.world.resource::<mc2_game::physics::Physics>();
    renderer.bodies.clear();
    renderer.bodies.extend(physics.world.bodies.iter().map(|b| {
        let (grid_origin, grid_rotation) = b.grid_pose_at(alpha);
        mc2_render::bodies::BodyInstance {
            key: b.id.0,
            shape: b.shape.clone(),
            grid_origin,
            grid_rotation,
        }
    }));
}

/// One HUD line on debris physics.
pub fn physics_line(game: &Game, renderer: &mc2_render::Renderer) -> String {
    let physics = game.world.resource::<mc2_game::physics::Physics>();
    let s = physics.world.stats;
    let drawn = renderer.world.bodies.stats;
    format!(
        "physics: {} bodies ({} awake, {} drawn), {} fuses, contacts {} + {} pairs, tick {:.2} ms (pairs {:.2}), blasts {}",
        s.bodies,
        s.awake,
        drawn.drawn,
        physics.fuses.len(),
        s.world_contacts,
        s.pair_contacts,
        s.step_ms,
        s.pair_ms,
        physics.blasts_total
    )
}
