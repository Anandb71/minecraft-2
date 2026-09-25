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

/// A camera standing on open land near the middle of the map (grassland,
/// clear of any tree), looking at the highest ground within a few
/// kilometres.
pub fn spawn_camera(terrain: &Arc<CoarseTerrain>) -> Camera {
    use mc2_worldgen::flora::{Biome, plants_near};
    let surface = mc2_worldgen::amplify::Surface::new(terrain.clone());
    let extent = terrain.params.extent_m();
    let sea = terrain.params.sea_level;
    let centre = extent * 0.5;
    let open = |x: f32, z: f32| {
        let s = surface.sample(x, z);
        if !matches!(
            s.biome(sea),
            Biome::Plains | Biome::Forest | Biome::BirchForest
        ) {
            return false;
        }
        let at = glam::Vec3::new(x, s.height, z);
        plants_near(&surface, at - 3.0, at + 3.0, terrain.seed, &|_, _| false).is_empty()
    };
    // The village nearest the middle of the map, if one is close: stand at
    // the corner of its plaza looking across it.
    let settlements = mc2_worldgen::settlement::Settlements::new(terrain.seed);
    let cell = mc2_worldgen::settlement::VILLAGE_CELL_M;
    let c = (centre / cell).floor() as i32;
    for ring in 0..4 {
        for k in c - ring..=c + ring {
            for i in c - ring..=c + ring {
                if (i - c).abs() != ring && (k - c).abs() != ring {
                    continue;
                }
                if let Some(v) = settlements.village(&surface, i, k) {
                    let (x, z) = (v.centre.x - 8.5, v.centre.z - 8.5);
                    let h = surface.sample(x, z).height;
                    let mut camera = Camera {
                        position: DVec3::new(f64::from(x), f64::from(h) + 1.75, f64::from(z)),
                        ..Default::default()
                    };
                    camera.look_at(DVec3::new(
                        f64::from(v.centre.x + 20.0),
                        f64::from(v.centre.y + 2.5),
                        f64::from(v.centre.z + 12.0),
                    ));
                    return camera;
                }
            }
        }
    }
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
            if h > sea + 12.0 && h < 300.0 && slope < 0.25 && open(x, z) {
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
    // Cars are drawn from their parts, wheels apart from the body.
    let cars = game.world.resource::<mc2_game::vehicles::Drawn>();
    renderer.bodies.extend(
        physics
            .host
            .bodies()
            .filter(|b| !cars.bodies.contains(&b.id))
            .map(|b| {
                let (grid_origin, grid_rotation) = b.grid_pose_at(alpha);
                mc2_render::bodies::BodyInstance {
                    key: b.id.0,
                    shape: b.shape.clone(),
                    grid_origin,
                    grid_rotation,
                }
            }),
    );
    renderer
        .bodies
        .extend(cars.parts.iter().map(|p| mc2_render::bodies::BodyInstance {
            key: p.key,
            shape: p.shape.clone(),
            grid_origin: p.corner,
            grid_rotation: p.rotation,
        }));
    let people = game.world.resource::<mc2_game::character::Drawn>();
    renderer
        .bodies
        .extend(people.0.iter().map(|p| mc2_render::bodies::BodyInstance {
            key: p.key,
            shape: p.shape.clone(),
            grid_origin: p.corner,
            grid_rotation: p.rotation,
        }));
}

/// One HUD line on structural integrity.
pub fn structure_line(game: &Game) -> String {
    let s = game
        .world
        .resource::<mc2_game::structure::Structure>()
        .stats;
    format!(
        "structure: {} regions ({} nodes, gather {:.2} ms, solve {:.2} ms, worst {:.2}), {} failures, {} collapses, {} pieces, {} dirty, {} falling",
        s.regions,
        s.last_nodes,
        s.last_gather_ms,
        s.last_solve_ms,
        s.worst_ratio,
        s.failures,
        s.islands,
        s.pieces,
        s.dirty,
        s.falling
    )
}

/// One HUD line on debris physics.
pub fn physics_line(game: &Game, renderer: &mc2_render::Renderer) -> String {
    let physics = game.world.resource::<mc2_game::physics::Physics>();
    let frame = physics.host.frame();
    let s = frame.stats;
    let drawn = renderer.world.bodies.stats;
    format!(
        "physics ({}): {} bodies ({} awake, {} drawn), {} fuses, contacts {} + {} pairs, step {:.2} ms (pairs {:.2}), job {:.2} ms / {} steps, {} cells mirrored, blasts {}",
        if physics.host.is_threaded() {
            "thread"
        } else {
            "inline"
        },
        physics.host.body_count(),
        s.awake,
        drawn.drawn,
        physics.fuses.len(),
        s.world_contacts,
        s.pair_contacts,
        s.step_ms,
        s.pair_ms,
        frame.job_ms,
        frame.steps,
        frame.mirrored_cells,
        physics.blasts_total
    )
}

/// The weather as the renderer draws it: cloud cover and wind, fog, rain
/// or snow by the climate where the camera is (nothing falls under a
/// roof), wet ground, and lightning, for a camera at `at`.
pub fn weather(game: &Game, renderer: &mut mc2_render::Renderer, at: DVec3) {
    use mc2_game::weather::Weather;
    let w = game.world.resource::<Weather>();
    let p = at;
    let cold = game
        .world
        .resource::<Streaming>()
        .0
        .as_ref()
        .is_some_and(|s| {
            let q = s.generator().surface.sample(p.x as f32, p.z as f32);
            q.temp < 0.32 || p.y as f32 > mc2_worldgen::amplify::SNOWLINE_M
        });
    let voxels = &game.world.resource::<Voxels>().0;
    let eye = (p * 16.0).floor().as_ivec3();
    let covered = (1..48).any(|k| voxels.voxel(eye + glam::IVec3::Y * (k * 8)).is_solid());
    let falling = if covered { 0.0 } else { w.rain };
    renderer.weather = mc2_render::camera::WeatherLook {
        rain: if cold { 0.0 } else { falling },
        snow: if cold { falling } else { 0.0 },
        flash: w.flash,
        wetness: if cold { 0.0 } else { w.wetness },
        wind: w.wind.to_array(),
        cover: w.cover,
    };
    renderer.clouds.coverage = w.cover;
    renderer.clouds.precipitation = w.rain;
    renderer.clouds.wind = (w.wind * 4.0 + glam::Vec2::new(4.0, 2.0)).to_array();
    renderer.fog.density = mc2_render::fog::FogSettings::default().density * (1.0 + 5.0 * w.rain);
}

/// Fire and weather for the profiler overlay.
pub fn weather_line(game: &Game) -> String {
    let w = game.world.resource::<mc2_game::weather::Weather>();
    let f = game.world.resource::<mc2_game::fire::Fire>();
    format!(
        "weather: {} (cover {:.2}, rain {:.2}, wet {:.2}, wind {:.1} m/s)  fire: {} burning, {} caught, {} burnt out",
        w.sky.name(),
        w.cover,
        w.rain,
        w.wetness,
        w.wind.length(),
        f.burning(),
        f.caught,
        f.burnt_out
    )
}
