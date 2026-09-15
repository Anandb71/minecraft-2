//! Golden image tests: renderer output compared against stored references.
//!
//! References are rendered on the software adapter (WARP on Windows,
//! lavapipe on Linux) so CI and developer machines agree. Regenerate with
//! `MC2_BLESS=1 cargo test -p mc2-render --test golden`.

use mc2_gpu::capture::{GoldenTolerance, check_golden, read_rgba8};
use mc2_gpu::{Gpu, GpuOptions};
use mc2_render::Renderer;
use mc2_render::quality::Quality;
use mc2_render::renderer::{RenderMode, RendererOptions};
use mc2_render::voxel_gpu::GpuWorldConfig;
use std::path::PathBuf;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden")
}

fn software_gpu() -> Option<Gpu> {
    if std::env::var("MC2_SKIP_GOLDEN").is_ok_and(|v| v == "1") {
        eprintln!("MC2_SKIP_GOLDEN=1: references are rendered on WARP, skipping here");
        return None;
    }
    match Gpu::headless(&GpuOptions {
        force_fallback: true,
    }) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            if std::env::var("MC2_REQUIRE_GPU").is_ok_and(|v| v == "1") {
                panic!("golden tests need a software adapter: {e}");
            }
            eprintln!("skipping golden test: {e}");
            None
        }
    }
}

fn render(gpu: &Gpu, renderer: &mut Renderer, size: (u32, u32), frames: u32) -> Vec<u8> {
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("golden target"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    for _ in 0..frames {
        renderer.render(gpu, target.clone());
    }
    read_rgba8(&gpu.device, &gpu.queue, &target)
}

#[test]
fn calibration_display_transform() {
    let Some(gpu) = software_gpu() else {
        return;
    };
    let size = (320, 180);
    let mut renderer = Renderer::new(
        &gpu,
        FORMAT,
        size,
        RendererOptions {
            mode: RenderMode::Calibration,
            world: GpuWorldConfig {
                tree_words: 1 << 16,
                voxel_words: 1 << 16,
                ..Default::default()
            },
            quality: Quality {
                render_scale: 1.0,
                ..Default::default()
            },
        },
    );
    renderer.frame.time = 1.0;
    let rgba = render(&gpu, &mut renderer, size, 1);
    let result = check_golden(
        &golden_dir(),
        "calibration",
        size.0,
        size.1,
        &rgba,
        &GoldenTolerance::default(),
    );
    if let Err(e) = result {
        panic!("{e}");
    }
}

#[test]
fn world_debug_shade() {
    let Some(gpu) = software_gpu() else {
        return;
    };
    let size = (320, 180);
    let mut renderer = Renderer::new(
        &gpu,
        FORMAT,
        size,
        RendererOptions {
            mode: RenderMode::World,
            world: GpuWorldConfig {
                tree_words: 1 << 20,
                voxel_words: 4 << 20,
                upload_budget: usize::MAX,
                proximity_m: 1.0e4,
                structure_budget_ms: f32::INFINITY,
            },
            quality: Quality {
                render_scale: 1.0,
                ..Default::default()
            },
        },
    );
    let mut world =
        mc2_voxel::samples::rolling_terrain(&mut mc2_voxel::samples::Rng(0x9e37_79b9_7f4a_7c15));
    let mut camera = mc2_render::camera::Camera {
        position: glam::DVec3::new(6.0, 19.0, 4.0),
        ..Default::default()
    };
    camera.look_at(glam::DVec3::new(34.0, 12.0, 26.0));
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("golden target"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    // Unlit material view: this golden pins the marcher and streaming, not
    // lighting, whose accumulation depends on how many frames streaming took.
    renderer.debug_mode = 3;
    renderer.frame.auto_exposure = false;
    // Render until streaming is quiescent, so the image never depends on
    // how many bricks one frame happened to upload.
    let mut quiet = 0;
    for _ in 0..200 {
        renderer.prepare(&gpu, &mut world, &camera, 1.0 / 60.0);
        renderer.render(&gpu, target.clone());
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let s = renderer.world.stats;
        quiet = if s.bricks_uploaded_last_frame == 0 && s.feedback_requests_last_frame == 0 {
            quiet + 1
        } else {
            0
        };
        if quiet >= 3 {
            break;
        }
    }
    let rgba = read_rgba8(&gpu.device, &gpu.queue, &target);
    if let Err(e) = check_golden(
        &golden_dir(),
        "world_debug_shade",
        size.0,
        size.1,
        &rgba,
        &GoldenTolerance::default(),
    ) {
        panic!("{e}");
    }
}

/// A 3x3 patch of full-detail chunk columns of generated terrain (coarse
/// erosion on a small grid) and a camera looking across it.
fn generated_world() -> (mc2_voxel::world::VoxelWorld, mc2_render::camera::Camera) {
    use mc2_worldgen::chunkgen::{ChunkGenerator, Lod};
    use mc2_worldgen::erosion::ErosionParams;
    use mc2_worldgen::terrain::{CoarseTerrain, TerrainParams};
    let params = TerrainParams {
        size: 128,
        cell_m: 128.0,
        sea_level: 96.0,
        erosion: ErosionParams {
            iterations: 60,
            ..Default::default()
        },
    };
    let terrain = std::sync::Arc::new(CoarseTerrain::generate(7, params, &mut |_, _| {}));
    let generator = ChunkGenerator::new(terrain.clone());
    let (cx, cz) = (256, 258);
    let mut world = mc2_voxel::world::VoxelWorld::new();
    for z in cz - 1..=cz + 1 {
        for x in cx - 1..=cx + 1 {
            let (lo, hi) = generator.column_range(x, z);
            for y in 0..16 {
                let bottom = (y * 32) as f32;
                if bottom > hi || bottom + 32.0 < lo {
                    continue;
                }
                let pos = mc2_voxel::coords::ChunkPos(glam::IVec3::new(x, y, z));
                let tree = generator.generate(pos, Lod::Full);
                if !tree.is_empty() {
                    world.insert_chunk(pos, tree);
                }
            }
        }
    }
    let centre_x = (cx as f64 + 0.5) * 32.0;
    let centre_z = (cz as f64 + 0.5) * 32.0;
    let ground = f64::from(terrain.height_at(centre_x as f32, centre_z as f32));
    let mut camera = mc2_render::camera::Camera {
        position: glam::DVec3::new(centre_x - 30.0, ground + 18.0, centre_z - 30.0),
        ..Default::default()
    };
    camera.look_at(glam::DVec3::new(centre_x + 10.0, ground, centre_z + 10.0));
    (world, camera)
}

fn world_renderer(gpu: &Gpu, size: (u32, u32)) -> Renderer {
    Renderer::new(
        gpu,
        FORMAT,
        size,
        RendererOptions {
            mode: RenderMode::World,
            world: GpuWorldConfig {
                tree_words: 4 << 20,
                voxel_words: 16 << 20,
                upload_budget: usize::MAX,
                proximity_m: 1.0e4,
                structure_budget_ms: f32::INFINITY,
            },
            quality: Quality {
                render_scale: 1.0,
                ..Default::default()
            },
        },
    )
}

fn golden_target(gpu: &Gpu, size: (u32, u32)) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("golden target"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Renders until streaming is quiescent (three frames without uploads), so
/// the image never depends on how many bricks one frame happened to upload.
fn render_until_quiet(
    gpu: &Gpu,
    renderer: &mut Renderer,
    world: &mut mc2_voxel::world::VoxelWorld,
    camera: &mc2_render::camera::Camera,
    target: &wgpu::Texture,
) {
    let mut quiet = 0;
    for _ in 0..300 {
        renderer.prepare(gpu, world, camera, 1.0 / 60.0);
        renderer.render(gpu, target.clone());
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let s = renderer.world.stats;
        quiet = if s.bricks_uploaded_last_frame == 0 && s.feedback_requests_last_frame == 0 {
            quiet + 1
        } else {
            0
        };
        if quiet >= 3 {
            break;
        }
    }
}

/// Generated terrain: pins worldgen, streaming and the marcher at once,
/// through the unlit material view.
#[test]
fn generated_terrain() {
    let Some(gpu) = software_gpu() else {
        return;
    };
    let (mut world, camera) = generated_world();
    let size = (320, 180);
    let mut renderer = world_renderer(&gpu, size);
    let target = golden_target(&gpu, size);
    renderer.debug_mode = 3;
    renderer.frame.auto_exposure = false;
    render_until_quiet(&gpu, &mut renderer, &mut world, &camera, &target);
    let rgba = read_rgba8(&gpu.device, &gpu.queue, &target);
    if let Err(e) = check_golden(
        &golden_dir(),
        "generated_terrain",
        size.0,
        size.1,
        &rgba,
        &GoldenTolerance::default(),
    ) {
        panic!("{e}");
    }
}

/// The lit pipeline on generated terrain: streams to quiescence, then
/// restarts the frame counter and accumulates a fixed number of frames so
/// every random sequence and history length is the same on every run.
fn lit_golden(name: &str, celestial: mc2_render::camera::Celestial, lantern: bool) {
    let Some(gpu) = software_gpu() else {
        return;
    };
    let (mut world, camera) = generated_world();
    if lantern {
        // A 6x6x6 voxel glowing block resting on the ground in view.
        let target = glam::DVec3::new(8208.0 + 10.0, 0.0, 8272.0 + 10.0);
        let mut y = 511;
        let (vx, vz) = ((target.x * 16.0) as i32, (target.z * 16.0) as i32);
        while y > 0 && world.voxel(glam::IVec3::new(vx, y * 16, vz)).is_air() {
            y -= 1;
        }
        let base = glam::IVec3::new(vx, y * 16 + 16, vz);
        world.fill_box(base, base + 5, mc2_voxel::material::ids::GLOWSTONE);
    }
    let size = (320, 180);
    let mut renderer = world_renderer(&gpu, size);
    renderer.celestial = celestial;
    let target = golden_target(&gpu, size);
    render_until_quiet(&gpu, &mut renderer, &mut world, &camera, &target);
    renderer.frame.frame_index = 0;
    for _ in 0..64 {
        renderer.prepare(&gpu, &mut world, &camera, 1.0 / 60.0);
        renderer.render(&gpu, target.clone());
    }
    let rgba = read_rgba8(&gpu.device, &gpu.queue, &target);
    if let Err(e) = check_golden(
        &golden_dir(),
        name,
        size.0,
        size.1,
        &rgba,
        &GoldenTolerance::default(),
    ) {
        panic!("{e}");
    }
}

/// Low afternoon sun: atmosphere LUTs, aerial perspective, traced sun and
/// sky visibility, exposure.
#[test]
fn lit_terrain_afternoon() {
    let sun_dir = glam::Vec3::new(-0.55, 0.32, 0.77).normalize();
    lit_golden(
        "lit_terrain_afternoon",
        mc2_render::camera::Celestial {
            sun_dir,
            moon_dir: -sun_dir,
            ..Default::default()
        },
        false,
    );
}

/// Full moon night with an emissive block: moon sky LUTs, stars and ReSTIR.
#[test]
fn lit_terrain_night() {
    let moon_dir = glam::Vec3::new(0.3, 0.6, 0.74).normalize();
    lit_golden(
        "lit_terrain_night",
        mc2_render::camera::Celestial {
            sun_dir: -moon_dir,
            moon_dir,
            ..Default::default()
        },
        true,
    );
}
