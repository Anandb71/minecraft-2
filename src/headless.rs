//! Offscreen capture and benchmark modes. Also how the golden tests and the
//! README screenshots are produced without a window.

use crate::cli::{Args, Mode};
use crate::world::{TerrainLoader, spawn_camera, stream, stream_settled, stream_stats};
use mc2_core::RollingStats;
use mc2_game::{Game, Voxels};
use mc2_gpu::{Gpu, GpuOptions};
use mc2_render::Renderer;
use mc2_render::renderer::RendererOptions;
use std::time::Instant;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn target(gpu: &Gpu, size: (u32, u32)) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless target"),
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

pub fn run(args: &Args) -> Result<(), String> {
    let gpu = Gpu::headless(&GpuOptions {
        force_fallback: args.software,
    })
    .map_err(|e| e.to_string())?;
    eprintln!("adapter: {}", gpu.describe());
    let mut renderer = Renderer::new(&gpu, FORMAT, args.size, RendererOptions::default());
    renderer.debug_mode = args.debug_view;
    renderer.beam = !args.no_beam;
    let t = Instant::now();
    let mut game = Game::new();
    let mut loader = TerrainLoader::start(args.seed, args.world_dir.clone(), Default::default());
    let terrain = loader.wait(&mut game)?;
    let mut camera = spawn_camera(&terrain);
    if let Some((pos, look)) = args.camera {
        camera.position = glam::DVec3::from_array(pos);
        camera.look_at(glam::DVec3::from_array(look));
    }
    eprintln!(
        "terrain ready in {:.2}s, spawn at {:.0} {:.0} {:.0}",
        t.elapsed().as_secs_f32(),
        camera.position.x,
        camera.position.y,
        camera.position.z
    );
    // Stream until the neighbourhood is generated and uploaded.
    let tex_warm = target(&gpu, args.size);
    let t = Instant::now();
    let mut quiet = 0;
    while quiet < 10 && t.elapsed().as_secs() < 180 {
        stream(&mut game, camera.position);
        renderer.prepare(
            &gpu,
            &mut game.world.resource_mut::<Voxels>().0,
            &camera,
            1.0 / 60.0,
        );
        renderer.render(&gpu, tex_warm.clone());
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| e.to_string())?;
        mc2_core::profiler::end_frame();
        let settled = stream_settled(&game);
        let uploads = renderer.world.stats.bricks_uploaded_last_frame
            + renderer.world.stats.feedback_requests_last_frame;
        quiet = if settled && uploads == 0 {
            quiet + 1
        } else {
            0
        };
    }
    if let Some(demo) = args.demo {
        let feet = camera.position - glam::DVec3::Y * mc2_game::player::EYE;
        game.spawn_player(feet, camera.yaw, camera.pitch);
        crate::demo::run(&mut game, demo);
        let view = game.view();
        camera.position = view.position;
        camera.yaw = view.yaw;
        camera.pitch = view.pitch;
    }
    if let Some(s) = stream_stats(&game) {
        eprintln!(
            "streamed {} chunks ({} full) in {:.2}s",
            s.loaded,
            s.full,
            t.elapsed().as_secs_f32()
        );
    }
    let tex = target(&gpu, args.size);
    let mut frame_ms = RollingStats::default();
    let start = Instant::now();
    let mut last = Instant::now();
    for i in 0..args.frames {
        renderer.frame.time = i as f32 / 60.0;
        stream(&mut game, camera.position);
        renderer.prepare(
            &gpu,
            &mut game.world.resource_mut::<Voxels>().0,
            &camera,
            1.0 / 60.0,
        );
        renderer.frame.hud.clear();
        if args.demo.is_some() {
            let screen = renderer.output_size();
            crate::game_hud::draw(
                &mut game,
                &mut renderer.frame.hud,
                &mut renderer.frame.gizmos,
                screen,
            );
        }
        if args.hud {
            let cpu = mc2_core::profiler::rows();
            let stats = renderer.world.stats;
            let extra = [format!(
                "chunks {}  bricks {}  uploaded {} ({:.1} MB)  tree {:.1} MB  voxels {:.1} MB",
                stats.chunks,
                stats.bricks_resident,
                stats.bricks_uploaded_last_frame,
                stats.bytes_uploaded_last_frame as f32 / 1e6,
                stats.tree_mb,
                stats.voxel_mb
            )];
            let input = mc2_render::overlay::OverlayInput {
                gpu: &renderer.profiler,
                cpu: &cpu,
                frame_ms: &frame_ms,
                adapter: &gpu.info.name,
                resolution: args.size,
                render_resolution: renderer.render_size(),
                extra: &extra,
            };
            let mut hud = std::mem::take(&mut renderer.frame.hud);
            mc2_render::overlay::draw_profiler(&mut hud, &input, 1.0);
            renderer.frame.hud = hud;
        }
        renderer.render(&gpu, tex.clone());
        // Headless has no present to pace against; wait so timings are real.
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| e.to_string())?;
        mc2_core::profiler::end_frame();
        frame_ms.push(last.elapsed().as_secs_f32() * 1000.0);
        last = Instant::now();
    }
    let elapsed = start.elapsed().as_secs_f32();

    match &args.mode {
        Mode::Capture(path) => {
            let rgba = mc2_gpu::capture::read_rgba8(&gpu.device, &gpu.queue, &tex);
            mc2_gpu::capture::save_png(path, args.size.0, args.size.1, &rgba)
                .map_err(|e| e.to_string())?;
            eprintln!("wrote {}", path.display());
        }
        Mode::Bench => {
            print_report(&renderer, &frame_ms, args, elapsed);
            if args.debug_view == 2
                && let Some(tex) = renderer.graph_texture("vis depth")
            {
                let raw = mc2_gpu::capture::read_texture(&gpu.device, &gpu.queue, tex);
                let mut its: Vec<f32> = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                its.sort_by(f32::total_cmp);
                let mean = its.iter().sum::<f32>() / its.len() as f32;
                println!(
                    "march iterations per ray: mean {mean:.1}, p50 {}, p99 {}, max {}",
                    its[its.len() / 2],
                    its[its.len() * 99 / 100],
                    its[its.len() - 1]
                );
            }
        }
        Mode::Window => unreachable!("headless::run called in window mode"),
    }
    Ok(())
}

fn print_report(renderer: &Renderer, frame_ms: &RollingStats, args: &Args, elapsed: f32) {
    println!(
        "{} frames at {}x{} in {:.2}s  frame mean {:.2} ms  p99 {:.2} ms",
        args.frames,
        args.size.0,
        args.size.1,
        elapsed,
        frame_ms.mean(),
        frame_ms.p99()
    );
    let s = renderer.world.stats;
    println!(
        "world: {} chunks, {} bricks resident, tree {:.1} MB, voxels {:.1} MB",
        s.chunks, s.bricks_resident, s.tree_mb, s.voxel_mb
    );
    println!("{:<32} {:>8} {:>8} {:>8}", "GPU pass", "mean", "p99", "max");
    for r in renderer.profiler.rows() {
        println!(
            "{:<32} {:>8.3} {:>8.3} {:>8.3}",
            r.name,
            r.stats.mean(),
            r.stats.p99(),
            r.stats.max()
        );
    }
    println!(
        "{:<32} {:>8} {:>8} {:>8}",
        "CPU scope", "mean", "p99", "max"
    );
    for r in mc2_core::profiler::rows() {
        println!(
            "{:<32} {:>8.3} {:>8.3} {:>8.3}",
            r.name,
            r.stats.mean(),
            r.stats.p99(),
            r.stats.max()
        );
    }
}
