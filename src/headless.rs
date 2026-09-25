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
    let mut renderer = Renderer::new(
        &gpu,
        FORMAT,
        args.size,
        RendererOptions {
            quality: crate::cli::quality(args),
            ..Default::default()
        },
    );
    renderer.debug_mode = args.debug_view;
    if let Some((focus_m, f_number)) = args.dof {
        renderer.frame.post.dof = Some(mc2_render::post::DepthOfField {
            focus_m,
            focal_mm: 35.0,
            f_number,
        });
    }
    renderer.beam = !args.no_beam;
    let t = Instant::now();
    let mut game = Game::new();
    if args.physics_thread {
        game.world.resource_mut::<mc2_game::physics::Physics>().host =
            mc2_game::physics_host::PhysicsHost::threaded();
    }
    let mut loader = TerrainLoader::start(args.seed, args.world_dir.clone(), Default::default());
    let terrain = loader.wait(&mut game)?;
    {
        // Captures and benchmarks see a frozen sky.
        let mut clock = game.clock();
        clock.paused = true;
        if let Some(hour) = args.time {
            clock.set_hour(hour);
        }
    }
    if let Some(sky) = args.weather {
        let mut w = game.world.resource_mut::<mc2_game::weather::Weather>();
        w.set(sky);
        w.frozen = true;
    }
    renderer.celestial = crate::world::celestial(&game);
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
    renderer.frame.hud.set_icons(crate::icons::icons().atlas());
    let mut water = crate::water_sim::WaterSim::new(&gpu, &renderer.frame.shaders);
    let mut frame_ms = RollingStats::default();
    let start = Instant::now();
    let mut last = Instant::now();
    for i in 0..args.frames {
        renderer.frame.time = i as f32 / 60.0;
        stream(&mut game, camera.position);
        water.frame(&gpu, &renderer.frame.shaders, &mut game, 1.0 / 60.0);
        crate::world::weather(&game, &mut renderer, camera.position);
        if !args.hide_bodies {
            crate::world::pose_bodies(&game, &mut renderer);
        }
        renderer.prepare(
            &gpu,
            &mut game.world.resource_mut::<Voxels>().0,
            &camera,
            1.0 / 60.0,
        );
        renderer.frame.hud.clear();
        if args.demo.is_some() && !args.clean {
            let screen = renderer.output_size();
            crate::game_hud::draw(
                &mut game,
                &mut renderer.frame.hud,
                &mut renderer.frame.gizmos,
                screen,
                false,
            );
        }
        if args.demo.is_some() {
            let screen = renderer.output_size();
            crate::demo::overlay(&mut game, &mut renderer.frame.hud, screen, args.demo);
        }
        if args.hud {
            let cpu = mc2_core::profiler::rows();
            let stats = renderer.world.stats;
            let extra = [
                format!(
                    "chunks {}  bricks {}  uploaded {} ({:.1} MB)  tree {:.1} MB  voxels {:.1} MB",
                    stats.chunks,
                    stats.bricks_resident,
                    stats.bricks_uploaded_last_frame,
                    stats.bytes_uploaded_last_frame as f32 / 1e6,
                    stats.tree_mb,
                    stats.voxel_mb
                ),
                crate::world::physics_line(&game, &renderer),
                crate::world::structure_line(&game),
                crate::world::weather_line(&game),
                water.line(&game),
            ];
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

    // MC2_GI_COMPARE=<frames> measures the indirect method against an
    // unbiased reference accumulated over that many frames.
    if let Some(frames) = std::env::var("MC2_GI_COMPARE")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        crate::measure::gi_compare(&gpu, &mut renderer, &mut game, &camera, &tex, frames)?;
    }

    // MC2_DUMP=1 also writes intermediate lighting targets next to a capture.
    if let Mode::Capture(path) = &args.mode
        && std::env::var("MC2_DUMP").is_ok()
    {
        for (label, scale) in [
            ("sky view", 20.0),
            ("sky transmittance", 1.0),
            ("sky multiscatter", 50.0),
            ("light visibility", 1.0),
            ("direct lights", 0.0),
            ("gi noisy", 0.0),
            ("gi pong", 0.0),
            ("surface radiance", 0.0),
        ] {
            if let Some(tex) = renderer.graph_texture(label) {
                let name = format!("dump_{}.png", label.replace(' ', "_"));
                dump_rgba16f(&gpu, tex, &path.with_file_name(name), scale)?;
            }
        }
    }
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

/// Writes an rgba16float texture as a PNG, scaled and clamped, for debugging.
fn dump_rgba16f(
    gpu: &Gpu,
    tex: &wgpu::Texture,
    path: &std::path::Path,
    scale: f32,
) -> Result<(), String> {
    let raw = mc2_gpu::capture::read_texture(&gpu.device, &gpu.queue, tex);
    let half = |b: [u8; 2]| crate::measure::half(u16::from_le_bytes(b));
    let size = tex.size();
    // Scale 0 exposes the 95th percentile brightest channel at white.
    let scale = if scale > 0.0 {
        scale
    } else {
        let mut peaks: Vec<f32> = raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|px| {
                (0..3)
                    .map(|c| half([px[c * 2], px[c * 2 + 1]]))
                    .fold(0.0, f32::max)
            })
            .filter(|v| v.is_finite())
            .collect();
        peaks.sort_by(f32::total_cmp);
        let p95 = peaks.get(peaks.len() * 95 / 100).copied().unwrap_or(1.0);
        1.0 / p95.max(1e-6)
    };
    let mut rgba = Vec::with_capacity(raw.len() / 2);
    for px in raw.as_chunks::<8>().0 {
        for c in 0..3 {
            let v = half([px[c * 2], px[c * 2 + 1]]) * scale;
            rgba.push((v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8);
        }
        rgba.push(255);
    }
    eprintln!("dump {} ({}x{})", path.display(), size.width, size.height);
    mc2_gpu::capture::save_png(path, size.width, size.height, &rgba).map_err(|e| e.to_string())
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
