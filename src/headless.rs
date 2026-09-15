//! Offscreen capture and benchmark modes. Also how the golden tests and the
//! README screenshots are produced without a window.

use crate::cli::{Args, Mode};
use crate::scene;
use mc2_core::RollingStats;
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
    let t = Instant::now();
    let (mut world, camera) = scene::demo();
    eprintln!("scene built in {:.2}s", t.elapsed().as_secs_f32());
    let tex = target(&gpu, args.size);
    let mut frame_ms = RollingStats::default();
    let start = Instant::now();
    let mut last = Instant::now();
    for i in 0..args.frames {
        renderer.frame.time = i as f32 / 60.0;
        renderer.prepare(&gpu, &mut world, &camera, 1.0 / 60.0);
        renderer.frame.hud.clear();
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
        Mode::Bench => print_report(&renderer, &frame_ms, args, elapsed),
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
