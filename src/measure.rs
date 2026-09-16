//! Offline measurements behind design decisions, run from headless captures.

use crate::world::stream;
use mc2_game::{Game, Voxels};
use mc2_gpu::Gpu;
use mc2_render::Renderer;
use mc2_render::camera::Camera;
use mc2_render::indirect::GiMethod;

/// Decodes an rgba16float texture to linear rgb.
pub fn read_rgb16f(gpu: &Gpu, tex: &wgpu::Texture) -> Vec<[f32; 3]> {
    let raw = mc2_gpu::capture::read_texture(&gpu.device, &gpu.queue, tex);
    raw.as_chunks::<8>()
        .0
        .iter()
        .map(|px| std::array::from_fn(|c| half(u16::from_le_bytes([px[c * 2], px[c * 2 + 1]]))))
        .collect()
}

pub fn half(h: u16) -> f32 {
    let sign = if h >> 15 == 1 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f32::from(h & 0x3ff);
    sign * match e {
        0 => m / 1024.0 * 2f32.powi(-14),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

fn luminance(c: [f32; 3]) -> f64 {
    f64::from(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
}

fn frame(
    gpu: &Gpu,
    renderer: &mut Renderer,
    game: &mut Game,
    camera: &Camera,
    target: &wgpu::Texture,
) {
    stream(game, camera.position);
    renderer.prepare(
        gpu,
        &mut game.world.resource_mut::<Voxels>().0,
        camera,
        1.0 / 60.0,
    );
    renderer.render(gpu, target.clone());
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    mc2_core::profiler::end_frame();
}

/// Error of the active indirect method's denoised output against an
/// unbiased reference: ReSTIR GI's initial samples without any reuse,
/// averaged over `reference_frames` frames of the same static view.
/// Also reports temporal instability over 30 frames.
pub fn gi_compare(
    gpu: &Gpu,
    renderer: &mut Renderer,
    game: &mut Game,
    camera: &Camera,
    target: &wgpu::Texture,
    reference_frames: u32,
) -> Result<(), String> {
    let method = renderer.quality().gi;
    let read = |renderer: &Renderer, label: &str| {
        renderer
            .graph_texture(label)
            .map(|t| read_rgb16f(gpu, t))
            .ok_or(format!("no texture `{label}`"))
    };
    // Temporal instability: mean absolute change between frames relative to
    // the mean, on the denoised output.
    let mut previous = read(renderer, "gi pong")?;
    let mut change = 0.0f64;
    let mut level = 0.0f64;
    for _ in 0..30 {
        frame(gpu, renderer, game, camera, target);
        let current = read(renderer, "gi pong")?;
        for (a, b) in current.iter().zip(&previous) {
            change += (luminance(*a) - luminance(*b)).abs();
            level += luminance(*a);
        }
        previous = current;
    }
    let result = previous;
    if method == GiMethod::RestirGi {
        // Reservoir health: M and W distributions after temporal and spatial reuse.
        for label in ["gi temporal reservoir", "gi spatial reservoir"] {
            let tex = renderer.graph_texture(label).ok_or("no reservoir")?;
            let raw = mc2_gpu::capture::read_texture(&gpu.device, &gpu.queue, tex);
            let texels: Vec<[f32; 4]> = raw
                .as_chunks::<16>()
                .0
                .iter()
                .map(|t| {
                    std::array::from_fn(|c| {
                        f32::from_le_bytes(t[c * 4..c * 4 + 4].try_into().unwrap())
                    })
                })
                .collect();
            let mut m: Vec<f32> = texels.iter().map(|t| t[1]).filter(|v| *v > 0.0).collect();
            let mut w: Vec<f32> = texels.iter().map(|t| t[2]).filter(|v| *v > 0.0).collect();
            m.sort_by(f32::total_cmp);
            w.sort_by(f32::total_cmp);
            let pct = |v: &[f32], p: usize| v.get(v.len() * p / 100).copied().unwrap_or(0.0);
            println!(
                "{label}: {} of {} live; M p10/p50/p90 {:.1}/{:.1}/{:.1}; W p50/p99 {:.3}/{:.3}",
                m.len(),
                texels.len(),
                pct(&m, 10),
                pct(&m, 50),
                pct(&m, 90),
                pct(&w, 50),
                pct(&w, 99)
            );
        }
    }

    let mut q = renderer.quality();
    q.gi = GiMethod::RestirGi;
    renderer.set_quality(q);
    let debug = renderer.debug_mode;
    renderer.debug_mode = 5;
    // Even and odd frames summed apart: their disagreement measures the
    // reference's own noise.
    let mut halves = [vec![0.0f64; result.len()], vec![0.0f64; result.len()]];
    for i in 0..reference_frames {
        frame(gpu, renderer, game, camera, target);
        let half = &mut halves[(i % 2) as usize];
        for (s, v) in half.iter_mut().zip(read(renderer, "gi noisy")?) {
            let l = luminance(v);
            if l.is_finite() {
                *s += l;
            }
        }
    }
    renderer.debug_mode = debug;
    q.gi = method;
    renderer.set_quality(q);

    let half_n = f64::from(reference_frames / 2);
    let mut sq = 0.0f64;
    let mut noise_sq = 0.0f64;
    let mut ref_sum = 0.0;
    let mut res_sum = 0.0;
    let mut count = 0.0f64;
    for i in 0..result.len() {
        let (a, b) = (halves[0][i] / half_n, halves[1][i] / half_n);
        let reference = 0.5 * (a + b);
        if reference <= 0.0 {
            continue;
        }
        let value = luminance(result[i]);
        sq += (value - reference).powi(2);
        // Var(mean of both halves) = Var(a - b) / 4.
        noise_sq += (a - b).powi(2) / 4.0;
        ref_sum += reference;
        res_sum += value;
        count += 1.0;
    }
    let mean = ref_sum / count.max(1.0);
    println!(
        "gi compare {method:?}: relative RMSE {:.3} (reference noise {:.3}), bias {:+.1}%, temporal change {:.4} per frame ({} reference frames, {} pixels)",
        (sq / count.max(1.0)).sqrt() / mean,
        (noise_sq / count.max(1.0)).sqrt() / mean,
        (res_sum / ref_sum - 1.0) * 100.0,
        change / level.max(1e-9),
        reference_frames,
        count
    );
    Ok(())
}
