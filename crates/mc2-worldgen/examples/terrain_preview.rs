//! Renders the coarse world terrain as a hillshaded map with rivers and sea.
//! `cargo run --release -p mc2-worldgen --example terrain_preview -- <size> <out.png>`

use mc2_worldgen::terrain::{CoarseTerrain, TerrainParams};
use std::io::Write;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let size: usize = args.next().map_or(256, |s| s.parse().expect("size"));
    let out = args.next().unwrap_or_else(|| "captures/terrain.png".into());
    let mut params = TerrainParams::world();
    params.cell_m = params.extent_m() / size as f32;
    params.size = size;
    let t = Instant::now();
    let terrain = CoarseTerrain::generate(42, params, &mut |stage, f| {
        print!("\r{stage} {:>3.0}%", f * 100.0);
        let _ = std::io::stdout().flush();
    });
    println!("\ngenerated {size}^2 in {:.2}s", t.elapsed().as_secs_f32());
    let (lo, hi) = terrain.height.min_max();
    let flow_max = terrain.flow.data.iter().cloned().fold(0.0f32, f32::max);
    let mut sorted = terrain.flow.data.clone();
    sorted.sort_by(f32::total_cmp);
    let flow_p95 = sorted[sorted.len() * 95 / 100].max(1e-6);
    println!("height {lo:.1}..{hi:.1} m, peak flow {flow_max:.4}");

    let mut rgba = Vec::with_capacity(size * size * 4);
    for y in 0..size {
        for x in 0..size {
            let h = terrain.height.get(x, y);
            let (gx, gy) = terrain.height.gradient(x, y);
            let (gx, gy) = (gx / params.cell_m, gy / params.cell_m);
            let n = [-gx * 4.0, 1.0, -gy * 4.0];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let shade = ((n[0] * -0.6 + n[1] * 0.7 + n[2] * -0.4) / len).clamp(0.0, 1.0);
            // Rivers: the top 5% of discharge, brightening with log flow.
            let flow = ((terrain.flow.get(x, y) / flow_p95).ln()
                / (flow_max / flow_p95).ln().max(1e-3))
            .clamp(0.0, 1.0);
            let sed = (terrain.sediment.get(x, y) / 10.0).clamp(0.0, 1.0);
            let mut c = if h < params.sea_level {
                let depth = ((params.sea_level - h) / 80.0).clamp(0.0, 1.0);
                [0.15 - depth * 0.1, 0.35 - depth * 0.2, 0.6 - depth * 0.25]
            } else {
                let t = ((h - params.sea_level) / 350.0).clamp(0.0, 1.0);
                let base = [0.35 + t * 0.4, 0.5 + t * 0.2, 0.25 + t * 0.45];
                [
                    base[0] * (0.3 + 0.7 * shade) + sed * 0.15,
                    base[1] * (0.3 + 0.7 * shade) + sed * 0.1,
                    base[2] * (0.3 + 0.7 * shade),
                ]
            };
            if h >= params.sea_level {
                let r = flow;
                c = [
                    c[0] * (1.0 - r) + 0.1 * r,
                    c[1] * (1.0 - r) + 0.3 * r,
                    c[2] * (1.0 - r) + 0.9 * r,
                ];
            }
            rgba.extend(c.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8));
            rgba.push(255);
        }
    }
    let path = std::path::Path::new(&out);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).expect("output dir");
    }
    let file = std::io::BufWriter::new(std::fs::File::create(path).expect("create png"));
    let mut enc = png::Encoder::new(file, size as u32, size as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(&rgba))
        .expect("write png");
    println!("wrote {out}");
}
