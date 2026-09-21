//! Names headless cameras for the worldgen v2 shots.
//!
//! Uses the same terrain cache as the game (`worlds/shots/terrain_<seed>.bin`).
//!
//! ```text
//! cargo run --release -p mc2-worldgen --example shot_spots
//! ```

use mc2_voxel::coords::ChunkPos;
use mc2_voxel::material::ids;
use mc2_worldgen::amplify::Surface;
use mc2_worldgen::cache::load_or_generate;
use mc2_worldgen::chunkgen::{ChunkGenerator, Lod};
use mc2_worldgen::karst::{Karst, soluble};
use mc2_worldgen::strata::Strata;
use mc2_worldgen::terrain::{CoarseTerrain, TerrainParams};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

const SEED: u64 = 42;
const VOXEL_M: f32 = 1.0 / 16.0;

fn main() {
    let dir = PathBuf::from("worlds/shots");
    std::fs::create_dir_all(&dir).expect("world dir");
    let path = dir.join(format!("terrain_{SEED}.bin"));
    let t = std::time::Instant::now();
    let terrain = load_or_generate(&path, SEED, TerrainParams::world(), &mut |stage, f| {
        print!("\r{stage} {:>3.0}%", f * 100.0);
        let _ = std::io::stdout().flush();
    })
    .expect("terrain");
    println!(
        "\nterrain in {:.1}s, cache {}",
        t.elapsed().as_secs_f32(),
        path.display()
    );
    let terrain = Arc::new(terrain);
    let surface = Surface::new(terrain.clone());
    let strata = Strata::new(SEED);
    let karst = Karst::new(SEED);
    let sea = terrain.params.sea_level;
    let extent = terrain.params.extent_m();

    let mouth = find_mouth(&surface, &strata, &karst, sea, extent);
    println!(
        "mouth at {:.1} {:.1} {:.1} slope {:.2}",
        mouth.0, mouth.1, mouth.2, mouth.3
    );
    let (stand, look) = outside_mouth(&surface, mouth);
    print_cam("cave-mouth", stand, look);

    let mut generator = ChunkGenerator::new(terrain.clone());
    generator.flora = false;
    if let Some(inside) = find_inside(&generator, &surface, mouth) {
        let look = (mouth.0 as f64, inside.1, mouth.2 as f64);
        print_cam("cave-inside", inside, look);
    } else {
        eprintln!("no dry interior found near the mouth");
    }

    let (stand, look) = range_view(&terrain, &surface, sea, extent);
    print_cam("range", stand, look);

    let (stand, look) = north_face(&surface, sea, extent);
    print_cam("north-face", stand, look);
}

fn print_cam(name: &str, eye: (f64, f64, f64), look: (f64, f64, f64)) {
    println!(
        "{name} --camera {eye_x:.2},{eye_y:.2},{eye_z:.2},{lx:.2},{ly:.2},{lz:.2}",
        eye_x = eye.0,
        eye_y = eye.1,
        eye_z = eye.2,
        lx = look.0,
        ly = look.1,
        lz = look.2
    );
}

/// (x, surface height, z, slope) of a steep carbonate joint that daylights.
fn find_mouth(
    surface: &Surface,
    strata: &Strata,
    karst: &Karst,
    sea: f32,
    extent: f32,
) -> (f32, f32, f32, f32) {
    let mut best = None;
    let mut score_best = 0.0f32;
    let mut x = 2500.0f32;
    while x < extent - 2500.0 {
        let mut z = 2500.0f32;
        while z < extent - 2500.0 {
            let s = surface.sample(x, z);
            if s.slope > 0.5 && s.height > sea + 12.0 && s.height < 280.0 {
                let y = s.height - 1.4;
                let host = strata.layer(x, y, z).material;
                let open = karst.openness(x, y, z, s.height, s.slope);
                if soluble(host) && open > 0.5 {
                    let score = open * s.slope;
                    if score > score_best {
                        score_best = score;
                        best = Some((x, s.height, z, s.slope));
                    }
                }
            }
            z += 48.0;
        }
        x += 48.0;
    }
    best.expect("no hillside cave on this seed")
}

fn outside_mouth(
    surface: &Surface,
    mouth: (f32, f32, f32, f32),
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let (x, h, z, _) = mouth;
    let dx = surface.sample(x + 4.0, z).height - surface.sample(x - 4.0, z).height;
    let dz = surface.sample(x, z + 4.0).height - surface.sample(x, z - 4.0).height;
    let len = (dx * dx + dz * dz).sqrt().max(0.2);
    // Stand downhill, looking back into the hill.
    let (ux, uz) = (dx / len, dz / len);
    let sx = x - ux * 7.0;
    let sz = z - uz * 7.0;
    let sh = surface.sample(sx, sz).height;
    let eye = (sx as f64, (sh + 1.7) as f64, sz as f64);
    let look = (x as f64, (h - 1.2) as f64, z as f64);
    (eye, look)
}

fn find_inside(
    generator: &ChunkGenerator,
    surface: &Surface,
    mouth: (f32, f32, f32, f32),
) -> Option<(f64, f64, f64)> {
    use std::collections::HashMap;
    let (x, _, z, _) = mouth;
    let dx = surface.sample(x + 4.0, z).height - surface.sample(x - 4.0, z).height;
    let dz = surface.sample(x, z + 4.0).height - surface.sample(x, z - 4.0).height;
    let len = (dx * dx + dz * dz).sqrt().max(0.2);
    let (ux, uz) = (dx / len, dz / len);
    let mut trees = HashMap::<ChunkPos, mc2_voxel::tree::ChunkTree>::new();
    // Step uphill into the hill, a few metres under the roof.
    for step in 2..14 {
        let px = x + ux * step as f32 * 1.5;
        let pz = z + uz * step as f32 * 1.5;
        let ground = surface.sample(px, pz).height;
        for down in [2.0f32, 4.0, 6.0, 9.0] {
            let y = ground - down;
            let vx = (px / VOXEL_M) as i32;
            let vy = (y / VOXEL_M) as i32;
            let vz = (pz / VOXEL_M) as i32;
            let at = glam::IVec3::new(vx, vy, vz);
            let pos = ChunkPos::of_voxel(at);
            let tree = trees
                .entry(pos)
                .or_insert_with(|| generator.generate(pos, Lod::Full));
            let local = at - pos.origin();
            if !tree.voxel(local).is_air() {
                continue;
            }
            let clear = [-1, 0, 1].iter().all(|&oy| {
                let p = local + glam::IVec3::Y * oy;
                (0..512).contains(&p.y) && tree.voxel(p).is_air()
            });
            let dry = tree.voxel(local - glam::IVec3::Y) != ids::WATER;
            if clear && dry {
                let eye_y = (vy as f32 + 0.5) * VOXEL_M;
                return Some((
                    (vx as f32 + 0.5) as f64 * f64::from(VOXEL_M),
                    f64::from(eye_y),
                    (vz as f32 + 0.5) as f64 * f64::from(VOXEL_M),
                ));
            }
        }
    }
    None
}

fn range_view(
    terrain: &CoarseTerrain,
    surface: &Surface,
    sea: f32,
    extent: f32,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let n = terrain.params.size;
    let mut peak = (0.0f32, sea, 0.0f32);
    for j in 40..n - 40 {
        for i in 40..n - 40 {
            let h = terrain.height.get(i, j);
            if h > peak.1 {
                let x = (i as f32 + 0.5) * terrain.params.cell_m;
                let z = (j as f32 + 0.5) * terrain.params.cell_m;
                peak = (x, h, z);
            }
        }
    }
    // Stand a kilometre toward the map centre, so the shot isn't the world edge.
    let centre = extent * 0.5;
    let mut dir = (centre - peak.0, centre - peak.2);
    let len = (dir.0 * dir.0 + dir.1 * dir.1).sqrt().max(1.0);
    dir = (dir.0 / len, dir.1 / len);
    let sx = peak.0 + dir.0 * 900.0;
    let sz = peak.2 + dir.1 * 900.0;
    let sh = surface.sample(sx, sz).height;
    let eye = (sx as f64, (sh + 8.0) as f64, sz as f64);
    let look = (peak.0 as f64, (peak.1 - 40.0) as f64, peak.2 as f64);
    (eye, look)
}

fn north_face(surface: &Surface, sea: f32, extent: f32) -> ((f64, f64, f64), (f64, f64, f64)) {
    let mut best = None;
    let mut score = 0.0f32;
    let mut x = 3000.0f32;
    while x < extent - 3000.0 {
        let mut z = 3000.0f32;
        while z < extent - 3000.0 {
            let s = surface.sample(x, z);
            if s.aspect > 0.45 && s.slope > 0.35 && s.height > sea + 40.0 && s.height < 420.0 {
                let here = s.aspect * s.slope * (1.0 - (s.height - 340.0).abs() / 200.0);
                if here > score {
                    score = here;
                    best = Some((x, s.height, z));
                }
            }
            z += 64.0;
        }
        x += 64.0;
    }
    let (x, h, z) = best.expect("no north face");
    // +z is north. Stand north of the face and look south at it.
    let sx = x;
    let sz = z + 28.0;
    let sh = surface.sample(sx, sz).height;
    let eye = (sx as f64, (sh.max(h - 6.0) + 2.0) as f64, sz as f64);
    let look = (x as f64, (h + 4.0) as f64, z as f64);
    (eye, look)
}
