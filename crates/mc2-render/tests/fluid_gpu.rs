//! The GPU fluid must agree with the CPU reference in `mc2-fluid`, step for
//! step: the same passes, the same tiles allocated after each step.

use glam::IVec3;
use mc2_fluid::{FluidWorld, Params, TILE, Terrain};
use mc2_gpu::TimestampScope;
use mc2_render::fluid::FluidGpu;

/// A basin: floor at y = 0, walls around x and z in `0..=size`.
struct Basin {
    size: IVec3,
    extra: Vec<IVec3>,
}

impl Terrain for Basin {
    fn solid(&self, c: IVec3) -> bool {
        c.y <= 0
            || c.x <= 0
            || c.z <= 0
            || c.x >= self.size.x
            || c.z >= self.size.z
            || self.extra.contains(&c)
    }
}

fn cells(lo: IVec3, hi: IVec3) -> Vec<IVec3> {
    let mut v = Vec::new();
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                v.push(IVec3::new(x, y, z));
            }
        }
    }
    v
}

/// Water column heights (cells) by (x, z) over the basin, and total mass.
struct Heights {
    h: Vec<f32>,
    mass: f64,
}

fn cpu_heights(w: &FluidWorld, size: IVec3) -> Heights {
    let mut h = Vec::new();
    for z in 1..size.z {
        for x in 1..size.x {
            h.push((1..24).map(|y| w.fill(IVec3::new(x, y, z))).sum());
        }
    }
    Heights { h, mass: w.mass() }
}

fn gpu_heights(f: &FluidGpu, gpu: &mc2_gpu::Gpu, size: IVec3) -> Heights {
    let tiles = f.read_tiles(&gpu.device, &gpu.queue);
    let mut h = vec![0.0; ((size.x - 1) * (size.z - 1)) as usize];
    let mut mass = 0.0f64;
    for t in &tiles {
        for i in 0..mc2_fluid::TILE_CELLS {
            let c = t.pos * TILE + mc2_fluid::local_of(i);
            mass += f64::from(t.water(i));
            if c.x >= 1 && c.x < size.x && c.z >= 1 && c.z < size.z && c.y >= 1 {
                h[((c.z - 1) * (size.x - 1) + c.x - 1) as usize] += t.fill(i);
            }
        }
    }
    Heights { h, mass }
}

fn run(terrain: &Basin, water: Vec<IVec3>, steps: u32, check_every: u32) {
    let Some(gpu) = mc2_gpu::device::test_gpu() else {
        return;
    };
    let shaders = mc2_render::shaders::library();
    let mut gpu_fluid = FluidGpu::new(&gpu.device, &shaders, 64, Params::default());
    let mut cpu_fluid = FluidWorld::default();
    gpu_fluid.add_water(&water, terrain);
    cpu_fluid.add_water(water, terrain);
    for step in 1..=steps {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        gpu_fluid.encode(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &shaders,
            1,
            &TimestampScope::none(),
        );
        gpu.queue.submit([encoder.finish()]);
        gpu_fluid.after_submit();
        gpu_fluid.wait(&gpu.device);
        gpu_fluid.maintain(terrain);
        cpu_fluid.step(terrain);
        if step % check_every != 0 {
            continue;
        }
        let (c, g) = (
            cpu_heights(&cpu_fluid, terrain.size),
            gpu_heights(&gpu_fluid, &gpu, terrain.size),
        );
        let mass_error = (c.mass - g.mass).abs() / c.mass;
        let diffs = || c.h.iter().zip(&g.h).map(|(a, b)| (a - b).abs());
        let mean = diffs().sum::<f32>() / c.h.len() as f32;
        let worst = diffs().fold(0.0f32, f32::max);
        eprintln!(
            "step {step}: mass cpu {:.3} gpu {:.3}; column heights differ by {mean:.1e} mean,              {worst:.1e} worst; tiles {} / {}",
            c.mass,
            g.mass,
            cpu_fluid.stats.tiles,
            gpu_fluid.tiles()
        );
        assert!(
            mass_error < 1e-4,
            "step {step}: mass {} vs {}",
            c.mass,
            g.mass
        );
        // Only rounding may differ; it grows slowly as the water sloshes.
        assert!(
            mean < 1e-3 && worst < 1e-2,
            "step {step}: heights differ by {mean} / {worst}"
        );
        assert_eq!(cpu_fluid.stats.tiles, gpu_fluid.tiles(), "step {step}");
    }
}

#[test]
fn gpu_still_water_matches_cpu() {
    let terrain = Basin {
        size: IVec3::new(9, 0, 9),
        extra: Vec::new(),
    };
    run(
        &terrain,
        cells(IVec3::new(1, 1, 1), IVec3::new(8, 3, 8)),
        60,
        20,
    );
}

#[test]
fn gpu_spreading_column_matches_cpu() {
    // A column across a tile edge, spreading over a basin two tiles wide
    // that needs tiles allocated as it goes.
    let terrain = Basin {
        size: IVec3::new(13, 0, 13),
        extra: Vec::new(),
    };
    run(
        &terrain,
        cells(IVec3::new(5, 1, 5), IVec3::new(8, 9, 8)),
        240,
        40,
    );
}
