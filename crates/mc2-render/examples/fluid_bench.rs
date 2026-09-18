//! GPU fluid timing: a block of water 16 m by 32 m and 12 m deep collapses
//! into a basin 64 m by 32 m, four lattice steps (a 60 Hz frame) at a time.
//! Prints the GPU time of the fluid scope, the CPU time spent on the tile
//! table and recording, how many tiles exist and step, and every two
//! seconds the water's mass against the start (read back, so the run
//! stalls for it).
//!
//! cargo run -p mc2-render --release --example fluid_bench

use glam::IVec3;
use mc2_core::RollingStats;
use mc2_fluid::{Params, Terrain};
use mc2_gpu::{Gpu, GpuOptions, GpuProfiler};
use mc2_render::fluid::FluidGpu;
use std::time::Instant;

const SIZE: IVec3 = IVec3::new(129, 0, 65);

struct Basin;

impl Terrain for Basin {
    fn solid(&self, c: IVec3) -> bool {
        c.y <= 0 || c.x <= 0 || c.z <= 0 || c.x >= SIZE.x || c.z >= SIZE.z
    }
}

fn main() {
    let gpu = Gpu::headless(&GpuOptions::default()).expect("gpu");
    println!("adapter: {}", gpu.describe());
    let shaders = mc2_render::shaders::library();
    let mut profiler = GpuProfiler::new(&gpu.device, &gpu.queue, gpu.timestamps);
    let mut fluid = FluidGpu::new(&gpu.device, &shaders, 4096, Params::default());
    let mut water = Vec::new();
    for z in 1..SIZE.z {
        for y in 1..=24 {
            for x in 1..=32 {
                water.push(IVec3::new(x, y, z));
            }
        }
    }
    fluid.add_water(&water, &Basin);
    let mass = |fluid: &FluidGpu| -> f64 {
        let tiles = fluid.read_tiles(&gpu.device, &gpu.queue);
        tiles
            .iter()
            .map(|t| {
                (0..t.kind.len())
                    .map(|i| f64::from(t.water(i)))
                    .sum::<f64>()
            })
            .sum()
    };
    let mut start_mass = None;
    let mut cpu = RollingStats::default();
    for frame in 1..=720u32 {
        let start = Instant::now();
        fluid.maintain(&Basin);
        profiler.begin_frame();
        let scope = profiler.scope("sim.fluid");
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        fluid.encode(&gpu.device, &gpu.queue, &mut encoder, &shaders, 4, &scope);
        profiler.end_frame(&mut encoder);
        cpu.push(start.elapsed().as_secs_f32() * 1000.0);
        gpu.queue.submit([encoder.finish()]);
        profiler.after_submit();
        fluid.after_submit();
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        fluid.harvest();
        if frame % 60 == 0 {
            let row = profiler.rows().iter().find(|r| r.name == "sim.fluid");
            let (last, mean, p99) = row.map_or((0.0, 0.0, 0.0), |r| {
                (r.stats.last(), r.stats.mean(), r.stats.p99())
            });
            println!(
                "t {:>4.1} s  gpu {last:>6.3} ms (mean {mean:.3}, p99 {p99:.3})  cpu {:.3} ms  tiles {:>4}  awake {:>4}  lost {:.3}",
                f64::from(frame) / 60.0,
                cpu.mean(),
                fluid.stats.tiles,
                fluid.stats.awake,
                fluid.stats.lost_mass,
            );
        }
        if frame == 1 || frame % 120 == 0 {
            let m = mass(&fluid);
            let m0 = *start_mass.get_or_insert(m);
            println!(
                "  mass {m:.1}, {:+.2}% since the start",
                (m - m0) / m0 * 100.0
            );
        }
    }
}
