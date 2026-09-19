//! The water simulation in the window and in headless captures: the GPU
//! lattice stepped at 240 Hz, fed the world's edits and new water, its
//! levels written back into the voxel world as they arrive.

use mc2_game::water::{Water, WorldTerrain};
use mc2_game::{Game, Streaming, Voxels};
use mc2_gpu::{Gpu, ShaderLibrary, TimestampScope};
use mc2_render::fluid::{FluidGpu, Params, STEP_S};

/// Tiles the pool holds, about 100 KB each on the GPU.
const MAX_TILES: u32 = 4096;
/// Lattice steps a frame at most, so a stall does not spiral.
const MAX_STEPS: u32 = 8;

pub struct WaterSim {
    fluid: FluidGpu,
    /// Simulated seconds owed to the lattice.
    owed: f64,
}

impl WaterSim {
    pub fn new(gpu: &Gpu, shaders: &ShaderLibrary) -> Self {
        Self {
            fluid: FluidGpu::new(&gpu.device, shaders, MAX_TILES, Params::default()),
            owed: 0.0,
        }
    }

    /// One frame: new water and terrain edits to the lattice, the steps
    /// `dt` owes it, and the levels that have come back into the world.
    pub fn frame(&mut self, gpu: &Gpu, shaders: &ShaderLibrary, game: &mut Game, dt: f32) {
        mc2_core::scope!("water.frame");
        let (add, changed) = {
            let mut w = game.world.resource_mut::<Water>();
            w.tick();
            (std::mem::take(&mut w.add), std::mem::take(&mut w.changed))
        };
        {
            let voxels = game.world.resource::<Voxels>();
            let water = game.world.resource::<Water>();
            let terrain = WorldTerrain {
                world: &voxels.0,
                water,
            };
            if !add.is_empty() {
                self.fluid.add_water(&add, &terrain);
            }
            for (lo, hi) in changed {
                self.fluid.terrain_changed(lo, hi, &terrain);
            }
            self.fluid.maintain(&terrain);
        }
        if self.fluid.tiles() == 0 {
            self.owed = 0.0;
            return;
        }
        self.owed += f64::from(dt);
        let steps = ((self.owed / STEP_S) as u32).min(MAX_STEPS);
        self.owed = (self.owed - f64::from(steps) * STEP_S).min(STEP_S * f64::from(MAX_STEPS));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.fluid.encode(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            shaders,
            steps,
            &TimestampScope::none(),
        );
        self.fluid
            .encode_levels(&gpu.device, &gpu.queue, &mut encoder, shaders);
        gpu.queue.submit([encoder.finish()]);
        self.fluid.after_submit();
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        self.fluid.harvest();
        if let Some(levels) = self.fluid.take_levels() {
            mc2_core::scope!("water.apply");
            game.world
                .resource_scope(|world, mut water: bevy_ecs::prelude::Mut<Water>| {
                    world.resource_scope(
                        |world, mut streaming: bevy_ecs::prelude::Mut<Streaming>| {
                            let mut voxels = world.resource_mut::<Voxels>();
                            water.cells_written = 0;
                            for t in &levels {
                                water.apply(&mut voxels.0, streaming.0.as_mut(), t.pos, &t.eighths);
                            }
                        },
                    );
                });
        }
    }

    /// One line for the profiler overlay.
    pub fn line(&self, game: &Game) -> String {
        let s = &self.fluid.stats;
        format!(
            "water: {} tiles, {} awake, {} cells written, {} steps",
            s.tiles,
            s.awake,
            game.world.resource::<Water>().cells_written,
            s.steps
        )
    }
}
