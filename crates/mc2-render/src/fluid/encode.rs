//! Recording a frame of fluid work: this frame's uploads, the edits, then
//! each step as one compute pass; and harvesting the reports that come
//! back.

use super::{
    ARGS_BYTES, CELLS, FluidGpu, GpuParams, MAP_DONE, MAP_FREE, MAP_PENDING, PARTS, Q,
    REPORT_BYTES, TileCells, TileReport,
};
use mc2_fluid::{SLEEP_STEPS, STILL_SPEED, TILE_CELLS};
use mc2_gpu::{ShaderLibrary, TimestampScope, groups};
use std::sync::atomic::Ordering;

/// Pass `index` of `total`, the first carrying the begin timestamp and the
/// last the end.
fn begin<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    scope: &'e TimestampScope,
    index: &mut u32,
    total: u32,
) -> wgpu::ComputePass<'e> {
    let first = *index == 0;
    *index += 1;
    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("sim.fluid"),
        timestamp_writes: scope.compute_part(first, *index == total),
    })
}

impl FluidGpu {
    fn gpu_params(&self, parity: u32) -> GpuParams {
        let p = &self.params;
        GpuParams {
            gravity: p.gravity.to_array(),
            tau: p.tau,
            smagorinsky: p.smagorinsky,
            rho_gas: p.rho_gas,
            fill_slack: p.fill_slack,
            wall_slip: p.wall_slip,
            parity,
            slot_count: self.slot_count,
            max_slots: self.max_slots,
            sleep_steps: SLEEP_STEPS,
            still_speed: STILL_SPEED,
            edit_count: self.edits.len() as u32,
            touched_count: self.touched.len() as u32,
            max_speed: super::MAX_SPEED,
        }
    }

    /// Writes this frame's table changes, tile states, edits and params.
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let b = &self.buffers;
        for (s, kinds) in &self.fresh {
            let cell = u64::from(*s) * CELLS;
            let tile = u64::from(*s) * CELLS * 4;
            for pops in &b.pops {
                encoder.clear_buffer(pops, tile * Q, Some(CELLS * Q * 4));
            }
            for buf in [&b.mass, &b.conv, &b.excess, &b.massex] {
                encoder.clear_buffer(buf, tile, Some(CELLS * 4));
            }
            encoder.clear_buffer(&b.rho_u, cell * 16, Some(CELLS * 16));
            let half = u64::from(self.max_slots) * CELLS * 4;
            encoder.clear_buffer(&b.fill, tile, Some(CELLS * 4));
            encoder.clear_buffer(&b.fill, half + tile, Some(CELLS * 4));
            // A cell's `next` starts as its kind with no transition.
            queue.write_buffer(&b.kind, tile, bytemuck::cast_slice(kinds));
            queue.write_buffer(&b.next, tile, bytemuck::cast_slice(kinds));
        }
        for &s in &self.dirty_rows {
            let row = &self.near[s as usize * 27..(s as usize + 1) * 27];
            queue.write_buffer(&b.near, u64::from(s) * 27 * 4, bytemuck::cast_slice(row));
        }
        for (s, state) in &self.states {
            queue.write_buffer(
                &b.tile_state,
                u64::from(*s) * REPORT_BYTES,
                bytemuck::bytes_of(state),
            );
        }
        // Woken tiles: stillness back to zero (the third word).
        for &s in &self.woken {
            if self.slots[s as usize].is_some() {
                queue.write_buffer(
                    &b.tile_state,
                    u64::from(s) * REPORT_BYTES + 8,
                    &0u32.to_le_bytes(),
                );
            }
        }
        let touched: Vec<[u32; 4]> = self.touched.iter().map(|&s| [s, 0, 0, 0]).collect();
        let entries = (self.edits.len() + touched.len()) as u64;
        if entries > self.edit_capacity {
            self.edit_capacity = entries.next_power_of_two();
            self.buffers.edits = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fluid edits"),
                size: self.edit_capacity * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.groups = Self::bind_groups(device, &self.layout, &self.buffers);
        }
        let b = &self.buffers;
        if !self.edits.is_empty() {
            queue.write_buffer(&b.edits, 0, bytemuck::cast_slice(&self.edits));
        }
        if !touched.is_empty() {
            queue.write_buffer(
                &b.edits,
                self.edits.len() as u64 * 16,
                bytemuck::cast_slice(&touched),
            );
        }
        for p in 0..2 {
            queue.write_buffer(
                &b.params[p],
                0,
                bytemuck::bytes_of(&self.gpu_params(p as u32)),
            );
        }
    }

    /// Records this frame's uploads and edits and `steps` lattice steps,
    /// timed as one scope, then copies the report out.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        shaders: &ShaderLibrary,
        steps: u32,
        scope: &TimestampScope,
    ) {
        if !self.started {
            // Indirect y and z never change: 8 workgroups per tile.
            let args: [u32; 8] = [0, PARTS, 1, 0, PARTS, 1, 0, 0];
            queue.write_buffer(&self.buffers.args, 0, bytemuck::cast_slice(&args));
            queue.write_buffer(&self.buffers.indirect, 0, bytemuck::cast_slice(&args));
            self.started = true;
        }
        self.upload(device, queue, encoder);
        let edited = !self.edits.is_empty() || !self.touched.is_empty() || !self.woken.is_empty();
        let total = steps + u32::from(edited);
        let slot_groups = groups(self.slot_count.max(1), 64);
        let mut index = 0;
        let p = &mut self.pipelines;
        let b = &self.buffers;
        if edited {
            let mut pass = begin(encoder, scope, &mut index, total);
            pass.set_bind_group(0, &self.groups[self.parity as usize], &[]);
            if !self.edits.is_empty() {
                pass.set_pipeline(p.edit.get(device, shaders));
                pass.dispatch_workgroups(groups(self.edits.len() as u32, 64), 1, 1);
            }
            if !self.touched.is_empty() {
                pass.set_pipeline(p.close.get(device, shaders));
                pass.dispatch_workgroups(self.touched.len() as u32, PARTS, 1);
            }
            pass.set_pipeline(p.reset_active.get(device, shaders));
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(p.wake.get(device, shaders));
            pass.dispatch_workgroups(slot_groups, 1, 1);
            drop(pass);
            encoder.copy_buffer_to_buffer(&b.args, 0, &b.indirect, 0, 24);
        }
        for _ in 0..steps {
            let mut pass = begin(encoder, scope, &mut index, total);
            pass.set_bind_group(0, &self.groups[self.parity as usize], &[]);
            pass.set_pipeline(p.copy_asleep.get(device, shaders));
            pass.dispatch_workgroups_indirect(&b.indirect, 12);
            for pipeline in [
                &mut p.stream,
                &mut p.flag,
                &mut p.apply,
                &mut p.share,
                &mut p.gather,
            ] {
                pass.set_pipeline(pipeline.get(device, shaders));
                pass.dispatch_workgroups_indirect(&b.indirect, 0);
            }
            pass.set_pipeline(p.reset_lists.get(device, shaders));
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(p.activity.get(device, shaders));
            pass.dispatch_workgroups(slot_groups, 1, 1);
            drop(pass);
            encoder.copy_buffer_to_buffer(&b.args, 0, &b.indirect, 0, 24);
            self.parity ^= 1;
        }
        self.stats.steps += u64::from(steps);
        if let Some(st) = self
            .staging
            .iter_mut()
            .find(|s| s.state.load(Ordering::Acquire) == MAP_FREE)
        {
            let states = u64::from(self.max_slots) * REPORT_BYTES;
            encoder.copy_buffer_to_buffer(
                &b.tile_state,
                0,
                &st.buffer,
                0,
                u64::from(self.slot_count.max(1)) * REPORT_BYTES,
            );
            encoder.copy_buffer_to_buffer(&b.args, 0, &st.buffer, states, ARGS_BYTES);
            st.state.store(MAP_PENDING, Ordering::Release);
            st.recorded = true;
        }
        self.fresh.clear();
        self.dirty_rows.clear();
        self.states.clear();
        self.woken.clear();
        self.edits.clear();
        self.touched.clear();
    }

    /// Starts mapping the report recorded this frame; call after submit.
    pub fn after_submit(&mut self) {
        for st in &mut self.staging {
            if !st.recorded {
                continue;
            }
            st.recorded = false;
            let state = st.state.clone();
            st.buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    state.store(
                        if r.is_ok() { MAP_DONE } else { MAP_FREE },
                        Ordering::Release,
                    );
                });
        }
    }

    /// Takes in every report that has arrived (the newest wins).
    pub fn harvest(&mut self) {
        let states = u64::from(self.max_slots) * REPORT_BYTES;
        for i in 0..self.staging.len() {
            if self.staging[i].state.load(Ordering::Acquire) != MAP_DONE {
                continue;
            }
            let (reports, args) = {
                let view = self.staging[i]
                    .buffer
                    .slice(..)
                    .get_mapped_range()
                    .expect("fluid report mapped");
                let reports: Vec<TileReport> = bytemuck::cast_slice(
                    &view[..(u64::from(self.slot_count) * REPORT_BYTES) as usize],
                )
                .to_vec();
                let args: [u32; 8] = bytemuck::pod_read_unaligned(
                    &view[states as usize..(states + ARGS_BYTES) as usize],
                );
                (reports, args)
            };
            self.staging[i].buffer.unmap();
            self.staging[i].state.store(MAP_FREE, Ordering::Release);
            for (s, r) in reports.iter().enumerate() {
                if self.slots[s].is_none() || r.alloc == 0 {
                    continue;
                }
                self.water[s] = r.water != 0;
                self.still[s] = r.still;
                self.need[s] = r.need[0] | r.need[1];
            }
            self.stats.awake = args[0];
            self.stats.lost_mass = f64::from(args[6] as i32) / 65536.0;
        }
        self.stats.tiles = self.index.len();
    }

    /// Waits for every report in flight and takes them in (tests, tools).
    pub fn wait(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        self.harvest();
    }

    /// Reads back every allocated tile's cells. Blocks.
    pub fn read_tiles(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<TileCells> {
        let n = u64::from(self.slot_count.max(1)) * CELLS;
        let read = |buf: &wgpu::Buffer, bytes: u64| -> Vec<u8> {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fluid read"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
            queue.submit([encoder.finish()]);
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let data = staging
                .slice(..)
                .get_mapped_range()
                .expect("fluid read mapped")
                .to_vec();
            staging.unmap();
            data
        };
        let kind: Vec<u32> = bytemuck::cast_slice(&read(&self.buffers.kind, n * 4)).to_vec();
        let mass: Vec<f32> = bytemuck::cast_slice(&read(&self.buffers.mass, n * 4)).to_vec();
        let rho_u: Vec<[f32; 4]> =
            bytemuck::cast_slice(&read(&self.buffers.rho_u, n * 16)).to_vec();
        let t = TILE_CELLS;
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(s, pos)| {
                pos.map(|pos| TileCells {
                    pos,
                    kind: kind[s * t..(s + 1) * t].to_vec(),
                    mass: mass[s * t..(s + 1) * t].to_vec(),
                    rho_u: rho_u[s * t..(s + 1) * t].to_vec(),
                })
            })
            .collect()
    }
}
