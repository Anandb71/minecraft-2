//! Water levels read back for the world: every allocated tile's cells as
//! eighths of their height, packed on the GPU, copied out and mapped a
//! frame or two later. The game writes them into the voxel world, so the
//! water it draws is the water that flows.

use super::{CELLS, FluidGpu, MAP_DONE, MAP_FREE, MAP_PENDING};
use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use mc2_gpu::{HotCompute, ShaderLibrary, bind, bind_group, groups, layout};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

const STAGING: usize = 3;
/// Bytes of levels per tile: a byte a cell.
const TILE_BYTES: u64 = CELLS;

/// One tile's water: `eighths[i]` is how many eighths of cell `i` (in
/// `mc2_fluid::index_of` order) hold water.
pub struct TileLevels {
    pub pos: IVec3,
    pub eighths: Vec<u8>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LevelParams {
    slot_count: u32,
    _pad: [u32; 3],
}

struct Readback {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    recorded: bool,
    /// Order of recording, so the newest of several arrivals wins.
    seq: u64,
    /// Which tile each slot held when the copy was recorded.
    slots: Vec<Option<IVec3>>,
}

pub(super) struct Levels {
    params: wgpu::Buffer,
    packed: wgpu::Buffer,
    group: wgpu::BindGroup,
    pipeline: HotCompute,
    readbacks: Vec<Readback>,
    recorded: u64,
}

impl Levels {
    fn new(device: &wgpu::Device, shaders: &ShaderLibrary, gpu: &FluidGpu) -> Self {
        let bytes = u64::from(gpu.max_slots) * TILE_BYTES;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid level params"),
            size: std::mem::size_of::<LevelParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let packed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid levels"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let layout = layout(
            device,
            "fluid levels",
            wgpu::ShaderStages::COMPUTE,
            &[
                bind::uniform(),
                bind::storage(true),
                bind::storage(true),
                bind::storage(true),
                bind::storage(false),
            ],
        );
        let b = &gpu.buffers;
        let group = bind_group(
            device,
            "fluid levels",
            &layout,
            &[
                params.as_entire_binding(),
                b.kind.as_entire_binding(),
                b.mass.as_entire_binding(),
                b.rho_u.as_entire_binding(),
                packed.as_entire_binding(),
            ],
        );
        let pipeline = HotCompute::new(
            device,
            shaders,
            "fluid_levels.wgsl",
            "pack_levels",
            &[&layout],
        );
        let readbacks = (0..STAGING)
            .map(|_| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fluid levels readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(MAP_FREE)),
                recorded: false,
                seq: 0,
                slots: Vec::new(),
            })
            .collect();
        Self {
            params,
            packed,
            group,
            pipeline,
            readbacks,
            recorded: 0,
        }
    }
}

impl FluidGpu {
    /// Records packing every allocated tile's levels and copying them out,
    /// when a readback buffer is free. Call after `encode`, in the same
    /// encoder, and `after_submit` once it is submitted.
    pub fn encode_levels(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        shaders: &ShaderLibrary,
    ) {
        if self.levels.is_none() {
            self.levels = Some(Levels::new(device, shaders, self));
        }
        let slot_count = self.slot_count;
        let slots = self.slots[..slot_count as usize].to_vec();
        let Some(lv) = self.levels.as_mut() else {
            return;
        };
        let Some(rb) = lv
            .readbacks
            .iter_mut()
            .find(|r| r.state.load(Ordering::Acquire) == MAP_FREE)
        else {
            return;
        };
        if slot_count == 0 {
            return;
        }
        queue.write_buffer(
            &lv.params,
            0,
            bytemuck::bytes_of(&LevelParams {
                slot_count,
                _pad: [0; 3],
            }),
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sim.fluid.levels"),
                timestamp_writes: None,
            });
            pass.set_pipeline(lv.pipeline.get(device, shaders));
            pass.set_bind_group(0, &lv.group, &[]);
            let words = slot_count * (CELLS as u32 / 4);
            pass.dispatch_workgroups(groups(words, 64), 1, 1);
        }
        let bytes = u64::from(slot_count) * TILE_BYTES;
        encoder.copy_buffer_to_buffer(&lv.packed, 0, &rb.buffer, 0, bytes);
        rb.state.store(MAP_PENDING, Ordering::Release);
        rb.recorded = true;
        rb.slots = slots;
        lv.recorded += 1;
        rb.seq = lv.recorded;
    }

    /// Starts mapping the levels recorded this frame (from `after_submit`).
    pub(super) fn map_levels(&mut self) {
        let Some(lv) = self.levels.as_mut() else {
            return;
        };
        for rb in &mut lv.readbacks {
            if !rb.recorded {
                continue;
            }
            rb.recorded = false;
            let state = rb.state.clone();
            let bytes = rb.slots.len() as u64 * TILE_BYTES;
            rb.buffer
                .slice(..bytes)
                .map_async(wgpu::MapMode::Read, move |r| {
                    state.store(
                        if r.is_ok() { MAP_DONE } else { MAP_FREE },
                        Ordering::Release,
                    );
                });
        }
    }

    /// The newest levels to have arrived since the last call, if any.
    pub fn take_levels(&mut self) -> Option<Vec<TileLevels>> {
        let lv = self.levels.as_mut()?;
        let mut newest: Option<(u64, Vec<TileLevels>)> = None;
        for rb in &mut lv.readbacks {
            if rb.state.load(Ordering::Acquire) != MAP_DONE {
                continue;
            }
            let bytes = rb.slots.len() as u64 * TILE_BYTES;
            let tiles = {
                let view = rb
                    .buffer
                    .slice(..bytes)
                    .get_mapped_range()
                    .expect("fluid levels mapped");
                rb.slots
                    .iter()
                    .enumerate()
                    .filter_map(|(s, pos)| {
                        let at = s * TILE_BYTES as usize;
                        pos.map(|pos| TileLevels {
                            pos,
                            eighths: view[at..at + TILE_BYTES as usize].to_vec(),
                        })
                    })
                    .collect()
            };
            rb.buffer.unmap();
            rb.state.store(MAP_FREE, Ordering::Release);
            if newest.as_ref().is_none_or(|(seq, _)| rb.seq > *seq) {
                newest = Some((rb.seq, tiles));
            }
        }
        newest.map(|(_, tiles)| tiles)
    }
}
