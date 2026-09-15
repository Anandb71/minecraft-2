//! GPU timestamp profiler.
//!
//! Every frame graph pass asks for a [`TimestampScope`]; its begin and end
//! timestamps land in one shared query set. At the end of the frame the set
//! is resolved into a staging buffer taken from a small ring. Buffers are
//! mapped asynchronously and harvested frames later, so the profiler never
//! stalls the GPU. If every staging buffer is still in flight the frame goes
//! untimed rather than blocking.

use mc2_core::RollingStats;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

const MAX_SCOPES: u32 = 64;
const RING: usize = 4;

const SLOT_FREE: u8 = 0;
const SLOT_PENDING: u8 = 1;
const SLOT_MAPPED: u8 = 2;
const SLOT_FAILED: u8 = 3;

struct Slot {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    names: Vec<&'static str>,
    /// Resolve was recorded into this frame's encoder and awaits `after_submit`.
    needs_map: bool,
}

#[derive(Clone, Debug)]
pub struct GpuRow {
    pub name: &'static str,
    pub stats: RollingStats,
}

/// Query indices for one pass. Cheap to clone; holds an Arc'd query set.
#[derive(Clone)]
pub struct TimestampScope {
    query_set: Option<wgpu::QuerySet>,
    begin: u32,
}

impl TimestampScope {
    pub fn none() -> Self {
        Self {
            query_set: None,
            begin: 0,
        }
    }

    pub fn compute(&self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.query_set
            .as_ref()
            .map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(self.begin),
                end_of_pass_write_index: Some(self.begin + 1),
            })
    }

    pub fn render(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.query_set
            .as_ref()
            .map(|qs| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(self.begin),
                end_of_pass_write_index: Some(self.begin + 1),
            })
    }
}

pub struct GpuProfiler {
    query_set: Option<wgpu::QuerySet>,
    resolve: Option<wgpu::Buffer>,
    slots: Vec<Slot>,
    frame_names: Vec<&'static str>,
    frame_active: bool,
    rows: Vec<GpuRow>,
    period_ns: f32,
    frame_total: RollingStats,
}

impl GpuProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, enabled: bool) -> Self {
        let bytes = u64::from(MAX_SCOPES) * 2 * 8;
        let (query_set, resolve, slots) = if enabled {
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gpu profiler queries"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_SCOPES * 2,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu profiler resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let slots = (0..RING)
                .map(|i| Slot {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("gpu profiler staging {i}")),
                        size: bytes,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    state: Arc::new(AtomicU8::new(SLOT_FREE)),
                    names: Vec::new(),
                    needs_map: false,
                })
                .collect();
            (Some(qs), Some(resolve), slots)
        } else {
            (None, None, Vec::new())
        };
        Self {
            query_set,
            resolve,
            slots,
            frame_names: Vec::new(),
            frame_active: false,
            rows: Vec::new(),
            period_ns: queue.get_timestamp_period(),
            frame_total: RollingStats::default(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.query_set.is_some()
    }

    /// Starts a frame. Timing is skipped when no staging buffer is free.
    pub fn begin_frame(&mut self) {
        self.harvest();
        self.frame_names.clear();
        self.frame_active = self.enabled()
            && self
                .slots
                .iter()
                .any(|s| s.state.load(Ordering::Acquire) == SLOT_FREE && !s.needs_map);
    }

    pub fn scope(&mut self, name: &'static str) -> TimestampScope {
        if !self.frame_active || self.frame_names.len() as u32 >= MAX_SCOPES {
            return TimestampScope::none();
        }
        let begin = self.frame_names.len() as u32 * 2;
        self.frame_names.push(name);
        TimestampScope {
            query_set: self.query_set.clone(),
            begin,
        }
    }

    /// Records the resolve and staging copy into the last encoder of the frame.
    pub fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.frame_active || self.frame_names.is_empty() {
            return;
        }
        let (Some(qs), Some(resolve)) = (&self.query_set, &self.resolve) else {
            return;
        };
        let Some(slot) = self
            .slots
            .iter_mut()
            .find(|s| s.state.load(Ordering::Acquire) == SLOT_FREE && !s.needs_map)
        else {
            return;
        };
        let count = self.frame_names.len() as u32 * 2;
        encoder.resolve_query_set(qs, 0..count, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, &slot.buffer, 0, u64::from(count) * 8);
        slot.names = std::mem::take(&mut self.frame_names);
        slot.needs_map = true;
    }

    /// Must follow `queue.submit` of the encoder passed to `end_frame`.
    pub fn after_submit(&mut self) {
        for slot in &mut self.slots {
            if !slot.needs_map {
                continue;
            }
            slot.needs_map = false;
            slot.state.store(SLOT_PENDING, Ordering::Release);
            let state = slot.state.clone();
            let len = slot.names.len() as u64 * 16;
            slot.buffer
                .map_async(wgpu::MapMode::Read, 0..len, move |r| {
                    let s = if r.is_ok() { SLOT_MAPPED } else { SLOT_FAILED };
                    state.store(s, Ordering::Release);
                });
        }
    }

    fn harvest(&mut self) {
        // Oldest mapped slots first is not required: each slot is a whole frame.
        for i in 0..self.slots.len() {
            let state = self.slots[i].state.load(Ordering::Acquire);
            if state == SLOT_FAILED {
                self.slots[i].buffer.unmap();
                self.slots[i].state.store(SLOT_FREE, Ordering::Release);
                continue;
            }
            if state != SLOT_MAPPED {
                continue;
            }
            let len = self.slots[i].names.len() as u64 * 16;
            let stamps: Vec<u64> = match self.slots[i].buffer.get_mapped_range(0..len) {
                // Mapped memory carries no alignment guarantee; decode bytewise.
                Ok(view) => view
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
                Err(_) => Vec::new(),
            };
            self.slots[i].buffer.unmap();
            self.slots[i].state.store(SLOT_FREE, Ordering::Release);
            let names = std::mem::take(&mut self.slots[i].names);
            // A pass dispatched twice in one frame reports its summed cost.
            let mut summed: Vec<(&'static str, f32)> = Vec::with_capacity(names.len());
            for (n, name) in names.iter().enumerate() {
                let (Some(&a), Some(&b)) = (stamps.get(n * 2), stamps.get(n * 2 + 1)) else {
                    continue;
                };
                let ms = b.saturating_sub(a) as f32 * self.period_ns / 1.0e6;
                match summed.iter_mut().find(|(s, _)| s == name) {
                    Some((_, acc)) => *acc += ms,
                    None => summed.push((name, ms)),
                }
            }
            self.frame_total.push(summed.iter().map(|(_, ms)| ms).sum());
            for (name, ms) in summed {
                self.push_row(name, ms);
            }
        }
    }

    fn push_row(&mut self, name: &'static str, ms: f32) {
        match self.rows.iter_mut().find(|r| r.name == name) {
            Some(row) => row.stats.push(ms),
            None => {
                let mut stats = RollingStats::default();
                stats.push(ms);
                self.rows.push(GpuRow { name, stats });
            }
        }
    }

    pub fn rows(&self) -> &[GpuRow] {
        &self.rows
    }

    /// Sum of all timed passes per frame.
    pub fn frame_total(&self) -> &RollingStats {
        &self.frame_total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_pass_produces_timing_row() {
        let Some(gpu) = crate::device::test_gpu() else {
            return;
        };
        if !gpu.timestamps {
            eprintln!("adapter lacks timestamp queries");
            return;
        }
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("noop"),
                source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("noop"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let mut prof = GpuProfiler::new(&gpu.device, &gpu.queue, true);
        for _ in 0..6 {
            prof.begin_frame();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            let scope = prof.scope("test.noop");
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("noop"),
                    timestamp_writes: scope.compute(),
                });
                pass.set_pipeline(&pipeline);
                pass.dispatch_workgroups(1, 1, 1);
            }
            prof.end_frame(&mut enc);
            gpu.queue.submit([enc.finish()]);
            prof.after_submit();
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
        }
        prof.begin_frame();
        let row = prof.rows().iter().find(|r| r.name == "test.noop");
        assert!(
            row.is_some_and(|r| r.stats.len() >= 4),
            "timing rows missing"
        );
    }
}
