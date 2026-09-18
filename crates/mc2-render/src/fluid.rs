//! Water on the GPU: the free-surface lattice Boltzmann of `mc2-fluid`, run
//! pass for pass in compute shaders over a pool of 8^3 tiles.
//!
//! The CPU owns the tile table (which slot holds which tile, and each slot's
//! 26 neighbours) and sends edits: water added, terrain closed or opened.
//! The GPU decides alone which tiles step: still tiles sleep, motion wakes
//! them, and each step dispatches indirectly over the awake list. What it
//! reports back (a frame or two later) says which tiles hold water and
//! which tiles around them that water needs; the CPU allocates those and
//! frees tiles long dry. Until a tile exists and is awake, water beside it
//! waits (the passes block conversions there), so allocating late delays
//! water but never loses it.

use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use mc2_core::{FxHashMap, FxHashSet};
use mc2_fluid::{Params, TILE_CELLS};
use mc2_gpu::{HotCompute, ShaderLibrary, bind, bind_group, layout};
use std::sync::Arc;
use std::sync::atomic::AtomicU8;


const Q: u64 = 19;
const CELLS: u64 = TILE_CELLS as u64;
const NONE: u32 = u32::MAX;
const SOLID: u32 = 3;
const EDIT_WATER: u32 = 0;
const EDIT_SOLID: u32 = 1;
const EDIT_OPEN: u32 = 2;
/// Workgroups per tile, 64 cells each.
const PARTS: u32 = (TILE_CELLS / 64) as u32;
/// A tile asleep this many steps, dry and with no wet neighbour, is freed.
const FREE_STILL: u32 = 1200;
/// Freed slots wait this many frames before reuse, in case an awake list
/// built before the free still names them.
const COOL_FRAMES: u64 = 6;
/// New tiles per frame at most.
const ALLOC_PER_FRAME: usize = 64;
const STAGING: usize = 3;
const MAP_FREE: u8 = 0;
const MAP_PENDING: u8 = 1;
const MAP_DONE: u8 = 2;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    gravity: [f32; 3],
    tau: f32,
    smagorinsky: f32,
    rho_gas: f32,
    fill_slack: f32,
    wall_slip: f32,
    parity: u32,
    slot_count: u32,
    max_slots: u32,
    sleep_steps: u32,
    still_speed: f32,
    edit_count: u32,
    touched_count: u32,
    _pad: u32,
}

/// One slot as the GPU keeps it (`TileState` in fluid_common.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TileReport {
    pub alloc: u32,
    pub awake: u32,
    pub still: u32,
    pub water: u32,
    pub moving: [u32; 2],
    pub need: [u32; 2],
}

const REPORT_BYTES: u64 = std::mem::size_of::<TileReport>() as u64;
const ARGS_BYTES: u64 = 32;

#[derive(Clone, Copy, Debug, Default)]
pub struct FluidGpuStats {
    pub tiles: usize,
    /// Tiles stepping, as of the last report.
    pub awake: u32,
    /// Mass lost for want of interface cells, lattice units.
    pub lost_mass: f64,
    pub steps: u64,
}

/// The cells of one tile, read back for tests and tools.
pub struct TileCells {
    pub pos: IVec3,
    pub kind: Vec<u32>,
    pub mass: Vec<f32>,
    pub rho_u: Vec<[f32; 4]>,
}

impl TileCells {
    /// Fill of cell `i`: 1 for liquid, the interface fill, 0 otherwise.
    pub fn fill(&self, i: usize) -> f32 {
        match self.kind[i] {
            2 => 1.0,
            1 => (self.mass[i] / self.rho_u[i][3].max(1e-6)).clamp(0.0, 1.0),
            _ => 0.0,
        }
    }

    /// Water mass of cell `i`, lattice units.
    pub fn water(&self, i: usize) -> f32 {
        match self.kind[i] {
            2 => self.rho_u[i][3],
            1 => self.mass[i],
            _ => 0.0,
        }
    }
}

struct Buffers {
    pops: [wgpu::Buffer; 2],
    kind: wgpu::Buffer,
    mass: wgpu::Buffer,
    rho_u: wgpu::Buffer,
    fill: wgpu::Buffer,
    conv: wgpu::Buffer,
    next: wgpu::Buffer,
    excess: wgpu::Buffer,
    massex: wgpu::Buffer,
    near: wgpu::Buffer,
    tile_state: wgpu::Buffer,
    args: wgpu::Buffer,
    indirect: wgpu::Buffer,
    lists: wgpu::Buffer,
    edits: wgpu::Buffer,
    params: [wgpu::Buffer; 2],
}

struct Pipelines {
    copy_asleep: HotCompute,
    stream: HotCompute,
    flag: HotCompute,
    apply: HotCompute,
    share: HotCompute,
    gather: HotCompute,
    reset_lists: HotCompute,
    activity: HotCompute,
    reset_active: HotCompute,
    wake: HotCompute,
    edit: HotCompute,
    close: HotCompute,
}

struct Staging {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    /// Copied into by the frame being recorded; map after submit.
    recorded: bool,
}

pub struct FluidGpu {
    pub params: Params,
    max_slots: u32,
    buffers: Buffers,
    groups: [wgpu::BindGroup; 2],
    layout: wgpu::BindGroupLayout,
    pipelines: Pipelines,
    /// Capacity of the edits buffer, in entries.
    edit_capacity: u64,
    // The tile table.
    slots: Vec<Option<IVec3>>,
    index: FxHashMap<IVec3, u32>,
    free: Vec<u32>,
    cooling: Vec<(u32, u64)>,
    slot_count: u32,
    near: Vec<u32>,
    // The last report, per slot.
    water: Vec<bool>,
    still: Vec<u32>,
    need: Vec<u32>,
    // This frame's uploads.
    /// New slots and their cell kinds.
    fresh: Vec<(u32, Vec<u32>)>,
    dirty_rows: FxHashSet<u32>,
    states: Vec<(u32, TileReport)>,
    woken: FxHashSet<u32>,
    edits: Vec<[u32; 4]>,
    touched: FxHashSet<u32>,
    parity: u32,
    frame: u64,
    started: bool,
    staging: Vec<Staging>,
    pub stats: FluidGpuStats,
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

impl FluidGpu {
    /// A pool of `max_slots` tiles (about 100 KB each), clamped to what the
    /// device can bind.
    pub fn new(
        device: &wgpu::Device,
        shaders: &ShaderLibrary,
        max_slots: u32,
        params: Params,
    ) -> Self {
        let pop_bytes = CELLS * Q * 4;
        let bind_limit = device.limits().max_storage_buffer_binding_size;
        let max_slots = max_slots.min((bind_limit / pop_bytes) as u32).max(1);
        let slots = u64::from(max_slots);
        let cells = slots * CELLS;
        let edit_capacity = 1 << 14;
        let buffers = Buffers {
            pops: [
                storage(device, "fluid pops a", cells * Q * 4),
                storage(device, "fluid pops b", cells * Q * 4),
            ],
            kind: storage(device, "fluid kind", cells * 4),
            mass: storage(device, "fluid mass", cells * 4),
            rho_u: storage(device, "fluid rho u", cells * 16),
            fill: storage(device, "fluid fill", cells * 8),
            conv: storage(device, "fluid conv", cells * 4),
            next: storage(device, "fluid next", cells * 4),
            excess: storage(device, "fluid excess", cells * 4),
            massex: storage(device, "fluid massex", cells * 4),
            near: storage(device, "fluid near", slots * 27 * 4),
            tile_state: storage(device, "fluid tile state", slots * REPORT_BYTES),
            args: storage(device, "fluid args", ARGS_BYTES),
            indirect: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fluid indirect"),
                size: ARGS_BYTES,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            lists: storage(device, "fluid lists", slots * 8),
            edits: storage(device, "fluid edits", edit_capacity * 16),
            params: ["fluid params 0", "fluid params 1"].map(|label| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: std::mem::size_of::<GpuParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }),
        };
        let mut types = vec![bind::uniform()];
        types.extend((1..=10).map(|_| bind::storage(false)));
        types.push(bind::storage(true));
        types.extend((12..=14).map(|_| bind::storage(false)));
        types.push(bind::storage(true));
        let layout = layout(device, "fluid", wgpu::ShaderStages::COMPUTE, &types);
        let hc = |file: &'static str, entry: &'static str| {
            HotCompute::new(device, shaders, file, entry, &[&layout])
        };
        let pipelines = Pipelines {
            copy_asleep: hc("fluid_stream.wgsl", "copy_asleep"),
            stream: hc("fluid_stream.wgsl", "stream_collide"),
            flag: hc("fluid_surface.wgsl", "flag"),
            apply: hc("fluid_surface.wgsl", "apply"),
            share: hc("fluid_surface.wgsl", "share"),
            gather: hc("fluid_surface.wgsl", "gather"),
            reset_lists: hc("fluid_tiles.wgsl", "reset_lists"),
            activity: hc("fluid_tiles.wgsl", "activity"),
            reset_active: hc("fluid_tiles.wgsl", "reset_active"),
            wake: hc("fluid_tiles.wgsl", "wake"),
            edit: hc("fluid_tiles.wgsl", "edit"),
            close: hc("fluid_tiles.wgsl", "close_surface"),
        };
        let groups = Self::bind_groups(device, &layout, &buffers);
        let staging = (0..STAGING)
            .map(|_| Staging {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fluid report"),
                    size: slots * REPORT_BYTES + ARGS_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(MAP_FREE)),
                recorded: false,
            })
            .collect();
        let n = max_slots as usize;
        Self {
            params,
            max_slots,
            buffers,
            groups,
            layout,
            pipelines,
            edit_capacity,
            slots: vec![None; n],
            index: FxHashMap::default(),
            free: Vec::new(),
            cooling: Vec::new(),
            slot_count: 0,
            near: vec![NONE; n * 27],
            water: vec![false; n],
            still: vec![0; n],
            need: vec![0; n],
            fresh: Vec::new(),
            dirty_rows: FxHashSet::default(),
            states: Vec::new(),
            woken: FxHashSet::default(),
            edits: Vec::new(),
            touched: FxHashSet::default(),
            parity: 0,
            frame: 0,
            started: false,
            staging,
            stats: FluidGpuStats::default(),
        }
    }

    /// Bind groups for even and odd steps, which swap the population halves.
    fn bind_groups(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        b: &Buffers,
    ) -> [wgpu::BindGroup; 2] {
        [0usize, 1].map(|p| {
            bind_group(
                device,
                "fluid",
                layout,
                &[
                    b.params[p].as_entire_binding(),
                    b.pops[p].as_entire_binding(),
                    b.pops[1 - p].as_entire_binding(),
                    b.kind.as_entire_binding(),
                    b.mass.as_entire_binding(),
                    b.rho_u.as_entire_binding(),
                    b.fill.as_entire_binding(),
                    b.conv.as_entire_binding(),
                    b.next.as_entire_binding(),
                    b.excess.as_entire_binding(),
                    b.massex.as_entire_binding(),
                    b.near.as_entire_binding(),
                    b.tile_state.as_entire_binding(),
                    b.args.as_entire_binding(),
                    b.lists.as_entire_binding(),
                    b.edits.as_entire_binding(),
                ],
            )
        })
    }

    pub fn max_slots(&self) -> u32 {
        self.max_slots
    }

    pub fn tiles(&self) -> usize {
        self.index.len()
    }
}
