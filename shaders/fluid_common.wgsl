// Free-surface lattice Boltzmann over sparse 8^3 tiles of half-metre cells:
// the bindings every fluid pass shares, the D3Q19 lattice and the lookups
// that cross tile edges. A GPU port of crates/mc2-fluid, pass for pass.
//
// Cells are addressed globally as slot * 512 + index within the tile.
// Populations are stored per tile, population-major, so neighbouring
// invocations read neighbouring words: (slot * 19 + q) * 512 + index.

struct FluidParams {
    gravity: vec3<f32>,
    tau: f32,
    smagorinsky: f32,
    rho_gas: f32,
    fill_slack: f32,
    wall_slip: f32,
    // Step parity: which population buffer, fill half, and moving and need
    // words are this step's sources.
    parity: u32,
    slot_count: u32,
    max_slots: u32,
    sleep_steps: u32,
    still_speed: f32,
    edit_count: u32,
    touched_count: u32,
    // Fastest the water may move, lattice units (MAX_SPEED).
    max_speed: f32,
}

struct TileState {
    // 1 while the slot holds a tile.
    alloc: u32,
    awake: u32,
    still: u32,
    // 1 while the tile holds water, as of the last step it ran.
    water: u32,
    // Step words by parity: MOVED and the pass bits below.
    moving: array<atomic<u32>, 2>,
    need: array<atomic<u32>, 2>,
}

// Indirect dispatch arguments for the awake tiles and for tiles that just
// fell asleep, then the mass lost for want of interface cells (fixed point).
struct Args {
    active_x: atomic<u32>,
    active_y: u32,
    active_z: u32,
    copy_x: atomic<u32>,
    copy_y: u32,
    copy_z: u32,
    lost: atomic<i32>,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: FluidParams;
@group(0) @binding(1) var<storage, read_write> pops_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> pops_dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> kind: array<u32>;
@group(0) @binding(4) var<storage, read_write> mass: array<f32>;
@group(0) @binding(5) var<storage, read_write> rho_u: array<vec4<f32>>;
// Two halves, by parity: the fill each cell had at the start of the step.
@group(0) @binding(6) var<storage, read_write> fill: array<f32>;
// Per cell: what streaming asked the cell to become (low two bits), and
// marks its neighbours leave on it (FILLING_NEAR, EMPTIED_NEAR).
@group(0) @binding(7) var<storage, read_write> conv: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> next: array<u32>;
@group(0) @binding(9) var<storage, read_write> excess: array<f32>;
@group(0) @binding(10) var<storage, read_write> massex: array<f32>;
@group(0) @binding(11) var<storage, read> near: array<u32>;
@group(0) @binding(12) var<storage, read_write> tile_state: array<TileState>;
@group(0) @binding(13) var<storage, read_write> args: Args;
// The awake list, then (from max_slots) the list of tiles to copy.
@group(0) @binding(14) var<storage, read_write> lists: array<u32>;
// Edits from the CPU: (cell, op, value bits, 0), then touched slots.
@group(0) @binding(15) var<storage, read> edits: array<vec4<u32>>;

const TILE_CELLS: u32 = 512u;
const Q: u32 = 19u;
const NONE: u32 = 0xffffffffu;

const GAS: u32 = 0u;
const INTERFACE: u32 = 1u;
const LIQUID: u32 = 2u;
const SOLID: u32 = 3u;

const TO_LIQUID: u32 = 1u;
const TO_GAS: u32 = 2u;
const FROM_GAS: u32 = 3u;

const MARGIN: i32 = 3;

// Marks in `conv`: a neighbour is filling (to liquid) this step, or has
// emptied (to gas). Set by the converting cell, read by the one it marks.
const WANT: u32 = 3u;
const FILLING_NEAR: u32 = 4u;
const EMPTIED_NEAR: u32 = 8u;

// The 19 lattice velocities (rest, the 6 faces, then the 12 edges, each
// followed by its opposite; `C` in crates/mc2-fluid lattice.rs) as bits by
// index: which components are non-zero, and which of those negative. Pure
// arithmetic: indexing a constant array at run time costs a copy of it.
const NONZERO = vec3<u32>(0x1e786u, 0x67998u, 0x79e60u);
const NEGATIVE = vec3<u32>(0x14504u, 0x43110u, 0x29440u);

fn velocity(q: u32) -> vec3<i32> {
    let bit = vec3<u32>(1u << q);
    let nonzero = vec3<i32>((NONZERO & bit) != vec3<u32>(0u));
    let negative = vec3<i32>((NEGATIVE & bit) != vec3<u32>(0u));
    return nonzero * (1 - 2 * negative);
}

// Bits a tile's step word collects: something moved (keeps it awake), and
// which surface passes have work near it.
const MOVED: u32 = 1u;
const HANDING_ON: u32 = 2u;

fn weight(q: u32) -> f32 {
    if q == 0u {
        return 1.0 / 3.0;
    }
    return select(1.0 / 36.0, 1.0 / 18.0, q < 7u);
}

fn opp(q: u32) -> u32 {
    if q == 0u {
        return 0u;
    }
    return select(q - 1u, q + 1u, (q & 1u) == 1u);
}

fn equilibrium(q: u32, rho: f32, u: vec3<f32>) -> f32 {
    let cu = dot(vec3<f32>(velocity(q)), u);
    return weight(q) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * dot(u, u));
}

fn guo(q: u32, tau: f32, u: vec3<f32>, force: vec3<f32>) -> f32 {
    let c = vec3<f32>(velocity(q));
    return (1.0 - 0.5 / tau) * weight(q) * dot(3.0 * (c - u) + 9.0 * dot(c, u) * c, force);
}

fn pack(k: u32, transition: u32) -> u32 {
    return k | (transition << 2u);
}

fn local_of(i: u32) -> vec3<i32> {
    return vec3<i32>(i32(i & 7u), i32((i >> 3u) & 7u), i32(i >> 6u));
}

fn index_of(l: vec3<i32>) -> u32 {
    return u32(l.x + 8 * (l.y + 8 * l.z));
}

fn near_slot(d: vec3<i32>) -> u32 {
    return u32((d.x + 1) + 3 * ((d.y + 1) + 3 * (d.z + 1)));
}

// Global index of the cell at offset `d` from cell `l` of slot `s`, or NONE
// where that tile is not allocated.
fn neighbour(s: u32, l: vec3<i32>, d: vec3<i32>) -> u32 {
    let n = l + d;
    let step = clamp(n >> vec3<u32>(3u), vec3<i32>(-1), vec3<i32>(1));
    var slot = s;
    if any(step != vec3<i32>(0)) {
        slot = near[s * 27u + near_slot(step)];
        if slot == NONE {
            return NONE;
        }
    }
    return slot * TILE_CELLS + index_of(n & vec3<i32>(7));
}

fn pop_index(cell: u32, q: u32) -> u32 {
    return ((cell / TILE_CELLS) * Q + q) * TILE_CELLS + cell % TILE_CELLS;
}

fn fill_index(cell: u32, half: u32) -> u32 {
    return half * params.max_slots * TILE_CELLS + cell;
}

// The neighbouring tiles water in cell `l` needs allocated, as bits by
// near_slot (the tile itself included).
fn need_of(l: vec3<i32>) -> u32 {
    // Vector writes through a dynamic index are not l-values under FXC.
    let lo = select(vec3<i32>(0), vec3<i32>(-1), l < vec3<i32>(MARGIN));
    let hi = select(vec3<i32>(0), vec3<i32>(1), l >= vec3<i32>(8 - MARGIN));
    var bits = 0u;
    for (var z = lo.z; z <= hi.z; z++) {
        for (var y = lo.y; y <= hi.y; y++) {
            for (var x = lo.x; x <= hi.x; x++) {
                bits |= 1u << near_slot(vec3<i32>(x, y, z));
            }
        }
    }
    return bits;
}

var<workgroup> around_bits: atomic<u32>;

// The step bits of tile `s` and the tiles around it, gathered by the whole
// workgroup (call from uniform control flow).
fn step_bits_around(s: u32, li: u32) -> u32 {
    if li < 27u {
        let n = near[s * 27u + li];
        if n != NONE {
            atomicOr(&around_bits, atomicLoad(&tile_state[n].moving[params.parity]));
        }
    }
    workgroupBarrier();
    return atomicLoad(&around_bits);
}
