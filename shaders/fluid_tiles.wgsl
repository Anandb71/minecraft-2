// Fluid tiles: which tiles step next (sleep and wake), and the edits the CPU
// sends between steps (water added, terrain opened or closed).
#import "fluid_common.wgsl"

// Before `activity` rebuilds the lists.
@compute @workgroup_size(1)
fn reset_lists() {
    atomicStore(&args.active_x, 0u);
    atomicStore(&args.copy_x, 0u);
}

// Before `wake`, which rebuilds only the awake list: the tiles to copy
// still wait for the next step.
@compute @workgroup_size(1)
fn reset_active() {
    atomicStore(&args.active_x, 0u);
}

// One invocation per slot: a tile stays awake while anything in it or
// around it moved within the last second. The awake list is rebuilt for
// the next step; tiles falling asleep are listed to have their state
// copied to both halves.
@compute @workgroup_size(64)
fn activity(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x;
    if s >= params.slot_count || tile_state[s].alloc == 0u {
        return;
    }
    let p = params.parity;
    var stirred = false;
    for (var d = 0u; d < 27u; d++) {
        let n = near[s * 27u + d];
        if n != NONE && atomicLoad(&tile_state[n].moving[p]) != 0u {
            stirred = true;
        }
    }
    var still = 0u;
    if !stirred {
        still = min(tile_state[s].still + 1u, 0x7fffffffu);
    }
    tile_state[s].still = still;
    let was_awake = tile_state[s].awake != 0u;
    if was_awake {
        tile_state[s].water = select(0u, 1u, atomicLoad(&tile_state[s].need[p]) != 0u);
    }
    let awake = still <= params.sleep_steps;
    tile_state[s].awake = select(0u, 1u, awake);
    if awake {
        lists[atomicAdd(&args.active_x, 1u)] = s;
    } else if was_awake {
        lists[params.max_slots + atomicAdd(&args.copy_x, 1u)] = s;
    }
    atomicStore(&tile_state[s].moving[1u - p], 0u);
    atomicStore(&tile_state[s].need[1u - p], 0u);
}

// Between frames, after edits: rebuilds the awake list without advancing
// anyone's stillness, so tiles the CPU woke (still = 0) step at once.
@compute @workgroup_size(64)
fn wake(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x;
    if s >= params.slot_count || tile_state[s].alloc == 0u {
        return;
    }
    let awake = tile_state[s].still <= params.sleep_steps;
    tile_state[s].awake = select(0u, 1u, awake);
    if awake {
        lists[atomicAdd(&args.active_x, 1u)] = s;
    }
}

const EDIT_WATER: u32 = 0u;
const EDIT_SOLID: u32 = 1u;
const EDIT_OPEN: u32 = 2u;

// One invocation per edit: still water at a given density, or terrain
// closing or opening a cell. Both halves of the state are written, since
// the tile may be asleep.
@compute @workgroup_size(64)
fn edit(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.edit_count {
        return;
    }
    let e = edits[id.x];
    let c = e.x;
    let k = kind[c];
    if e.y == EDIT_WATER {
        if k == SOLID {
            return;
        }
        let rho = bitcast<f32>(e.z);
        kind[c] = LIQUID;
        next[c] = pack(LIQUID, 0u);
        conv[c] = 0u;
        mass[c] = rho;
        rho_u[c] = vec4<f32>(0.0, 0.0, 0.0, rho);
        for (var q = 0u; q < Q; q++) {
            pops_src[pop_index(c, q)] = weight(q) * rho;
            pops_dst[pop_index(c, q)] = weight(q) * rho;
        }
        fill[fill_index(c, 0u)] = 1.0;
        fill[fill_index(c, 1u)] = 1.0;
        excess[c] = 0.0;
        massex[c] = 0.0;
    } else if e.y == EDIT_SOLID && k == GAS {
        kind[c] = SOLID;
        next[c] = pack(SOLID, 0u);
    } else if e.y == EDIT_OPEN && k == SOLID {
        kind[c] = GAS;
        next[c] = pack(GAS, 0u);
    }
}

// After edits, over the touched tiles and their neighbours: liquid facing
// gas or unallocated space becomes a full interface cell, so the interface
// layer stays closed. `next` holds every kind (edits keep it current), so
// no cell reads a kind another is rewriting.
@compute @workgroup_size(64)
fn close_surface(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let s = edits[params.edit_count + wg.x].x;
    let i = wg.y * 64u + li;
    let c = s * TILE_CELLS + i;
    if kind[c] != LIQUID {
        return;
    }
    let l = local_of(i);
    for (var q = 1u; q < Q; q++) {
        let n = neighbour(s, l, C[q]);
        if n == NONE || (next[n] & 3u) == GAS {
            kind[c] = INTERFACE;
            mass[c] = rho_u[c].w;
            return;
        }
    }
}
