// Fluid step, passes 1-4: the free surface moving. Each pass reads of its
// neighbours only what an earlier pass wrote (see crates/mc2-fluid
// sim/surface.rs, which runs the same passes on the CPU).
//
// 1. flag: each cell's transition, from what streaming asked for.
// 2. apply: conversions take effect; converting cells set aside excess.
// 3. share: excess split among the water around each converting cell.
// 4. gather: cells collect those shares; tiles note which tiles around
//    them their water needs.
//
// Cells that convert mark their neighbours (bits in `conv`), so a cell
// learns what happened around it from its own word instead of asking all
// 18 neighbours; gather clears the marks for the next step. One workgroup
// covers 64 cells of one awake tile.
#import "fluid_common.wgsl"

var<workgroup> bits: atomic<u32>;
var<workgroup> needed: atomic<u32>;

struct Cell {
    slot: u32,
    global: u32,
    local: vec3<i32>,
}

fn cell_of(wg: vec3<u32>, li: u32) -> Cell {
    let s = lists[wg.x];
    let i = wg.y * 64u + li;
    return Cell(s, s * TILE_CELLS + i, local_of(i));
}

// Adds this workgroup's bits to its tile's step word.
fn publish(c: Cell, li: u32) {
    workgroupBarrier();
    let b = atomicLoad(&bits);
    if li == 0u && b != 0u {
        atomicOr(&tile_state[c.slot].moving[params.parity], b);
    }
}

@compute @workgroup_size(64)
fn flag(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let c = cell_of(wg, li);
    let k = kind[c.global];
    let marks = atomicLoad(&conv[c.global]);
    let want = marks & WANT;
    let filling_near = (marks & FILLING_NEAR) != 0u;
    var transition = 0u;
    if k == GAS && filling_near {
        transition = FROM_GAS;
    } else if k == INTERFACE && want == TO_LIQUID {
        transition = TO_LIQUID;
    } else if k == INTERFACE && want == TO_GAS && !filling_near {
        transition = TO_GAS;
        // Liquid around it becomes interface.
        for (var q = 1u; q < Q; q++) {
            let n = neighbour(c.slot, c.local, velocity(q));
            if n != NONE {
                atomicOr(&conv[n], EMPTIED_NEAR);
            }
        }
    }
    next[c.global] = pack(k, transition);
}

@compute @workgroup_size(64)
fn apply(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let c = cell_of(wg, li);
    let here = next[c.global];
    let transition = here >> 2u;
    if transition == FROM_GAS {
        // The mean of the water around it, not counting cells that are
        // emptying or new themselves.
        var rho = 0.0;
        var u = vec3<f32>(0.0);
        var n = 0.0;
        for (var q = 1u; q < Q; q++) {
            let o = neighbour(c.slot, c.local, velocity(q));
            if o == NONE {
                continue;
            }
            let there = next[o];
            let k = there & 3u;
            if (k == LIQUID || k == INTERFACE) && (there >> 2u) != TO_GAS {
                rho += rho_u[o].w;
                u += rho_u[o].xyz;
                n += 1.0;
            }
        }
        if n > 0.0 {
            rho /= n;
            u /= n;
        } else {
            rho = 1.0;
        }
        kind[c.global] = INTERFACE;
        mass[c.global] = 0.0;
        rho_u[c.global] = vec4<f32>(u, rho);
        for (var q = 0u; q < Q; q++) {
            pops_dst[pop_index(c.global, q)] = equilibrium(q, rho, u);
        }
        atomicOr(&bits, MOVED);
    } else if transition == TO_LIQUID {
        let rho = rho_u[c.global].w;
        excess[c.global] = mass[c.global] - rho;
        kind[c.global] = LIQUID;
        mass[c.global] = rho;
        atomicOr(&bits, MOVED | HANDING_ON);
    } else if transition == TO_GAS {
        excess[c.global] = mass[c.global];
        kind[c.global] = GAS;
        mass[c.global] = 0.0;
        atomicOr(&bits, MOVED | HANDING_ON);
    } else if (here & 3u) == LIQUID && (atomicLoad(&conv[c.global]) & EMPTIED_NEAR) != 0u {
        kind[c.global] = INTERFACE;
        mass[c.global] = rho_u[c.global].w;
        atomicOr(&bits, MOVED);
    }
    publish(c, li);
}

@compute @workgroup_size(64)
fn share(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let c = cell_of(wg, li);
    let transition = next[c.global] >> 2u;
    massex[c.global] = 0.0;
    if transition == TO_LIQUID || transition == TO_GAS {
        var targets = 0u;
        for (var q = 1u; q < Q; q++) {
            let o = neighbour(c.slot, c.local, velocity(q));
            if o != NONE && (kind[o] == INTERFACE || kind[o] == LIQUID) {
                targets += 1u;
            }
        }
        if targets == 0u {
            atomicAdd(&args.lost, i32(round(excess[c.global] * 65536.0)));
        } else {
            massex[c.global] = excess[c.global] / f32(targets);
        }
    }
}

@compute @workgroup_size(64)
fn gather(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let c = cell_of(wg, li);
    let handed_near = (step_bits_around(c.slot, li) & HANDING_ON) != 0u;
    let k = kind[c.global];
    var f = 0.0;
    if (k == INTERFACE || k == LIQUID) && handed_near {
        var share = 0.0;
        for (var q = 1u; q < Q; q++) {
            let o = neighbour(c.slot, c.local, velocity(q));
            if o != NONE {
                share += massex[o];
            }
        }
        if share != 0.0 {
            mass[c.global] += share;
            if k == LIQUID {
                // A liquid cell's mass is its density: the share goes into
                // its populations by weight, which leaves its momentum be.
                rho_u[c.global].w += share;
                for (var q = 0u; q < Q; q++) {
                    let at = pop_index(c.global, q);
                    pops_dst[at] += weight(q) * share;
                }
            }
        }
    }
    if k == INTERFACE {
        f = clamp(mass[c.global] / max(rho_u[c.global].w, 1e-6), 0.0, 1.0);
    } else if k == LIQUID {
        f = 1.0;
    }
    fill[fill_index(c.global, 1u - params.parity)] = f;
    // Marks are read; clear them for the next step's streaming.
    atomicStore(&conv[c.global], 0u);
    if k == INTERFACE || k == LIQUID {
        atomicOr(&needed, need_of(c.local));
    }
    workgroupBarrier();
    if li == 0u {
        let n = atomicLoad(&needed);
        if n != 0u {
            atomicOr(&tile_state[c.slot].need[params.parity], n);
        }
    }
}
