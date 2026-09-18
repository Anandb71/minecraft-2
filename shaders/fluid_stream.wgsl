// Fluid step, pass 0: stream by pulling, exchange interface mass, collide
// (BGK with a Smagorinsky subgrid term and Guo gravity). Interface cells
// that overfilled or emptied note it in `conv`; a cell with a neighbour
// asleep or unallocated may not convert yet and keeps its tile stirred.
// One workgroup covers 64 cells of one awake tile.
#import "fluid_common.wgsl"

var<workgroup> bits: atomic<u32>;
// The tile's 27 neighbours (itself included) and whether each is awake,
// loaded once per workgroup rather than once per cell and direction.
var<workgroup> around: array<u32, 27>;
var<workgroup> around_awake: array<u32, 27>;

// `neighbour` through the workgroup's copy of the table; also says whether
// that tile is awake.
fn neighbour_near(l: vec3<i32>, d: vec3<i32>) -> vec2<u32> {
    let n = l + d;
    let t = near_slot(clamp(n >> vec3<u32>(3u), vec3<i32>(-1), vec3<i32>(1)));
    let slot = around[t];
    if slot == NONE {
        return vec2<u32>(NONE, 0u);
    }
    return vec2<u32>(slot * TILE_CELLS + index_of(n & vec3<i32>(7)), around_awake[t]);
}

fn fill_src(cell: u32) -> f32 {
    return fill[fill_index(cell, params.parity)];
}

// Where a flat wall reflects population `q` of cell `l` from, when the
// cell behind it is solid: (cell, population), or NONE for faces, corners,
// and when that cell holds no water. Only the 12 edge populations can slide;
// they come in blocks of four per plane (xy, xz, yz from 7, 9, 11): ++ at
// the block start, -- after it, +- six on, -+ seven on. Flipping the
// plane's first axis turns offset o into o ^ 7, flipping its second o ^ 6.
fn specular(l: vec3<i32>, q: u32) -> vec2<u32> {
    if q < 7u {
        return vec2<u32>(NONE);
    }
    let plane = ((q - 7u) % 6u) / 2u;
    let a1 = select(0u, 1u, plane == 2u);
    let a2 = select(2u, 1u, plane == 0u);
    let c = velocity(q);
    var b1 = vec3<i32>(0);
    b1[a1] = -c[a1];
    var b2 = vec3<i32>(0);
    b2[a2] = -c[a2];
    let n1 = neighbour_near(l, b1).x;
    let n2 = neighbour_near(l, b2).x;
    let s1 = n1 != NONE && kind[n1] == SOLID;
    let s2 = n2 != NONE && kind[n2] == SOLID;
    if s1 == s2 {
        return vec2<u32>(NONE);
    }
    // A wall across a2 reflects from the cell back along a1, and the other
    // way round.
    let m = select(n2, n1, s2);
    if m == NONE || (kind[m] != LIQUID && kind[m] != INTERFACE) {
        return vec2<u32>(NONE);
    }
    let start = 7u + 2u * plane;
    return vec2<u32>(m, start + ((q - start) ^ select(7u, 6u, s2)));
}

// Relaxation time raised by the strain in the non-equilibrium populations,
// from the second moment of the populations (`diag`: the c_a c_a terms,
// `off`: xy, xz, yz) less that of the equilibrium, rho / 3 + rho u u.
fn subgrid_tau(rho: f32, u: vec3<f32>, diag: vec3<f32>, off: vec3<f32>) -> f32 {
    let d = diag - (rho / 3.0 + rho * u * u);
    let o = off - rho * vec3<f32>(u.x * u.y, u.x * u.z, u.y * u.z);
    let s = sqrt(dot(d, d) + 2.0 * dot(o, o));
    let tau0 = params.tau;
    let c2 = params.smagorinsky * params.smagorinsky;
    return 0.5 * (tau0 + sqrt(tau0 * tau0 + 18.0 * sqrt(2.0) * c2 * s / max(rho, 1e-6)));
}

// What streaming one cell gathers besides its populations.
struct Streaming {
    slot: u32,
    cell: u32,
    local: vec3<i32>,
    kind: u32,
    fill_here: f32,
    rho_gas: f32,
    u_old: vec3<f32>,
    dm: f32,
    gas_near: bool,
    fluid_near: bool,
    blocked: bool,
}

// Population `q` arriving at the cell: from the neighbour behind it, off
// a wall, or rebuilt from the gas; counts interface mass exchange.
fn pull(q: u32, st: ptr<function, Streaming>) -> f32 {
    let c = (*st).cell;
    let k = (*st).kind;
    let found = neighbour_near((*st).local, -velocity(q));
    let src = found.x;
    // Beyond the allocated tiles lies gas.
    var nk = GAS;
    if src == NONE {
        (*st).blocked = true;
    } else {
        (*st).blocked = (*st).blocked || found.y == 0u;
        nk = kind[src];
    }
    if nk == LIQUID && k == LIQUID {
        return pops_src[pop_index(src, q)];
    }
    // What this cell sent the other way, which comes back from a wall or
    // the gas, and counts in an interface cell's mass exchange.
    let out_q = pops_src[pop_index(c, opp(q))];
    if nk == SOLID {
        let m = specular((*st).local, q);
        if m.x == NONE {
            return out_q;
        }
        let slide = pops_src[pop_index(m.x, m.y)];
        if k == INTERFACE {
            var exchange = params.wall_slip * (slide - out_q);
            if kind[m.x] != LIQUID {
                exchange *= 0.5 * ((*st).fill_here + fill_src(m.x));
            }
            (*st).dm += exchange;
        }
        return params.wall_slip * slide + (1.0 - params.wall_slip) * out_q;
    }
    if nk == GAS {
        (*st).gas_near = (*st).gas_near || src != NONE;
        let u = (*st).u_old;
        return equilibrium(q, (*st).rho_gas, u) + equilibrium(opp(q), (*st).rho_gas, u) - out_q;
    }
    (*st).fluid_near = (*st).fluid_near || nk == LIQUID;
    let f = pops_src[pop_index(src, q)];
    if k == INTERFACE {
        var exchange = f - out_q;
        if nk != LIQUID {
            exchange *= 0.5 * ((*st).fill_here + fill_src(src));
        }
        (*st).dm += exchange;
    }
    return f;
}

// Density, momentum and second moment (diagonal, then xy, xz, yz).
struct Moments {
    rho: f32,
    j: vec3<f32>,
    diag: vec3<f32>,
    off: vec3<f32>,
    // Velocity with half the force, and the force density, once known.
    u: vec3<f32>,
    force: vec3<f32>,
}

fn moment(q: u32, f: f32, m: ptr<function, Moments>) {
    let c = vec3<f32>(velocity(q));
    (*m).rho += f;
    (*m).j += c * f;
    (*m).diag += c * c * f;
    (*m).off += vec3<f32>(c.x * c.y, c.x * c.z, c.y * c.z) * f;
}

fn relax(q: u32, f: f32, cell: u32, m: ptr<function, Moments>, tau: f32) {
    let feq = equilibrium(q, (*m).rho, (*m).u);
    pops_dst[pop_index(cell, q)] = f - (f - feq) / tau + guo(q, tau, (*m).u, (*m).force);
}

// The 19 populations are separate values, each call below with a literal
// index: an array indexed at run time would live in scratch memory, which
// costs several times the rest of the kernel.
@compute @workgroup_size(64)
fn stream_collide(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let s = lists[wg.x];
    if li < 27u {
        let n = near[s * 27u + li];
        around[li] = n;
        around_awake[li] = select(0u, tile_state[n].awake, n != NONE);
    }
    workgroupBarrier();
    let i = wg.y * 64u + li;
    let c = s * TILE_CELLS + i;
    let k = kind[c];
    conv[c] = 0u;
    if k == LIQUID || k == INTERFACE {
        var fill_here = 1.0;
        if k == INTERFACE {
            fill_here = fill_src(c);
        }
        // The gas pressure plus the weight of the part of the cell's water
        // standing above its centre.
        let rho_gas = params.rho_gas + 3.0 * length(params.gravity) * (fill_here - 0.5);
        var st = Streaming(s, c, local_of(i), k, fill_here, rho_gas, rho_u[c].xyz, 0.0, false, false, false);
        let f0 = pops_src[pop_index(c, 0u)];
        let f1 = pull(1u, &st);
        let f2 = pull(2u, &st);
        let f3 = pull(3u, &st);
        let f4 = pull(4u, &st);
        let f5 = pull(5u, &st);
        let f6 = pull(6u, &st);
        let f7 = pull(7u, &st);
        let f8 = pull(8u, &st);
        let f9 = pull(9u, &st);
        let f10 = pull(10u, &st);
        let f11 = pull(11u, &st);
        let f12 = pull(12u, &st);
        let f13 = pull(13u, &st);
        let f14 = pull(14u, &st);
        let f15 = pull(15u, &st);
        let f16 = pull(16u, &st);
        let f17 = pull(17u, &st);
        let f18 = pull(18u, &st);
        var m = Moments(0.0, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
        moment(0u, f0, &m);
        moment(1u, f1, &m);
        moment(2u, f2, &m);
        moment(3u, f3, &m);
        moment(4u, f4, &m);
        moment(5u, f5, &m);
        moment(6u, f6, &m);
        moment(7u, f7, &m);
        moment(8u, f8, &m);
        moment(9u, f9, &m);
        moment(10u, f10, &m);
        moment(11u, f11, &m);
        moment(12u, f12, &m);
        moment(13u, f13, &m);
        moment(14u, f14, &m);
        moment(15u, f15, &m);
        moment(16u, f16, &m);
        moment(17u, f17, &m);
        moment(18u, f18, &m);
        let r = m.rho;
        m.force = params.gravity * r;
        if r > 1e-6 {
            m.u = (m.j + 0.5 * m.force) / r;
        }
        let v = m.u;
        let tau = subgrid_tau(r, v, m.diag, m.off);
        relax(0u, f0, c, &m, tau);
        relax(1u, f1, c, &m, tau);
        relax(2u, f2, c, &m, tau);
        relax(3u, f3, c, &m, tau);
        relax(4u, f4, c, &m, tau);
        relax(5u, f5, c, &m, tau);
        relax(6u, f6, c, &m, tau);
        relax(7u, f7, c, &m, tau);
        relax(8u, f8, c, &m, tau);
        relax(9u, f9, c, &m, tau);
        relax(10u, f10, c, &m, tau);
        relax(11u, f11, c, &m, tau);
        relax(12u, f12, c, &m, tau);
        relax(13u, f13, c, &m, tau);
        relax(14u, f14, c, &m, tau);
        relax(15u, f15, c, &m, tau);
        relax(16u, f16, c, &m, tau);
        relax(17u, f17, c, &m, tau);
        relax(18u, f18, c, &m, tau);
        rho_u[c] = vec4<f32>(v, r);
        if length(v) > params.still_speed {
            atomicOr(&bits, MOVED);
        }
        if k == LIQUID {
            mass[c] = r;
        } else {
            let mc = mass[c] + st.dm;
            mass[c] = mc;
            var want = 0u;
            if mc > (1.0 + params.fill_slack) * r || !st.gas_near {
                want = TO_LIQUID;
            } else if mc < -params.fill_slack * r || (!st.fluid_near && mc < 0.1 * r) {
                want = TO_GAS;
            }
            if want != 0u && st.blocked {
                atomicOr(&bits, MOVED);
            } else if want != 0u {
                conv[c] = want;
                atomicOr(&bits, CONVERTING);
            }
        }
    }
    workgroupBarrier();
    let b = atomicLoad(&bits);
    if li == 0u && b != 0u {
        atomicOr(&tile_state[s].moving[params.parity], b);
    }
}

// Before streaming: tiles that fell asleep at the end of the last step hold
// their state only in this step's source half; copy it to the other so
// both read the same while they sleep.
@compute @workgroup_size(64)
fn copy_asleep(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let s = lists[params.max_slots + wg.x];
    let c = s * TILE_CELLS + wg.y * 64u + li;
    for (var q = 0u; q < Q; q++) {
        pops_dst[pop_index(c, q)] = pops_src[pop_index(c, q)];
    }
    fill[fill_index(c, 1u - params.parity)] = fill[fill_index(c, params.parity)];
}
