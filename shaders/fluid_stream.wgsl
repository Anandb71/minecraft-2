// Fluid step, pass 0: stream by pulling, exchange interface mass, collide
// (BGK with a Smagorinsky subgrid term and Guo gravity). Interface cells
// that overfilled or emptied note it in `conv`; a cell with a neighbour
// asleep or unallocated may not convert yet and keeps its tile stirred.
// One workgroup covers 64 cells of one awake tile.
#import "fluid_common.wgsl"

var<workgroup> stirred: atomic<u32>;

fn fill_src(cell: u32) -> f32 {
    return fill[fill_index(cell, params.parity)];
}

// Where a flat wall reflects population `q` of cell `l` in slot `s` from,
// when the cell behind it is solid: (cell, population), or NONE for faces,
// corners, and when that cell holds no water.
fn specular(s: u32, l: vec3<i32>, q: u32) -> vec2<u32> {
    if q < 7u {
        return vec2<u32>(NONE);
    }
    let c = C[q];
    var a1 = 1u;
    var a2 = 2u;
    if c.x != 0 && c.y != 0 {
        a1 = 0u;
        a2 = 1u;
    } else if c.x != 0 {
        a1 = 0u;
        a2 = 2u;
    }
    var b1 = vec3<i32>(0);
    b1[a1] = -c[a1];
    var b2 = vec3<i32>(0);
    b2[a2] = -c[a2];
    let n1 = neighbour(s, l, b1);
    let n2 = neighbour(s, l, b2);
    let s1 = n1 != NONE && kind[n1] == SOLID;
    let s2 = n2 != NONE && kind[n2] == SOLID;
    var m = NONE;
    var normal = 0u;
    if !s1 && s2 {
        m = n1;
        normal = a2;
    } else if s1 && !s2 {
        m = n2;
        normal = a1;
    }
    if m == NONE || (kind[m] != LIQUID && kind[m] != INTERFACE) {
        return vec2<u32>(NONE);
    }
    return vec2<u32>(m, mirror(q, normal));
}

// Relaxation time raised by the strain in the non-equilibrium populations.
fn les_tau(f: ptr<function, array<f32, 19>>, rho: f32, u: vec3<f32>) -> f32 {
    var d = vec3<f32>(0.0);
    var o = vec3<f32>(0.0);
    for (var q = 0u; q < Q; q++) {
        let neq = (*f)[q] - equilibrium(q, rho, u);
        let c = vec3<f32>(C[q]);
        d += c * c * neq;
        o += vec3<f32>(c.x * c.y, c.x * c.z, c.y * c.z) * neq;
    }
    let s = sqrt(dot(d, d) + 2.0 * dot(o, o));
    let tau0 = params.tau;
    let c2 = params.smagorinsky * params.smagorinsky;
    return 0.5 * (tau0 + sqrt(tau0 * tau0 + 18.0 * sqrt(2.0) * c2 * s / max(rho, 1e-6)));
}

@compute @workgroup_size(64)
fn stream_collide(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let s = lists[wg.x];
    let i = wg.y * 64u + li;
    let c = s * TILE_CELLS + i;
    let k = kind[c];
    conv[c] = 0u;
    if k == LIQUID || k == INTERFACE {
        let l = local_of(i);
        var fill_here = 1.0;
        if k == INTERFACE {
            fill_here = fill_src(c);
        }
        let u_old = rho_u[c].xyz;
        // The gas pressure plus the weight of the part of the cell's water
        // standing above its centre.
        let rho_gas = params.rho_gas + 3.0 * length(params.gravity) * (fill_here - 0.5);
        var fi: array<f32, 19>;
        fi[0] = pops_src[pop_index(c, 0u)];
        var dm = 0.0;
        var gas_near = false;
        var fluid_near = false;
        var blocked = false;
        for (var q = 1u; q < Q; q++) {
            let out_q = pops_src[pop_index(c, opp(q))];
            let src = neighbour(s, l, -C[q]);
            if src == NONE {
                // Beyond the allocated tiles lies gas.
                blocked = true;
                fi[q] = equilibrium(q, rho_gas, u_old) + equilibrium(opp(q), rho_gas, u_old) - out_q;
                continue;
            }
            blocked = blocked || tile_state[src / TILE_CELLS].awake == 0u;
            let nk = kind[src];
            if nk == SOLID {
                let m = specular(s, l, q);
                if m.x == NONE {
                    fi[q] = out_q;
                } else {
                    let slide = pops_src[pop_index(m.x, m.y)];
                    fi[q] = params.wall_slip * slide + (1.0 - params.wall_slip) * out_q;
                    if k == INTERFACE {
                        var exchange = params.wall_slip * (slide - out_q);
                        if kind[m.x] != LIQUID {
                            exchange *= 0.5 * (fill_here + fill_src(m.x));
                        }
                        dm += exchange;
                    }
                }
            } else if nk == GAS {
                gas_near = true;
                fi[q] = equilibrium(q, rho_gas, u_old) + equilibrium(opp(q), rho_gas, u_old) - out_q;
            } else {
                fluid_near = fluid_near || nk == LIQUID;
                let incoming = pops_src[pop_index(src, q)];
                fi[q] = incoming;
                if k == INTERFACE {
                    var exchange = incoming - out_q;
                    if nk != LIQUID {
                        exchange *= 0.5 * (fill_here + fill_src(src));
                    }
                    dm += exchange;
                }
            }
        }
        var r = 0.0;
        var j = vec3<f32>(0.0);
        for (var q = 0u; q < Q; q++) {
            r += fi[q];
            j += vec3<f32>(C[q]) * fi[q];
        }
        let force = params.gravity * r;
        var v = vec3<f32>(0.0);
        if r > 1e-6 {
            v = (j + 0.5 * force) / r;
        }
        let tau = les_tau(&fi, r, v);
        for (var q = 0u; q < Q; q++) {
            let feq = equilibrium(q, r, v);
            pops_dst[pop_index(c, q)] = fi[q] - (fi[q] - feq) / tau + guo(q, tau, v, force);
        }
        rho_u[c] = vec4<f32>(v, r);
        if length(v) > params.still_speed {
            atomicOr(&stirred, 1u);
        }
        if k == LIQUID {
            mass[c] = r;
        } else {
            let m = mass[c] + dm;
            mass[c] = m;
            var want = 0u;
            if m > (1.0 + params.fill_slack) * r || !gas_near {
                want = TO_LIQUID;
            } else if m < -params.fill_slack * r || (!fluid_near && m < 0.1 * r) {
                want = TO_GAS;
            }
            if want != 0u && blocked {
                atomicOr(&stirred, 1u);
            } else {
                conv[c] = want;
            }
        }
    }
    workgroupBarrier();
    if li == 0u && atomicLoad(&stirred) != 0u {
        atomicOr(&tile_state[s].moving[params.parity], 1u);
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
