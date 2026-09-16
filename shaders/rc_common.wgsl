// Radiance cascades (Sannikov 2023, section 4.5): screen-space probes on the
// visibility buffer holding world-space radiance intervals. Cascade i places
// a probe every SPACING0 * 2^i render pixels with DIRS0 * 2^i octahedral
// directions per axis over the full sphere, and covers the distance interval
// [T0 (4^i - 1) / 3, T0 (4^(i+1) - 1) / 3]. The paper's penumbra condition
// doubles the interval with the spacing; fourfold intervals trade some
// angular resolution far away for reach (341 m in six cascades) and measured
// 3% bias against 14% (D19). Every cascade's atlas has the same size, so the
// whole hierarchy costs a constant per cascade.
//
// Atlas texel (x, y) of cascade i is probe (x, y) / dirs, direction
// (x, y) % dirs; rgb holds radiance, a the interval's transparency.

struct RcParams {
    cascade: u32,
    cascades: u32,
    // Render pixels between probes at this cascade.
    spacing: u32,
    // Directions per axis at this cascade.
    dirs: u32,
    t_min_m: f32,
    t_max_m: f32,
    // GI signal resolution shift (gather pass).
    shift: u32,
    _pad: u32,
}

const RC_SPACING0: u32 = 16u;
const RC_DIRS0: u32 = 4u;

fn rc_direction(texel_in_probe: vec2<u32>, dirs: u32) -> vec3<f32> {
    let uv = (vec2<f32>(texel_in_probe) + 0.5) / f32(dirs);
    return oct_decode(uv);
}

// Octahedral texel of a direction at a given resolution.
fn rc_texel(dir: vec3<f32>, dirs: u32) -> vec2<u32> {
    let uv = oct_encode(dir);
    return min(vec2<u32>(uv * f32(dirs)), vec2<u32>(dirs - 1u));
}

// Render pixel a probe sits on.
fn rc_probe_pixel(probe: vec2<u32>, spacing: u32) -> vec2<i32> {
    let p = probe * spacing + spacing / 2u;
    return min(vec2<i32>(p), vec2<i32>(frame.render_size) - 1);
}

fn rc_probe_count(spacing: u32) -> vec2<u32> {
    return (vec2<u32>(frame.render_size) + spacing - 1u) / spacing;
}
