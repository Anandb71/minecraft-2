// Rigid bodies: dense voxel grids with a pose, traced after the world.
// Layouts match crates/mc2-render/src/bodies.rs.
//
// A uniform grid over the bodies' combined bounds lists the bodies whose
// boxes touch each cell; a ray that misses those bounds pays one slab test.
// Inside a body the ray is transformed into the grid frame (orthonormal
// axes, voxel units, so t is unchanged) and marched over 4^3 occupancy
// blocks, then voxels.
//
// A body hit reports kind HIT_BODY, its grid voxel, the grid axis of the
// face it entered, and the body's table index in `lod`.
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"

struct GpuBody {
    origin: vec3<f32>,
    tag: u32,
    axis_x: vec3<f32>,
    voxels: u32,
    axis_y: vec3<f32>,
    coarse: u32,
    axis_z: vec3<f32>,
    _pad0: u32,
    size: vec3<i32>,
    _pad1: u32,
    prev_origin: vec3<f32>,
    _pad2: u32,
    prev_axis_x: vec3<f32>,
    _pad3: u32,
    prev_axis_y: vec3<f32>,
    _pad4: u32,
    prev_axis_z: vec3<f32>,
    _pad5: u32,
}

struct BodyGrid {
    min: vec3<f32>,
    cell: f32,
    dims: vec3<i32>,
    count: u32,
    // Cells (list offset << 12 | count), then the body index lists.
    words: array<u32>,
}

@group(1) @binding(5) var<storage, read> bodies: array<GpuBody>;
@group(1) @binding(6) var<storage, read> body_voxels: array<u32>;
@group(1) @binding(7) var<storage, read> body_grid: BodyGrid;

const HIT_BODY: u32 = 2u;
const BODY_GRID_STEPS: u32 = 100u;
// Blocks plus voxels one body may cost a ray.
const BODY_STEPS: u32 = 320u;

// Avoids infinities for axis-parallel rays; the sign is irrelevant there.
fn safe_dir(d: vec3<f32>) -> vec3<f32> {
    return select(d, vec3<f32>(1e-9), abs(d) < vec3<f32>(1e-9));
}

fn body_block_occupied(coarse: u32, blocks: vec3<i32>, c: vec3<i32>) -> bool {
    let bit = u32(c.x + blocks.x * (c.y + blocks.y * c.z));
    return ((body_voxels[coarse + bit / 32u] >> (bit % 32u)) & 1u) != 0u;
}

fn body_material(voxels: u32, size: vec3<i32>, v: vec3<i32>) -> u32 {
    let i = u32(v.x + size.x * (v.y + size.y * v.z));
    return (body_voxels[voxels + i / 2u] >> (16u * (i % 2u))) & 0xffffu;
}

// Marches body k over [t0, t1] from camera-relative origin `o` (voxels).
fn trace_body(k: u32, o: vec3<f32>, dir: vec3<f32>, t0: f32, t1: f32, iterations: ptr<function, u32>) -> Hit {
    let size = bodies[k].size;
    let voxels = bodies[k].voxels;
    let coarse = bodies[k].coarse;
    let ax = bodies[k].axis_x;
    let ay = bodies[k].axis_y;
    let az = bodies[k].axis_z;
    let rel = o - bodies[k].origin;
    let p = vec3<f32>(dot(rel, ax), dot(rel, ay), dot(rel, az));
    let d = safe_dir(vec3<f32>(dot(dir, ax), dot(dir, ay), dot(dir, az)));
    let inv = 1.0 / d;
    let ta = -p * inv;
    let tb = (vec3<f32>(size) - p) * inv;
    let tn3 = min(ta, tb);
    let tf3 = max(ta, tb);
    let tn = max(max(tn3.x, tn3.y), tn3.z);
    let tf = min(min(tf3.x, tf3.y), tf3.z);
    let start = max(tn, t0);
    let end = min(tf, t1);
    if start >= end {
        return miss_hit(0u);
    }
    var axis = 2u;
    if tn3.x >= tn3.y && tn3.x >= tn3.z {
        axis = 0u;
    } else if tn3.y >= tn3.z {
        axis = 1u;
    }
    let step = vec3<i32>(sign(d));
    let upper = select(vec3<f32>(0.0), vec3<f32>(1.0), d > vec3<f32>(0.0));
    let blocks = (size + 3) / 4;
    let delta1 = abs(inv);
    let delta4 = delta1 * 4.0;
    var c = clamp(vec3<i32>(floor((p + d * start) / 4.0)), vec3<i32>(0), blocks - 1);
    var c_tmax = ((vec3<f32>(c) + upper) * 4.0 - p) * inv;
    var t_c = start;
    var steps = 0u;
    loop {
        steps += 1u;
        if steps > BODY_STEPS {
            break;
        }
        let c_exit = min(min(c_tmax.x, c_tmax.y), min(c_tmax.z, end));
        if body_block_occupied(coarse, blocks, c) {
            let lo = c * 4;
            let hi = min(lo + 4, size) - 1;
            var v = clamp(vec3<i32>(floor(p + d * t_c)), lo, hi);
            var v_tmax = (vec3<f32>(v) + upper - p) * inv;
            var t_v = t_c;
            var v_axis = axis;
            loop {
                steps += 1u;
                let m = body_material(voxels, size, v);
                if m != 0u {
                    *iterations += steps;
                    return Hit(HIT_BODY, t_v, v, v_axis, m, k, *iterations, 0u);
                }
                let a = min_axis(v_tmax);
                t_v = v_tmax[a];
                if t_v >= c_exit || steps > BODY_STEPS {
                    break;
                }
                v += axis_step(a, step);
                v_tmax += axis_unit(a) * delta1;
                v_axis = a;
                if any(v < lo) || any(v > hi) {
                    break;
                }
            }
        }
        let a = min_axis(c_tmax);
        t_c = c_tmax[a];
        if t_c >= end {
            break;
        }
        c += axis_step(a, step);
        c_tmax += axis_unit(a) * delta4;
        axis = a;
        if any(c < vec3<i32>(0)) || any(c >= blocks) {
            break;
        }
    }
    *iterations += steps;
    return miss_hit(0u);
}

// Nearest body hit along the ray over [t_start, t_end] (voxels).
fn trace_bodies(r: Ray, t_start: f32, t_end: f32) -> Hit {
    var best = miss_hit(0u);
    if body_grid.count == 0u {
        return best;
    }
    let o = vec3<f32>(r.base - frame.camera_voxel) + r.frac;
    let d = safe_dir(r.dir);
    let inv = 1.0 / d;
    let dims = body_grid.dims;
    let cell = body_grid.cell;
    let gmin = body_grid.min;
    let ta = (gmin - o) * inv;
    let tb = (gmin + vec3<f32>(dims) * cell - o) * inv;
    let tn3 = min(ta, tb);
    let tf3 = max(ta, tb);
    var t = max(max(max(tn3.x, tn3.y), tn3.z), t_start);
    let end = min(min(min(tf3.x, tf3.y), tf3.z), t_end);
    if t >= end {
        return best;
    }
    let step = vec3<i32>(sign(d));
    let upper = select(vec3<f32>(0.0), vec3<f32>(1.0), d > vec3<f32>(0.0));
    var c = clamp(vec3<i32>(floor((o + d * t - gmin) / cell)), vec3<i32>(0), dims - 1);
    var tmax = (gmin + (vec3<f32>(c) + upper) * cell - o) * inv;
    let delta = abs(inv) * cell;
    var iterations = 0u;
    for (var s = 0u; s < BODY_GRID_STEPS; s++) {
        let exit = min(min(tmax.x, tmax.y), min(tmax.z, end));
        let w = body_grid.words[u32(c.x + dims.x * (c.y + dims.y * c.z))];
        let first = w >> 12u;
        let n = w & 0xfffu;
        for (var i = 0u; i < n; i++) {
            let h = trace_body(body_grid.words[first + i], o, r.dir, t, min(end, best.t), &iterations);
            if h.kind != HIT_NONE && h.t < best.t {
                best = h;
            }
        }
        // Bodies listed in later cells cannot be nearer than a hit here.
        if best.t <= exit || exit >= end {
            break;
        }
        let a = min_axis(tmax);
        c += axis_step(a, step);
        tmax += axis_unit(a) * delta;
        t = exit;
        if any(c < vec3<i32>(0)) || any(c >= dims) {
            break;
        }
    }
    return Hit(best.kind, best.t, best.voxel, best.axis, best.material, best.lod, iterations, best.leaf);
}

// The world, then bodies in front of it. Bodies are searched from
// `body_t_min`: a primary ray starts the world march at the beam distance,
// but a body can sit in front of that.
fn trace_scene(r: Ray, body_t_min: f32) -> Hit {
    let world = march(r);
    let body = trace_bodies(r, body_t_min, min(world.t, r.t_max));
    if body.kind != HIT_NONE {
        return Hit(body.kind, body.t, body.voxel, body.axis, body.material, body.lod, body.iterations + world.iterations, 0u);
    }
    return world;
}

// World axis of a hit face before orientation.
fn hit_axis_world(h: Hit) -> vec3<f32> {
    if h.kind == HIT_BODY {
        let k = h.lod;
        return select(select(bodies[k].axis_z, bodies[k].axis_y, h.axis == 1u), bodies[k].axis_x, h.axis == 0u);
    }
    return axis_unit(h.axis);
}

// Normal of the face a ray entered.
fn hit_normal(h: Hit, dir: vec3<f32>) -> vec3<f32> {
    let a = hit_axis_world(h);
    return select(a, -a, dot(dir, a) > 0.0);
}

// Visibility id word: material, entry face (grid axes for bodies), body
// tag and kind.
fn hit_id_word(h: Hit, dir: vec3<f32>) -> u32 {
    let face = h.axis * 2u + select(0u, 1u, dot(dir, hit_axis_world(h)) < 0.0);
    var tag = 0u;
    if h.kind == HIT_BODY {
        tag = bodies[h.lod].tag & 0x7ffu;
    }
    return h.material | (face << 16u) | (tag << 19u) | (h.kind << 30u);
}

// Where a point on body k (camera-relative voxels) was last frame, in the
// same frame of reference.
fn body_prev_point(k: u32, rel_voxels: vec3<f32>) -> vec3<f32> {
    let d = rel_voxels - bodies[k].origin;
    let g = vec3<f32>(dot(d, bodies[k].axis_x), dot(d, bodies[k].axis_y), dot(d, bodies[k].axis_z));
    return bodies[k].prev_origin + bodies[k].prev_axis_x * g.x + bodies[k].prev_axis_y * g.y + bodies[k].prev_axis_z * g.z;
}
