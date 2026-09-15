// Hierarchical voxel ray marcher.
//
// Levels, by child cell size in voxels:
//   5: world grid of sectors (8192, 32 x 1 x 32)
//   4: sector root children (2048)
//   3: 128 m node children (512, chunk roots)
//   2: chunk root children (128)
//   1: 8 m node children (32)
//   0: 2 m node children, brick cells (8)
// then brick subblocks (4, 2x2x2 with a 64-bit mask each) and voxels (1).
//
// Each level runs an Amanatides-Woo DDA. The active level's state lives in
// plain local vectors; descending pushes it onto small per-field stacks and
// leaving a child pops it back, so the parent resumes exactly instead of
// re-deriving its cell from a position. A child DDA finds its first cell by
// clamping the entry point into the child's bounds (never by biasing t).
// Positions are relative to an integer origin voxel; cell bounds are formed
// by integer subtraction before conversion to float.
//
// Written for the weakest compilers we ship to: no writes through struct or
// array element chains (FXC rejects them as l-values), no struct copies in
// the hot loop.
#import "voxel_data.wgsl"

const MAX_ITERATIONS: u32 = 1024u;
const HIT_NONE: u32 = 0u;
const HIT_VOXEL: u32 = 1u;
const HIT_LOD: u32 = 3u;

struct Ray {
    // Integer origin voxel and offset from it, in voxels.
    base: vec3<i32>,
    frac: vec3<f32>,
    dir: vec3<f32>,
    t_max: f32,
    // Stop descending when a cell is smaller than this many voxels per unit t.
    lod_scale: f32,
    // Record non-resident data the ray touched.
    feedback: bool,
}

struct Hit {
    kind: u32,
    t: f32,
    voxel: vec3<i32>,
    axis: u32,
    material: u32,
    lod: u32,
    iterations: u32,
    // Leaf word of the brick hit, 0 for uniform and LOD hits.
    leaf: u32,
}

fn miss_hit(iterations: u32) -> Hit {
    return Hit(HIT_NONE, 1e30, vec3<i32>(0), 0u, 0u, 0u, iterations, 0u);
}

fn min_axis(v: vec3<f32>) -> u32 {
    if v.x <= v.y && v.x <= v.z {
        return 0u;
    }
    if v.y <= v.z {
        return 1u;
    }
    return 2u;
}

fn first_cell(r: Ray, rel_min: vec3<f32>, size: f32, dims: vec3<i32>, t: f32) -> vec3<i32> {
    let p = r.frac + r.dir * t;
    return clamp(vec3<i32>(floor((p - rel_min) / size)), vec3<i32>(0), dims - 1);
}

fn first_tmax(r: Ray, inv: vec3<f32>, rel_min: vec3<f32>, size: f32, cell: vec3<i32>) -> vec3<f32> {
    let upper = select(vec3<f32>(0.0), vec3<f32>(1.0), r.dir > vec3<f32>(0.0));
    return (rel_min + (vec3<f32>(cell) + upper) * size - r.frac) * inv;
}

fn voxel_at(r: Ray, t: f32, lo: vec3<i32>, hi: vec3<i32>) -> vec3<i32> {
    let p = r.frac + r.dir * t;
    return clamp(r.base + vec3<i32>(floor(p)), lo, hi);
}

// Vector writes through a dynamic index are not l-values under FXC; build
// axis vectors with select instead.
fn axis_mask(a: u32) -> vec3<bool> {
    return vec3<bool>(a == 0u, a == 1u, a == 2u);
}

fn axis_step(a: u32, s: vec3<i32>) -> vec3<i32> {
    return select(vec3<i32>(0), s, axis_mask(a));
}

fn axis_unit(a: u32) -> vec3<f32> {
    return select(vec3<f32>(0.0), vec3<f32>(1.0), axis_mask(a));
}

// Marches one brick between t0 and t1. `axis_in` is the face the ray entered by.
fn march_brick(r: Ray, inv: vec3<f32>, step: vec3<i32>, leaf: u32, brick_min: vec3<i32>, t0: f32, t1: f32, axis_in: u32, iterations: ptr<function, u32>) -> Hit {
    let base = leaf & BRICK_PTR_MASK;
    let rel_brick = vec3<f32>(brick_min - r.base);
    let delta4 = 4.0 * abs(inv);
    let delta1 = abs(inv);
    var axis = axis_in;
    var s = first_cell(r, rel_brick, 4.0, vec3<i32>(2), t0);
    var s_tmax = first_tmax(r, inv, rel_brick, 4.0, s);
    var t_sub = t0;
    loop {
        *iterations += 1u;
        let s_index = u32(s.x + s.z * 2 + s.y * 4);
        let occ_lo = voxels[base + 1u + s_index * 2u];
        let occ_hi = voxels[base + 2u + s_index * 2u];
        let sub_exit = min(min(s_tmax.x, s_tmax.y), min(s_tmax.z, t1));
        if (occ_lo | occ_hi) != 0u {
            let rel_sub = rel_brick + vec3<f32>(s * 4);
            var v = first_cell(r, rel_sub, 1.0, vec3<i32>(4), t_sub);
            var v_tmax = first_tmax(r, inv, rel_sub, 1.0, v);
            var t_v = t_sub;
            var v_axis = axis;
            loop {
                *iterations += 1u;
                let bit = u32(v.x + v.z * 4 + v.y * 16);
                var occupied: bool;
                if bit < 32u {
                    occupied = ((occ_lo >> bit) & 1u) != 0u;
                } else {
                    occupied = ((occ_hi >> (bit - 32u)) & 1u) != 0u;
                }
                if occupied {
                    let local = s * 4 + v;
                    let m = brick_material(leaf, local);
                    return Hit(HIT_VOXEL, t_v, brick_min + local, v_axis, m, 0u, *iterations, leaf);
                }
                let a = min_axis(v_tmax);
                let t_next = v_tmax[a];
                if t_next >= sub_exit {
                    break;
                }
                v += axis_step(a, step);
                if any(v < vec3<i32>(0)) || any(v > vec3<i32>(3)) {
                    break;
                }
                t_v = t_next;
                v_tmax += axis_unit(a) * delta1;
                v_axis = a;
            }
        }
        let a = min_axis(s_tmax);
        let t_next = s_tmax[a];
        if t_next >= t1 {
            break;
        }
        s += axis_step(a, step);
        if any(s < vec3<i32>(0)) || any(s > vec3<i32>(1)) {
            break;
        }
        t_sub = t_next;
        s_tmax += axis_unit(a) * delta4;
        axis = a;
    }
    return miss_hit(*iterations);
}

// Clips a ray to the world box; returns (t_enter, t_exit, entry axis).
fn clip_world(r: Ray, inv: vec3<f32>) -> vec4<f32> {
    let lo = vec3<f32>(vec3<i32>(0) - r.base) - r.frac;
    let hi = vec3<f32>(vec3<i32>(WORLD_SECTORS_XZ * SECTOR_VOXELS, SECTOR_VOXELS, WORLD_SECTORS_XZ * SECTOR_VOXELS) - r.base) - r.frac;
    let ta = lo * inv;
    let tb = hi * inv;
    let tn = min(ta, tb);
    let tf = max(ta, tb);
    var t0 = 0.0;
    var axis = 0u;
    if tn.x > t0 {
        t0 = tn.x;
        axis = 0u;
    }
    if tn.y > t0 {
        t0 = tn.y;
        axis = 1u;
    }
    if tn.z > t0 {
        t0 = tn.z;
        axis = 2u;
    }
    let t1 = min(min(tf.x, tf.y), min(tf.z, r.t_max));
    return vec4<f32>(t0, t1, f32(axis), 0.0);
}

fn march(r_in: Ray) -> Hit {
    var r = r_in;
    // Exact zeros would put 0 * inf into the boundary maths.
    r.dir = select(r.dir, vec3<f32>(1e-9), abs(r.dir) < vec3<f32>(1e-9));
    let inv = 1.0 / r.dir;
    let abs_inv = abs(inv);
    let step = vec3<i32>(sign(r.dir));
    var iterations = 0u;

    let clip = clip_world(r, inv);
    if clip.x > clip.y {
        return miss_hit(0u);
    }
    let t_end = clip.y;
    var axis = u32(clip.z);

    var st_node: array<u32, 6>;
    var st_min: array<vec3<i32>, 6>;
    var st_cell: array<vec3<i32>, 6>;
    var st_tmax: array<vec3<f32>, 6>;

    // Active level state.
    var level = 5u;
    var node = 0u;
    var node_min = vec3<i32>(0);
    var size = SECTOR_VOXELS;
    var dims = vec3<i32>(WORLD_SECTORS_XZ, 1, WORLD_SECTORS_XZ);
    var delta = f32(size) * abs_inv;
    var t_cell = clip.x;
    var cell = first_cell(r, vec3<f32>(-r.base), f32(size), dims, t_cell);
    var tmax = first_tmax(r, inv, vec3<f32>(-r.base), f32(size), cell);

    loop {
        iterations += 1u;
        if iterations > MAX_ITERATIONS {
            return miss_hit(iterations);
        }
        let t_exit = min(min(tmax.x, tmax.y), min(tmax.z, t_end));
        var child = 0u;
        var child_min = vec3<i32>(0);
        var descend = false;

        if level == 5u {
            let root = sectors[u32(cell.x + cell.z * WORLD_SECTORS_XZ)];
            if root != 0u {
                child = root;
                child_min = cell * SECTOR_VOXELS;
                descend = true;
            }
        } else {
            let idx = u32(cell.x + cell.z * 4 + cell.y * 16);
            if node_has_child(node, idx) {
                let w0 = tree[node];
                let cell_min = node_min + cell * size;
                let cell_max = cell_min + vec3<i32>(size - 1);
                if (w0 & NOT_RESIDENT) != 0u {
                    if r.feedback {
                        request_upload(node);
                    }
                    let lod = tree[node + 3u];
                    return Hit(HIT_LOD, t_cell, voxel_at(r, t_cell, cell_min, cell_max), axis, lod & 0xffffu, lod, iterations, 0u);
                }
                let slot = node_child_slot(node, idx);
                let ptr = w0 & PTR_MASK;
                if (w0 & LEAF_PARENT) != 0u {
                    let word = ptr + slot;
                    let leaf = tree[word];
                    if (leaf & UNIFORM) != 0u {
                        return Hit(HIT_VOXEL, t_cell, voxel_at(r, t_cell, cell_min, cell_max), axis, leaf & 0xffffu, 0u, iterations, 0u);
                    }
                    if (leaf & NOT_RESIDENT) != 0u {
                        if r.feedback {
                            request_upload(word);
                        }
                        return Hit(HIT_LOD, t_cell, voxel_at(r, t_cell, cell_min, cell_max), axis, leaf & 0xffffu, 0u, iterations, 0u);
                    }
                    // Sub-pixel brick: shade it from its first palette entry.
                    if f32(size) < t_cell * r.lod_scale {
                        let m = (voxels[(leaf & BRICK_PTR_MASK) + 17u] >> 16u) & 0xffffu;
                        return Hit(HIT_LOD, t_cell, voxel_at(r, t_cell, cell_min, cell_max), axis, m, 0u, iterations, 0u);
                    }
                    let h = march_brick(r, inv, step, leaf, cell_min, t_cell, t_exit, axis, &iterations);
                    if h.kind != HIT_NONE {
                        return h;
                    }
                } else {
                    let c = ptr + slot * 4u;
                    if f32(size) < t_cell * r.lod_scale {
                        let lod = tree[c + 3u];
                        return Hit(HIT_LOD, t_cell, voxel_at(r, t_cell, cell_min, cell_max), axis, lod & 0xffffu, lod, iterations, 0u);
                    }
                    child = c;
                    child_min = cell_min;
                    descend = true;
                }
            }
        }

        if descend {
            st_node[level] = node;
            st_min[level] = node_min;
            st_cell[level] = cell;
            st_tmax[level] = tmax;
            level -= 1u;
            node = child;
            node_min = child_min;
            size = size >> 2u;
            if level == 4u {
                size = 2048;
            }
            dims = vec3<i32>(4);
            delta = f32(size) * abs_inv;
            let rel_min = vec3<f32>(node_min - r.base);
            cell = first_cell(r, rel_min, f32(size), dims, t_cell);
            tmax = first_tmax(r, inv, rel_min, f32(size), cell);
            continue;
        }

        // Advance, popping levels whose grid the ray has left.
        loop {
            let a = min_axis(tmax);
            let t_next = tmax[a];
            if t_next >= t_end {
                return miss_hit(iterations);
            }
            cell += axis_step(a, step);
            t_cell = t_next;
            tmax += axis_unit(a) * delta;
            axis = a;
            if all(cell >= vec3<i32>(0)) && all(cell < dims) {
                break;
            }
            level += 1u;
            if level > 5u {
                return miss_hit(iterations);
            }
            node = st_node[level];
            node_min = st_min[level];
            cell = st_cell[level];
            tmax = st_tmax[level];
            if level == 5u {
                size = SECTOR_VOXELS;
                dims = vec3<i32>(WORLD_SECTORS_XZ, 1, WORLD_SECTORS_XZ);
            } else {
                size = 8 << (2u * level);
            }
            delta = f32(size) * abs_inv;
        }
    }
    return miss_hit(iterations);
}
