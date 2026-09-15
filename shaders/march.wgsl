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
// Each level runs an Amanatides-Woo DDA whose state (cell, tMax) is kept on
// an explicit stack, so leaving a child resumes the parent exactly instead
// of re-deriving it from a position. A child DDA finds its first cell by
// clamping the entry point into the child's bounds (never by biasing t).
// Positions are relative to an integer origin voxel; cell bounds are formed
// by integer subtraction before conversion to float.
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

fn level_size(level: u32) -> i32 {
    return 8 << (2u * level);
}

fn level_dims(level: u32) -> vec3<i32> {
    if level == 5u {
        return vec3<i32>(WORLD_SECTORS_XZ, 1, WORLD_SECTORS_XZ);
    }
    return vec3<i32>(4);
}

struct Dda {
    cell: vec3<i32>,
    t_max: vec3<f32>,
}

fn dda_init(r: Ray, inv: vec3<f32>, min_voxel: vec3<i32>, size: i32, dims: vec3<i32>, t: f32) -> Dda {
    let rel_min = vec3<f32>(min_voxel - r.base);
    let s = f32(size);
    let p = r.frac + r.dir * t;
    let cell = clamp(vec3<i32>(floor((p - rel_min) / s)), vec3<i32>(0), dims - 1);
    let upper = select(vec3<f32>(0.0), vec3<f32>(1.0), r.dir > vec3<f32>(0.0));
    let boundary = rel_min + (vec3<f32>(cell) + upper) * s;
    return Dda(cell, (boundary - r.frac) * inv);
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

fn voxel_at(r: Ray, t: f32, lo: vec3<i32>, hi: vec3<i32>) -> vec3<i32> {
    let p = r.frac + r.dir * t;
    return clamp(r.base + vec3<i32>(floor(p)), lo, hi);
}

// Marches one brick between t0 and t1. `axis` is the face the ray entered by.
fn march_brick(r: Ray, inv: vec3<f32>, step: vec3<i32>, leaf: u32, brick_min: vec3<i32>, t0: f32, t1: f32, axis_in: u32, iterations: ptr<function, u32>) -> Hit {
    let base = leaf & BRICK_PTR_MASK;
    var axis = axis_in;
    var sub = dda_init(r, inv, brick_min, 4, vec3<i32>(2), t0);
    var t_sub = t0;
    let delta4 = 4.0 * abs(inv);
    let delta1 = abs(inv);
    loop {
        *iterations += 1u;
        let s = sub.cell;
        let s_index = u32(s.x + s.z * 2 + s.y * 4);
        let occ = brick_occupancy(base, s_index);
        let sub_exit = min(min(sub.t_max.x, sub.t_max.y), min(sub.t_max.z, t1));
        if (occ.x | occ.y) != 0u {
            let sub_min = brick_min + s * 4;
            var vd = dda_init(r, inv, sub_min, 1, vec3<i32>(4), t_sub);
            var t_v = t_sub;
            var v_axis = axis;
            loop {
                *iterations += 1u;
                let v = vd.cell;
                let bit = u32(v.x + v.z * 4 + v.y * 16);
                var occupied: bool;
                if bit < 32u {
                    occupied = ((occ.x >> bit) & 1u) != 0u;
                } else {
                    occupied = ((occ.y >> (bit - 32u)) & 1u) != 0u;
                }
                if occupied {
                    let local = s * 4 + v;
                    let m = brick_material(leaf, local);
                    return Hit(HIT_VOXEL, t_v, brick_min + local, v_axis, m, 0u, *iterations, leaf);
                }
                let a = min_axis(vd.t_max);
                if vd.t_max[a] >= sub_exit {
                    break;
                }
                vd.cell[a] += step[a];
                if vd.cell[a] < 0 || vd.cell[a] > 3 {
                    break;
                }
                t_v = vd.t_max[a];
                vd.t_max[a] += delta1[a];
                v_axis = a;
            }
        }
        let a = min_axis(sub.t_max);
        if sub.t_max[a] >= t1 {
            break;
        }
        sub.cell[a] += step[a];
        if sub.cell[a] < 0 || sub.cell[a] > 1 {
            break;
        }
        t_sub = sub.t_max[a];
        sub.t_max[a] += delta4[a];
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
    for (var a = 0u; a < 3u; a++) {
        if tn[a] > t0 {
            t0 = tn[a];
            axis = a;
        }
    }
    let t1 = min(min(tf.x, tf.y), min(tf.z, r.t_max));
    return vec4<f32>(t0, t1, f32(axis), 0.0);
}

fn march(r_in: Ray) -> Hit {
    var r = r_in;
    // Exact zeros would put 0 * inf into the boundary maths.
    r.dir = select(r.dir, vec3<f32>(1e-9), abs(r.dir) < vec3<f32>(1e-9));
    let inv = 1.0 / r.dir;
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
    var st_dda: array<Dda, 6>;
    var st_t: array<f32, 6>;

    var level = 5u;
    st_min[5] = vec3<i32>(0);
    st_dda[5] = dda_init(r, inv, vec3<i32>(0), SECTOR_VOXELS, level_dims(5u), clip.x);
    st_t[5] = clip.x;

    loop {
        iterations += 1u;
        if iterations > MAX_ITERATIONS {
            return miss_hit(iterations);
        }
        let d = st_dda[level];
        let cell = d.cell;
        let t_cell = st_t[level];
        let t_exit = min(min(d.t_max.x, d.t_max.y), min(d.t_max.z, t_end));
        let size = level_size(level);
        var descend = false;

        if level == 5u {
            let root = sectors[u32(cell.x + cell.z * WORLD_SECTORS_XZ)];
            if root != 0u {
                st_node[4] = root;
                st_min[4] = st_min[5] + cell * SECTOR_VOXELS;
                descend = true;
            }
        } else {
            let node = st_node[level];
            let idx = u32(cell.x + cell.z * 4 + cell.y * 16);
            if node_has_child(node, idx) {
                let w0 = tree[node];
                let cell_min = st_min[level] + cell * size;
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
                    let child = ptr + slot * 4u;
                    if f32(size) < t_cell * r.lod_scale {
                        let lod = tree[child + 3u];
                        return Hit(HIT_LOD, t_cell, voxel_at(r, t_cell, cell_min, cell_max), axis, lod & 0xffffu, lod, iterations, 0u);
                    }
                    st_node[level - 1u] = child;
                    st_min[level - 1u] = cell_min;
                    descend = true;
                }
            }
        }

        if descend {
            level -= 1u;
            st_dda[level] = dda_init(r, inv, st_min[level], level_size(level), vec3<i32>(4), t_cell);
            st_t[level] = t_cell;
            continue;
        }

        // Advance, popping levels whose grid the ray has left.
        loop {
            let a = min_axis(st_dda[level].t_max);
            let t_next = st_dda[level].t_max[a];
            if t_next >= t_end {
                return miss_hit(iterations);
            }
            st_dda[level].cell[a] += step[a];
            st_t[level] = t_next;
            st_dda[level].t_max[a] += f32(level_size(level)) * abs(inv[a]);
            axis = a;
            let c = st_dda[level].cell[a];
            if c >= 0 && c < level_dims(level)[a] {
                break;
            }
            level += 1u;
            if level > 5u {
                return miss_hit(iterations);
            }
        }
    }
    return miss_hit(iterations);
}
