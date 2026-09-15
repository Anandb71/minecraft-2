// Voxel structure bindings (group 1) and accessors. Word layout is defined
// in crates/mc2-voxel/src/gpu_layout.rs; keep the two in sync.

struct Material {
    albedo: vec3<f32>,
    roughness: f32,
    emission: vec3<f32>,
    metallic: f32,
    ior: f32,
    kind: u32,
    flags: u32,
    _pad: f32,
}

@group(1) @binding(0) var<storage, read> tree: array<u32>;
@group(1) @binding(1) var<storage, read> voxels: array<u32>;
@group(1) @binding(2) var<storage, read> sectors: array<u32>;
@group(1) @binding(3) var<storage, read> materials: array<Material>;
@group(1) @binding(4) var<storage, read_write> feedback: array<u32>;

const LEAF_PARENT: u32 = 0x80000000u;
const NOT_RESIDENT: u32 = 0x40000000u;
const PTR_MASK: u32 = 0x3fffffffu;
const UNIFORM: u32 = 0x80000000u;
const WIDE: u32 = 0x20000000u;
const BRICK_PTR_MASK: u32 = 0x1fffffffu;

const WORLD_SECTORS_XZ: i32 = 32;
const SECTOR_VOXELS: i32 = 8192;

const KIND_AIR: u32 = 0u;
const KIND_SOLID: u32 = 1u;
const KIND_FOLIAGE: u32 = 2u;
const KIND_TRANSPARENT: u32 = 3u;
const KIND_LIQUID: u32 = 4u;

fn node_has_child(node: u32, idx: u32) -> bool {
    if idx < 32u {
        return ((tree[node + 1u] >> idx) & 1u) != 0u;
    }
    return ((tree[node + 2u] >> (idx - 32u)) & 1u) != 0u;
}

fn node_child_slot(node: u32, idx: u32) -> u32 {
    let lo = tree[node + 1u];
    if idx < 32u {
        return countOneBits(lo & ((1u << idx) - 1u));
    }
    return countOneBits(lo) + countOneBits(tree[node + 2u] & ((1u << (idx - 32u)) - 1u));
}

// True when every cell of the 2x2x2 group containing child `idx` is empty.
fn node_group_empty(node: u32, idx: u32) -> bool {
    let base = idx & 0x2au;
    let group = 0x00330033u;
    let lo = tree[node + 1u];
    let hi = tree[node + 2u];
    // The 2^3 group spans bits base .. base+21; split across the halves.
    var m: u32;
    if base >= 32u {
        m = (hi >> (base - 32u)) & group;
    } else if base + 21u < 32u {
        m = (lo >> base) & group;
    } else {
        let lo_part = (lo >> base) & group;
        let hi_part = (hi << (32u - base)) & group;
        m = lo_part | hi_part;
    }
    return m == 0u;
}

fn brick_occupancy(base: u32, subblock: u32) -> vec2<u32> {
    return vec2<u32>(voxels[base + 1u + subblock * 2u], voxels[base + 2u + subblock * 2u]);
}

fn brick_material(leaf: u32, local: vec3<i32>) -> u32 {
    let base = leaf & BRICK_PTR_MASK;
    let i = u32(local.x + local.z * 8 + local.y * 64);
    var p: u32;
    if (leaf & WIDE) != 0u {
        p = (voxels[base + 145u + i / 4u] >> (8u * (i % 4u))) & 0xffu;
    } else {
        p = (voxels[base + 25u + i / 8u] >> (4u * (i % 8u))) & 0xfu;
    }
    return (voxels[base + 17u + p / 2u] >> (16u * (p & 1u))) & 0xffffu;
}

fn brick_state(leaf: u32, local: vec3<i32>) -> u32 {
    let base = leaf & BRICK_PTR_MASK;
    let state = voxels[base];
    if state == 0u {
        return 0u;
    }
    let i = u32(local.x + local.z * 8 + local.y * 64);
    return (voxels[state + i / 4u] >> (8u * (i % 4u))) & 0xffu;
}

fn request_upload(word: u32) {
    let n = arrayLength(&feedback);
    let h = (word * 2654435761u) % n;
    feedback[h] = word + 1u;
}
