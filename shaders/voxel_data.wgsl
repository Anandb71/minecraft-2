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
const UNIFORM_NODE: u32 = 0x20000000u;
const PTR_MASK: u32 = 0x1fffffffu;
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

// Mask helpers over a node's cached child mask halves.
fn mask_has(lo: u32, hi: u32, idx: u32) -> bool {
    if idx < 32u {
        return ((lo >> idx) & 1u) != 0u;
    }
    return ((hi >> (idx - 32u)) & 1u) != 0u;
}

fn mask_slot(lo: u32, hi: u32, idx: u32) -> u32 {
    if idx < 32u {
        return countOneBits(lo & ((1u << idx) - 1u));
    }
    return countOneBits(lo) + countOneBits(hi & ((1u << (idx - 32u)) - 1u));
}

// Group bases are 0, 2, 8, 10 (low half) or 32, 34, 40, 42 (high half),
// so the 22-bit group span never straddles the two words.
fn mask_group_empty(lo: u32, hi: u32, idx: u32) -> bool {
    let base = idx & 0x2au;
    if base >= 32u {
        return ((hi >> (base - 32u)) & 0x00330033u) == 0u;
    }
    return ((lo >> base) & 0x00330033u) == 0u;
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
