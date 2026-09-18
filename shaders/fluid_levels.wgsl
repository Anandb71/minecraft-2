// Water levels for the world: each cell's fill in eighths of its height
// (the voxel layers it fills), four cells to a word, over every allocated
// slot. The CPU writes them into the voxel world.

// Its own bindings, so not fluid_common's; the same cell kinds.
const TILE_CELLS: u32 = 512u;
const INTERFACE: u32 = 1u;
const LIQUID: u32 = 2u;

struct LevelParams {
    slot_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> lp: LevelParams;
@group(0) @binding(1) var<storage, read> kind_in: array<u32>;
@group(0) @binding(2) var<storage, read> mass_in: array<f32>;
@group(0) @binding(3) var<storage, read> rho_u_in: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> levels: array<u32>;

@compute @workgroup_size(64)
fn pack_levels(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= lp.slot_count * (TILE_CELLS / 4u) {
        return;
    }
    var word = 0u;
    for (var k = 0u; k < 4u; k++) {
        let c = id.x * 4u + k;
        let kd = kind_in[c];
        var eighths = 0u;
        if kd == LIQUID {
            eighths = 8u;
        } else if kd == INTERFACE {
            let f = clamp(mass_in[c] / max(rho_u_in[c].w, 1e-6), 0.0, 1.0);
            eighths = u32(round(f * 8.0));
        }
        word |= eighths << (k * 8u);
    }
    levels[id.x] = word;
}
