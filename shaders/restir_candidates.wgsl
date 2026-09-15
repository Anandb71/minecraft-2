// ReSTIR step 1: M = 32 candidates per pixel drawn by emitter power through
// an alias table, resampled by the unshadowed contribution, then the
// survivor's visibility is traced and occluded samples dropped (W = 0)
// before they can spread through temporal and spatial reuse.
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"
#import "brdf.wgsl"
#import "restir_common.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var out_a: texture_storage_2d<rgba32float, write>;
@group(2) @binding(4) var out_b: texture_storage_2d<rgba32float, write>;
#import "restir_surface.wgsl"

const CANDIDATES: u32 = 32u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.render_size);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let id = textureLoad(vis_id, pixel, 0);
    var r = empty_reservoir();
    let slots = arrayLength(&alias_table);
    if (id.w >> 30u) == 0u || slots == 0u {
        textureStore(out_a, pixel, pack_a(r));
        textureStore(out_b, pixel, pack_b(r));
        return;
    }
    let s = shading_at(pixel, id);
    var rng = rng_seed(gid.xy, frame.frame_index ^ 0x5e5717u);
    for (var i = 0u; i < CANDIDATES; i++) {
        let bin = min(u32(rng_next(&rng) * f32(slots)), slots - 1u);
        var slot = bin;
        if rng_next(&rng) >= alias_table[bin].probability {
            slot = alias_table[bin].alias_slot;
        }
        let light = lights[slot];
        let source_pdf = alias_table[slot].pdf;
        if light.count == 0u || source_pdf <= 0.0 {
            r.m += 1.0;
            continue;
        }
        let offset = random_offset(vec2<f32>(rng_next(&rng), rng_next(&rng)));
        let p_hat = target_weight(light_contribution(s, light, offset));
        reservoir_update(&r, slot, offset, p_hat / source_pdf, p_hat, rng_next(&rng));
    }
    if r.light != 0xffffffffu && r.target_pdf > 0.0 {
        r.w = r.w_sum / (r.m * r.target_pdf);
        let hit = march(shadow_ray_to(s, face_of(id), sample_point(lights[r.light], r.offset)));
        if hit.kind != HIT_NONE {
            r.w = 0.0;
        }
    } else {
        r.w = 0.0;
    }
    textureStore(out_a, pixel, pack_a(r));
    textureStore(out_b, pixel, pack_b(r));
}
