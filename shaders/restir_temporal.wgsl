// ReSTIR step 2: temporal reuse. The previous frame's reservoir at the
// reprojected pixel is combined (Alg. 4) when that pixel saw the exact same
// voxel, which voxel ids make an exact test rather than a depth heuristic.
// The history's M is clamped to 20x the current M.
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"
#import "brdf.wgsl"
#import "restir_common.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var cur_a: texture_2d<f32>;
@group(2) @binding(4) var cur_b: texture_2d<f32>;
@group(2) @binding(5) var prev_id: texture_2d<u32>;
@group(2) @binding(6) var prev_a: texture_2d<f32>;
@group(2) @binding(7) var prev_b: texture_2d<f32>;
@group(2) @binding(8) var out_a: texture_storage_2d<rgba32float, write>;
@group(2) @binding(9) var out_b: texture_storage_2d<rgba32float, write>;
#import "restir_surface.wgsl"

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.render_size);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let id = textureLoad(vis_id, pixel, 0);
    let cur = unpack(textureLoad(cur_a, pixel, 0), textureLoad(cur_b, pixel, 0));
    if (id.w >> 30u) == 0u {
        textureStore(out_a, pixel, pack_a(cur));
        textureStore(out_b, pixel, pack_b(cur));
        return;
    }
    let mv = textureLoad(vis_motion, pixel, 0).rg;
    let uv = (vec2<f32>(pixel) + 0.5) / frame.render_size;
    let prev_pixel = vec2<i32>(floor((uv + mv) * frame.render_size));
    var prev = empty_reservoir();
    var have_prev = false;
    if all(prev_pixel >= vec2<i32>(0)) && all(prev_pixel < vec2<i32>(size)) {
        let pid = textureLoad(prev_id, prev_pixel, 0);
        if all(pid.xyz == id.xyz) && (pid.w >> 30u) != 0u {
            prev = unpack(textureLoad(prev_a, prev_pixel, 0), textureLoad(prev_b, prev_pixel, 0));
            have_prev = prev.light != 0xffffffffu;
        }
    }
    if !have_prev {
        textureStore(out_a, pixel, pack_a(cur));
        textureStore(out_b, pixel, pack_b(cur));
        return;
    }
    let s = shading_at(pixel, id);
    var rng = rng_seed(gid.xy, frame.frame_index ^ 0x7e3a11u);
    var out = empty_reservoir();
    if cur.light != 0xffffffffu {
        let p_hat = target_weight(light_contribution(s, lights[cur.light], cur.offset));
        reservoir_update(&out, cur.light, cur.offset, p_hat * cur.w * cur.m, p_hat, rng_next(&rng));
    }
    let prev_m = min(prev.m, 20.0 * max(cur.m, 1.0));
    let light = lights[prev.light];
    if light.count > 0u {
        let p_hat = target_weight(light_contribution(s, light, prev.offset));
        reservoir_update(&out, prev.light, prev.offset, p_hat * prev.w * prev_m, p_hat, rng_next(&rng));
    }
    out.m = cur.m + prev_m;
    if out.light != 0xffffffffu && out.target_pdf > 0.0 {
        out.w = out.w_sum / (out.m * out.target_pdf);
    } else {
        out.w = 0.0;
    }
    textureStore(out_a, pixel, pack_a(out));
    textureStore(out_b, pixel, pack_b(out));
}
