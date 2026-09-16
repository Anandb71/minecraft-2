// ReSTIR GI, temporal resampling (Algorithm 3): the fresh sample joins last
// frame's temporal reservoir at the reprojected pixel when that pixel saw the
// same surface. History M is clamped to 30 so a bright sample cannot stick.
#import "common.wgsl"
#import "frame.wgsl"

@group(1) @binding(0) var vis_id: texture_2d<u32>;
@group(1) @binding(1) var vis_depth: texture_2d<f32>;
@group(1) @binding(2) var vis_motion: texture_2d<f32>;
@group(1) @binding(3) var prev_id: texture_2d<u32>;
@group(1) @binding(4) var init_pos: texture_2d<f32>;
@group(1) @binding(5) var init_rad: texture_2d<f32>;
@group(1) @binding(6) var prev_pos: texture_2d<f32>;
@group(1) @binding(7) var prev_rad: texture_2d<f32>;
@group(1) @binding(8) var prev_res: texture_2d<f32>;
@group(1) @binding(9) var out_pos: texture_storage_2d<rgba32float, write>;
@group(1) @binding(10) var out_rad: texture_storage_2d<rgba32float, write>;
@group(1) @binding(11) var out_res: texture_storage_2d<rgba32float, write>;
@group(1) @binding(12) var<uniform> params: SvgfParams;
#import "svgf_common.wgsl"

#import "restir_gi_common.wgsl"

const TEMPORAL_M_MAX: f32 = 30.0;

fn store(p: vec2<i32>, r: GiReservoir) {
    textureStore(out_pos, p, vec4<f32>(r.position, r.kind));
    textureStore(out_rad, p, vec4<f32>(r.radiance, pack_normal(r.normal)));
    textureStore(out_res, p, vec4<f32>(r.w_sum, r.m, r.w, r.target_value));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = signal_size();
    let p = vec2<i32>(gid.xy);
    if any(p >= size) {
        return;
    }
    let c = gbuffer(p);
    if (c.id.w >> 30u) == 0u {
        store(p, empty_gi_reservoir());
        return;
    }
    var rng = rng_seed(vec2<u32>(p), frame.frame_index ^ 0x7e3au);
    let fresh = read_reservoir(textureLoad(init_pos, p, 0), textureLoad(init_rad, p, 0), vec4<f32>(0.0));
    var r = empty_gi_reservoir();
    let fresh_target = gi_target(fresh, c.position, c.normal);
    r = gi_update(r, fresh, fresh_target / GI_SOURCE_PDF, rng_next(&rng));
    r.m = 1.0;

    // Last frame's reservoir at the reprojected GI pixel. Debug view 5
    // (the unbiased reference for measurements) keeps fresh samples only.
    let g = p << vec2<u32>(params.shift);
    let mv = textureLoad(vis_motion, g, 0).rg;
    let uv = (vec2<f32>(p) + 0.5) / vec2<f32>(size);
    let q = vec2<i32>(floor((uv + mv) * vec2<f32>(size)));
    if all(q >= vec2<i32>(0)) && all(q < size) && frame.debug_mode != GI_REFERENCE_VIEW {
        let pid = textureLoad(prev_id, q << vec2<u32>(params.shift), 0);
        if same_surface(c.id, pid, c.depth, f32(2u << params.shift)) {
            var h = read_reservoir(textureLoad(prev_pos, q, 0), textureLoad(prev_rad, q, 0), textureLoad(prev_res, q, 0));
            if h.kind == GI_SURFACE {
                // Into this frame's camera space.
                h.position += frame.prev_camera_delta;
            }
            if h.kind != GI_NONE {
                let m = min(h.m, TEMPORAL_M_MAX);
                let t = gi_target(h, c.position, c.normal);
                r = gi_update(r, h, t * h.w * m, rng_next(&rng));
                r.m += m;
            }
        }
    }
    r.target_value = gi_target(r, c.position, c.normal);
    r.w = select(0.0, r.w_sum / (r.m * r.target_value), r.target_value > 0.0);
    store(p, r);
}
