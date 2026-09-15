// ReSTIR step 3: spatial reuse. k = 5 random neighbours within 30 pixels,
// rejected when camera distance differs by more than 10% or normals by more
// than 25 degrees (the paper's biased heuristic), combined with Alg. 4.
// Run twice, ping-ponging between two reservoir images.
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"
#import "brdf.wgsl"
#import "restir_common.wgsl"

struct SpatialParams {
    iteration: u32,
    _pad: vec3<u32>,
}

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var in_a: texture_2d<f32>;
@group(2) @binding(4) var in_b: texture_2d<f32>;
@group(2) @binding(5) var out_a: texture_storage_2d<rgba32float, write>;
@group(2) @binding(6) var out_b: texture_storage_2d<rgba32float, write>;
@group(2) @binding(7) var<uniform> params: SpatialParams;
#import "restir_surface.wgsl"

const NEIGHBOURS: i32 = 5;
const RADIUS_PX: f32 = 30.0;
const COS_25_DEG: f32 = 0.9063;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.render_size);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let id = textureLoad(vis_id, pixel, 0);
    let centre = unpack(textureLoad(in_a, pixel, 0), textureLoad(in_b, pixel, 0));
    if (id.w >> 30u) == 0u {
        textureStore(out_a, pixel, pack_a(centre));
        textureStore(out_b, pixel, pack_b(centre));
        return;
    }
    let s = shading_at(pixel, id);
    let depth = textureLoad(vis_depth, pixel, 0).r;
    var rng = rng_seed(gid.xy, frame.frame_index * 7u + params.iteration * 131u);
    var out = empty_reservoir();
    var m_total = centre.m;
    if centre.light != 0xffffffffu {
        let p_hat = target_weight(light_contribution(s, lights[centre.light], centre.offset));
        reservoir_update(&out, centre.light, centre.offset, p_hat * centre.w * centre.m, p_hat, rng_next(&rng));
    }
    for (var k = 0; k < NEIGHBOURS; k++) {
        let angle = rng_next(&rng) * TAU;
        let radius = sqrt(rng_next(&rng)) * RADIUS_PX;
        let q = pixel + vec2<i32>(vec2<f32>(cos(angle), sin(angle)) * radius);
        if any(q < vec2<i32>(0)) || any(q >= vec2<i32>(size)) || all(q == pixel) {
            continue;
        }
        let qid = textureLoad(vis_id, q, 0);
        if (qid.w >> 30u) == 0u {
            continue;
        }
        let qdepth = textureLoad(vis_depth, q, 0).r;
        let qn = oct_decode(textureLoad(vis_motion, q, 0).ba);
        if abs(qdepth - depth) > 0.1 * depth || dot(qn, s.normal) < COS_25_DEG {
            continue;
        }
        let r = unpack(textureLoad(in_a, q, 0), textureLoad(in_b, q, 0));
        m_total += r.m;
        if r.light == 0xffffffffu {
            continue;
        }
        let p_hat = target_weight(light_contribution(s, lights[r.light], r.offset));
        reservoir_update(&out, r.light, r.offset, p_hat * r.w * r.m, p_hat, rng_next(&rng));
    }
    out.m = m_total;
    if out.light != 0xffffffffu && out.target_pdf > 0.0 && out.m > 0.0 {
        out.w = out.w_sum / (out.m * out.target_pdf);
    } else {
        out.w = 0.0;
    }
    textureStore(out_a, pixel, pack_a(out));
    textureStore(out_b, pixel, pack_b(out));
}
