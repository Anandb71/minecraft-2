// ReSTIR GI, spatial resampling (Algorithm 4): temporal reservoirs of nearby
// pixels on similar geometry (normals within 25 degrees, depth within 5%)
// are merged with the reconnection Jacobian of Eq. 11. Visibility is traced
// once, for a winning neighbour sample; if it is hidden the pixel keeps its
// own temporal reservoir. The normalisation counts only the reservoirs whose
// pixel faces the chosen sample.
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"
#import "bodies.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var t_pos: texture_2d<f32>;
@group(2) @binding(4) var t_rad: texture_2d<f32>;
@group(2) @binding(5) var t_res: texture_2d<f32>;
@group(2) @binding(6) var out_pos: texture_storage_2d<rgba32float, write>;
@group(2) @binding(7) var out_rad: texture_storage_2d<rgba32float, write>;
@group(2) @binding(8) var out_res: texture_storage_2d<rgba32float, write>;
@group(2) @binding(9) var<uniform> params: SvgfParams;
#import "svgf_common.wgsl"
#import "restir_gi_common.wgsl"

const NEIGHBOURS: i32 = 5;
const RADIUS_PX: f32 = 16.0;
const COS_25_DEG: f32 = 0.9063;
const SPATIAL_M_MAX: f32 = 500.0;

fn store(p: vec2<i32>, r: GiReservoir) {
    textureStore(out_pos, p, vec4<f32>(r.position, r.kind));
    textureStore(out_rad, p, vec4<f32>(r.radiance, pack_normal(r.normal)));
    textureStore(out_res, p, vec4<f32>(r.w_sum, r.m, r.w, r.target_value));
}

fn read_at(q: vec2<i32>) -> GiReservoir {
    return read_reservoir(textureLoad(t_pos, q, 0), textureLoad(t_rad, q, 0), textureLoad(t_res, q, 0));
}

// |J_{q -> r}|: how differently the receiver r would have sampled the point
// that q's visible point sampled. Directions to the sky need no correction.
fn jacobian(s: GiReservoir, from_visible: vec3<f32>, to_visible: vec3<f32>) -> f32 {
    if s.kind != GI_SURFACE {
        return 1.0;
    }
    let to_q = from_visible - s.position;
    let to_r = to_visible - s.position;
    let dq2 = max(dot(to_q, to_q), 1e-6);
    let dr2 = max(dot(to_r, to_r), 1e-6);
    let cos_q = abs(dot(s.normal, to_q)) / sqrt(dq2);
    let cos_r = abs(dot(s.normal, to_r)) / sqrt(dr2);
    return (cos_r / max(cos_q, 1e-4)) * (dq2 / dr2);
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
    var rng = rng_seed(vec2<u32>(p), frame.frame_index ^ 0x5a17u);
    let own = read_at(p);
    if frame.debug_mode == GI_REFERENCE_VIEW {
        store(p, own);
        return;
    }
    var r = empty_gi_reservoir();
    r = gi_update(r, own, gi_target(own, c.position, c.normal) * own.w * own.m, rng_next(&rng));
    var chosen = -1;
    // Accepted neighbours, kept for the normalisation.
    var n_pos: array<vec3<f32>, 5>;
    var n_nrm: array<vec3<f32>, 5>;
    var n_m: array<f32, 5>;
    var accepted = 0;
    for (var k = 0; k < NEIGHBOURS; k++) {
        let angle = rng_next(&rng) * TAU;
        let radius = max(sqrt(rng_next(&rng)) * RADIUS_PX, 1.0);
        let q = p + vec2<i32>(vec2<f32>(cos(angle), sin(angle)) * radius);
        if any(q < vec2<i32>(0)) || any(q >= size) || all(q == p) {
            continue;
        }
        let g = gbuffer(q);
        if (g.id.w >> 30u) == 0u || dot(g.normal, c.normal) < COS_25_DEG || abs(g.depth - c.depth) > 0.05 * c.depth {
            continue;
        }
        let s = read_at(q);
        if s.kind == GI_NONE || s.m <= 0.0 {
            continue;
        }
        let target_here = gi_target(s, c.position, c.normal) / jacobian(s, g.position, c.position);
        let before = r.w_sum;
        r = gi_update(r, s, target_here * s.w * s.m, rng_next(&rng));
        if r.w_sum > before && r.position.x == s.position.x && r.position.y == s.position.y && r.position.z == s.position.z && r.kind == s.kind {
            chosen = accepted;
        }
        n_pos[accepted] = g.position;
        n_nrm[accepted] = g.normal;
        n_m[accepted] = s.m;
        accepted += 1;
    }
    r.target_value = gi_target(r, c.position, c.normal);
    if r.kind == GI_NONE || r.target_value <= 0.0 {
        store(p, empty_gi_reservoir());
        return;
    }
    // A sample taken over from a neighbour may be hidden from here.
    if chosen >= 0 {
        let origin = frame.camera_frac + c.position * VOXELS_PER_METRE + face_normal(c.id) * 0.05;
        let dir = sample_direction(r, c.position);
        var ray: Ray;
        ray.base = frame.camera_voxel;
        ray.frac = origin;
        ray.dir = dir;
        ray.t_min = 0.0;
        if r.kind == GI_SURFACE {
            ray.t_max = max(length(r.position - c.position) * VOXELS_PER_METRE - 1.0, 0.0);
        } else {
            ray.t_max = GI_RANGE_M * VOXELS_PER_METRE;
        }
        ray.lod_scale = frame.pixel_angle * frame.lod_pixels * 8.0;
        ray.feedback = false;
        ray.coarse = false;
        if trace_scene(ray, ray.t_min).kind != HIT_NONE {
            // Keep this pixel's own temporal reservoir rather than going dark.
            store(p, own);
            return;
        }
    }
    // Z: M of every reservoir whose surface faces the chosen sample.
    var z = own.m;
    for (var k = 0; k < accepted; k++) {
        if dot(n_nrm[k], sample_direction(r, n_pos[k])) > 0.0 {
            z += n_m[k];
        }
    }
    r.m = min(z, SPATIAL_M_MAX);
    r.w = r.w_sum / (max(z, 1.0) * r.target_value);
    store(p, r);
}
