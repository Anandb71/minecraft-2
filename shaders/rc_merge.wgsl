// Radiance cascades, merging cascade i+1 into cascade i (Eq. 13): each texel
// keeps its own interval and, where that interval is transparent, adds the
// coarser cascade's merged radiance for the same direction, averaged over
// the four finer directions it splits into and interpolated between the four
// nearest coarser probes. Probe weights are bilinear times a bilateral term
// on distance from the finer probe's tangent plane, which keeps light from
// leaking across depth edges.
#import "common.wgsl"
#import "frame.wgsl"

@group(1) @binding(0) var vis_id: texture_2d<u32>;
@group(1) @binding(1) var vis_depth: texture_2d<f32>;
@group(1) @binding(2) var vis_motion: texture_2d<f32>;
@group(1) @binding(3) var raw: texture_2d<f32>;
@group(1) @binding(4) var coarser: texture_2d<f32>;
@group(1) @binding(5) var out_merged: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var<uniform> rc: RcParams;
#import "rc_common.wgsl"

struct ProbeSurface {
    position: vec3<f32>,
    normal: vec3<f32>,
    valid: bool,
}

fn probe_surface(pixel: vec2<i32>) -> ProbeSurface {
    var s: ProbeSurface;
    let id = textureLoad(vis_id, pixel, 0);
    s.valid = (id.w >> 30u) != 0u;
    let depth = textureLoad(vis_depth, pixel, 0).r;
    s.position = camera_ray_dir(vec2<f32>(pixel)) * depth;
    s.normal = oct_decode(textureLoad(vis_motion, pixel, 0).ba);
    return s;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probes = rc_probe_count(rc.spacing);
    let probe = gid.xy / rc.dirs;
    if any(probe >= probes) {
        return;
    }
    let own = textureLoad(raw, gid.xy, 0);
    if own.a <= 0.0 {
        textureStore(out_merged, gid.xy, own);
        return;
    }
    let pixel = rc_probe_pixel(probe, rc.spacing);
    let here = probe_surface(pixel);
    let spacing = rc.spacing * 2u;
    let dirs = rc.dirs * 2u;
    let coarse_probes = rc_probe_count(spacing);
    // Position among coarser probe centres.
    let g = (vec2<f32>(pixel) + 0.5 - f32(spacing) * 0.5) / f32(spacing);
    let base = vec2<i32>(floor(g));
    let f = g - vec2<f32>(base);
    let d = gid.xy % rc.dirs;
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let cp = clamp(base + o, vec2<i32>(0), vec2<i32>(coarse_probes) - 1);
        let cpixel = rc_probe_pixel(vec2<u32>(cp), spacing);
        let there = probe_surface(cpixel);
        if !there.valid {
            continue;
        }
        let bilinear = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        let footprint = max(length(here.position), 0.1) * frame.pixel_angle * f32(spacing);
        let plane = abs(dot(here.normal, there.position - here.position));
        let w = max(bilinear, 1e-3) * exp(-plane / footprint);
        let origin = vec2<u32>(cp) * dirs + d * 2u;
        var dir_sum = textureLoad(coarser, origin, 0).rgb;
        dir_sum += textureLoad(coarser, origin + vec2<u32>(1u, 0u), 0).rgb;
        dir_sum += textureLoad(coarser, origin + vec2<u32>(0u, 1u), 0).rgb;
        dir_sum += textureLoad(coarser, origin + vec2<u32>(1u, 1u), 0).rgb;
        sum += dir_sum * 0.25 * w;
        weight += w;
    }
    let far = select(vec3<f32>(0.0), sum / max(weight, 1e-6), weight > 0.0);
    textureStore(out_merged, gid.xy, vec4<f32>(own.rgb + own.a * far, 0.0));
}
