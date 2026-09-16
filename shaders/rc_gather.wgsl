// Radiance cascades, gathering: demodulated irradiance at each GI pixel from
// the four nearest merged cascade-0 probes (bilinear times bilateral), each
// integrated over its directions with the pixel's cosine.
#import "common.wgsl"
#import "frame.wgsl"

@group(1) @binding(0) var vis_id: texture_2d<u32>;
@group(1) @binding(1) var vis_depth: texture_2d<f32>;
@group(1) @binding(2) var vis_motion: texture_2d<f32>;
@group(1) @binding(3) var merged0: texture_2d<f32>;
@group(1) @binding(4) var out_irradiance: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var<uniform> rc: RcParams;
#import "rc_common.wgsl"

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = (vec2<u32>(frame.render_size) + (1u << rc.shift) - 1u) >> vec2<u32>(rc.shift);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(gid.xy << vec2<u32>(rc.shift));
    let id = textureLoad(vis_id, pixel, 0);
    if (id.w >> 30u) == 0u {
        textureStore(out_irradiance, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let depth = textureLoad(vis_depth, pixel, 0).r;
    let position = camera_ray_dir(vec2<f32>(pixel)) * depth;
    let normal = oct_decode(textureLoad(vis_motion, pixel, 0).ba);
    let spacing = rc.spacing;
    let dirs = rc.dirs;
    let probes = rc_probe_count(spacing);
    let g = (vec2<f32>(pixel) + 0.5 - f32(spacing) * 0.5) / f32(spacing);
    let base = vec2<i32>(floor(g));
    let f = g - vec2<f32>(base);
    let solid_angle = 4.0 * PI / f32(dirs * dirs);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let cp = clamp(base + o, vec2<i32>(0), vec2<i32>(probes) - 1);
        let ppixel = rc_probe_pixel(vec2<u32>(cp), spacing);
        let pid = textureLoad(vis_id, ppixel, 0);
        if (pid.w >> 30u) == 0u {
            continue;
        }
        let ppos = camera_ray_dir(vec2<f32>(ppixel)) * textureLoad(vis_depth, ppixel, 0).r;
        let pn = oct_decode(textureLoad(vis_motion, ppixel, 0).ba);
        let footprint = max(depth, 0.1) * frame.pixel_angle * f32(spacing);
        let plane = abs(dot(normal, ppos - position));
        let bilinear = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        let w = max(bilinear, 1e-3) * exp(-plane / footprint) * max(dot(pn, normal), 0.0);
        if w <= 0.0 {
            continue;
        }
        var e = vec3<f32>(0.0);
        for (var dy = 0u; dy < dirs; dy++) {
            for (var dx = 0u; dx < dirs; dx++) {
                let dir = rc_direction(vec2<u32>(dx, dy), dirs);
                let c = max(dot(dir, normal), 0.0);
                e += textureLoad(merged0, vec2<u32>(cp) * dirs + vec2<u32>(dx, dy), 0).rgb * c;
            }
        }
        sum += e * solid_angle * w;
        weight += w;
    }
    let irradiance = select(vec3<f32>(0.0), sum / max(weight, 1e-6), weight > 0.0);
    textureStore(out_irradiance, gid.xy, vec4<f32>(irradiance, 1.0));
}
