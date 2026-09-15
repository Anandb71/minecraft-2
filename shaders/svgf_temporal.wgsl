// SVGF temporal accumulation and variance estimation (sections 4.1, 4.2).
// Reprojects last frame's filtered colour and luminance moments with a 2x2
// bilinear footprint whose taps are each tested for the same surface,
// accumulates with alpha = max(1 / length, 0.2), and estimates variance from
// the integrated moments. With fewer than four frames of history the
// variance comes from a 7x7 bilateral spatial estimate instead.
#import "common.wgsl"
#import "frame.wgsl"

@group(1) @binding(0) var vis_id: texture_2d<u32>;
@group(1) @binding(1) var vis_depth: texture_2d<f32>;
@group(1) @binding(2) var vis_motion: texture_2d<f32>;
@group(1) @binding(3) var prev_id: texture_2d<u32>;
@group(1) @binding(4) var noisy: texture_2d<f32>;
@group(1) @binding(5) var history_color: texture_2d<f32>;
@group(1) @binding(6) var history_moments: texture_2d<f32>;
// rgb integrated colour, a variance.
@group(1) @binding(7) var out_color: texture_storage_2d<rgba16float, write>;
// x first moment, y second moment, z history length.
@group(1) @binding(8) var out_moments: texture_storage_2d<rgba16float, write>;
@group(1) @binding(9) var<uniform> params: SvgfParams;
#import "svgf_common.wgsl"

const ALPHA: f32 = 0.2;
const MAX_HISTORY: f32 = 32.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = signal_size();
    let p = vec2<i32>(gid.xy);
    if any(p >= size) {
        return;
    }
    let c = gbuffer(p);
    let x = textureLoad(noisy, p, 0).rgb;
    if (c.id.w >> 30u) == 0u {
        textureStore(out_color, p, vec4<f32>(x, 0.0));
        textureStore(out_moments, p, vec4<f32>(0.0));
        return;
    }
    let l = luminance(x);

    // Bilinear reprojection over consistent taps.
    let g = p << vec2<u32>(params.shift);
    let mv = textureLoad(vis_motion, g, 0).rg;
    let uv = (vec2<f32>(p) + 0.5) / vec2<f32>(size);
    let prev = (uv + mv) * vec2<f32>(size) - 0.5;
    let base = vec2<i32>(floor(prev));
    let f = prev - vec2<f32>(base);
    var sum_color = vec3<f32>(0.0);
    var sum_moments = vec3<f32>(0.0);
    var sum_w = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let q = base + o;
        if any(q < vec2<i32>(0)) || any(q >= size) {
            continue;
        }
        let pid = textureLoad(prev_id, q << vec2<u32>(params.shift), 0);
        if !same_surface(c.id, pid, c.depth) {
            continue;
        }
        let bw = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        sum_color += textureLoad(history_color, q, 0).rgb * bw;
        sum_moments += textureLoad(history_moments, q, 0).xyz * bw;
        sum_w += bw;
    }
    var history_len = 0.0;
    var color = x;
    var m1 = l;
    var m2 = l * l;
    if sum_w > 1e-3 {
        let hc = sum_color / sum_w;
        let hm = sum_moments / sum_w;
        history_len = min(hm.z + 1.0, MAX_HISTORY);
        let alpha = max(1.0 / history_len, ALPHA);
        color = mix(hc, x, alpha);
        m1 = mix(hm.x, l, alpha);
        m2 = mix(hm.y, l * l, alpha);
    } else {
        history_len = 1.0;
    }
    var variance = max(m2 - m1 * m1, 0.0);

    if history_len < 4.0 {
        // Spatial fallback: bilateral moments over 7x7.
        var s1 = 0.0;
        var s2 = 0.0;
        var sw = 0.0;
        for (var dy = -3; dy <= 3; dy++) {
            for (var dx = -3; dx <= 3; dx++) {
                let q = p + vec2<i32>(dx, dy);
                if any(q < vec2<i32>(0)) || any(q >= size) {
                    continue;
                }
                let w = geometry_weight(c, gbuffer(q), length(vec2<f32>(f32(dx), f32(dy))));
                let lq = luminance(textureLoad(noisy, q, 0).rgb);
                s1 += lq * w;
                s2 += lq * lq * w;
                sw += w;
            }
        }
        let mean = s1 / max(sw, 1e-4);
        // Few frames also mean few samples: boost toward stronger filtering.
        variance = max(s2 / max(sw, 1e-4) - mean * mean, 0.0) * (4.0 / history_len);
    }
    textureStore(out_color, p, vec4<f32>(color, variance));
    textureStore(out_moments, p, vec4<f32>(m1, m2, history_len, 1.0));
}
