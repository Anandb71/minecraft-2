// One SVGF edge-avoiding a-trous iteration (sections 4.3, 4.4): a 5x5
// B3-spline kernel with holes of `step` pixels, weighted by geometry and by
// luminance distance normalised with the 3x3 Gaussian-prefiltered standard
// deviation. Variance is filtered with squared weights for the next level.
#import "common.wgsl"
#import "frame.wgsl"

@group(1) @binding(0) var vis_id: texture_2d<u32>;
@group(1) @binding(1) var vis_depth: texture_2d<f32>;
@group(1) @binding(2) var vis_motion: texture_2d<f32>;
// rgb colour, a variance.
@group(1) @binding(3) var src: texture_2d<f32>;
@group(1) @binding(4) var out_color: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var<uniform> params: SvgfParams;
#import "svgf_common.wgsl"

// 1D B3-spline weights 3/8, 1/4, 1/16 by distance from the centre.
fn kernel(i: i32) -> f32 {
    let a = abs(i);
    return select(select(0.0625, 0.25, a == 1), 0.375, a == 0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = signal_size();
    let p = vec2<i32>(gid.xy);
    if any(p >= size) {
        return;
    }
    let centre = textureLoad(src, p, 0);
    // Uncovered pixels were written as zero colour and zero variance.
    if all(centre == vec4<f32>(0.0)) {
        textureStore(out_color, p, centre);
        return;
    }
    let c = gbuffer(p);
    if (c.id.w >> 30u) == 0u {
        textureStore(out_color, p, centre);
        return;
    }
    // 3x3 Gaussian of variance for the luminance edge-stopping function.
    var var_blur = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), size - 1);
            let w = select(select(0.0625, 0.125, dx == 0 || dy == 0), 0.25, dx == 0 && dy == 0);
            var_blur += textureLoad(src, q, 0).a * w;
        }
    }
    let l = luminance(centre.rgb);
    let step = i32(params.step);
    let denom = SIGMA_L * sqrt(max(var_blur, 0.0)) + 1e-4;
    var sum = centre.rgb * kernel(0) * kernel(0);
    var sum_var = centre.a * pow(kernel(0) * kernel(0), 2.0);
    var sum_w = kernel(0) * kernel(0);
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let q = p + vec2<i32>(dx, dy) * step;
            if any(q < vec2<i32>(0)) || any(q >= size) {
                continue;
            }
            let s = textureLoad(src, q, 0);
            let pixels = length(vec2<f32>(f32(dx), f32(dy))) * f32(step);
            let wg = geometry_weight(c, gbuffer(q), pixels);
            let wl = exp(-abs(luminance(s.rgb) - l) / denom);
            let w = kernel(dx) * kernel(dy) * wg * wl;
            sum += s.rgb * w;
            sum_var += s.a * w * w;
            sum_w += w;
        }
    }
    textureStore(out_color, p, vec4<f32>(sum / sum_w, sum_var / (sum_w * sum_w)));
}
