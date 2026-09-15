// Temporal upsampling: jittered render-resolution samples accumulated into a
// display-resolution history (Karis 2014, with the upsampling weights of
// temporal super resolution). Each display pixel gathers the 3x3 render
// pixels around its position under a Gaussian of their jittered centres,
// reprojects history with a Catmull-Rom filter, clips it to the variance box
// of the gathered neighbourhood in YCoCg, and blends by how well this
// frame's samples cover the pixel. Blending happens on tonemapped weights so
// a firefly cannot stain the history.
#import "common.wgsl"
#import "frame.wgsl"

@group(1) @binding(0) var scene: texture_2d<f32>;
@group(1) @binding(1) var vis_depth: texture_2d<f32>;
@group(1) @binding(2) var vis_motion: texture_2d<f32>;
@group(1) @binding(3) var history: texture_2d<f32>;
@group(1) @binding(4) var linear_clamp: sampler;
@group(1) @binding(5) var out_color: texture_storage_2d<rgba16float, write>;
// Last frame's metered exposure (g), so tonemapped weights see display values.
@group(1) @binding(6) var exposure_tex: texture_2d<f32>;

// Variance box half-width in standard deviations.
const CLIP_GAMMA: f32 = 1.25;
// Blend factor bounds: a sample exactly on the pixel centre replaces this
// much of history; a pixel no sample landed near still takes a little.
const ALPHA_MAX: f32 = 0.2;
const ALPHA_MIN: f32 = 0.03;

fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
    );
}

fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Compresses HDR so bright outliers weigh like ordinary pixels.
fn tonemap_weight(c: vec3<f32>, exposure: f32) -> f32 {
    return 1.0 / (1.0 + luminance(c) * exposure);
}

// Catmull-Rom history sample with 5 bilinear taps (the corners are dropped).
fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(history));
    let p = uv * size;
    let t1 = floor(p - 0.5) + 0.5;
    let f = p - t1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let offset12 = w2 / w12;
    let t0 = (t1 - 1.0) / size;
    let t3 = (t1 + 2.0) / size;
    let t12 = (t1 + offset12) / size;
    var c = vec3<f32>(0.0);
    c += textureSampleLevel(history, linear_clamp, vec2<f32>(t12.x, t0.y), 0.0).rgb * w12.x * w0.y;
    c += textureSampleLevel(history, linear_clamp, vec2<f32>(t0.x, t12.y), 0.0).rgb * w0.x * w12.y;
    c += textureSampleLevel(history, linear_clamp, t12, 0.0).rgb * w12.x * w12.y;
    c += textureSampleLevel(history, linear_clamp, vec2<f32>(t3.x, t12.y), 0.0).rgb * w3.x * w12.y;
    c += textureSampleLevel(history, linear_clamp, vec2<f32>(t12.x, t3.y), 0.0).rgb * w12.x * w3.y;
    let w = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(c / w, vec3<f32>(0.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_size = vec2<u32>(frame.output_size);
    if any(gid.xy >= out_size) {
        return;
    }
    let render = frame.render_size;
    let uv = (vec2<f32>(gid.xy) + 0.5) / frame.output_size;
    // Display pixel centre in render pixel coordinates; render sample k sits
    // at k + 0.5 + jitter.
    let p = uv * render - 0.5 - frame.jitter;
    let base = vec2<i32>(floor(p + 0.5));
    // Reconstruction filter in render pixels; coverage (which sets the blend)
    // is measured in display pixels, so upsampling blends proportionally less.
    let scale = max(frame.output_size.x / render.x, 1.0);
    let sigma = 0.47;
    let metered = textureLoad(exposure_tex, vec2<i32>(0), 0).g;
    let exposure = select(1.0, metered, metered > 0.0) * frame.exposure;
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var nearest_motion = vec2<f32>(0.0);
    var nearest_depth = 1e10;
    var best_gauss = 0.0;
    let max_px = vec2<i32>(render) - 1;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = clamp(base + vec2<i32>(dx, dy), vec2<i32>(0), max_px);
            let c = textureLoad(scene, q, 0).rgb;
            let d = vec2<f32>(q) - p;
            let g = exp(-0.5 * dot(d, d) / (sigma * sigma));
            let w = g * tonemap_weight(c, exposure);
            sum += c * w;
            weight += w;
            let d_out = d * scale;
            best_gauss = max(best_gauss, exp(-0.5 * dot(d_out, d_out) / (sigma * sigma)));
            let y = rgb_to_ycocg(c);
            m1 += y;
            m2 += y * y;
            // Motion from the nearest surface in the neighbourhood, so edges
            // of foreground objects carry their own history.
            let depth = textureLoad(vis_depth, q, 0).r;
            if depth < nearest_depth {
                nearest_depth = depth;
                nearest_motion = textureLoad(vis_motion, q, 0).rg;
            }
        }
    }
    let current = sum / max(weight, 1e-6);
    let mean = m1 / 9.0;
    let sigma_c = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));

    let prev = uv + nearest_motion;
    var result = current;
    // History alpha is 1 wherever a previous frame wrote it; a reallocated
    // history reads 0 and must not be blended in.
    let history_valid = textureSampleLevel(history, linear_clamp, prev, 0.0).a > 0.99;
    if history_valid && all(prev >= vec2<f32>(0.0)) && all(prev <= vec2<f32>(1.0)) {
        var h = rgb_to_ycocg(sample_history(prev));
        let lo = mean - CLIP_GAMMA * sigma_c;
        let hi = mean + CLIP_GAMMA * sigma_c;
        // Clip toward the box centre rather than clamping per axis.
        let centre = 0.5 * (lo + hi);
        let extent = max(0.5 * (hi - lo), vec3<f32>(1e-4));
        let offset = h - centre;
        let ratio = abs(offset) / extent;
        let m = max(ratio.x, max(ratio.y, ratio.z));
        if m > 1.0 {
            h = centre + offset / m;
        }
        let history_rgb = max(ycocg_to_rgb(h), vec3<f32>(0.0));
        let alpha = mix(ALPHA_MIN, ALPHA_MAX, best_gauss);
        let wc = alpha * tonemap_weight(current, exposure);
        let wh = (1.0 - alpha) * tonemap_weight(history_rgb, exposure);
        result = (current * wc + history_rgb * wh) / max(wc + wh, 1e-6);
    }
    textureStore(out_color, gid.xy, vec4<f32>(result, 1.0));
}
