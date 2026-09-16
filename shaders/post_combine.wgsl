// Display-resolution post: camera motion blur, depth of field (photo mode)
// and bloom, on the temporally upsampled HDR image.
#import "common.wgsl"
#import "frame.wgsl"

struct PostParams {
    bloom_intensity: f32,
    // Fraction of the frame the shutter stays open; 0 disables blur.
    shutter: f32,
    // Thin-lens depth of field; aperture 0 disables it.
    focus_m: f32,
    aperture_mm: f32,
    focal_mm: f32,
    sensor_mm: f32,
    max_coc_px: f32,
    _pad: f32,
}

@group(1) @binding(0) var color: texture_2d<f32>;
@group(1) @binding(1) var bloom: texture_2d<f32>;
@group(1) @binding(2) var vis_depth: texture_2d<f32>;
@group(1) @binding(3) var vis_motion: texture_2d<f32>;
@group(1) @binding(4) var linear_clamp: sampler;
@group(1) @binding(5) var out_color: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var<uniform> post: PostParams;

const BLUR_TAPS: i32 = 8;
const DOF_TAPS: i32 = 32;
const GOLDEN_ANGLE: f32 = 2.39996323;

fn render_texel(uv: vec2<f32>) -> vec2<i32> {
    return clamp(vec2<i32>(uv * frame.render_size), vec2<i32>(0), vec2<i32>(frame.render_size) - 1);
}

// Circle of confusion diameter in display pixels for a subject distance.
fn coc_px(depth_m: f32) -> f32 {
    let f = post.focal_mm / 1000.0;
    let a = post.aperture_mm / 1000.0;
    let s = max(post.focus_m, f * 1.01);
    let d = max(depth_m, f * 1.01);
    let coc_m = a * f * abs(d - s) / (d * (s - f));
    return min(coc_m / (post.sensor_mm / 1000.0) * frame.output_size.y, post.max_coc_px);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.output_size);
    if any(gid.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / frame.output_size;
    var c = textureSampleLevel(color, linear_clamp, uv, 0.0).rgb;

    if post.aperture_mm > 0.0 {
        // Scatter-as-gather: a sample contributes where its own circle
        // reaches this pixel.
        let depth = textureLoad(vis_depth, render_texel(uv), 0).r;
        let centre_coc = coc_px(depth);
        var sum = c;
        var weight = 1.0;
        let radius = post.max_coc_px;
        for (var i = 1; i < DOF_TAPS; i++) {
            let r = sqrt(f32(i) / f32(DOF_TAPS)) * radius;
            let theta = f32(i) * GOLDEN_ANGLE;
            let offset = vec2<f32>(cos(theta), sin(theta)) * r;
            let suv = uv + offset / frame.output_size;
            let sdepth = textureLoad(vis_depth, render_texel(suv), 0).r;
            let scoc = coc_px(sdepth);
            // Background samples may not bleed over a sharper foreground.
            let reach = select(scoc, min(scoc, centre_coc), sdepth > depth);
            let w = smoothstep(r - 1.0, r + 1.0, reach * 0.5 + 0.5);
            sum += textureSampleLevel(color, linear_clamp, suv, 0.0).rgb * w;
            weight += w;
        }
        c = sum / weight;
    }

    if post.shutter > 0.0 {
        // Camera motion: blur along the screen path the point took while
        // the shutter was open (motion vectors point to last frame).
        let motion = textureLoad(vis_motion, render_texel(uv), 0).rg * post.shutter;
        let length_px = length(motion * frame.output_size);
        if length_px > 0.5 {
            var sum = c;
            var n = 1.0;
            for (var i = 1; i < BLUR_TAPS; i++) {
                let t = f32(i) / f32(BLUR_TAPS - 1) - 0.5;
                sum += textureSampleLevel(color, linear_clamp, uv + motion * t, 0.0).rgb;
                n += 1.0;
            }
            c = sum / n;
        }
    }

    let b = textureSampleLevel(bloom, linear_clamp, uv, 0.0).rgb;
    c = mix(c, b, post.bloom_intensity);
    textureStore(out_color, gid.xy, vec4<f32>(c, 1.0));
}
