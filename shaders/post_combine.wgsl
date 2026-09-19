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
    c = precipitation(c, uv, b);
    // Lightning lights everything at once.
    c *= 1.0 + 2.5 * frame.weather.z;
    textureStore(out_color, gid.xy, vec4<f32>(c, 1.0));
}

fn precip_hash(p: vec2<i32>, salt: u32) -> f32 {
    var h = u32(p.x) * 0x8da6b343u ^ u32(p.y) * 0xd8163841u ^ salt * 0x9e3779b9u;
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    return f32(h >> 8u) / 16777216.0;
}

// Rain streaks and snowflakes in four layers of air, 1.5 to 12 m out, in
// a grid over the view direction's angles (so they stay put as the camera
// turns), sheared along the wind, hidden behind anything nearer. `local`
// is the blurred scene around the pixel: drops catch what light there is.
fn precipitation(c: vec3<f32>, uv: vec2<f32>, local: vec3<f32>) -> vec3<f32> {
    let rain = frame.weather.x;
    let snow = frame.weather.y;
    if rain + snow <= 0.001 {
        return c;
    }
    let raw_depth = textureLoad(vis_depth, render_texel(uv), 0).r;
    let depth = select(raw_depth, 1.0e9, raw_depth <= 0.0);
    let dir = camera_ray_dir(uv * frame.render_size);
    let az = atan2(dir.x, dir.z);
    let el = asin(clamp(dir.y, -1.0, 1.0));
    // Wind across the view, metres a second.
    let across = frame.wind.x * cos(az) - frame.wind.y * sin(az);
    let light = max(dot(local, vec3<f32>(0.2126, 0.7152, 0.0722)), 1.0e-4);
    var out = c;
    for (var k = 0u; k < 4u; k++) {
        let d = 1.5 * f32(1u << k);
        if depth < d {
            break;
        }
        let fade = 1.0 / (1.0 + 0.6 * f32(k));
        if rain > 0.0 {
            // Drops fall at 9 m/s, 0.1 m apart across, cells 0.6 m tall.
            let slant = across / 9.0;
            let st = vec2<f32>((az + el * slant) * d / 0.1, (el + frame.time * 9.0 / d) * d / 0.6);
            let cell = vec2<i32>(floor(st));
            let f = fract(st);
            if precip_hash(cell, k) < rain * 0.6 {
                let x = 0.2 + 0.6 * precip_hash(cell, k + 7u);
                let width = 0.045 / (1.0 + f32(k));
                let a = smoothstep(width, 0.0, abs(f.x - x)) * smoothstep(0.0, 0.08, f.y)
                    * smoothstep(0.6, 0.35, f.y);
                out += a * fade * light * 0.28 * vec3<f32>(0.85, 0.9, 1.0);
            }
        }
        if snow > 0.0 {
            // Flakes drift down at 1.2 m/s, swaying, 0.25 m apart.
            let st0 = vec2<f32>(az * d / 0.25, (el + frame.time * 1.2 / d) * d / 0.25);
            let cell = vec2<i32>(floor(st0));
            if precip_hash(cell, k + 20u) < snow * 0.5 {
                let phase = precip_hash(cell, k + 30u) * 6.2832;
                let centre = vec2<f32>(
                    0.5 + 0.3 * sin(frame.time * 1.3 + phase),
                    0.3 + 0.4 * precip_hash(cell, k + 40u),
                );
                let r = 0.1 / (1.0 + 0.4 * f32(k));
                let a = smoothstep(r, r * 0.3, length(fract(st0) - centre));
                out = mix(out, vec3<f32>(light * 1.6), a * fade * 0.85);
            }
        }
    }
    return out;
}
