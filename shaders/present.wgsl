// Fullscreen resolve of the HDR image onto the display format: exposure,
// night vision, the display transform, then the lens and film touches that
// belong after it (sharpening, vignette, grain).
#import "tonemap.wgsl"

struct PresentUniforms {
    exposure: f32,
    tonemap: u32,
    auto_exposure: u32,
    frame_index: u32,
    // Contrast-adaptive sharpening strength, 0..1.
    sharpen: f32,
    // Corner darkening, 0..1.
    vignette: f32,
    // Film grain amplitude in display units.
    grain: f32,
    // 1 enables the Purkinje shift at low adaptation.
    purkinje: u32,
}

@group(0) @binding(0) var<uniform> present: PresentUniforms;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var linear_clamp: sampler;
@group(0) @binding(3) var auto_exposure: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    // One triangle covering the viewport.
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VsOut;
    out.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

fn hash(p: vec2<u32>, seed: u32) -> f32 {
    var h = p.x * 1973u + p.y * 9277u + seed * 26699u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h) / 4294967295.0;
}

// Scene-referred display value of one texel, before sharpening.
fn display(uv: vec2<f32>, exposure: f32, night: f32) -> vec3<f32> {
    var hdr = textureSampleLevel(scene, linear_clamp, uv, 0.0).rgb * exposure;
    if night > 0.0 {
        // Rods see one channel, peaking toward blue-green: desaturate and
        // cool the image as adaptation falls.
        let scotopic = dot(hdr, vec3<f32>(0.08, 0.53, 0.39));
        hdr = mix(hdr, scotopic * vec3<f32>(0.78, 0.92, 1.25), night);
    }
    if present.tonemap == 1u {
        return agx(hdr);
    }
    return hdr;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // present.exposure is a manual multiplier on top of automatic exposure;
    // the auto channel holds 1 until metering has run.
    let meter = textureLoad(auto_exposure, vec2<i32>(0), 0);
    let auto_on = present.auto_exposure == 1u;
    let exposure = present.exposure * select(1.0, meter.g, auto_on);
    // Adapted EV100 in meter.r: fully photopic above EV 3, scotopic below -3.
    let night = select(0.0, 0.85 * (1.0 - smoothstep(-3.0, 3.0, meter.r)), auto_on && present.purkinje == 1u);

    let size = vec2<f32>(textureDimensions(scene));
    let t = 1.0 / size;
    var c = display(in.uv, exposure, night);
    if present.sharpen > 0.0 {
        // AMD's contrast-adaptive sharpening on the cross neighbourhood.
        let n = display(in.uv + vec2<f32>(0.0, -t.y), exposure, night);
        let s = display(in.uv + vec2<f32>(0.0, t.y), exposure, night);
        let e = display(in.uv + vec2<f32>(t.x, 0.0), exposure, night);
        let w = display(in.uv + vec2<f32>(-t.x, 0.0), exposure, night);
        let mn = min(c, min(min(n, s), min(e, w)));
        let mx = max(c, max(max(n, s), max(e, w)));
        let amount = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3<f32>(1e-4)), vec3<f32>(0.0), vec3<f32>(1.0)));
        let peak = -1.0 / mix(8.0, 5.0, present.sharpen);
        let wgt = amount * peak;
        c = clamp((c + (n + s + e + w) * wgt) / (1.0 + 4.0 * wgt), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    // Natural vignetting: cos^4 of the off-axis angle, scaled.
    let d = (in.uv - 0.5) * vec2<f32>(size.x / size.y, 1.0);
    let cos_off = 1.0 / sqrt(1.0 + dot(d, d) * 1.2);
    c *= mix(1.0, pow(cos_off, 4.0), present.vignette);
    // Luminance grain, strongest in the mid-tones.
    let g = (hash(vec2<u32>(in.pos.xy), present.frame_index) - 0.5) * present.grain;
    let mid = 4.0 * dot(c, vec3<f32>(0.333)) * (1.0 - dot(c, vec3<f32>(0.333)));
    c += vec3<f32>(g * mid);
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
