// Bloom (Jimenez 2014): a downsample chain with the 13-tap filter, the
// first level Karis-averaged so single bright pixels cannot flicker, then an
// upsample chain adding a 3x3 tent of each coarser level.
#import "common.wgsl"

struct BloomParams {
    // 0 downsample, 1 upsample.
    mode: u32,
    // Karis average on this downsample.
    karis: u32,
    // Upsample tent radius in source texels.
    radius: f32,
    _pad: f32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
// Upsample: the level being added to (same size as the output).
@group(0) @binding(1) var base: texture_2d<f32>;
@group(0) @binding(2) var linear_clamp: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: BloomParams;

fn karis_weight(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + luminance(c));
}

fn tap(uv: vec2<f32>, texel: vec2<f32>, o: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(src, linear_clamp, uv + o * texel, 0.0).rgb;
}

fn downsample(uv: vec2<f32>) -> vec3<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src));
    let a = tap(uv, t, vec2<f32>(-2.0, -2.0));
    let b = tap(uv, t, vec2<f32>(0.0, -2.0));
    let c = tap(uv, t, vec2<f32>(2.0, -2.0));
    let d = tap(uv, t, vec2<f32>(-2.0, 0.0));
    let e = tap(uv, t, vec2<f32>(0.0, 0.0));
    let f = tap(uv, t, vec2<f32>(2.0, 0.0));
    let g = tap(uv, t, vec2<f32>(-2.0, 2.0));
    let h = tap(uv, t, vec2<f32>(0.0, 2.0));
    let i = tap(uv, t, vec2<f32>(2.0, 2.0));
    let j = tap(uv, t, vec2<f32>(-1.0, -1.0));
    let k = tap(uv, t, vec2<f32>(1.0, -1.0));
    let l = tap(uv, t, vec2<f32>(-1.0, 1.0));
    let m = tap(uv, t, vec2<f32>(1.0, 1.0));
    if params.karis == 1u {
        // Five overlapping 2x2 groups, each weighted by 1 / (1 + luma).
        let g0 = (j + k + l + m) * 0.25;
        let g1 = (a + b + d + e) * 0.25;
        let g2 = (b + c + e + f) * 0.25;
        let g3 = (d + e + g + h) * 0.25;
        let g4 = (e + f + h + i) * 0.25;
        let w0 = karis_weight(g0) * 0.5;
        let w1 = karis_weight(g1) * 0.125;
        let w2 = karis_weight(g2) * 0.125;
        let w3 = karis_weight(g3) * 0.125;
        let w4 = karis_weight(g4) * 0.125;
        return (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    }
    return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

fn upsample(uv: vec2<f32>) -> vec3<f32> {
    let t = params.radius / vec2<f32>(textureDimensions(src));
    var sum = tap(uv, t, vec2<f32>(0.0, 0.0)) * 4.0;
    sum += (tap(uv, t, vec2<f32>(-1.0, 0.0)) + tap(uv, t, vec2<f32>(1.0, 0.0)) + tap(uv, t, vec2<f32>(0.0, -1.0)) + tap(uv, t, vec2<f32>(0.0, 1.0))) * 2.0;
    sum += tap(uv, t, vec2<f32>(-1.0, -1.0)) + tap(uv, t, vec2<f32>(1.0, -1.0)) + tap(uv, t, vec2<f32>(-1.0, 1.0)) + tap(uv, t, vec2<f32>(1.0, 1.0));
    return sum / 16.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if any(gid.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    if params.mode == 0u {
        textureStore(dst, gid.xy, vec4<f32>(downsample(uv), 1.0));
    } else {
        let here = textureLoad(base, gid.xy, 0).rgb;
        textureStore(dst, gid.xy, vec4<f32>(here + upsample(uv), 1.0));
    }
}
