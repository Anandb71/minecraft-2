// Fullscreen resolve of the scene HDR target onto the display format.
#import "tonemap.wgsl"

struct PresentUniforms {
    exposure: f32,
    tonemap: u32,
    auto_exposure: u32,
    _pad: f32,
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

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // present.exposure is a manual multiplier on top of automatic exposure;
    // the auto channel holds 1 until metering has run.
    let metered = select(1.0, textureLoad(auto_exposure, vec2<i32>(0), 0).g, present.auto_exposure == 1u);
    let hdr = textureSampleLevel(scene, linear_clamp, in.uv, 0.0).rgb * present.exposure * metered;
    var ldr = hdr;
    if present.tonemap == 1u {
        ldr = agx(hdr);
    }
    return vec4<f32>(clamp(ldr, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
