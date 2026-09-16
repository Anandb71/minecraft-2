// ReSTIR GI, shading: demodulated indirect irradiance from the spatial
// reservoir's sample, radiance x cosine x W (Eq. 6 without the albedo, which
// composition applies after denoising).
#import "common.wgsl"
#import "frame.wgsl"

@group(1) @binding(0) var vis_id: texture_2d<u32>;
@group(1) @binding(1) var vis_depth: texture_2d<f32>;
@group(1) @binding(2) var vis_motion: texture_2d<f32>;
@group(1) @binding(3) var s_pos: texture_2d<f32>;
@group(1) @binding(4) var s_rad: texture_2d<f32>;
@group(1) @binding(5) var s_res: texture_2d<f32>;
@group(1) @binding(6) var out_irradiance: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var<uniform> params: SvgfParams;
#import "svgf_common.wgsl"
#import "restir_gi_common.wgsl"

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = signal_size();
    let p = vec2<i32>(gid.xy);
    if any(p >= size) {
        return;
    }
    let c = gbuffer(p);
    let r = read_reservoir(textureLoad(s_pos, p, 0), textureLoad(s_rad, p, 0), textureLoad(s_res, p, 0));
    if (c.id.w >> 30u) == 0u || r.kind == GI_NONE || r.w <= 0.0 {
        textureStore(out_irradiance, p, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let cos_theta = max(dot(c.normal, sample_direction(r, c.position)), 0.0);
    // E = integral of L cos over the hemisphere, estimated as f(y) W with
    // W the reservoir's stand-in for 1 / pdf.
    let e = r.radiance * cos_theta * r.w;
    textureStore(out_irradiance, p, vec4<f32>(e, 1.0));
}
