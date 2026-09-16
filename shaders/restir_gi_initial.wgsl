// ReSTIR GI, initial samples (Ouyang et al. 2021, Algorithm 2): one uniform
// hemisphere ray per GI pixel from its visible point; the sample is the hit
// point, its normal and the radiance it sends back.
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"
#import "atmosphere.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var prev_id: texture_2d<u32>;
@group(2) @binding(4) var surface_prev: texture_2d<f32>;
@group(2) @binding(5) var sky_map: texture_2d<f32>;
@group(2) @binding(6) var sky_ambient: texture_2d<f32>;
@group(2) @binding(7) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(8) var sky_view_lut: texture_2d<f32>;
@group(2) @binding(9) var sky_view_moon: texture_2d<f32>;
@group(2) @binding(10) var lut_sampler: sampler;
@group(2) @binding(11) var out_pos: texture_storage_2d<rgba32float, write>;
@group(2) @binding(12) var out_rad: texture_storage_2d<rgba32float, write>;
@group(2) @binding(13) var<uniform> params: SvgfParams;
#import "svgf_common.wgsl"
#import "gi_common.wgsl"
#import "restir_gi_common.wgsl"

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = signal_size();
    let p = vec2<i32>(gid.xy);
    if any(p >= size) {
        return;
    }
    let c = gbuffer(p);
    if (c.id.w >> 30u) == 0u {
        textureStore(out_pos, p, vec4<f32>(0.0, 0.0, 0.0, GI_NONE));
        textureStore(out_rad, p, vec4<f32>(0.0));
        return;
    }
    var rng = rng_seed(vec2<u32>(p), frame.frame_index ^ 0x61u);
    let dir = sample_hemisphere_uniform(c.normal, vec2<f32>(rng_next(&rng), rng_next(&rng)));
    let origin = frame.camera_frac + c.position * VOXELS_PER_METRE + face_normal(c.id) * 0.05;
    // Sky light arrives through composition's traced sky visibility; indirect
    // rays carry only light bounced off surfaces.
    let s = trace_gi(origin, dir, 0.0, GI_RANGE_M, false);
    let packed = pack_sample(s);
    textureStore(out_pos, p, packed.pos);
    textureStore(out_rad, p, packed.rad);
}
