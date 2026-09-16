// Froxel fog, integration: front to back along every froxel column, the
// in-scattered luminance reaching the camera from the slice's far edge
// (rgb) and the transmittance to it (a), integrated per slice with the
// energy-conserving form of Hillaire 2015.
#import "common.wgsl"
#import "frame.wgsl"

@group(1) @binding(0) var injected: texture_3d<f32>;
@group(1) @binding(1) var out_integrated: texture_storage_3d<rgba16float, write>;
@group(1) @binding(2) var<uniform> fog: FogParams;
#import "fog_common.wgsl"

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= fog.width || gid.y >= fog.height {
        return;
    }
    var scattered = vec3<f32>(0.0);
    var transmittance = 1.0;
    var previous_distance = 0.0;
    for (var z = 0u; z < fog.depth; z++) {
        let far = fog_slice_distance(f32(z + 1u) / f32(fog.depth));
        let thickness = far - previous_distance;
        previous_distance = far;
        let v = textureLoad(injected, vec3<u32>(gid.xy, z), 0);
        let sigma_t = max(v.a, 1e-7);
        let step_t = exp(-sigma_t * thickness);
        scattered += transmittance * (v.rgb - v.rgb * step_t) / sigma_t;
        transmittance *= step_t;
        textureStore(out_integrated, vec3<u32>(gid.xy, z), vec4<f32>(scattered, transmittance));
    }
}
