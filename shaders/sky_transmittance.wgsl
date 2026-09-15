// Transmittance LUT: transmittance from a point at a given altitude to the
// top of the atmosphere along a given zenith angle. Built once.
#import "atmosphere.wgsl"

@group(0) @binding(0) var out_lut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out_lut);
    if any(id.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let hc = transmittance_params(uv);
    let origin = vec3<f32>(0.0, hc.x, 0.0);
    let dir = vec3<f32>(sqrt(max(0.0, 1.0 - hc.y * hc.y)), hc.y, 0.0);
    textureStore(out_lut, id.xy, vec4<f32>(integrate_transmittance(origin, dir), 1.0));
}
