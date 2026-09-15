// Sky-view LUT, per frame: sky luminance around the camera per unit sun
// illuminance, in Hillaire's horizon-dense latitude mapping.
#import "frame.wgsl"
#import "atmosphere.wgsl"

@group(1) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(1) @binding(1) var multiscatter_lut: texture_2d<f32>;
@group(1) @binding(2) var lut_sampler: sampler;
@group(1) @binding(3) var out_lut: texture_storage_2d<rgba16float, write>;

fn camera_height_km() -> f32 {
    let y_m = (f32(frame.camera_voxel.y) + frame.camera_frac.y) / VOXELS_PER_METRE;
    return PLANET_RADIUS + max(y_m, 1.0) / 1000.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out_lut);
    if any(id.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let height = camera_height_km();
    let p = sky_view_params(uv, height);
    let sun = normalize(frame.sun_dir);
    let sun_zenith = clamp(sun.y, -1.0, 1.0);
    // Local frame: sun in the XY plane.
    let sun_local = vec3<f32>(sqrt(max(0.0, 1.0 - sun_zenith * sun_zenith)), sun_zenith, 0.0);
    let vz = p.view_zenith_cos;
    let vs = sqrt(max(0.0, 1.0 - vz * vz));
    let lc = p.light_view_cos;
    let ls = sqrt(max(0.0, 1.0 - lc * lc));
    let dir = vec3<f32>(vs * lc, vz, vs * ls);
    let origin = vec3<f32>(0.0, height, 0.0);
    let result = integrate_scattering(origin, dir, sun_local, 1e9, 30, transmittance_lut, multiscatter_lut, lut_sampler);
    textureStore(out_lut, id.xy, vec4<f32>(result.rgb, 1.0));
}
