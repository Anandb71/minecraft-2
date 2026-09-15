// Aerial perspective froxels, per frame: in-scattering (rgb, per unit
// illuminance of the sun or moon) and mean transmittance (a) from the camera to each slice.
// 32 quadratic slices reach 8 km so the first kilometre keeps most of them.
#import "frame.wgsl"
#import "atmosphere.wgsl"

@group(1) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(1) @binding(1) var multiscatter_lut: texture_2d<f32>;
@group(1) @binding(2) var lut_sampler: sampler;
@group(1) @binding(4) var<uniform> params: SkyParams;

// Which light this dispatch integrates: 0 sun, 1 moon.
struct SkyParams {
    light: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn light_dir() -> vec3<f32> {
    return normalize(select(frame.sun_dir, frame.moon_dir, params.light == 1u));
}
@group(1) @binding(3) var out_volume: texture_storage_3d<rgba16float, write>;

const AERIAL_DEPTH_KM: f32 = 8.0;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out_volume);
    if any(id >= size) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size.xy);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let far = frame.inv_view_proj * vec4<f32>(ndc, 0.5, 1.0);
    let world_dir = normalize(far.xyz / far.w);
    let slice = (f32(id.z) + 0.5) / f32(size.z);
    let distance_km = AERIAL_DEPTH_KM * slice * slice;
    let y_m = (f32(frame.camera_voxel.y) + frame.camera_frac.y) / VOXELS_PER_METRE;
    let origin = vec3<f32>(0.0, PLANET_RADIUS + max(y_m, 1.0) / 1000.0, 0.0);
    let steps = i32(id.z) / 2 + 2;
    let result = integrate_scattering(origin, world_dir, light_dir(), distance_km, steps, transmittance_lut, multiscatter_lut, lut_sampler);
    textureStore(out_volume, id, result);
}
