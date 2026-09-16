// Cloud shadow map update: for a quarter of the texels each frame, the
// transmittance of a sun ray from the cloud base plane up through the layer.
// Texels are world cells addressed modulo the map size; each is resolved to
// the copy of its cell nearest the camera, so the map follows the camera
// without shifting.
#import "common.wgsl"
#import "frame.wgsl"

@group(1) @binding(0) var shape_noise: texture_3d<f32>;
@group(1) @binding(1) var detail_noise: texture_3d<f32>;
@group(1) @binding(2) var noise_sampler: sampler;
@group(1) @binding(3) var out_shadow: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var<uniform> clouds: CloudParams;
#import "clouds_params.wgsl"
#import "clouds_common.wgsl"

const MAP: i32 = 256;
const SAMPLES: i32 = 10;

fn slot(frame_index: u32) -> vec2<u32> {
    let k = (0x2130u >> ((frame_index % 4u) * 4u)) & 0xfu;
    return vec2<u32>(k & 1u, k >> 1u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= vec2<u32>(128u)) {
        return;
    }
    let texel = vec2<i32>(gid.xy * 2u + slot(frame.frame_index));
    let camera_cell = vec2<i32>(floor(clouds.camera_world.xz / clouds.shadow_texel_m));
    let camera_texel = ((camera_cell % MAP) + MAP) % MAP;
    // Offset from the camera's texel, wrapped into [-MAP/2, MAP/2).
    let offset = ((texel - camera_texel + MAP + MAP / 2) % MAP) - MAP / 2;
    let cell = camera_cell + offset;
    let sun = normalize(frame.sun_dir);
    let night = sun.y < -0.05;
    let light = select(sun, normalize(frame.moon_dir), night);
    if light.y <= 0.02 {
        textureStore(out_shadow, texel, vec4<f32>(1.0));
        return;
    }
    let start = vec3<f32>((vec2<f32>(cell) + 0.5) * clouds.shadow_texel_m, clouds.base_m).xzy;
    let length_m = (clouds.top_m - clouds.base_m) / light.y;
    let step = length_m / f32(SAMPLES);
    var depth = 0.0;
    for (var i = 0; i < SAMPLES; i++) {
        let p = start + light * (f32(i) + 0.5) * step;
        depth += cloud_density(p, p.y, false) * step;
    }
    let t = exp(-depth * clouds.extinction);
    textureStore(out_shadow, texel, vec4<f32>(t, t, t, 1.0));
}
