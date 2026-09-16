// Froxel fog, light injection (Hillaire 2015): per froxel, height fog
// extinction and the light it scatters toward the camera, from the sun or
// moon (atmospheric transmittance, sky map visibility, cloud shadow,
// Henyey-Greenstein phase) and from the sky (ambient cube, sky openness,
// isotropic phase). Samples are jittered within the froxel every frame and
// blended with last frame's volume reprojected into this one.
#import "common.wgsl"
#import "frame.wgsl"
#import "atmosphere.wgsl"

@group(1) @binding(0) var history: texture_3d<f32>;
@group(1) @binding(1) var out_volume: texture_storage_3d<rgba16float, write>;
@group(1) @binding(2) var linear_clamp: sampler;
@group(1) @binding(3) var transmittance_lut: texture_2d<f32>;
@group(1) @binding(4) var sky_ambient: texture_2d<f32>;
@group(1) @binding(5) var sky_map: texture_2d<f32>;
@group(1) @binding(6) var cloud_shadow_map: texture_2d<f32>;
@group(1) @binding(7) var<uniform> clouds: CloudParams;
@group(1) @binding(8) var<uniform> fog: FogParams;
#import "ambient.wgsl"
#import "skymap.wgsl"
#import "clouds_params.wgsl"
#import "cloud_shadow.wgsl"
#import "fog_common.wgsl"

const HISTORY_WEIGHT: f32 = 0.9;

fn hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(fog.width, fog.height, fog.depth);
    if any(gid >= dims) {
        return;
    }
    var rng = rng_seed(gid.xy + vec2<u32>(gid.z * 7919u, 0u), frame.frame_index);
    let jitter = vec3<f32>(rng_next(&rng), rng_next(&rng), rng_next(&rng));
    let uv = (vec2<f32>(gid.xy) + jitter.xy) / vec2<f32>(dims.xy);
    let w = (f32(gid.z) + jitter.z) / f32(dims.z);
    let distance = fog_slice_distance(w);
    let dir = fog_direction(uv);
    let rel = dir * distance;
    let camera_m = (vec3<f32>(frame.camera_voxel) + frame.camera_frac) / VOXELS_PER_METRE;
    let world_m = camera_m + rel;
    let sigma_t = fog_density(world_m.y);
    var scatter = vec3<f32>(0.0);
    if sigma_t > 0.0 {
        let world_voxels = world_m * VOXELS_PER_METRE;
        let up = vec3<f32>(0.0, 1.0, 0.0);
        let altitude_km = max(world_m.y, 0.0) / 1000.0;
        let sun = normalize(frame.sun_dir);
        let moon = normalize(frame.moon_dir);
        var light = vec3<f32>(0.0);
        if sun.y > -0.05 {
            let t = textureSampleLevel(transmittance_lut, linear_clamp, transmittance_uv(PLANET_RADIUS + altitude_km, sun.y), 0.0).rgb;
            // A point in the air: any "surface" normal facing the light works
            // for the height-field visibility test.
            let visible = sky_map_light_visibility(world_voxels, sun, sun) * cloud_shadow(world_m, sun);
            light += frame.sun_illuminance * t * visible * hg(dot(dir, sun), fog.anisotropy);
        }
        if moon.y > -0.05 && sun.y < 0.05 {
            let t = textureSampleLevel(transmittance_lut, linear_clamp, transmittance_uv(PLANET_RADIUS + altitude_km, moon.y), 0.0).rgb;
            let visible = sky_map_light_visibility(world_voxels, moon, moon) * cloud_shadow(world_m, moon);
            light += frame.moon_illuminance * t * visible * hg(dot(dir, moon), fog.anisotropy);
        }
        let sky_open = sky_map_sky_visibility(world_voxels, up);
        light += sky_irradiance(up) * sky_open / (4.0 * PI);
        scatter = light * sigma_t * fog.albedo;
    }
    var result = vec4<f32>(scatter, sigma_t);

    // Reproject last frame's value at this froxel's centre.
    if fog.history_valid == 1u {
        let centre_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims.xy);
        let centre_rel = fog_direction(centre_uv) * fog_slice_distance((f32(gid.z) + 0.5) / f32(dims.z));
        let prev_rel = centre_rel - frame.prev_camera_delta;
        let clip = frame.prev_view_proj * vec4<f32>(prev_rel, 1.0);
        let prev_ndc = clip.xy / clip.w;
        let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
        let prev_w = fog_slice_coord(length(prev_rel));
        if clip.w > 0.0 && all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0)) {
            let h = textureSampleLevel(history, linear_clamp, vec3<f32>(prev_uv, prev_w), 0.0);
            result = mix(result, h, HISTORY_WEIGHT);
        }
    }
    textureStore(out_volume, gid, result);
}
