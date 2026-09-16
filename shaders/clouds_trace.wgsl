// Cloud ray march at half render resolution, one texel of every 2x2 block
// per frame (the resolve pass reprojects the rest, Schneider and Vos's
// amortisation). Cheap base-shape samples at long steps until a cloud is
// near, then full samples with detail and a light cone; stops once the ray
// is nearly opaque. Output: in-scattered luminance (rgb, cd/m^2) and
// transmittance (a), both faded with distance into the sky behind.
#import "common.wgsl"
#import "frame.wgsl"
#import "atmosphere.wgsl"

@group(1) @binding(0) var shape_noise: texture_3d<f32>;
@group(1) @binding(1) var detail_noise: texture_3d<f32>;
@group(1) @binding(2) var noise_sampler: sampler;
@group(1) @binding(3) var transmittance_lut: texture_2d<f32>;
@group(1) @binding(4) var lut_sampler: sampler;
@group(1) @binding(5) var sky_ambient: texture_2d<f32>;
@group(1) @binding(6) var out_trace: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var<uniform> clouds: CloudParams;
#import "clouds_params.wgsl"
#import "clouds_common.wgsl"
#import "ambient.wgsl"

const MAX_DISTANCE_M: f32 = 35000.0;
const HAZE_DISTANCE_M: f32 = 45000.0;

// Distances along `dir` from the camera to a sphere of radius `r` around the
// planet centre; (-1, -1) when missed.
fn sphere_hits(camera_alt: f32, dir: vec3<f32>, r: f32) -> vec2<f32> {
    let o = vec3<f32>(0.0, camera_alt + EARTH_RADIUS_M, 0.0);
    let b = dot(o, dir);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return vec2<f32>(-1.0);
    }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

fn cloud_slot(frame_index: u32) -> vec2<u32> {
    let k = (0x2130u >> ((frame_index % 4u) * 4u)) & 0xfu;
    return vec2<u32>(k & 1u, k >> 1u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = (vec2<u32>(frame.render_size) + 1u) / 2u;
    let blocks = (size + 1u) / 2u;
    if any(gid.xy >= blocks) {
        return;
    }
    let texel = min(gid.xy * 2u + cloud_slot(frame.frame_index), size - 1u);
    let dir = camera_ray_dir(vec2<f32>(texel * 2u) + 0.5);
    let camera_alt = (f32(frame.camera_voxel.y) + frame.camera_frac.y) / VOXELS_PER_METRE;

    // The layer between the base and top shells.
    let base_r = EARTH_RADIUS_M + clouds.base_m;
    let top_r = EARTH_RADIUS_M + clouds.top_m;
    let base_hits = sphere_hits(camera_alt, dir, base_r);
    let top_hits = sphere_hits(camera_alt, dir, top_r);
    var start = 0.0;
    var end = 0.0;
    if camera_alt < clouds.base_m {
        // From below: rays that reach the base shell going outward.
        if base_hits.y <= 0.0 || dir.y < -0.05 {
            textureStore(out_trace, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
            return;
        }
        start = base_hits.y;
        end = top_hits.y;
    } else if camera_alt < clouds.top_m {
        end = select(top_hits.y, base_hits.x, base_hits.x > 0.0);
    } else {
        if top_hits.x <= 0.0 {
            textureStore(out_trace, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
            return;
        }
        start = top_hits.x;
        end = select(top_hits.y, base_hits.x, base_hits.x > 0.0);
    }
    end = min(end, MAX_DISTANCE_M);
    if end <= start {
        textureStore(out_trace, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Light: the sun, or the moon once the sun has set.
    let sun = normalize(frame.sun_dir);
    let moon = normalize(frame.moon_dir);
    let night = sun.y < -0.05;
    let light = select(sun, moon, night);
    let illuminance = select(frame.sun_illuminance, vec3<f32>(frame.moon_illuminance), night);
    let cos_theta = dot(dir, light);
    let ambient_top = sky_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * INV_PI;
    let ambient_side = (sky_irradiance(vec3<f32>(1.0, 0.0, 0.0)) + sky_irradiance(vec3<f32>(-1.0, 0.0, 0.0))) * 0.5 * INV_PI;

    var rng = rng_seed(texel, frame.frame_index);
    let steps = max(clouds.steps, 8u);
    let step = (end - start) / f32(steps);
    var t = start + step * rng_next(&rng);
    let camera_world = clouds.camera_world;
    var transmittance = 1.0;
    var luminance = vec3<f32>(0.0);
    var empty_run = 0u;
    var fine = false;
    for (var i = 0u; i < steps * 2u; i++) {
        if t >= end || transmittance < 0.02 {
            break;
        }
        let rel = dir * t;
        let world = camera_world + rel;
        let altitude = shell_altitude(rel, camera_alt);
        if !fine {
            // Cheap search at double stride until a base shape appears.
            if cloud_density(world, altitude, true) > 0.0 {
                fine = true;
                t = max(t - step, start);
                empty_run = 0u;
            } else {
                t += step * 2.0;
            }
            continue;
        }
        let density = cloud_density(world, altitude, false);
        if density <= 0.0 {
            empty_run += 1u;
            if empty_run > 6u {
                fine = false;
            }
            t += step;
            continue;
        }
        empty_run = 0u;
        let sigma = density * clouds.extinction;
        let optical = light_optical_depth(world, camera_alt, rel, light, rng_next(&rng));
        let altitude_km = max(altitude, 0.0) / 1000.0;
        let sun_t = textureSampleLevel(transmittance_lut, lut_sampler, transmittance_uv(PLANET_RADIUS + altitude_km, light.y), 0.0).rgb;
        let direct = illuminance * sun_t * cloud_scattering(optical, cos_theta, density);
        let h = height_fraction(altitude);
        let ambient = mix(ambient_side, ambient_top, h) * mix(0.35, 1.0, h);
        let scatter = (direct + ambient * 0.25) * sigma;
        let step_t = exp(-sigma * step);
        // Energy-conserving integration over the step (Hillaire 2015).
        luminance += transmittance * (scatter - scatter * step_t) / max(sigma, 1e-7);
        transmittance *= step_t;
        t += step;
    }
    // Distant clouds dissolve into the sky behind them.
    let haze = 1.0 - exp(-start / HAZE_DISTANCE_M);
    luminance *= 1.0 - haze;
    transmittance = mix(transmittance, 1.0, haze);
    textureStore(out_trace, gid.xy, vec4<f32>(luminance, transmittance));
}
