// Composes the lit image from the visibility buffer: sun and moon with
// traced visibility, sky ambient, emitted light gathered by ReSTIR, material
// emission, then aerial perspective. Sky pixels come from the sky-view LUT
// plus the sun disc.
#import "common.wgsl"
#import "frame.wgsl"
#import "voxel_data.wgsl"
#import "atmosphere.wgsl"
#import "brdf.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var light_vis: texture_2d<f32>;
@group(2) @binding(4) var direct_lights: texture_2d<f32>;
@group(2) @binding(5) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(6) var sky_view_lut: texture_2d<f32>;
@group(2) @binding(7) var aerial: texture_3d<f32>;
@group(2) @binding(8) var lut_sampler: sampler;
@group(2) @binding(9) var out_hdr: texture_storage_2d<rgba16float, write>;

const AERIAL_DEPTH_KM: f32 = 8.0;

fn camera_height_km() -> f32 {
    let y_m = (f32(frame.camera_voxel.y) + frame.camera_frac.y) / VOXELS_PER_METRE;
    return PLANET_RADIUS + max(y_m, 1.0) / 1000.0;
}

// Sky luminance toward `dir` per unit sun illuminance.
fn sky_lut(dir: vec3<f32>) -> vec3<f32> {
    let height = camera_height_km();
    let sun = normalize(frame.sun_dir);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let view_zenith_cos = dot(dir, up);
    // Azimuth relative to the sun, measured in the horizontal plane.
    let sun_h = normalize(vec3<f32>(sun.x, 0.0, sun.z) + vec3<f32>(1e-5, 0.0, 0.0));
    let dir_h = normalize(vec3<f32>(dir.x, 0.0, dir.z) + vec3<f32>(1e-5, 0.0, 0.0));
    let light_view_cos = dot(sun_h, dir_h);
    let origin = vec3<f32>(0.0, height, 0.0);
    let hits_ground = ray_sphere(origin, dir, PLANET_RADIUS) > 0.0;
    let uv = sky_view_uv(height, view_zenith_cos, light_view_cos, hits_ground);
    return textureSampleLevel(sky_view_lut, lut_sampler, uv, 0.0).rgb;
}

fn sun_transmittance_at(altitude_km: f32, cos_zenith: f32) -> vec3<f32> {
    let h = PLANET_RADIUS + max(altitude_km, 0.001);
    return textureSampleLevel(transmittance_lut, lut_sampler, transmittance_uv(h, cos_zenith), 0.0).rgb;
}

// Irradiance from the sky dome around a normal, eight fixed samples.
fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    for (var i = 0; i < 8; i++) {
        let phi = f32(i) / 8.0 * TAU;
        let d = normalize(n * 0.7 + (t * cos(phi) + b * sin(phi)) * 0.71);
        sum += sky_lut(normalize(vec3<f32>(d.x, max(d.y, 0.02), d.z)));
    }
    return sum / 8.0 * PI;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.render_size);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let id = textureLoad(vis_id, pixel, 0);
    let dir = camera_ray_dir(vec2<f32>(pixel));
    let sun = normalize(frame.sun_dir);
    let kind = id.w >> 30u;

    if kind == 0u {
        var sky = sky_lut(dir) * frame.sun_illuminance;
        // Sun disc, limb darkened, through the atmosphere.
        let cos_disc = cos(frame.sun_angular_radius);
        let c = dot(dir, sun);
        if c > cos_disc {
            let r = sqrt(max(0.0, 1.0 - (1.0 - c) / (1.0 - cos_disc)));
            let solid_angle = TAU * (1.0 - cos_disc);
            let t = sun_transmittance_at(camera_height_km() - PLANET_RADIUS, sun.y);
            sky += frame.sun_illuminance / solid_angle * t * (0.4 + 0.6 * r);
        }
        // Night: a faint moonlit sky.
        sky += vec3<f32>(0.02, 0.03, 0.06) * frame.moon_illuminance * max(dir.y, 0.0);
        textureStore(out_hdr, pixel, vec4<f32>(sky, 1.0));
        return;
    }

    let depth_m = textureLoad(vis_depth, pixel, 0).r;
    let normal = oct_decode(textureLoad(vis_motion, pixel, 0).ba);
    let mat = materials[id.w & 0xffffu];
    let rel_m = dir * depth_m;
    let altitude_km = ((f32(frame.camera_voxel.y) + frame.camera_frac.y) / VOXELS_PER_METRE + rel_m.y) / 1000.0;
    let v = -dir;
    let lv = textureLoad(light_vis, pixel, 0);
    let voxel_hash = hash_to_unit(pcg(id.x * 73856093u ^ id.y * 19349663u ^ id.z * 83492791u));
    let albedo = mat.albedo * (0.92 + 0.16 * voxel_hash);
    let diffuse = albedo * (1.0 - mat.metallic) * INV_PI;
    let f0 = mix(vec3<f32>(0.04), mat.albedo, mat.metallic);

    var light = vec3<f32>(0.0);
    // Sun.
    let nl_sun = max(dot(normal, sun), 0.0);
    if nl_sun > 0.0 && lv.r > 0.0 {
        let t = sun_transmittance_at(altitude_km, sun.y);
        let e = frame.sun_illuminance * t * lv.r;
        light += e * (diffuse * nl_sun + ggx_specular(normal, v, sun, mat.roughness, f0));
    }
    // Moon.
    let moon = normalize(frame.moon_dir);
    let nl_moon = max(dot(normal, moon), 0.0);
    if nl_moon > 0.0 && lv.g > 0.0 {
        let e = vec3<f32>(0.6, 0.7, 1.0) * frame.moon_illuminance * lv.g;
        light += e * diffuse * nl_moon;
    }
    // Sky light scaled by traced sky visibility; bounce light is added by
    // the indirect pass.
    let sky_e = sky_irradiance(normal) * frame.sun_illuminance;
    light += (sky_e * diffuse + vec3<f32>(0.004, 0.006, 0.012) * frame.moon_illuminance * albedo) * lv.b;
    // Emissive voxels gathered by ReSTIR.
    light += textureLoad(direct_lights, pixel, 0).rgb;
    // Emission.
    light += mat.emission;

    // Aerial perspective.
    let uv = (vec2<f32>(pixel) + 0.5) / frame.render_size;
    let slice = sqrt(clamp(depth_m / 1000.0 / AERIAL_DEPTH_KM, 0.0, 1.0));
    let ap = textureSampleLevel(aerial, lut_sampler, vec3<f32>(uv, slice), 0.0);
    let color = light * ap.a + ap.rgb * frame.sun_illuminance;
    textureStore(out_hdr, pixel, vec4<f32>(color, 1.0));
}

