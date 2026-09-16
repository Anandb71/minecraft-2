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
@group(2) @binding(10) var sky_view_moon: texture_2d<f32>;
@group(2) @binding(11) var aerial_moon: texture_3d<f32>;
// Sky irradiance for the six axis normals, one texel each: +X -X +Y -Y +Z -Z.
@group(2) @binding(12) var sky_ambient: texture_2d<f32>;
// Denoised indirect irradiance at GI resolution (half render resolution).
@group(2) @binding(13) var gi: texture_2d<f32>;
// Outgoing diffuse radiance of every surface: indirect rays reuse it next
// frame.
@group(2) @binding(14) var out_surface: texture_storage_2d<rgba16float, write>;
// Clouds at half resolution (rgb in-scattering, a transmittance), the
// cloud shadow map and the cloud layer parameters.
@group(2) @binding(15) var clouds_tex: texture_2d<f32>;
@group(2) @binding(16) var cloud_shadow_map: texture_2d<f32>;
@group(2) @binding(17) var<uniform> clouds: CloudParams;
#import "clouds_params.wgsl"
#import "cloud_shadow.wgsl"
#import "sky_common.wgsl"
#import "ambient.wgsl"

const AERIAL_DEPTH_KM: f32 = 8.0;
const MOON_ANGULAR_RADIUS: f32 = 0.004547;
const MOON_ALBEDO: f32 = 0.12;

// Indirect irradiance at a render pixel from the half-resolution GI signal:
// the four GI texels around it, weighted bilinearly and by agreement with
// this surface's plane and normal.
fn gi_upsample(pixel: vec2<i32>, position: vec3<f32>, normal: vec3<f32>, depth: f32) -> vec3<f32> {
    let gi_size = vec2<i32>(textureDimensions(gi));
    let g = (vec2<f32>(pixel) + 0.5) * 0.5 - 0.5;
    let base = vec2<i32>(floor(g));
    let f = g - vec2<f32>(base);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let q = clamp(base + o, vec2<i32>(0), gi_size - 1);
        let qp = min(q * 2, vec2<i32>(frame.render_size) - 1);
        let qid = textureLoad(vis_id, qp, 0);
        if (qid.w >> 30u) == 0u {
            continue;
        }
        let qpos = camera_ray_dir(vec2<f32>(qp)) * textureLoad(vis_depth, qp, 0).r;
        let qn = oct_decode(textureLoad(vis_motion, qp, 0).ba);
        let footprint = depth * frame.pixel_angle * 2.0 + 1e-3;
        let wg = exp(-abs(dot(normal, qpos - position)) / footprint) * pow(max(dot(qn, normal), 0.0), 32.0);
        let wb = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        let w = max(wb, 1e-3) * wg;
        sum += textureLoad(gi, q, 0).rgb * w;
        weight += w;
    }
    return select(vec3<f32>(0.0), sum / max(weight, 1e-6), weight > 1e-5);
}

// Star field fixed to the celestial sphere: one candidate star per cell of a
// 256x256 cube map, magnitudes drawn so counts triple per magnitude down to
// 6.5 (about 8000 naked-eye stars), each a Gaussian about one pixel wide.
fn stars(dir: vec3<f32>) -> vec3<f32> {
    // World to celestial: undo the latitude tilt, then the sidereal spin.
    let lat = frame.latitude;
    let north = vec3<f32>(0.0, 0.0, -1.0);
    let pole = vec3<f32>(0.0, sin(lat), 0.0) + north * cos(lat);
    let east = vec3<f32>(1.0, 0.0, 0.0);
    let meridian = cross(pole, east);
    let local = vec3<f32>(dot(dir, east), dot(dir, meridian), dot(dir, pole));
    let c = cos(frame.star_rotation);
    let sn = sin(frame.star_rotation);
    let cel = vec3<f32>(c * local.x - sn * local.y, sn * local.x + c * local.y, local.z);
    let a = abs(cel);
    var face = 0u;
    var uv = vec2<f32>(0.0);
    if a.x >= a.y && a.x >= a.z {
        face = select(1u, 0u, cel.x > 0.0);
        uv = cel.yz / a.x;
    } else if a.y >= a.z {
        face = select(3u, 2u, cel.y > 0.0);
        uv = cel.xz / a.y;
    } else {
        face = select(5u, 4u, cel.z > 0.0);
        uv = cel.xy / a.z;
    }
    let cells = 256.0;
    let g = (uv * 0.5 + 0.5) * cells;
    let cell = vec2<u32>(min(floor(g), vec2<f32>(cells - 1.0)));
    let h = pcg3d(vec3<u32>(cell, face * 7919u + 17u));
    if hash_to_unit(h.x) > 0.02 {
        return vec3<f32>(0.0);
    }
    let centre = vec2<f32>(cell) + vec2<f32>(0.25) + 0.5 * vec2<f32>(hash_to_unit(h.y), hash_to_unit(h.z));
    // A cube cell spans about 2 / cells radians near the face centre.
    let d = (g - centre) * (2.0 / cells);
    let sigma = max(frame.pixel_angle * 0.7, 1e-5);
    let magnitude = max(6.5 + log(max(hash_to_unit(pcg(h.x ^ h.z)), 1e-6)) / log(3.0), -1.5);
    let illuminance = 2.54e-6 * pow(10.0, -0.4 * magnitude);
    let radiance = illuminance / (TAU * sigma * sigma) * exp(-dot(d, d) / (2.0 * sigma * sigma));
    // Colour from a rough temperature spread: red dwarfs to blue giants.
    let temp = hash_to_unit(pcg(h.y ^ h.x));
    let tint = mix(vec3<f32>(1.0, 0.72, 0.52), vec3<f32>(0.78, 0.86, 1.0), temp);
    return radiance * tint;
}

// The moon's disc, lit by the sun as a Lambertian sphere with albedo 0.12
// and faint maria.
fn moon_disc(dir: vec3<f32>) -> vec3<f32> {
    let m = normalize(frame.moon_dir);
    if dot(dir, m) < cos(MOON_ANGULAR_RADIUS) {
        return vec3<f32>(0.0);
    }
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(m.y) > 0.99);
    let u = normalize(cross(up, m));
    let v = cross(m, u);
    let p = vec2<f32>(dot(dir, u), dot(dir, v)) / MOON_ANGULAR_RADIUS;
    let r2 = dot(p, p);
    if r2 >= 1.0 {
        return vec3<f32>(0.0);
    }
    let n = u * p.x + v * p.y - m * sqrt(1.0 - r2);
    let lit = max(dot(n, normalize(frame.sun_dir)), 0.0);
    let q = vec2<u32>((p * 0.5 + 0.5) * 12.0);
    let maria = 0.75 + 0.25 * hash_to_unit(pcg(q.x * 31u + q.y * 977u));
    let t = light_transmittance(camera_height_km() - PLANET_RADIUS, m);
    return frame.sun_illuminance * MOON_ALBEDO * INV_PI * lit * maria * t;
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
        var sky = sky_radiance(dir);
        let view_t = light_transmittance(camera_height_km() - PLANET_RADIUS, dir);
        // Sun disc, limb darkened, through the atmosphere.
        let cos_disc = cos(frame.sun_angular_radius);
        let c = dot(dir, sun);
        if c > cos_disc {
            let r = sqrt(max(0.0, 1.0 - (1.0 - c) / (1.0 - cos_disc)));
            let solid_angle = TAU * (1.0 - cos_disc);
            sky += frame.sun_illuminance / solid_angle * view_t * (0.4 + 0.6 * r);
        }
        sky += moon_disc(dir);
        sky += (stars(dir) + NIGHT_SKY) * view_t;
        // Clouds in front of everything beyond the atmosphere.
        // Half-resolution texel centres sit at even render pixels plus a half.
        let cloud_uv = (vec2<f32>(pixel) + 0.5) * 0.5 / vec2<f32>(textureDimensions(clouds_tex));
        let cloud = textureSampleLevel(clouds_tex, lut_sampler, cloud_uv, 0.0);
        sky = sky * cloud.a + cloud.rgb;
        textureStore(out_hdr, pixel, vec4<f32>(sky, 1.0));
        textureStore(out_surface, pixel, vec4<f32>(0.0));
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

    // Irradiance reaching the surface, then the reflected specular.
    var irradiance = vec3<f32>(0.0);
    var specular = vec3<f32>(0.0);
    let world_m = (vec3<f32>(frame.camera_voxel) + frame.camera_frac) / VOXELS_PER_METRE + rel_m;
    let nl_sun = max(dot(normal, sun), 0.0);
    if nl_sun > 0.0 && lv.r > 0.0 {
        let e = frame.sun_illuminance * light_transmittance(altitude_km, sun) * lv.r * cloud_shadow(world_m, sun);
        irradiance += e * nl_sun;
        specular += e * ggx_specular(normal, v, sun, mat.roughness, f0);
    }
    let moon = normalize(frame.moon_dir);
    let nl_moon = max(dot(normal, moon), 0.0);
    if nl_moon > 0.0 && lv.g > 0.0 {
        irradiance += frame.moon_illuminance * light_transmittance(altitude_km, moon) * lv.g * nl_moon * cloud_shadow(world_m, moon);
    }
    // Sky light scaled by traced sky visibility.
    irradiance += sky_irradiance(normal) * lv.b;
    // Emissive voxels gathered by ReSTIR, which does not run without lights.
    if frame.light_count > 0u {
        irradiance += textureLoad(direct_lights, pixel, 0).rgb;
    }
    // Indirect light.
    irradiance += gi_upsample(pixel, rel_m, normal, depth_m);
    let surface = mat.emission + diffuse * irradiance;
    textureStore(out_surface, pixel, vec4<f32>(surface, 1.0));
    let light = surface + specular;

    // Aerial perspective.
    let uv = (vec2<f32>(pixel) + 0.5) / frame.render_size;
    let slice = sqrt(clamp(depth_m / 1000.0 / AERIAL_DEPTH_KM, 0.0, 1.0));
    let ap = textureSampleLevel(aerial, lut_sampler, vec3<f32>(uv, slice), 0.0);
    var in_scatter = ap.rgb * frame.sun_illuminance;
    if moon_sky() {
        in_scatter += textureSampleLevel(aerial_moon, lut_sampler, vec3<f32>(uv, slice), 0.0).rgb * frame.moon_illuminance;
    }
    let color = light * ap.a + in_scatter;
    textureStore(out_hdr, pixel, vec4<f32>(color, 1.0));
}

