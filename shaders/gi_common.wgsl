// Indirect rays and the radiance leaving what they hit. Shared by ReSTIR GI
// and radiance cascades.
//
// A hit that last frame's camera saw on the same surface returns that
// frame's lit surface radiance: direct light with traced shadows, sky
// visibility, emitters and last frame's indirect light, so bounces compound
// over frames at no extra cost. A hit off screen is lit from the sky map
// (height-field sun visibility and sky openness) and the sky ambient cube.
//
// Expects, declared before import: prev_id (u32), surface_prev (f32),
// sky_map, sky_ambient, transmittance_lut, sky_view_lut, sky_view_moon,
// lut_sampler; and common, frame, voxel_data, march and atmosphere imports.
#import "sky_common.wgsl"
#import "ambient.wgsl"
#import "skymap.wgsl"
#import "gi_sample.wgsl"

fn gi_ray(origin: vec3<f32>, dir: vec3<f32>, t_min_m: f32, t_max_m: f32) -> Ray {
    var r: Ray;
    r.base = frame.camera_voxel;
    r.frac = origin;
    r.dir = dir;
    r.t_min = t_min_m * VOXELS_PER_METRE;
    r.t_max = t_max_m * VOXELS_PER_METRE;
    r.lod_scale = frame.pixel_angle * frame.lod_pixels * 8.0;
    r.feedback = false;
    r.coarse = false;
    return r;
}

// Outgoing diffuse radiance of a surface point toward any direction.
fn surface_radiance(position: vec3<f32>, normal: vec3<f32>, voxel: vec3<i32>, material: u32, face: u32) -> vec3<f32> {
    let mat = materials[material];
    // Seen last frame?
    let rel_m = (position - frame.camera_frac) / VOXELS_PER_METRE;
    let uv = prev_uv(rel_m);
    if all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0)) {
        let pp = vec2<i32>(uv * frame.render_size);
        let pid = textureLoad(prev_id, pp, 0);
        let hid = vec4<u32>(vec3<u32>(voxel), material | (face << 16u) | (1u << 30u));
        if same_surface(hid, pid, length(rel_m), 1.0) {
            return textureLoad(surface_prev, pp, 0).rgb;
        }
    }
    let world = vec3<f32>(frame.camera_voxel) + position;
    let altitude_km = world.y / VOXELS_PER_METRE / 1000.0;
    let sun = normalize(frame.sun_dir);
    var e = frame.sun_illuminance * light_transmittance(altitude_km, sun) * max(dot(normal, sun), 0.0)
        * sky_map_light_visibility(world, normal, sun);
    if moon_sky() {
        let moon = normalize(frame.moon_dir);
        e += frame.moon_illuminance * light_transmittance(altitude_km, moon) * max(dot(normal, moon), 0.0)
            * sky_map_light_visibility(world, normal, moon);
    }
    e += sky_irradiance(normal) * sky_map_sky_visibility(world, normal);
    return mat.emission + mat.albedo * (1.0 - mat.metallic) * INV_PI * e;
}

// Traces one indirect ray over [t_min_m, t_max_m] from a camera-relative
// origin (voxels). A miss past t_max_m within range returns no radiance
// with sky = 0 unless `sky_on_miss`.
fn trace_gi(origin: vec3<f32>, dir: vec3<f32>, t_min_m: f32, t_max_m: f32, sky_on_miss: bool) -> GiSample {
    let hit = march(gi_ray(origin, dir, t_min_m, t_max_m));
    var s: GiSample;
    if hit.kind == HIT_NONE {
        s.position = dir;
        s.normal = -dir;
        s.sky = 1u;
        s.radiance = select(vec3<f32>(0.0), sky_radiance(dir), sky_on_miss);
        return s;
    }
    s.position = origin + dir * hit.t;
    s.normal = select(vec3<f32>(0.0), -sign(dir), axis_mask(hit.axis));
    let face = hit.axis * 2u + select(0u, 1u, dir[hit.axis] < 0.0);
    s.sky = 0u;
    s.radiance = surface_radiance(s.position, s.normal, hit.voxel, hit.material, face);
    return s;
}

// Uniform direction on the hemisphere around n; pdf 1 / (2 pi).
fn sample_hemisphere_uniform(n: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
    let z = u.x;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = TAU * u.y;
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return normalize(t * r * cos(phi) + b * r * sin(phi) + n * z);
}
