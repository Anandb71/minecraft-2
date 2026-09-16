// Sky radiance and transmittance lookups shared by composition and indirect
// light. Expects `transmittance_lut`, `sky_view_lut`, `sky_view_moon` and
// `lut_sampler` bindings declared before import, after frame.wgsl and
// atmosphere.wgsl.

// Natural night sky brightness without moon, mostly airglow, cd/m^2.
const NIGHT_SKY: vec3<f32> = vec3<f32>(1.6e-4, 2.2e-4, 2.0e-4);

fn camera_height_km() -> f32 {
    let y_m = (f32(frame.camera_voxel.y) + frame.camera_frac.y) / VOXELS_PER_METRE;
    return PLANET_RADIUS + max(y_m, 1.0) / 1000.0;
}

fn moon_sky() -> bool {
    return (frame.sky_flags & SKY_MOON) != 0u;
}

// Sky luminance toward `dir` per unit illuminance of the light the LUT was
// built for.
fn sky_lut(lut: texture_2d<f32>, dir: vec3<f32>, light: vec3<f32>) -> vec3<f32> {
    let height = camera_height_km();
    let view_zenith_cos = dir.y;
    // Azimuth relative to the light, measured in the horizontal plane.
    let light_h = normalize(vec3<f32>(light.x, 0.0, light.z) + vec3<f32>(1e-5, 0.0, 0.0));
    let dir_h = normalize(vec3<f32>(dir.x, 0.0, dir.z) + vec3<f32>(1e-5, 0.0, 0.0));
    let light_view_cos = dot(light_h, dir_h);
    let origin = vec3<f32>(0.0, height, 0.0);
    let hits_ground = ray_sphere(origin, dir, PLANET_RADIUS) > 0.0;
    let uv = sky_view_uv(height, view_zenith_cos, light_view_cos, hits_ground);
    return textureSampleLevel(lut, lut_sampler, uv, 0.0).rgb;
}

// Sky luminance from both lights, cd/m^2.
fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    var l = sky_lut(sky_view_lut, dir, normalize(frame.sun_dir)) * frame.sun_illuminance;
    if moon_sky() {
        l += sky_lut(sky_view_moon, dir, normalize(frame.moon_dir)) * frame.moon_illuminance;
    }
    return l;
}

// Transmittance toward a light from an altitude, zero once the planet
// blocks it.
fn light_transmittance(altitude_km: f32, dir: vec3<f32>) -> vec3<f32> {
    let h = PLANET_RADIUS + max(altitude_km, 0.001);
    if ray_sphere(vec3<f32>(0.0, h, 0.0), dir, PLANET_RADIUS) > 0.0 {
        return vec3<f32>(0.0);
    }
    return textureSampleLevel(transmittance_lut, lut_sampler, transmittance_uv(h, dir.y), 0.0).rgb;
}
