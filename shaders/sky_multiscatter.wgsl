// Multiple scattering LUT (Hillaire 2020, section 5.5). For a sun zenith
// angle and altitude, integrate second-order scattering over 64 directions
// with an isotropic phase, including light bounced off the ground, and sum
// all higher orders as the geometric series Psi_2 / (1 - f_ms), where f_ms
// is the direction-averaged transmittance-weighted scattering. Built once.
#import "atmosphere.wgsl"

@group(0) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(1) var lut_sampler: sampler;
@group(0) @binding(2) var out_lut: texture_storage_2d<rgba16float, write>;

// Transmittance to the sun, zero in the planet's shadow.
fn sun_transmittance(p: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    if ray_sphere(p, sun, PLANET_RADIUS) > 0.0 {
        return vec3<f32>(0.0);
    }
    let h = length(p);
    let uv = transmittance_uv(h, dot(p / h, sun));
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out_lut);
    if any(id.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let cos_sun = uv.x * 2.0 - 1.0;
    let sun = vec3<f32>(0.0, cos_sun, sqrt(max(0.0, 1.0 - cos_sun * cos_sun)));
    let height = PLANET_RADIUS + clamp(uv.y, 0.0, 1.0) * (ATMOSPHERE_RADIUS - PLANET_RADIUS);
    let origin = vec3<f32>(0.0, max(height, PLANET_RADIUS + 0.01), 0.0);

    let sqrt_samples = 8;
    let isotropic = 1.0 / (4.0 * 3.14159265);
    var second_order = vec3<f32>(0.0);
    var transfer = vec3<f32>(0.0);
    let steps = 20;
    for (var i = 0; i < sqrt_samples; i++) {
        for (var j = 0; j < sqrt_samples; j++) {
            let theta = 2.0 * 3.14159265 * (f32(i) + 0.5) / f32(sqrt_samples);
            let phi = acos(1.0 - 2.0 * (f32(j) + 0.5) / f32(sqrt_samples));
            let dir = vec3<f32>(cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi));
            let t_ground = ray_sphere(origin, dir, PLANET_RADIUS);
            let t_top = ray_sphere(origin, dir, ATMOSPHERE_RADIUS);
            var t_max = t_top;
            if t_ground > 0.0 {
                t_max = t_ground;
            }
            if t_max <= 0.0 {
                continue;
            }
            let dt = t_max / f32(steps);
            var throughput = vec3<f32>(1.0);
            var luminance = vec3<f32>(0.0);
            var scattered = vec3<f32>(0.0);
            for (var s = 0; s < steps; s++) {
                let p = origin + dir * (f32(s) + 0.5) * dt;
                let m = sample_medium(p);
                let step_t = exp(-m.extinction * dt);
                let in_scatter = m.scattering * isotropic * sun_transmittance(p, sun);
                // Energy-conserving analytic integration over the segment.
                let segment = throughput * (1.0 - step_t) / max(m.extinction, vec3<f32>(1e-6));
                luminance += in_scatter * segment;
                scattered += m.scattering * segment;
                throughput *= step_t;
            }
            if t_ground > 0.0 {
                let ground = origin + dir * t_ground;
                let n = normalize(ground);
                luminance += throughput * sun_transmittance(ground, sun) * max(dot(n, sun), 0.0) * GROUND_ALBEDO / 3.14159265;
            }
            second_order += luminance / f32(sqrt_samples * sqrt_samples);
            transfer += scattered / f32(sqrt_samples * sqrt_samples);
        }
    }
    // Each further order is the previous one scattered along the averaged
    // path again, a factor f_ms, so all orders sum to Psi_2 / (1 - f_ms).
    let f_ms = clamp(transfer, vec3<f32>(0.0), vec3<f32>(0.99));
    let psi = second_order / (1.0 - f_ms);
    textureStore(out_lut, id.xy, vec4<f32>(psi, 1.0));
}
