// Volumetric cloud layer (Schneider and Vos 2015; lighting after Hillaire
// 2016 and Wrenninge's multiple scattering octaves). Positions are world
// metres with y up; the layer is a spherical shell over the planet so it
// sinks below the horizon with distance.
//
// Expects `shape_noise`, `detail_noise` (3D) and `noise_sampler`, and a
// `clouds: CloudParams` uniform declared before import (import
// clouds_params.wgsl first for the struct).

#import "clouds_params.wgsl"

const SHAPE_TILE_M: f32 = 6000.0;
const DETAIL_TILE_M: f32 = 600.0;
const COVERAGE_TILE_M: f32 = 24000.0;
const EARTH_RADIUS_M: f32 = 6360000.0;

fn remap(v: f32, lo: f32, hi: f32, new_lo: f32, new_hi: f32) -> f32 {
    return new_lo + (v - lo) / max(hi - lo, 1e-5) * (new_hi - new_lo);
}

// Altitude above the planet of a point given relative to a camera at
// altitude `camera_alt_m`, following the curvature.
fn shell_altitude(p: vec3<f32>, camera_alt_m: f32) -> f32 {
    let centre_to_p = vec3<f32>(p.x, p.y + camera_alt_m + EARTH_RADIUS_M, p.z);
    return length(centre_to_p) - EARTH_RADIUS_M;
}

fn height_fraction(altitude: f32) -> f32 {
    return clamp((altitude - clouds.base_m) / (clouds.top_m - clouds.base_m), 0.0, 1.0);
}

// Density over height for stratus (thin, low) blending to cumulus (tall,
// rounded tops).
fn height_gradient(h: f32) -> f32 {
    let stratus = smoothstep(0.0, 0.05, h) * (1.0 - smoothstep(0.12, 0.3, h));
    let cumulus = smoothstep(0.0, 0.12, h) * (1.0 - smoothstep(0.45, 1.0, h));
    return mix(stratus, cumulus, clouds.cloud_type);
}

// Weather: coverage varies over tens of kilometres around the global value.
fn local_coverage(world_xz: vec2<f32>) -> f32 {
    let uvw = vec3<f32>((world_xz + clouds.wind * 0.3) / COVERAGE_TILE_M, 0.37).xzy;
    let n = textureSampleLevel(shape_noise, noise_sampler, uvw, 0.0).r;
    return clamp(clouds.coverage + (n - 0.5) * 0.6, 0.0, 1.0);
}

// Cloud density at a world point; `cheap` skips the detail erosion.
fn cloud_density(world: vec3<f32>, altitude: f32, cheap: bool) -> f32 {
    let h = height_fraction(altitude);
    if h <= 0.0 || h >= 1.0 {
        return 0.0;
    }
    let p = world + vec3<f32>(clouds.wind.x, 0.0, clouds.wind.y);
    // Tops lean downwind.
    let lean = vec3<f32>(clouds.wind.x, 0.0, clouds.wind.y) * h * 0.02;
    let shape = textureSampleLevel(shape_noise, noise_sampler, (p + lean) / SHAPE_TILE_M, 0.0);
    let low_freq = shape.g * 0.625 + shape.b * 0.25 + shape.a * 0.125;
    var base = remap(shape.r, -(1.0 - low_freq), 1.0, 0.0, 1.0);
    // The Perlin-Worley dilations above leave base values clustered around
    // 0.75; stretch them so coverage has the full range to cut into.
    base = clamp(remap(base, 0.62, 0.95, 0.0, 1.0), 0.0, 1.0);
    base *= height_gradient(h);
    let coverage = local_coverage(world.xz);
    base = clamp(remap(base, 1.0 - coverage, 1.0, 0.0, 1.0), 0.0, 1.0) * coverage;
    // Rain clouds sit lower and denser.
    base *= mix(1.0, 1.4, clouds.precipitation);
    if cheap || base <= 0.0 {
        return base;
    }
    let detail = textureSampleLevel(detail_noise, noise_sampler, p / DETAIL_TILE_M, 0.0);
    let high_freq = detail.r * 0.625 + detail.g * 0.25 + detail.b * 0.125;
    // Wispy bottoms, billowy tops.
    let modifier = mix(high_freq, 1.0 - high_freq, clamp(h * 10.0, 0.0, 1.0));
    return clamp(remap(base, modifier * 0.35, 1.0, 0.0, 1.0), 0.0, 1.0);
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

// Forward-scattering lobe for silver linings plus a weak back lobe.
fn cloud_phase(cos_theta: f32, g_scale: f32) -> f32 {
    return mix(henyey_greenstein(cos_theta, -0.2 * g_scale), henyey_greenstein(cos_theta, 0.8 * g_scale), 0.7);
}

// Optical depth toward the light: a widening cone of samples, the last far
// away to catch shadows from distant clouds.
fn light_optical_depth(world: vec3<f32>, camera_alt_m: f32, rel: vec3<f32>, light: vec3<f32>, seed: f32) -> f32 {
    let n = max(clouds.light_steps, 1u);
    var depth = 0.0;
    var t = 0.0;
    let thickness = clouds.top_m - clouds.base_m;
    for (var i = 0u; i < n; i++) {
        // Steps of 60, 120, 180 m... about 1.3 km over six samples.
        let step = 60.0 * f32(i + 1u);
        t += step;
        // Cone spread grows with distance.
        let jitter = vec3<f32>(
            fract(seed * 12.9898 + f32(i) * 0.618) - 0.5,
            0.0,
            fract(seed * 78.233 + f32(i) * 0.381) - 0.5,
        ) * t * 0.3;
        let q = world + light * t + jitter;
        let alt = shell_altitude(rel + light * t + jitter, camera_alt_m);
        depth += cloud_density(q, alt, false) * step;
    }
    // Far sample.
    let far_t = t + thickness * 0.5;
    let far_alt = shell_altitude(rel + light * far_t, camera_alt_m);
    depth += cloud_density(world + light * far_t, far_alt, true) * thickness * 0.3;
    return depth * clouds.extinction;
}

// Scattered light at a point toward the viewer, per unit light illuminance:
// three octaves of Wrenninge's multiple scattering approximation with a
// view-dependent powder term (dark edges facing the light).
fn cloud_scattering(optical_depth: f32, cos_theta: f32, density: f32) -> f32 {
    var sum = 0.0;
    var a = 1.0;
    var b = 1.0;
    var c = 1.0;
    let absorption = mix(1.0, 2.5, clouds.precipitation);
    for (var i = 0; i < 3; i++) {
        sum += b * exp(-optical_depth * a * absorption) * cloud_phase(cos_theta, c);
        a *= 0.5;
        b *= 0.5;
        c *= 0.5;
    }
    let powder = 1.0 - exp(-density * clouds.extinction * 60.0);
    let view_powder = mix(powder, 1.0, 0.5 + 0.5 * cos_theta);
    return sum * mix(0.6, 1.0, view_powder);
}
