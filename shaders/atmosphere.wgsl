// Physically based atmosphere after Hillaire 2020, "A Scalable and
// Production Ready Sky and Atmosphere Rendering Technique".
//
// Distances in kilometres, planet centred at the origin, +Y up. Earth-like
// media: Rayleigh with an 8 km scale height, Mie with 1.2 km, ozone as a
// tent 30 km wide centred at 25 km. Multiple scattering from the 2D LUT of
// orders >= 2 with an isotropic phase function, summed as a geometric series.

const PLANET_RADIUS: f32 = 6360.0;
const ATMOSPHERE_RADIUS: f32 = 6460.0;
const RAYLEIGH_SCATTERING: vec3<f32> = vec3<f32>(5.802e-3, 13.558e-3, 33.1e-3);
const MIE_SCATTERING: f32 = 3.996e-3;
const MIE_ABSORPTION: f32 = 4.40e-3;
const OZONE_ABSORPTION: vec3<f32> = vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3);
const MIE_G: f32 = 0.8;
const GROUND_ALBEDO: vec3<f32> = vec3<f32>(0.3);

struct Medium {
    scattering: vec3<f32>,
    extinction: vec3<f32>,
    rayleigh: vec3<f32>,
    mie: vec3<f32>,
}

fn sample_medium(p: vec3<f32>) -> Medium {
    let h = max(length(p) - PLANET_RADIUS, 0.0);
    let rayleigh = RAYLEIGH_SCATTERING * exp(-h / 8.0);
    let mie_density = exp(-h / 1.2);
    let ozone = OZONE_ABSORPTION * max(0.0, 1.0 - abs(h - 25.0) / 15.0);
    let mie = vec3<f32>(MIE_SCATTERING * mie_density);
    var m: Medium;
    m.rayleigh = rayleigh;
    m.mie = mie;
    m.scattering = rayleigh + mie;
    m.extinction = rayleigh + vec3<f32>((MIE_SCATTERING + MIE_ABSORPTION) * mie_density) + ozone;
    return m;
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 3.0 * (1.0 + cos_theta * cos_theta) / (16.0 * 3.14159265);
}

// Cornette-Shanks.
fn mie_phase(cos_theta: f32) -> f32 {
    let g = MIE_G;
    let k = 3.0 / (8.0 * 3.14159265) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cos_theta * cos_theta) / pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5);
}

// Distance to the nearest positive intersection with a sphere at the
// origin, or -1.
fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> f32 {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return -1.0;
    }
    let s = sqrt(disc);
    let t0 = -b - s;
    let t1 = -b + s;
    if t0 >= 0.0 {
        return t0;
    }
    if t1 >= 0.0 {
        return t1;
    }
    return -1.0;
}

// Transmittance LUT parameterisation (Bruneton and Neyret): u from the view
// zenith cosine through the horizon distance, v from altitude.
fn transmittance_uv(height: f32, cos_zenith: f32) -> vec2<f32> {
    let h = sqrt(max(0.0, ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS - PLANET_RADIUS * PLANET_RADIUS));
    let rho = sqrt(max(0.0, height * height - PLANET_RADIUS * PLANET_RADIUS));
    let discriminant = height * height * (cos_zenith * cos_zenith - 1.0) + ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS;
    let d = max(0.0, -height * cos_zenith + sqrt(max(discriminant, 0.0)));
    let d_min = ATMOSPHERE_RADIUS - height;
    let d_max = rho + h;
    let x_mu = (d - d_min) / max(d_max - d_min, 1e-4);
    let x_r = rho / h;
    return vec2<f32>(x_mu, x_r);
}

fn transmittance_params(uv: vec2<f32>) -> vec2<f32> {
    let h = sqrt(ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS - PLANET_RADIUS * PLANET_RADIUS);
    let rho = h * uv.y;
    let height = sqrt(rho * rho + PLANET_RADIUS * PLANET_RADIUS);
    let d_min = ATMOSPHERE_RADIUS - height;
    let d_max = rho + h;
    let d = d_min + uv.x * (d_max - d_min);
    var cos_zenith = 1.0;
    if d > 0.0 {
        cos_zenith = (h * h - rho * rho - d * d) / (2.0 * height * d);
    }
    return vec2<f32>(height, clamp(cos_zenith, -1.0, 1.0));
}

// Optical depth integrated numerically; used to build the transmittance LUT.
fn integrate_transmittance(origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let t_max = ray_sphere(origin, dir, ATMOSPHERE_RADIUS);
    if t_max < 0.0 {
        return vec3<f32>(1.0);
    }
    let steps = 40;
    let dt = t_max / f32(steps);
    var depth = vec3<f32>(0.0);
    for (var i = 0; i < steps; i++) {
        let p = origin + dir * (f32(i) + 0.5) * dt;
        depth += sample_medium(p).extinction * dt;
    }
    return exp(-depth);
}

// Sky-view LUT mapping (Hillaire 2020, section 5.3): the horizon gets most
// of the latitude texels, and longitude is measured from the sun, mirrored.
struct SkyViewDir {
    view_zenith_cos: f32,
    light_view_cos: f32,
}

fn sky_view_params(uv: vec2<f32>, view_height: f32) -> SkyViewDir {
    let v_horizon = sqrt(max(view_height * view_height - PLANET_RADIUS * PLANET_RADIUS, 0.0));
    let cos_beta = v_horizon / view_height;
    let beta = acos(cos_beta);
    let zenith_horizon = 3.14159265 - beta;
    var out: SkyViewDir;
    if uv.y < 0.5 {
        var c = 1.0 - 2.0 * uv.y;
        c = 1.0 - c * c;
        out.view_zenith_cos = cos(zenith_horizon * c);
    } else {
        var c = uv.y * 2.0 - 1.0;
        c = c * c;
        out.view_zenith_cos = cos(zenith_horizon + beta * c);
    }
    let x = uv.x * uv.x;
    out.light_view_cos = -(x * 2.0 - 1.0);
    return out;
}

fn sky_view_uv(view_height: f32, view_zenith_cos: f32, light_view_cos: f32, hits_ground: bool) -> vec2<f32> {
    let v_horizon = sqrt(max(view_height * view_height - PLANET_RADIUS * PLANET_RADIUS, 0.0));
    let cos_beta = v_horizon / view_height;
    let beta = acos(cos_beta);
    let zenith_horizon = 3.14159265 - beta;
    var uv: vec2<f32>;
    if !hits_ground {
        var c = acos(clamp(view_zenith_cos, -1.0, 1.0)) / zenith_horizon;
        c = 1.0 - sqrt(max(1.0 - c, 0.0));
        uv.y = c * 0.5;
    } else {
        var c = (acos(clamp(view_zenith_cos, -1.0, 1.0)) - zenith_horizon) / beta;
        c = sqrt(max(c, 0.0));
        uv.y = c * 0.5 + 0.5;
    }
    uv.x = sqrt(clamp(-light_view_cos * 0.5 + 0.5, 0.0, 1.0));
    return uv;
}

// Raymarched single scattering plus the multiple scattering LUT, per unit
// sun illuminance. Returns luminance (rgb) and mean transmittance (a).
fn integrate_scattering(
    origin: vec3<f32>,
    dir: vec3<f32>,
    sun: vec3<f32>,
    t_limit: f32,
    steps: i32,
    transmittance_lut: texture_2d<f32>,
    multiscatter_lut: texture_2d<f32>,
    lut_sampler: sampler,
) -> vec4<f32> {
    let t_ground = ray_sphere(origin, dir, PLANET_RADIUS);
    let t_top = ray_sphere(origin, dir, ATMOSPHERE_RADIUS);
    var t_max = t_top;
    if t_ground > 0.0 {
        t_max = t_ground;
    }
    t_max = min(t_max, t_limit);
    if t_max <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let cos_theta = dot(dir, sun);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta);
    let dt = t_max / f32(steps);
    var throughput = vec3<f32>(1.0);
    var luminance = vec3<f32>(0.0);
    for (var s = 0; s < steps; s++) {
        let p = origin + dir * (f32(s) + 0.3) * dt;
        let h = length(p);
        let up = p / h;
        let m = sample_medium(p);
        let sun_cos = dot(up, sun);
        let t_sun = textureSampleLevel(transmittance_lut, lut_sampler, transmittance_uv(h, sun_cos), 0.0).rgb;
        // Earth shadow: the sun below this point's horizon.
        let shadow = select(1.0, 0.0, ray_sphere(p, sun, PLANET_RADIUS) > 0.0);
        let ms_uv = vec2<f32>(sun_cos * 0.5 + 0.5, (h - PLANET_RADIUS) / (ATMOSPHERE_RADIUS - PLANET_RADIUS));
        let ms = textureSampleLevel(multiscatter_lut, lut_sampler, ms_uv, 0.0).rgb;
        let scatter = t_sun * shadow * (m.rayleigh * phase_r + m.mie * phase_m) + ms * m.scattering;
        let step_t = exp(-m.extinction * dt);
        luminance += throughput * (scatter - scatter * step_t) / max(m.extinction, vec3<f32>(1e-6));
        throughput *= step_t;
    }
    return vec4<f32>(luminance, dot(throughput, vec3<f32>(1.0 / 3.0)));
}
