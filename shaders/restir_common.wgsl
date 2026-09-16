// ReSTIR direct lighting over emissive voxel clusters (Bitterli et al. 2020).
//
// Each light is a cluster of emissive voxels (a torch flame, a brick of
// lava) treated as a sphere of its cluster radius. A sample is a light slot
// and a point on that sphere. Light slots are stable across frames, so a
// reservoir from last frame still names the same emitter.

struct Light {
    // World voxel of the cluster centre.
    voxel: vec3<i32>,
    // Emissive voxel count; 0 marks a free slot.
    count: u32,
    emission: vec3<f32>,
    radius_voxels: f32,
}

struct AliasEntry {
    probability: f32,
    alias_slot: u32,
    // Source pdf of choosing this slot.
    pdf: f32,
    _pad: f32,
}

@group(3) @binding(0) var<storage, read> lights: array<Light>;
@group(3) @binding(1) var<storage, read> alias_table: array<AliasEntry>;

struct Reservoir {
    light: u32,
    // Unit offset from the light centre to the sample point.
    offset: vec3<f32>,
    w_sum: f32,
    m: f32,
    w: f32,
    target_pdf: f32,
}

fn empty_reservoir() -> Reservoir {
    return Reservoir(0xffffffffu, vec3<f32>(0.0, 1.0, 0.0), 0.0, 0.0, 0.0, 0.0);
}

// Packed as two rgba32float texels: (light, w_sum, m, W) and (offset, p_hat).
fn pack_a(r: Reservoir) -> vec4<f32> {
    return vec4<f32>(f32(r.light), r.w_sum, r.m, r.w);
}

fn pack_b(r: Reservoir) -> vec4<f32> {
    return vec4<f32>(r.offset, r.target_pdf);
}

fn unpack(a: vec4<f32>, b: vec4<f32>) -> Reservoir {
    var r: Reservoir;
    r.light = select(u32(a.x), 0xffffffffu, a.x < 0.0 || a.x >= f32(arrayLength(&lights)));
    r.w_sum = a.y;
    r.m = a.z;
    r.w = a.w;
    r.offset = b.xyz;
    r.target_pdf = b.w;
    return r;
}

fn reservoir_update(r: ptr<function, Reservoir>, light: u32, offset: vec3<f32>, weight: f32, target_pdf: f32, u: f32) -> bool {
    (*r).w_sum += weight;
    (*r).m += 1.0;
    if weight > 0.0 && u * (*r).w_sum < weight {
        (*r).light = light;
        (*r).offset = offset;
        (*r).target_pdf = target_pdf;
        return true;
    }
    return false;
}

// Sample point of a reservoir, camera-relative voxels.
fn sample_point(light: Light, offset: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(light.voxel - frame.camera_voxel) + offset * light.radius_voxels;
}

struct Shading {
    rel_voxels: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    albedo: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
    metallic: f32,
}

// Unshadowed reflected luminance from a sample point: the target function.
fn light_contribution(s: Shading, light: Light, offset: vec3<f32>) -> vec3<f32> {
    let p = sample_point(light, offset);
    let to_light = p - s.rel_voxels;
    let d2_voxels = max(dot(to_light, to_light), 0.25);
    let l = to_light / sqrt(d2_voxels);
    let nl = dot(s.normal, l);
    if nl <= 0.0 {
        return vec3<f32>(0.0);
    }
    // The point faces the receiver on its sphere; cos at the light is the
    // angle between the sphere normal and the direction back.
    let cos_light = max(dot(offset, -l), 0.05);
    let d2_m = d2_voxels / (VOXELS_PER_METRE * VOXELS_PER_METRE);
    // Radiance times the emitter's projected area over distance squared.
    let radius_m = light.radius_voxels / VOXELS_PER_METRE;
    let area = 4.0 * PI * radius_m * radius_m;
    let geometry = cos_light * area / d2_m;
    let brdf = s.albedo * (1.0 - s.metallic) * INV_PI * nl + ggx_specular(s.normal, s.view, l, s.roughness, s.f0);
    return light.emission * brdf * geometry;
}

// Irradiance from a sample point: the demodulated signal the denoiser sees.
fn light_irradiance(s: Shading, light: Light, offset: vec3<f32>) -> vec3<f32> {
    let p = sample_point(light, offset);
    let to_light = p - s.rel_voxels;
    let d2_voxels = max(dot(to_light, to_light), 0.25);
    let l = to_light / sqrt(d2_voxels);
    let nl = dot(s.normal, l);
    if nl <= 0.0 {
        return vec3<f32>(0.0);
    }
    let cos_light = max(dot(offset, -l), 0.05);
    let d2_m = d2_voxels / (VOXELS_PER_METRE * VOXELS_PER_METRE);
    let radius_m = light.radius_voxels / VOXELS_PER_METRE;
    let area = 4.0 * PI * radius_m * radius_m;
    return light.emission * nl * cos_light * area / d2_m;
}

fn target_weight(c: vec3<f32>) -> f32 {
    return luminance(c);
}

// A uniformly random unit vector, used as the point on the light sphere.
fn random_offset(u: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - 2.0 * u.x;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = TAU * u.y;
    return vec3<f32>(r * cos(phi), z, r * sin(phi));
}

fn shadow_ray_to(s: Shading, face: vec3<f32>, point: vec3<f32>) -> Ray {
    let origin = s.rel_voxels + face * 0.02;
    let to = point - origin;
    let d = length(to);
    var r: Ray;
    r.base = frame.camera_voxel;
    r.frac = origin;
    r.dir = to / max(d, 1e-4);
    r.t_min = 0.0;
    // Stop short of the emitter's own voxels.
    r.t_max = max(d - 1.5, 0.0);
    r.lod_scale = frame.pixel_angle * frame.lod_pixels * 8.0;
    r.feedback = false;
    r.coarse = false;
    return r;
}
