// Shared shading helpers: surface reconstruction from the visibility buffer,
// secondary ray setup and sampling.
#import "brdf.wgsl"

struct Surface {
    // Camera-relative hit position, voxels.
    rel_voxels: vec3<f32>,
    // World voxel of the hit and its material.
    voxel: vec3<i32>,
    material: u32,
    normal: vec3<f32>,
    face: vec3<f32>,
    distance_m: f32,
    kind: u32,
}

fn load_surface(pixel: vec2<i32>, id: vec4<u32>, depth_m: f32, normal_oct: vec2<f32>) -> Surface {
    var s: Surface;
    s.kind = id.w >> 30u;
    s.material = id.w & 0xffffu;
    s.voxel = vec3<i32>(id.xyz);
    s.distance_m = depth_m;
    let dir = camera_ray_dir(vec2<f32>(pixel));
    s.rel_voxels = frame.camera_frac + dir * depth_m * VOXELS_PER_METRE;
    s.normal = oct_decode(normal_oct);
    s.face = id_face(id, s.normal);
    return s;
}

// Ray origin just outside the hit face so a secondary march never starts
// inside the voxel it left.
fn secondary_ray(s: Surface, dir: vec3<f32>, t_max_m: f32) -> Ray {
    var r: Ray;
    r.base = frame.camera_voxel;
    r.frac = s.rel_voxels + s.face * 0.02;
    r.dir = dir;
    r.t_min = 0.0;
    r.t_max = t_max_m * VOXELS_PER_METRE;
    // Secondary rays tolerate much coarser detail than primary ones.
    r.lod_scale = frame.pixel_angle * frame.lod_pixels * 8.0;
    r.feedback = false;
    r.coarse = false;
    // Sun, moon and sky reach through glass and into water.
    r.pass_kinds = PASS_CLEAR;
    return r;
}

// Uniformly distributed direction in a cone around `axis`.
fn sample_cone(axis: vec3<f32>, cos_max: f32, u: vec2<f32>) -> vec3<f32> {
    let cos_t = mix(1.0, cos_max, u.x);
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let phi = TAU * u.y;
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(axis.y) > 0.9);
    let t = normalize(cross(up, axis));
    let b = cross(axis, t);
    return normalize(t * cos(phi) * sin_t + b * sin(phi) * sin_t + axis * cos_t);
}

// Cosine-weighted hemisphere direction around `n`.
fn sample_cosine(n: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
    let r = sqrt(u.x);
    let phi = TAU * u.y;
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return normalize(t * r * cos(phi) + b * r * sin(phi) + n * sqrt(max(0.0, 1.0 - u.x)));
}

fn material_f0(m: Material) -> vec3<f32> {
    let dielectric = pow((m.ior - 1.0) / (m.ior + 1.0), 2.0);
    return mix(vec3<f32>(dielectric), m.albedo, m.metallic);
}

// World altitude of a camera-relative voxel position, metres.
fn altitude_m(rel_voxels: vec3<f32>) -> f32 {
    return (f32(frame.camera_voxel.y) + rel_voxels.y) / VOXELS_PER_METRE;
}
