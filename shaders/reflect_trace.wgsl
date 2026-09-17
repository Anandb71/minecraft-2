// Traced glossy reflections at GI resolution: one GGX visible-normal sample
// per pixel (Heitz 2018) for surfaces smoother than REFLECT_MAX_ROUGHNESS,
// traced through the world and shaded like indirect hits, sky included. The
// output is the estimator L F G2 / G1, which the VNDF pdf leaves after
// cancelling the rest of the BRDF; alpha marks pixels that were traced.
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"
#import "atmosphere.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var prev_id: texture_2d<u32>;
@group(2) @binding(4) var surface_prev: texture_2d<f32>;
@group(2) @binding(5) var sky_map: texture_2d<f32>;
@group(2) @binding(6) var sky_ambient: texture_2d<f32>;
@group(2) @binding(7) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(8) var sky_view_lut: texture_2d<f32>;
@group(2) @binding(9) var sky_view_moon: texture_2d<f32>;
@group(2) @binding(10) var lut_sampler: sampler;
@group(2) @binding(11) var out_reflection: texture_storage_2d<rgba16float, write>;
@group(2) @binding(12) var<uniform> params: SvgfParams;
#import "svgf_common.wgsl"
#import "gi_common.wgsl"

const REFLECT_MAX_ROUGHNESS: f32 = 0.5;
const REFLECT_RANGE_M: f32 = 256.0;

// Heitz 2018, Listing 1: a visible normal of the GGX distribution with
// roughness `alpha` for a view direction in the local frame (z = normal).
fn sample_vndf(ve: vec3<f32>, alpha: f32, u: vec2<f32>) -> vec3<f32> {
    let vh = normalize(vec3<f32>(alpha * ve.x, alpha * ve.y, ve.z));
    let lensq = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if lensq > 0.0 {
        t1 = vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(lensq);
    }
    let t2 = cross(vh, t1);
    let r = sqrt(u.x);
    let phi = TAU * u.y;
    let p1 = r * cos(phi);
    var p2 = r * sin(phi);
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;
    let nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}

// Smith masking for GGX.
fn smith_g1(cos_theta: f32, alpha: f32) -> f32 {
    let c2 = max(cos_theta * cos_theta, 1e-6);
    let tan2 = (1.0 - c2) / c2;
    return 2.0 / (1.0 + sqrt(1.0 + alpha * alpha * tan2));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = signal_size();
    let p = vec2<i32>(gid.xy);
    if any(p >= size) {
        return;
    }
    let c = gbuffer(p);
    if (c.id.w >> 30u) == 0u {
        textureStore(out_reflection, p, vec4<f32>(0.0));
        return;
    }
    let mat = materials[c.id.w & 0xffffu];
    if mat.roughness > REFLECT_MAX_ROUGHNESS {
        textureStore(out_reflection, p, vec4<f32>(0.0));
        return;
    }
    let n = c.normal;
    let v = -normalize(c.position);
    // Local frame around the normal.
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let tx = normalize(cross(up, n));
    let ty = cross(n, tx);
    let ve = vec3<f32>(dot(v, tx), dot(v, ty), dot(v, n));
    if ve.z <= 0.0 {
        textureStore(out_reflection, p, vec4<f32>(0.0));
        return;
    }
    let alpha = max(mat.roughness * mat.roughness, 0.002);
    var rng = rng_seed(vec2<u32>(p), frame.frame_index ^ 0x3ef1u);
    let he = sample_vndf(ve, alpha, vec2<f32>(rng_next(&rng), rng_next(&rng)));
    let h = tx * he.x + ty * he.y + n * he.z;
    let l = reflect(-v, h);
    let nl = dot(n, l);
    if nl <= 0.0 {
        textureStore(out_reflection, p, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let origin = frame.camera_frac + c.position * VOXELS_PER_METRE + id_face(c.id, c.normal) * 0.05;
    let s = trace_gi(origin, l, 0.0, REFLECT_RANGE_M, true);
    let f0 = mix(vec3<f32>(pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0)), mat.albedo, mat.metallic);
    let vh = max(dot(v, h), 0.0);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    // Separable Smith: G2 / G1(V) = G1(L).
    let weight = fresnel * smith_g1(nl, alpha);
    textureStore(out_reflection, p, vec4<f32>(s.radiance * weight, 1.0));
}
