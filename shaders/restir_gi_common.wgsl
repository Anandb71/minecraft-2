// ReSTIR GI samples and reservoirs. Samples are stored camera relative in
// metres (sky samples as directions), so last frame's samples move into this
// frame's space by adding the camera displacement.
//
//   pos  rgba32f  position (m) or direction, kind
//   rad  rgba32f  outgoing radiance, packed octahedral normal
//   res  rgba32f  w_sum, M, W, target value
#import "gi_sample.wgsl"

const GI_NONE: f32 = 0.0;
const GI_SURFACE: f32 = 1.0;
const GI_SKY: f32 = 2.0;
// Indirect rays stop here; farther light arrives through the sky term.
const GI_RANGE_M: f32 = 64.0;
// Debug view that disables reuse: one unbiased sample per pixel per frame.
const GI_REFERENCE_VIEW: u32 = 5u;
// Uniform hemisphere sampling.
const GI_SOURCE_PDF: f32 = 0.15915494;

struct PackedSample {
    pos: vec4<f32>,
    rad: vec4<f32>,
}

struct GiReservoir {
    // Sample.
    kind: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    radiance: vec3<f32>,
    // Reservoir.
    w_sum: f32,
    m: f32,
    w: f32,
    target_value: f32,
}

fn face_normal(id: vec4<u32>) -> vec3<f32> {
    let face_index = (id.w >> 16u) & 7u;
    let sign = select(-1.0, 1.0, (face_index & 1u) == 1u);
    return select(vec3<f32>(0.0), vec3<f32>(sign), axis_mask(face_index / 2u));
}

fn pack_normal(n: vec3<f32>) -> f32 {
    let e = vec2<u32>(clamp(oct_encode(n), vec2<f32>(0.0), vec2<f32>(1.0)) * 255.0 + 0.5);
    return f32(e.x | (e.y << 8u));
}

fn unpack_normal(v: f32) -> vec3<f32> {
    let bits = u32(v);
    return oct_decode(vec2<f32>(f32(bits & 255u), f32((bits >> 8u) & 255u)) / 255.0);
}

fn pack_sample(s: GiSample) -> PackedSample {
    var out: PackedSample;
    if s.sky == 1u {
        out.pos = vec4<f32>(s.position, GI_SKY);
    } else {
        out.pos = vec4<f32>((s.position - frame.camera_frac) / VOXELS_PER_METRE, GI_SURFACE);
    }
    out.rad = vec4<f32>(s.radiance, pack_normal(s.normal));
    return out;
}

fn empty_gi_reservoir() -> GiReservoir {
    var r: GiReservoir;
    r.kind = GI_NONE;
    return r;
}

fn read_reservoir(pos: vec4<f32>, rad: vec4<f32>, res: vec4<f32>) -> GiReservoir {
    var r: GiReservoir;
    r.kind = pos.w;
    r.position = pos.xyz;
    r.radiance = rad.rgb;
    r.normal = unpack_normal(rad.a);
    r.w_sum = res.x;
    r.m = res.y;
    r.w = res.z;
    r.target_value = res.w;
    return r;
}

// Direction from a camera-relative visible point (m) to a sample.
fn sample_direction(r: GiReservoir, visible: vec3<f32>) -> vec3<f32> {
    if r.kind == GI_SKY {
        return r.position;
    }
    return normalize(r.position - visible);
}

// Target function (Eq. 9 with albedo demodulated): radiance luminance times
// the cosine at the visible point.
fn gi_target(r: GiReservoir, visible: vec3<f32>, normal: vec3<f32>) -> f32 {
    if r.kind == GI_NONE {
        return 0.0;
    }
    return luminance(r.radiance) * max(dot(normal, sample_direction(r, visible)), 0.0);
}

// Weighted reservoir update: returns the reservoir with the candidate merged.
fn gi_update(r: GiReservoir, candidate: GiReservoir, weight: f32, u: f32) -> GiReservoir {
    var out = r;
    out.w_sum += weight;
    if weight > 0.0 && u * out.w_sum < weight {
        out.kind = candidate.kind;
        out.position = candidate.position;
        out.normal = candidate.normal;
        out.radiance = candidate.radiance;
    }
    return out;
}
