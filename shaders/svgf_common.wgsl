// Shared by the SVGF passes (Schied et al. 2017). A signal may run at render
// resolution or at half resolution (shift 1): signal pixel p reads the
// visibility buffer at p << shift. Expects vis_id, vis_depth and vis_motion
// bindings and a `params: SvgfParams` uniform declared before import.

struct SvgfParams {
    // Signal to visibility-buffer resolution shift.
    shift: u32,
    // A-trous step for this iteration, in signal pixels.
    step: u32,
    _pad0: u32,
    _pad1: u32,
}

struct GSample {
    id: vec4<u32>,
    depth: f32,
    normal: vec3<f32>,
    // Camera-relative position, metres.
    position: vec3<f32>,
}

fn gbuffer(p: vec2<i32>) -> GSample {
    let g = p << vec2<u32>(params.shift);
    var s: GSample;
    s.id = textureLoad(vis_id, g, 0);
    s.depth = textureLoad(vis_depth, g, 0).r;
    s.normal = oct_decode(textureLoad(vis_motion, g, 0).ba);
    s.position = camera_ray_dir(vec2<f32>(g)) * s.depth;
    return s;
}

fn signal_size() -> vec2<i32> {
    return (vec2<i32>(frame.render_size) + (1 << params.shift) - 1) >> vec2<u32>(params.shift);
}

// Geometric edge-stopping weight between a centre sample and a neighbour
// `pixels` signal pixels away: distance from the centre's tangent plane in
// units of the pixel footprint (the paper's depth gradient term, exact for
// the planes voxels are made of) times the normal term max(0, n.n)^128.
fn geometry_weight(c: GSample, q: GSample, pixels: f32) -> f32 {
    if (q.id.w >> 30u) == 0u {
        return 0.0;
    }
    let footprint = c.depth * frame.pixel_angle * f32(1u << params.shift);
    let plane = abs(dot(c.normal, q.position - c.position));
    let wz = exp(-plane / (SIGMA_Z * footprint * max(pixels, 1.0) + 1e-3));
    let wn = pow(max(0.0, dot(c.normal, q.normal)), SIGMA_N);
    return wz * wn;
}

const SIGMA_Z: f32 = 1.0;
const SIGMA_N: f32 = 128.0;
const SIGMA_L: f32 = 4.0;
