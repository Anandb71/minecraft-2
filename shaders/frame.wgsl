// Per-frame uniforms. Group 0, binding 0 in every pass that sees the world.
// Positions on the GPU are camera relative: the camera sits at an integer
// world voxel plus a fraction, and every world coordinate is formed as an
// integer difference first, so precision does not decay 8 km from origin.
#import "common.wgsl"

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_voxel: vec3<i32>,
    frame_index: u32,
    camera_frac: vec3<f32>,
    time: f32,
    prev_camera_delta: vec3<f32>,
    dt: f32,
    render_size: vec2<f32>,
    output_size: vec2<f32>,
    jitter: vec2<f32>,
    prev_jitter: vec2<f32>,
    sun_dir: vec3<f32>,
    exposure: f32,
    debug_mode: u32,
    // Initial ReSTIR candidates per pixel.
    restir_candidates: u32,
    pixel_angle: f32,
    lod_pixels: f32,
    beam: u32,
    // Visibility rays trace one pixel in trace_stride^2 once history settles.
    trace_stride: u32,
    // Active emitters; zero lets passes skip light sampling.
    light_count: u32,
    _pad0: u32,
    // Sun illuminance at the top of the atmosphere, linear RGB.
    sun_illuminance: vec3<f32>,
    sun_angular_radius: f32,
    moon_dir: vec3<f32>,
    moon_illuminance: f32,
    // 0 new moon, 0.5 full.
    moon_phase: f32,
    // Star field rotation about the celestial pole, and the observer
    // latitude that tilts the pole, radians.
    star_rotation: f32,
    latitude: f32,
    sky_flags: u32,
    // Rain and snow in the air, lightning flash, wetness of open surfaces.
    weather: vec4<f32>,
    // Wind at the ground (x, z m/s), cloud cover, unused.
    wind: vec4<f32>,
}

// sky_flags bit: moon-lit sky LUTs were rendered this frame.
const SKY_MOON: u32 = 1u;

@group(0) @binding(0) var<uniform> frame: Frame;

const VOXELS_PER_METRE: f32 = 16.0;

// Whether a previous frame's visibility id saw the same surface as `id`,
// seen at `depth_m`, when the two samples may be up to `pixels` render
// pixels apart (jitter alone moves them one; half-resolution signals two).
// Ids may differ by that footprint plus one voxel, on the same face, and
// must both be geometry.
fn same_surface(id: vec4<u32>, prev: vec4<u32>, depth_m: f32, pixels: f32) -> bool {
    if (prev.w >> 30u) == 0u || (id.w >> 30u) == 0u {
        return false;
    }
    if ((prev.w >> 16u) & 7u) != ((id.w >> 16u) & 7u) {
        return false;
    }
    // A body only matches itself (its tag), and never the world.
    if ((prev.w >> 19u) & 0x7ffu) != ((id.w >> 19u) & 0x7ffu) || ((prev.w >> 30u) == 2u) != ((id.w >> 30u) == 2u) {
        return false;
    }
    let footprint = depth_m * VOXELS_PER_METRE * frame.pixel_angle * pixels;
    let tolerance = vec3<i32>(i32(1.0 + ceil(footprint)));
    let d = abs(vec3<i32>(prev.xyz) - vec3<i32>(id.xyz));
    return all(d <= tolerance);
}

// Outward face of a visibility id: an axis for the world, the stored normal
// for bodies, whose faces turn with them.
fn id_face(id: vec4<u32>, normal: vec3<f32>) -> vec3<f32> {
    if (id.w >> 30u) == 2u {
        return normal;
    }
    let face_index = (id.w >> 16u) & 7u;
    let sign = select(-1.0, 1.0, (face_index & 1u) == 1u);
    return select(vec3<f32>(0.0), vec3<f32>(sign), axis_mask(face_index / 2u));
}

// Camera-relative direction (metres) through a pixel centre plus jitter.
fn camera_ray_dir(pixel: vec2<f32>) -> vec3<f32> {
    let uv = (pixel + 0.5 + frame.jitter) / frame.render_size;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let far = frame.inv_view_proj * vec4<f32>(ndc, 0.5, 1.0);
    return normalize(far.xyz / far.w);
}

// Screen uv of a camera-relative point (metres) last frame.
fn prev_uv(rel_metres: vec3<f32>) -> vec2<f32> {
    let clip = frame.prev_view_proj * vec4<f32>(rel_metres - frame.prev_camera_delta, 1.0);
    let ndc = clip.xy / clip.w;
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) - frame.prev_jitter / frame.render_size;
}

fn current_uv(rel_metres: vec3<f32>) -> vec2<f32> {
    let clip = frame.view_proj * vec4<f32>(rel_metres, 1.0);
    let ndc = clip.xy / clip.w;
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) - frame.jitter / frame.render_size;
}
