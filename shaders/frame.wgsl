// Per-frame uniforms. Group 0, binding 0 in every pass that sees the world.
// Positions on the GPU are camera relative: the camera sits at an integer
// world voxel plus a fraction, and every world coordinate is formed as an
// integer difference first, so precision does not decay 8 km from origin.

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
    quality: u32,
    pixel_angle: f32,
    lod_pixels: f32,
    beam: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> frame: Frame;

const VOXELS_PER_METRE: f32 = 16.0;

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
