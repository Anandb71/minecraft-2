// Shared by the visibility trace and resolve passes. Expects vis_id,
// vis_motion, prev_id and history bindings declared before import.

// Shadow penumbra widened beyond the true 0.27 degree sun disc.
const SOFTNESS: f32 = 4.0;
// Sky visibility rays stop here; farther occluders are left to indirect light.
const SKY_RANGE_M: f32 = 96.0;
// History length before a pixel waits its turn instead of tracing first.
const MIN_REUSE: f32 = 4.0;
const MAX_HISTORY: f32 = 24.0;

// Which cell of the stride x stride block traces this frame; for 2x2 the
// order visits diagonals first so each frame's samples stay spread out.
fn interleave_slot(frame_index: u32, stride: u32) -> u32 {
    if stride == 2u {
        // Nibbles 0, 3, 1, 2 packed low first.
        return (0x2130u >> ((frame_index % 4u) * 4u)) & 0xfu;
    }
    return (frame_index * 5u) % (stride * stride);
}

// Last frame's accumulated visibility (rgb) and sample count (a) for a pixel,
// or zero when the reprojected pixel saw a different voxel.
fn reproject_history(pixel: vec2<i32>, id: vec4<u32>) -> vec4<f32> {
    let size = vec2<i32>(frame.render_size);
    let mv = textureLoad(vis_motion, pixel, 0).rg;
    let uv = (vec2<f32>(pixel) + 0.5) / frame.render_size;
    let prev_pixel = vec2<i32>(floor((uv + mv) * frame.render_size));
    if any(prev_pixel < vec2<i32>(0)) || any(prev_pixel >= size) {
        return vec4<f32>(0.0);
    }
    let pid = textureLoad(prev_id, prev_pixel, 0);
    if any(pid.xyz != id.xyz) || (pid.w >> 30u) == 0u {
        return vec4<f32>(0.0);
    }
    return textureLoad(history, prev_pixel, 0);
}
