// Folds this frame's block samples into per-pixel visibility history. The
// traced pixel of each block accumulates its new sample; the others carry
// their reprojected history, and a pixel with no history borrows its block's
// sample until its own turn to trace (it is first in line next frame).
#import "common.wgsl"
#import "frame.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_motion: texture_2d<f32>;
@group(2) @binding(2) var prev_id: texture_2d<u32>;
@group(2) @binding(3) var history: texture_2d<f32>;
@group(2) @binding(4) var trace: texture_2d<f32>;
@group(2) @binding(5) var out_vis: texture_storage_2d<rgba16float, write>;
@group(2) @binding(6) var vis_depth: texture_2d<f32>;
#import "visibility_common.wgsl"

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.render_size);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let id = textureLoad(vis_id, pixel, 0);
    if (id.w >> 30u) == 0u {
        textureStore(out_vis, pixel, vec4<f32>(1.0));
        return;
    }
    let stride = max(frame.trace_stride, 1u);
    let block = gid.xy / stride;
    let local = gid.x % stride + (gid.y % stride) * stride;
    let sample = textureLoad(trace, vec2<i32>(block), 0);
    let h = reproject_history(pixel, id);
    if u32(sample.a + 0.5) == local {
        let n = min(h.a + 1.0, MAX_HISTORY);
        textureStore(out_vis, pixel, vec4<f32>(mix(h.rgb, sample.rgb, 1.0 / n), n));
    } else if h.a > 0.0 {
        textureStore(out_vis, pixel, h);
    } else {
        textureStore(out_vis, pixel, vec4<f32>(sample.rgb, 0.5));
    }
}
