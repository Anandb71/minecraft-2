// Sun, moon and sky visibility rays, one pixel per stride x stride block per
// frame. Each block traces the pixel whose history is shortest (a
// disocclusion) or else the block's scheduled pixel, so the ray count is
// fixed by the stride and steady-state quality comes from accumulation.
// Dispatching over blocks, not pixels, keeps every SIMD lane of a workgroup
// tracing; skipping pixels inside a full-resolution dispatch made each group
// pay for its slowest ray and saved little.
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"
#import "bodies.wgsl"
#import "lighting.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var prev_id: texture_2d<u32>;
@group(2) @binding(4) var history: texture_2d<f32>;
@group(2) @binding(5) var out_trace: texture_storage_2d<rgba16float, write>;
#import "visibility_common.wgsl"

// Sun and moon rays see bodies; sky rays see only the world, since a
// fragment hides a small part of the sky and every hemisphere ray through
// the debris would pay for the body grid.
fn light_visible(s: Surface, dir: vec3<f32>, range_m: f32, bodies: bool) -> f32 {
    if dot(s.face, dir) <= 0.0 {
        return 0.0;
    }
    let ray = secondary_ray(s, dir, range_m);
    var hit: Hit;
    if bodies {
        hit = trace_scene(ray, 0.0);
    } else {
        hit = march(ray);
    }
    return select(0.0, 1.0, hit.kind == HIT_NONE);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let stride = max(frame.trace_stride, 1u);
    let size = vec2<u32>(frame.render_size);
    let blocks = (size + stride - 1u) / stride;
    if any(gid.xy >= blocks) {
        return;
    }
    let base = gid.xy * stride;
    let n = stride * stride;
    let first = interleave_slot(frame.frame_index, stride);
    // Scheduled pixel unless a geometry pixel in the block lacks history.
    var chosen = first;
    var shortest = MAX_HISTORY + 1.0;
    for (var k = 0u; k < n; k++) {
        let local = (first + k) % n;
        let p = base + vec2<u32>(local % stride, local / stride);
        if any(p >= size) {
            continue;
        }
        let id = textureLoad(vis_id, vec2<i32>(p), 0);
        if (id.w >> 30u) == 0u {
            continue;
        }
        let len = reproject_history(vec2<i32>(p), id).a;
        if len < MIN_REUSE && len < shortest {
            shortest = len;
            chosen = local;
        }
    }
    let pixel = vec2<i32>(base + vec2<u32>(chosen % stride, chosen / stride));
    let out_texel = vec2<i32>(gid.xy);
    let id = textureLoad(vis_id, pixel, 0);
    if any(vec2<u32>(pixel) >= size) || (id.w >> 30u) == 0u {
        textureStore(out_trace, out_texel, vec4<f32>(1.0, 1.0, 1.0, f32(chosen)));
        return;
    }
    let s = load_surface(pixel, id, textureLoad(vis_depth, pixel, 0).r, textureLoad(vis_motion, pixel, 0).ba);
    var rng = rng_seed(vec2<u32>(pixel), frame.frame_index);
    let cos_max = cos(frame.sun_angular_radius * SOFTNESS);
    var sun_vis = 0.0;
    if frame.sun_dir.y > -0.1 {
        let sun = sample_cone(normalize(frame.sun_dir), cos_max, vec2<f32>(rng_next(&rng), rng_next(&rng)));
        sun_vis = light_visible(s, sun, 1500.0, true);
    }
    var moon_vis = 0.0;
    if frame.moon_dir.y > -0.1 && frame.sun_dir.y < 0.1 {
        let moon = sample_cone(normalize(frame.moon_dir), cos_max, vec2<f32>(rng_next(&rng), rng_next(&rng)));
        moon_vis = light_visible(s, moon, 1500.0, true);
    }
    // Cosine-weighted sky visibility: the fraction of the hemisphere's
    // irradiance that reaches the surface unoccluded.
    let sky_dir = sample_cosine(s.normal, vec2<f32>(rng_next(&rng), rng_next(&rng)));
    let sky_vis = light_visible(s, sky_dir, SKY_RANGE_M, false);
    textureStore(out_trace, out_texel, vec4<f32>(sun_vis, moon_vis, sky_vis, f32(chosen)));
}
