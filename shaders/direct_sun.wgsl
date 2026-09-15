// Sun, moon and sky visibility: one shadow ray per pixel toward a random
// point on each light's disc (widened for soft contact shadows) and one
// cosine-weighted sky ray, accumulated over frames. History is reused only where the previous frame saw the exact same
// voxel, which voxel ids make an exact test instead of a depth heuristic.
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"
#import "lighting.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var prev_id: texture_2d<u32>;
@group(2) @binding(4) var history: texture_2d<f32>;
@group(2) @binding(5) var out_vis: texture_storage_2d<rgba16float, write>;

// Shadow penumbra widened beyond the true 0.27 degree sun disc.
const SOFTNESS: f32 = 4.0;
// Sky visibility rays stop here; farther occluders are left to indirect light.
const SKY_RANGE_M: f32 = 96.0;

fn light_visible(s: Surface, dir: vec3<f32>, range_m: f32) -> f32 {
    if dot(s.face, dir) <= 0.0 {
        return 0.0;
    }
    let hit = march(secondary_ray(s, dir, range_m));
    return select(0.0, 1.0, hit.kind == HIT_NONE);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.render_size);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let id = textureLoad(vis_id, pixel, 0);
    if (id.w >> 30u) == 0u {
        textureStore(out_vis, pixel, vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }
    let mv = textureLoad(vis_motion, pixel, 0);
    let s = load_surface(pixel, id, textureLoad(vis_depth, pixel, 0).r, mv.ba);
    var rng = rng_seed(gid.xy, frame.frame_index);
    let cos_max = cos(frame.sun_angular_radius * SOFTNESS);
    let sun = sample_cone(normalize(frame.sun_dir), cos_max, vec2<f32>(rng_next(&rng), rng_next(&rng)));
    var sun_vis = 0.0;
    if frame.sun_dir.y > -0.1 {
        sun_vis = light_visible(s, sun, 1500.0);
    }
    var moon_vis = 0.0;
    if frame.moon_dir.y > -0.1 && frame.sun_dir.y < 0.1 {
        let moon = sample_cone(normalize(frame.moon_dir), cos_max, vec2<f32>(rng_next(&rng), rng_next(&rng)));
        moon_vis = light_visible(s, moon, 1500.0);
    }
    // Cosine-weighted sky visibility: the fraction of the hemisphere's
    // irradiance that reaches the surface unoccluded.
    let sky_dir = sample_cosine(s.normal, vec2<f32>(rng_next(&rng), rng_next(&rng)));
    let sky_vis = light_visible(s, sky_dir, SKY_RANGE_M);

    // Exact-voxel temporal reuse.
    let uv = (vec2<f32>(pixel) + 0.5) / frame.render_size;
    let prev_pixel = vec2<i32>(floor((uv + mv.rg) * frame.render_size));
    var history_len = 0.0;
    var prev = vec3<f32>(0.0);
    if all(prev_pixel >= vec2<i32>(0)) && all(prev_pixel < vec2<i32>(size)) {
        let pid = textureLoad(prev_id, prev_pixel, 0);
        if all(pid.xyz == id.xyz) && (pid.w >> 30u) != 0u {
            let h = textureLoad(history, prev_pixel, 0);
            prev = h.rgb;
            history_len = h.a;
        }
    }
    let n = min(history_len + 1.0, 24.0);
    let alpha = 1.0 / n;
    let accumulated = mix(prev, vec3<f32>(sun_vis, moon_vis, sky_vis), alpha);
    textureStore(out_vis, pixel, vec4<f32>(accumulated, n));
}
