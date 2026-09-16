// Radiance cascades, interval tracing: one ray per atlas texel of one
// cascade, from its probe over the cascade's distance interval. Directions
// into the probe's own surface are opaque and dark. Sky light is not
// gathered here (composition traces sky visibility), so the last cascade
// closes unobstructed intervals with darkness.
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
@group(2) @binding(11) var out_atlas: texture_storage_2d<rgba16float, write>;
@group(2) @binding(12) var<uniform> rc: RcParams;
#import "gi_common.wgsl"
#import "rc_common.wgsl"

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probes = rc_probe_count(rc.spacing);
    let probe = gid.xy / rc.dirs;
    if any(probe >= probes) {
        return;
    }
    let pixel = rc_probe_pixel(probe, rc.spacing);
    let id = textureLoad(vis_id, pixel, 0);
    if (id.w >> 30u) == 0u {
        // No surface under this probe: transparent, so merging passes the
        // coarser cascade through and gathering gives it no weight.
        textureStore(out_atlas, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let depth = textureLoad(vis_depth, pixel, 0).r;
    let normal = oct_decode(textureLoad(vis_motion, pixel, 0).ba);
    let dir = rc_direction(gid.xy % rc.dirs, rc.dirs);
    if dot(dir, normal) <= 0.0 {
        textureStore(out_atlas, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    let face_index = (id.w >> 16u) & 7u;
    let sign = select(-1.0, 1.0, (face_index & 1u) == 1u);
    let face = select(vec3<f32>(0.0), vec3<f32>(sign), axis_mask(face_index / 2u));
    let origin = frame.camera_frac + camera_ray_dir(vec2<f32>(pixel)) * depth * VOXELS_PER_METRE + face * 0.05;
    let last = rc.cascade + 1u == rc.cascades;
    let s = trace_gi(origin, dir, rc.t_min_m, rc.t_max_m, false);
    if s.sky == 1u {
        textureStore(out_atlas, gid.xy, vec4<f32>(0.0, 0.0, 0.0, select(1.0, 0.0, last)));
    } else {
        textureStore(out_atlas, gid.xy, vec4<f32>(s.radiance, 0.0));
    }
}
