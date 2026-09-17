// Primary visibility: one ray per pixel into the visibility buffer. Nothing
// is shaded here.
//
//   vis_id      rgba32uint  world voxel xyz, then material | face << 16 | kind << 30
//   vis_depth   r32float    hit distance along the ray, metres (1e9 for sky)
//   vis_motion  rgba16float screen-space motion uv (rg), octahedral normal (ba)
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"
#import "bodies.wgsl"

@group(2) @binding(0) var vis_id: texture_storage_2d<rgba32uint, write>;
@group(2) @binding(1) var vis_depth: texture_storage_2d<r32float, write>;
@group(2) @binding(2) var vis_motion: texture_storage_2d<rgba16float, write>;
@group(2) @binding(3) var beam_t: texture_2d<f32>;

const SKY_DEPTH: f32 = 1e9;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(frame.render_size);
    if any(id.xy >= size) {
        return;
    }
    let dir = camera_ray_dir(vec2<f32>(id.xy));
    var ray: Ray;
    ray.base = frame.camera_voxel;
    ray.frac = frame.camera_frac;
    ray.dir = dir;
    ray.t_max = 4096.0 * VOXELS_PER_METRE;
    ray.coarse = false;
    ray.t_min = 0.0;
    if frame.beam != 0u {
        let b = vec2<i32>(id.xy / 4u);
        let t0 = textureLoad(beam_t, b, 0).r;
        let t1 = textureLoad(beam_t, b + vec2<i32>(1, 0), 0).r;
        let t2 = textureLoad(beam_t, b + vec2<i32>(0, 1), 0).r;
        let t3 = textureLoad(beam_t, b + vec2<i32>(1, 1), 0).r;
        let tb = min(min(t0, t1), min(t2, t3));
        // Back off one brick plus the beam's own width at that distance.
        ray.t_min = max(tb - 8.0 - tb * frame.pixel_angle * 8.0, 0.0);
    }
    ray.lod_scale = frame.pixel_angle * frame.lod_pixels;
    ray.feedback = true;
    let hit = trace_scene(ray, 0.0);
    // Debug view 2 visualises traversal cost instead of depth.
    if frame.debug_mode == 2u {
        textureStore(vis_id, id.xy, vec4<u32>(0u, 0u, 0u, hit.kind << 30u));
        textureStore(vis_depth, id.xy, vec4<f32>(f32(hit.iterations)));
        textureStore(vis_motion, id.xy, vec4<f32>(0.0, 0.0, 0.5, 0.5));
        return;
    }

    if hit.kind == HIT_NONE {
        textureStore(vis_id, id.xy, vec4<u32>(0u));
        textureStore(vis_depth, id.xy, vec4<f32>(SKY_DEPTH));
        // Sky moves only with rotation: reproject the direction itself.
        let far = dir * 1e5;
        let motion = prev_uv(far + frame.prev_camera_delta) - current_uv(far);
        textureStore(vis_motion, id.xy, vec4<f32>(motion, 0.5, 0.5));
        return;
    }

    var normal = hit_normal(hit, dir);
    if hit.kind == HIT_LOD && hit.lod != 0u {
        // Aggregate normal of the filtered subtree, blended toward the face
        // by how spread its distribution is.
        let oct = vec2<f32>(f32((hit.lod >> 16u) & 63u), f32((hit.lod >> 22u) & 63u)) / 63.0;
        let f = oct * 2.0 - 1.0;
        var n = vec3<f32>(f.x, 1.0 - abs(f.x) - abs(f.y), f.y);
        let tt = max(-n.y, 0.0);
        n.x += select(tt, -tt, n.x >= 0.0);
        n.z += select(tt, -tt, n.z >= 0.0);
        let spread = f32(hit.lod >> 28u) / 15.0;
        normal = normalize(mix(normalize(n), normal, spread));
    }

    let t_m = hit.t / VOXELS_PER_METRE;
    let rel = dir * t_m;
    var prev_rel = rel;
    if hit.kind == HIT_BODY {
        // The point moved with its body since last frame.
        let now = frame.camera_frac + dir * hit.t;
        prev_rel = (body_prev_point(hit.lod, now) - frame.camera_frac) / VOXELS_PER_METRE;
    }
    let motion = prev_uv(prev_rel) - current_uv(rel);
    // The face the ray entered through; an aggregate LOD normal can point
    // the other way along this axis, so it is not used for the sign.
    textureStore(vis_id, id.xy, vec4<u32>(vec3<u32>(hit.voxel), hit_id_word(hit, dir)));
    textureStore(vis_depth, id.xy, vec4<f32>(t_m));
    textureStore(vis_motion, id.xy, vec4<f32>(motion, oct_encode(normal)));
}
