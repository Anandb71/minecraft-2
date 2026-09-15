// Step 3 debug shading of the visibility buffer: material albedo lit by a
// fixed sun and hemisphere sky, no shadows. Also visualises iteration
// counts and LOD hits via frame.debug_mode.
#import "common.wgsl"
#import "frame.wgsl"
#import "voxel_data.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var out_hdr: texture_storage_2d<rgba16float, write>;

fn sky(dir: vec3<f32>) -> vec3<f32> {
    let up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3<f32>(0.9, 0.85, 0.8), vec3<f32>(0.25, 0.45, 0.9), up) * 2.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(frame.render_size);
    if any(id.xy >= size) {
        return;
    }
    let pid = textureLoad(vis_id, id.xy, 0);
    let dir = camera_ray_dir(vec2<f32>(id.xy));
    let kind = pid.w >> 30u;
    if kind == 0u {
        textureStore(out_hdr, id.xy, vec4<f32>(sky(dir), 1.0));
        return;
    }
    let mat = materials[pid.w & 0xffffu];
    let n = oct_decode(textureLoad(vis_motion, id.xy, 0).ba);
    let sun = normalize(frame.sun_dir);
    let diffuse = max(dot(n, sun), 0.0) * vec3<f32>(3.0, 2.8, 2.5);
    let ambient = sky(n) * 0.25;
    // Per-voxel value jitter so voxel resolution is visible in debug views.
    let h = hash_to_unit(pcg(pid.x * 73856093u ^ pid.y * 19349663u ^ pid.z * 83492791u));
    var color = mat.albedo * (0.9 + 0.2 * h) * (diffuse + ambient) + mat.emission * 0.001;
    if frame.debug_mode == 1u {
        color = select(vec3<f32>(0.2, 0.6, 0.2), vec3<f32>(0.9, 0.2, 0.9), kind == 3u);
    }
    let depth = textureLoad(vis_depth, id.xy, 0).r;
    let fog = 1.0 - exp(-depth * 0.0015);
    color = mix(color, sky(dir), fog);
    textureStore(out_hdr, id.xy, vec4<f32>(color, 1.0));
}
