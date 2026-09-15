// Beam prepass (after Laine and Karras's beam optimisation). One coarse ray
// through every corner of a 4x4 pixel block marches with a cone-sized LOD
// cutoff and stops at the first occupied brick cell or coarser node. Primary
// rays start at the minimum over their block's four corners, less a margin,
// instead of descending through empty space from the world root.
#import "frame.wgsl"
#import "march.wgsl"

@group(2) @binding(0) var beam_t: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(beam_t);
    if any(id.xy >= size) {
        return;
    }
    // Corner of the 4x4 block: pixel coordinate id * 4, less the half pixel
    // camera_ray_dir adds for pixel centres.
    let dir = camera_ray_dir(vec2<f32>(id.xy * 4u) - 0.5);
    var ray: Ray;
    ray.base = frame.camera_voxel;
    ray.frac = frame.camera_frac;
    ray.dir = dir;
    ray.t_min = 0.0;
    ray.t_max = 4096.0 * VOXELS_PER_METRE;
    // A node is only trusted as a stopping point when it is at least twice
    // as wide as the beam (four pixels) at that distance.
    ray.lod_scale = frame.pixel_angle * 8.0;
    ray.feedback = false;
    ray.coarse = true;
    let hit = march(ray);
    textureStore(beam_t, id.xy, vec4<f32>(hit.t));
}
