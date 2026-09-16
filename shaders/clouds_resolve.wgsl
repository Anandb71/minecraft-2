// Cloud temporal resolve: every half-resolution texel reprojects last
// frame's clouds by view direction (clouds are kilometres away, so camera
// translation barely moves them); texels traced this frame blend their new
// sample in, the rest keep history, and anything without history takes its
// block's fresh sample.
#import "common.wgsl"
#import "frame.wgsl"

@group(1) @binding(0) var trace: texture_2d<f32>;
@group(1) @binding(1) var history: texture_2d<f32>;
@group(1) @binding(2) var linear_clamp: sampler;
@group(1) @binding(3) var out_clouds: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var<uniform> clouds: CloudParams;
#import "clouds_params.wgsl"

const BLEND: f32 = 0.35;

fn cloud_slot(frame_index: u32) -> vec2<u32> {
    let k = (0x2130u >> ((frame_index % 4u) * 4u)) & 0xfu;
    return vec2<u32>(k & 1u, k >> 1u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = (vec2<u32>(frame.render_size) + 1u) / 2u;
    if any(gid.xy >= size) {
        return;
    }
    let block = gid.xy / 2u;
    let fresh = textureLoad(trace, block, 0);
    let dir = camera_ray_dir(vec2<f32>(gid.xy * 2u) + 0.5);
    let far = dir * 1e5;
    let prev = prev_uv(far + frame.prev_camera_delta);
    let valid = clouds.history_valid == 1u && all(prev >= vec2<f32>(0.0)) && all(prev <= vec2<f32>(1.0));
    let traced = all(gid.xy % 2u == cloud_slot(frame.frame_index));
    var result = fresh;
    if valid {
        // History texels cover two render pixels; map screen uv onto them.
        let history_uv = prev * frame.render_size * 0.5 / vec2<f32>(textureDimensions(history));
        let h = textureSampleLevel(history, linear_clamp, history_uv, 0.0);
        result = select(h, mix(h, fresh, BLEND), traced);
    }
    textureStore(out_clouds, gid.xy, result);
}
