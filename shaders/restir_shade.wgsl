// ReSTIR step 4: shade with the surviving sample, f(y) W, tracing its
// visibility once more.
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"
#import "bodies.wgsl"
#import "brdf.wgsl"
#import "restir_common.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var in_a: texture_2d<f32>;
@group(2) @binding(4) var in_b: texture_2d<f32>;
@group(2) @binding(5) var out_light: texture_storage_2d<rgba16float, write>;
#import "restir_surface.wgsl"

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.render_size);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let id = textureLoad(vis_id, pixel, 0);
    let r = unpack(textureLoad(in_a, pixel, 0), textureLoad(in_b, pixel, 0));
    if (id.w >> 30u) == 0u || r.light == 0xffffffffu || r.w <= 0.0 {
        textureStore(out_light, pixel, vec4<f32>(0.0));
        return;
    }
    let s = shading_at(pixel, id);
    let light = lights[r.light];
    let irradiance = light_irradiance(s, light, r.offset);
    let hit = trace_scene(shadow_ray_to(s, id_face(id, s.normal), sample_point(light, r.offset)), 0.0);
    let visible = select(0.0, 1.0, hit.kind == HIT_NONE);
    textureStore(out_light, pixel, vec4<f32>(irradiance * r.w * visible, 1.0));
}
