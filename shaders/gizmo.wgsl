// World-space line gizmos (build preview, selection) drawn over the final
// image. Lines behind marched geometry are drawn faintly so a placement
// preview stays readable when partly buried.
#import "frame.wgsl"

@group(1) @binding(0) var vis_depth: texture_2d<f32>;

struct VsIn {
    @location(0) rel: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs(v: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = frame.view_proj * vec4<f32>(v.rel, 1.0);
    out.rel = v.rel;
    out.color = v.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.clip.xy / frame.output_size;
    let pixel = vec2<i32>(uv * frame.render_size);
    let scene = textureLoad(vis_depth, pixel, 0).r;
    let occluded = length(in.rel) > scene + 0.03;
    let alpha = select(in.color.a, in.color.a * 0.25, occluded);
    return vec4<f32>(in.color.rgb, alpha);
}
