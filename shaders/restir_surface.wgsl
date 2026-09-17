// Visibility buffer lookups shared by the ReSTIR passes. Expects bindings
// named vis_id, vis_depth and vis_motion to be declared before import.

fn shading_at(pixel: vec2<i32>, id: vec4<u32>) -> Shading {
    let dir = camera_ray_dir(vec2<f32>(pixel));
    let depth = textureLoad(vis_depth, pixel, 0).r;
    let mat = materials[id.w & 0xffffu];
    var s: Shading;
    s.rel_voxels = frame.camera_frac + dir * depth * VOXELS_PER_METRE;
    s.normal = oct_decode(textureLoad(vis_motion, pixel, 0).ba);
    s.view = -dir;
    s.albedo = mat.albedo;
    s.f0 = mix(vec3<f32>(0.04), mat.albedo, mat.metallic);
    s.roughness = mat.roughness;
    s.metallic = mat.metallic;
    return s;
}

