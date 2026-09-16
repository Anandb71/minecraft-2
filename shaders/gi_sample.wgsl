// One indirect sample: where an indirect ray ended and what it brought back.
struct GiSample {
    // Camera-relative position, voxels from camera_voxel; for sky samples
    // the direction instead.
    position: vec3<f32>,
    normal: vec3<f32>,
    radiance: vec3<f32>,
    // 0 surface, 1 sky.
    sky: u32,
}
