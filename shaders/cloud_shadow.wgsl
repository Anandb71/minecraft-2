// Cloud shadow lookup. Expects a `cloud_shadow_map` (r of an rgba16float
// 256^2 texture) and the `clouds: CloudParams` uniform (clouds_params.wgsl).

const SHADOW_MAP_SIZE: i32 = 256;

// Transmittance of the cloud layer toward the sun above a world point.
// The map stores, per world cell of the layer's base plane, what a ray
// leaving that plane toward the sun keeps; a point below looks up where its
// sun ray crosses the plane.
fn cloud_shadow(world: vec3<f32>, sun: vec3<f32>) -> f32 {
    if sun.y <= 0.01 {
        return 1.0;
    }
    let to_base = (clouds.base_m - world.y) / sun.y;
    let q = world.xz + sun.xz * max(to_base, 0.0);
    // Bilinear between cell centres, wrapping like the map's addressing.
    let g = q / clouds.shadow_texel_m - 0.5;
    let base = vec2<i32>(floor(g));
    let f = g - vec2<f32>(base);
    var sum = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let texel = (((base + o) % SHADOW_MAP_SIZE) + SHADOW_MAP_SIZE) % SHADOW_MAP_SIZE;
        let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        sum += textureLoad(cloud_shadow_map, texel, 0).r * w;
    }
    return sum;
}
