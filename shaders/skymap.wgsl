// Sky map lookups: the highest solid 0.5 m cell per column around the camera,
// toroidally addressed (see skymap.rs). Expects a `sky_map` r32float binding.

const SKY_MAP_SIZE: i32 = 1024;
const SKY_MAP_UNKNOWN: f32 = -1.0e3;

// Top of the highest solid cell above a world voxel column, metres.
fn sky_map_height(world_voxel_xz: vec2<i32>) -> f32 {
    let cell = vec2<i32>(floor(vec2<f32>(world_voxel_xz) / 8.0));
    let texel = ((cell % SKY_MAP_SIZE) + SKY_MAP_SIZE) % SKY_MAP_SIZE;
    return textureLoad(sky_map, texel, 0).r;
}

// Whether the sky is open above a surface point (world voxels) with normal
// `n`: the column just outside the surface has nothing above the point.
// Soft over half a metre so cliff edges fade instead of stepping.
fn sky_map_sky_visibility(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let q = p + n * 8.0;
    let h = sky_map_height(vec2<i32>(floor(q.xz)));
    if h < SKY_MAP_UNKNOWN {
        return 1.0;
    }
    return smoothstep(-0.5, 0.25, q.y / 16.0 - h);
}

// Whether the sun reaches a surface point: a 2D march toward the light over
// the height field with geometrically growing steps out to 220 m.
fn sky_map_light_visibility(p: vec3<f32>, n: vec3<f32>, light: vec3<f32>) -> f32 {
    if light.y <= 0.0 || dot(n, light) <= 0.0 {
        return 0.0;
    }
    let horizontal = length(light.xz);
    if horizontal < 1e-4 {
        // Overhead: only the column itself can shade.
        return sky_map_sky_visibility(p, vec3<f32>(0.0, 1.0, 0.0));
    }
    let dir = light.xz / horizontal;
    let rise = light.y / horizontal;
    let start = p + n * 4.0;
    var s = 0.75;
    for (var k = 0; k < 18; k++) {
        let xz = start.xz / 16.0 + dir * s;
        let h = sky_map_height(vec2<i32>(floor(xz * 16.0)));
        if h > start.y / 16.0 + rise * s {
            return 0.0;
        }
        s *= 1.4;
        if s > 220.0 {
            break;
        }
    }
    return 1.0;
}
