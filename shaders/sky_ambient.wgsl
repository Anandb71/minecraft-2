// Sky ambient cube, per frame: irradiance from the sky (sun- and moon-lit
// LUTs plus airglow) onto the six axis-aligned normals, integrated with 144
// cosine-distributed directions. Composition and indirect hits read it in
// place of sampling the sky-view LUT per pixel.
#import "common.wgsl"
#import "frame.wgsl"
#import "atmosphere.wgsl"

@group(1) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(1) @binding(1) var sky_view_lut: texture_2d<f32>;
@group(1) @binding(2) var sky_view_moon: texture_2d<f32>;
@group(1) @binding(3) var lut_sampler: sampler;
@group(1) @binding(4) var out_ambient: texture_storage_2d<rgba16float, write>;
#import "sky_common.wgsl"

@compute @workgroup_size(6, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= 6u {
        return;
    }
    let axis = gid.x / 2u;
    let sign = select(1.0, -1.0, (gid.x & 1u) == 1u);
    let n = select(vec3<f32>(0.0), vec3<f32>(sign), axis_mask(axis));
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    let rows = 12;
    var sum = vec3<f32>(0.0);
    for (var i = 0; i < rows; i++) {
        for (var j = 0; j < rows; j++) {
            // Stratified cosine-weighted directions.
            let u = (vec2<f32>(f32(i), f32(j)) + 0.5) / f32(rows);
            let r = sqrt(u.x);
            let phi = TAU * u.y;
            let d = t * r * cos(phi) + b * r * sin(phi) + n * sqrt(max(0.0, 1.0 - u.x));
            // Only the sky: light from below the horizon arrives by bouncing
            // off the ground, which indirect light traces.
            if d.y > 0.0 {
                sum += sky_radiance(normalize(d));
            }
        }
    }
    // E = pi * mean radiance over cosine-distributed directions.
    let irradiance = (sum / f32(rows * rows) + NIGHT_SKY) * PI;
    textureStore(out_ambient, vec2<u32>(gid.x, 0u), vec4<f32>(irradiance, 1.0));
}
