// Sky ambient cube lookup. Expects a `sky_ambient` binding (6x1, one texel
// per axis normal: +X -X +Y -Y +Z -Z).

// Sky irradiance for a normal, blended over the cube with squared normal
// components.
fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {
    let n2 = n * n;
    let px = textureLoad(sky_ambient, vec2<i32>(select(1, 0, n.x >= 0.0), 0), 0).rgb;
    let py = textureLoad(sky_ambient, vec2<i32>(select(3, 2, n.y >= 0.0), 0), 0).rgb;
    let pz = textureLoad(sky_ambient, vec2<i32>(select(5, 4, n.z >= 0.0), 0), 0).rgb;
    return px * n2.x + py * n2.y + pz * n2.z;
}
