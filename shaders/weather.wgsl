// Weather on surfaces: how wet a point open to the sky is, where water
// stands in puddles, and what that does to roughness and colour. Needs
// frame.wgsl (frame.weather.w is how wet open ground is).

fn weather_hash(p: vec2<i32>) -> f32 {
    var h = u32(p.x) * 0x8da6b343u ^ u32(p.y) * 0xd8163841u;
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    return f32(h >> 8u) / 16777216.0;
}

// Smooth value noise in the plane, 0..1.
fn weather_noise(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = weather_hash(i);
    let b = weather_hash(i + vec2<i32>(1, 0));
    let c = weather_hash(i + vec2<i32>(0, 1));
    let d = weather_hash(i + vec2<i32>(1, 1));
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

// How wet a surface is, 0..1: the weather's wetness where the sky reaches
// it (`sky` 0..1), tops more than walls.
fn wetness(normal: vec3<f32>, sky: f32) -> f32 {
    let up = smoothstep(0.2, 0.9, normal.y);
    return frame.weather.w * smoothstep(0.3, 0.8, sky) * (0.3 + 0.7 * up);
}

// Standing water on level ground, 0..1: hollows that fill as it gets wet.
fn puddle(world_m: vec3<f32>, normal: vec3<f32>, wet: f32) -> f32 {
    if normal.y < 0.95 || wet <= 0.0 {
        return 0.0;
    }
    let n = weather_noise(world_m.xz * 0.35) * 0.7 + weather_noise(world_m.xz * 1.7) * 0.3;
    let fill = 1.0 - 0.6 * wet;
    return smoothstep(fill, fill + 0.04, n) * wet;
}

// Roughness once wet: damp surfaces smoother, puddles a mirror.
fn wet_roughness(r: f32, wet: f32, pud: f32) -> f32 {
    return mix(r * mix(1.0, 0.5, wet), 0.03, pud);
}

// Colour once wet: porous things darken as water fills them.
fn wet_albedo(albedo: vec3<f32>, metallic: f32, wet: f32, pud: f32) -> vec3<f32> {
    return albedo * mix(1.0, 0.55, wet * (1.0 - metallic)) * (1.0 - 0.25 * pud);
}
