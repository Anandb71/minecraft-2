// Shared constants and helpers. Imported by nearly every shader.

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;
const INV_PI: f32 = 0.31830988618379;
const F32_MAX: f32 = 3.40282e38;

// PCG hash, Jarzynski and Olano 2020. Good avalanche for a single u32.
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg3d(v_in: vec3<u32>) -> vec3<u32> {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

fn hash_to_unit(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}

// Per-pixel, per-frame RNG stream.
struct Rng {
    state: u32,
}

fn rng_seed(pixel: vec2<u32>, frame: u32) -> Rng {
    return Rng(pcg(pixel.x + pcg(pixel.y + pcg(frame))));
}

fn rng_next(rng: ptr<function, Rng>) -> f32 {
    (*rng).state = pcg((*rng).state);
    return hash_to_unit((*rng).state);
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Octahedral normal encoding into [0,1]^2.
fn oct_wrap(v: vec2<f32>) -> vec2<f32> {
    return (1.0 - abs(v.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), v.xy >= vec2<f32>(0.0));
}

fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if n.z < 0.0 {
        p = oct_wrap(p);
    }
    return p * 0.5 + 0.5;
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    let f = e * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = clamp(-n.z, 0.0, 1.0);
    n = vec3<f32>(n.x + select(t, -t, n.x >= 0.0), n.y + select(t, -t, n.y >= 0.0), n.z);
    return normalize(n);
}
