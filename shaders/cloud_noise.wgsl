// Tiling cloud noise volumes (Schneider and Vos 2015), generated once.
//
//   shape   128^3 rgba8: Perlin-Worley, then Worley at three frequencies
//   detail   32^3 rgba8: Worley at three frequencies
//
// Every noise tiles with the texture: lattice coordinates wrap at the
// period, so clouds repeat seamlessly over the 128 m or 32 m of each map.
#import "common.wgsl"

@group(0) @binding(0) var out_shape: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(1) var out_detail: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> job: NoiseJob;

struct NoiseJob {
    // 0 shape, 1 detail.
    volume: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn hash3(p: vec3<i32>) -> vec3<f32> {
    let h = pcg3d(vec3<u32>(p) + vec3<u32>(0x9e3779b9u, 0x7f4a7c15u, 0x94d049bbu));
    return vec3<f32>(h) / 4294967296.0;
}

fn wrap(p: vec3<i32>, period: i32) -> vec3<i32> {
    return ((p % period) + period) % period;
}

// Tiling Worley noise: distance to the nearest feature point, 0 near points.
fn worley(p: vec3<f32>, period: i32) -> f32 {
    let cell = vec3<i32>(floor(p));
    let f = p - vec3<f32>(cell);
    var nearest = 1e9;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let o = vec3<i32>(x, y, z);
                let feature = vec3<f32>(o) + hash3(wrap(cell + o, period));
                let d = feature - f;
                nearest = min(nearest, dot(d, d));
            }
        }
    }
    return sqrt(nearest);
}

fn gradient(p: vec3<i32>) -> vec3<f32> {
    return normalize(hash3(p) * 2.0 - 1.0 + vec3<f32>(1e-4));
}

// Tiling gradient (Perlin) noise in [-1, 1] roughly.
fn perlin(p: vec3<f32>, period: i32) -> f32 {
    let cell = vec3<i32>(floor(p));
    let f = p - vec3<f32>(cell);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var sum = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let g = gradient(wrap(cell + o, period));
        let d = dot(g, f - vec3<f32>(o));
        let w = mix(1.0 - u, u, vec3<f32>(o));
        sum += d * w.x * w.y * w.z;
    }
    return sum;
}

// Worley fBm with three octaves, inverted so cells are billows (1 inside).
fn worley_fbm(p: vec3<f32>, period: i32) -> f32 {
    let a = 1.0 - worley(p, period);
    let b = 1.0 - worley(p * 2.0, period * 2);
    let c = 1.0 - worley(p * 4.0, period * 4);
    return clamp(a * 0.625 + b * 0.25 + c * 0.125, 0.0, 1.0);
}

fn perlin_fbm(p: vec3<f32>, period: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    var per = period;
    for (var i = 0; i < 5; i++) {
        sum += perlin(p * freq, per) * amp;
        freq *= 2.0;
        per *= 2;
        amp *= 0.5;
    }
    // Gradient noise sums cluster near zero; widen them to use the range.
    return clamp(sum * 1.1 + 0.5, 0.0, 1.0);
}

fn remap(v: f32, lo: f32, hi: f32, new_lo: f32, new_hi: f32) -> f32 {
    return new_lo + (v - lo) / (hi - lo) * (new_hi - new_lo);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if job.volume == 0u {
        if any(gid >= vec3<u32>(128u)) {
            return;
        }
        let uvw = (vec3<f32>(gid) + 0.5) / 128.0;
        // Perlin-Worley: Perlin fBm dilated by inverted Worley billows.
        let p = uvw * 4.0;
        let perlin_v = perlin_fbm(p, 4);
        let w = worley_fbm(p, 4);
        let perlin_worley = clamp(remap(perlin_v, w - 1.0, 1.0, 0.0, 1.0), 0.0, 1.0);
        let w1 = worley_fbm(uvw * 4.0, 4);
        let w2 = worley_fbm(uvw * 8.0, 8);
        let w3 = worley_fbm(uvw * 16.0, 16);
        textureStore(out_shape, gid, vec4<f32>(perlin_worley, w1, w2, w3));
    } else {
        if any(gid >= vec3<u32>(32u)) {
            return;
        }
        let uvw = (vec3<f32>(gid) + 0.5) / 32.0;
        let w1 = worley_fbm(uvw * 2.0, 2);
        let w2 = worley_fbm(uvw * 4.0, 4);
        let w3 = worley_fbm(uvw * 8.0, 8);
        textureStore(out_detail, gid, vec4<f32>(w1, w2, w3, 1.0));
    }
}
