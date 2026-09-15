// Automatic exposure: centre-weighted mean log luminance of the scene,
// converted to EV100 (Lagarde and de Rousiers) and adapted over time in log
// space. Writes the linear exposure multiplier into a 1x1 texture.
#import "common.wgsl"
#import "frame.wgsl"

@group(1) @binding(0) var scene: texture_2d<f32>;
@group(1) @binding(1) var previous: texture_2d<f32>;
@group(1) @binding(2) var out_exposure: texture_storage_2d<rgba16float, write>;

const MIN_EV: f32 = -6.0;
const MAX_EV: f32 = 17.0;
const ADAPT_UP: f32 = 2.5;
const ADAPT_DOWN: f32 = 1.2;

@compute @workgroup_size(1)
fn main() {
    let size = vec2<f32>(textureDimensions(scene));
    let grid = vec2<i32>(64, 36);
    var sum = 0.0;
    var weight = 0.0;
    for (var y = 0; y < grid.y; y++) {
        for (var x = 0; x < grid.x; x++) {
            let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / vec2<f32>(grid);
            let c = textureLoad(scene, vec2<i32>(uv * size), 0).rgb;
            let d = length(uv - 0.5);
            let w = 1.0 - smoothstep(0.15, 0.7, d) * 0.8;
            sum += log2(max(luminance(c), 1e-4)) * w;
            weight += w;
        }
    }
    let avg_log = sum / weight;
    // EV100 = log2(L * S / K) with S = 100, K = 12.5.
    let target_ev = clamp(avg_log + log2(100.0 / 12.5), MIN_EV, MAX_EV);
    let prev = textureLoad(previous, vec2<i32>(0), 0);
    var ev = target_ev;
    // Alpha channel marks a valid history (first frame snaps).
    if prev.a > 0.5 {
        let speed = select(ADAPT_DOWN, ADAPT_UP, target_ev > prev.r);
        ev = mix(prev.r, target_ev, 1.0 - exp(-frame.dt * speed));
    }
    let exposure = 1.0 / (1.2 * pow(2.0, ev));
    textureStore(out_exposure, vec2<i32>(0), vec4<f32>(ev, exposure, target_ev, 1.0));
}
