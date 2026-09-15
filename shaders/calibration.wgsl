// Harness calibration image: an exposure ramp spanning -8..+8 EV, primary and
// emitter-like saturated bars at rising intensity, and a moving bar for frame
// pacing. Used by golden tests to pin the display transform and the frame
// graph plumbing before any world exists.
#import "common.wgsl"

struct CalibrationUniforms {
    time: f32,
    frame: u32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> cal: CalibrationUniforms;
@group(0) @binding(1) var out_hdr: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out_hdr);
    if any(id.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    var color: vec3<f32>;
    if uv.y < 0.25 {
        // Neutral ramp from 2^-8 to 2^8.
        color = vec3<f32>(pow(2.0, mix(-8.0, 8.0, uv.x)));
    } else if uv.y < 0.75 {
        let band = u32((uv.y - 0.25) * 12.0);
        let hues = array<vec3<f32>, 6>(
            vec3<f32>(1.0, 0.05, 0.02), vec3<f32>(0.05, 1.0, 0.05), vec3<f32>(0.03, 0.08, 1.0),
            vec3<f32>(1.0, 0.45, 0.08), vec3<f32>(1.0, 0.85, 0.5), vec3<f32>(0.3, 0.6, 1.0),
        );
        color = hues[band] * pow(2.0, mix(-4.0, 10.0, uv.x));
    } else {
        // Checker with a bar sweeping once every four seconds.
        let cell = vec2<u32>(uv * vec2<f32>(32.0, 8.0));
        let checker = f32((cell.x + cell.y) & 1u) * 0.1 + 0.05;
        let bar = step(abs(fract(cal.time * 0.25) - uv.x), 0.01);
        color = vec3<f32>(checker) + vec3<f32>(bar * 4.0);
    }
    textureStore(out_hdr, id.xy, vec4<f32>(color, 1.0));
}
