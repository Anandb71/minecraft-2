// Froxel volume layout for local fog (Hillaire 2015): one froxel per 8x8
// render pixels, 64 slices spaced exponentially from 0.5 m to 192 m.

struct FogParams {
    // Extinction per metre at the fog height.
    density: f32,
    // Altitude below which fog is at full density, metres.
    height_m: f32,
    // Density halves every this many metres above `height_m`.
    falloff_m: f32,
    // Henyey-Greenstein anisotropy.
    anisotropy: f32,
    // Scattering over extinction.
    albedo: f32,
    // Froxel grid size.
    width: u32,
    height: u32,
    depth: u32,
    // 0 on the first frame after (re)allocation.
    history_valid: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

const FOG_NEAR_M: f32 = 0.5;
const FOG_FAR_M: f32 = 192.0;

// Distance of a (fractional) slice coordinate in [0, 1].
fn fog_slice_distance(w: f32) -> f32 {
    return FOG_NEAR_M * pow(FOG_FAR_M / FOG_NEAR_M, w);
}

// Slice coordinate in [0, 1] of a distance.
fn fog_slice_coord(d: f32) -> f32 {
    return clamp(log(max(d, FOG_NEAR_M) / FOG_NEAR_M) / log(FOG_FAR_M / FOG_NEAR_M), 0.0, 1.0);
}

// Unjittered view direction through a screen uv.
fn fog_direction(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let far = frame.inv_view_proj * vec4<f32>(ndc, 0.5, 1.0);
    return normalize(far.xyz / far.w);
}

// Fog extinction at a world altitude.
fn fog_density(altitude_m: f32) -> f32 {
    let above = max(altitude_m - fog.height_m, 0.0);
    return fog.density * pow(0.5, above / max(fog.falloff_m, 1.0));
}
