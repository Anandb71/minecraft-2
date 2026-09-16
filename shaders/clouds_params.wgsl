// Cloud layer parameters, written by clouds.rs every frame.

struct CloudParams {
    // Fraction of the sky covered, 0..1.
    coverage: f32,
    // 0 stratus to 1 cumulus.
    cloud_type: f32,
    // Wind displacement so far, metres (xz).
    wind: vec2<f32>,
    // Layer bottom and top above sea level, metres.
    base_m: f32,
    top_m: f32,
    // Extinction per metre at density 1.
    extinction: f32,
    // Ray march samples through the layer, and toward the light.
    steps: u32,
    light_steps: u32,
    // Extra absorption for rain-laden clouds, 0..1.
    precipitation: f32,
    // Cloud shadow map: metres per texel; the map is addressed toroidally
    // by world cell so it never needs shifting.
    shadow_texel_m: f32,
    // 0 on the first frame after (re)allocation: no history to reuse.
    history_valid: u32,
    // Camera position, world metres.
    camera_world: vec3<f32>,
    _pad0: f32,
}
