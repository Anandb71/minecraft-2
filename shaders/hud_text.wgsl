// Instanced HUD quads: 8x8 bitmap glyphs from a 16x8 cell atlas, and solid
// rectangles (glyph 127 is a filled cell).

struct HudUniforms {
    screen: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> hud: HudUniforms;
@group(0) @binding(1) var atlas: texture_2d<f32>;

struct Instance {
    @location(0) rect: vec4<f32>,
    @location(1) glyph: u32,
    @location(2) color: u32,
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) glyph: u32,
    @location(2) color: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, inst: Instance) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let c = corners[vi];
    let px = inst.rect.xy + c * inst.rect.zw;
    var out: VsOut;
    out.pos = vec4<f32>(px.x / hud.screen.x * 2.0 - 1.0, 1.0 - px.y / hud.screen.y * 2.0, 0.0, 1.0);
    out.uv = c;
    out.glyph = inst.glyph;
    out.color = unpack4x8unorm(inst.color);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let cell = vec2<i32>(i32(in.glyph % 16u), i32(in.glyph / 16u));
    let local = vec2<i32>(clamp(in.uv * 8.0, vec2<f32>(0.0), vec2<f32>(7.999)));
    let coverage = textureLoad(atlas, cell * 8 + local, 0).r;
    if coverage < 0.5 {
        discard;
    }
    return in.color;
}
