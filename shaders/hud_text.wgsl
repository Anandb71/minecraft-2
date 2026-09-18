// Instanced HUD quads: 8x8 bitmap glyphs from a 16x8 cell atlas, solid
// rectangles (glyph 127 is a filled cell) and icons (glyph 256 and up) from
// a mipmapped RGBA atlas.

struct HudUniforms {
    screen: vec2<f32>,
    // Icon cells per atlas row, and one cell's size in atlas UV.
    icon_grid: vec2<f32>,
}

const ICON_BASE: u32 = 256u;

@group(0) @binding(0) var<uniform> hud: HudUniforms;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var icons: texture_2d<f32>;
@group(0) @binding(3) var icon_sampler: sampler;

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
    // Icon texel first, in uniform control flow, so the sampler's mip
    // selection has its derivatives; text and rectangles ignore it.
    let index = select(0u, in.glyph - ICON_BASE, in.glyph >= ICON_BASE);
    let per_row = max(u32(hud.icon_grid.x), 1u);
    let icon_cell = vec2<f32>(f32(index % per_row), f32(index / per_row));
    // Half a texel in from the cell edge so neighbours never bleed.
    let inset = 0.5 / f32(textureDimensions(icons).x);
    let icon_uv = (icon_cell + in.uv) * hud.icon_grid.y;
    let lo = icon_cell * hud.icon_grid.y + inset;
    let hi = (icon_cell + 1.0) * hud.icon_grid.y - inset;
    let texel = textureSample(icons, icon_sampler, clamp(icon_uv, lo, hi));
    if in.glyph >= ICON_BASE {
        if texel.a < 0.004 {
            discard;
        }
        return texel * in.color;
    }

    let cell = vec2<i32>(i32(in.glyph % 16u), i32(in.glyph / 16u));
    let local = vec2<i32>(clamp(in.uv * 8.0, vec2<f32>(0.0), vec2<f32>(7.999)));
    let coverage = textureLoad(atlas, cell * 8 + local, 0).r;
    if coverage < 0.5 {
        discard;
    }
    return in.color;
}
