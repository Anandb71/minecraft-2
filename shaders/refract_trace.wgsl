// What is seen through clear surfaces: glass, ice and water. For each pixel
// whose surface is clear, a ray carries on past it, passing through that
// material, and the surface behind is shaded like an indirect hit (sky
// included). Into water the ray bends by Snell's law; through glass it goes
// straight on, since a pane's two faces undo each other's bend. Water
// absorbs along the way, red first (Beer-Lambert), and scatters sky and sun
// light back, so depth reads blue-green; glass tints a little. Full
// resolution: the view through a window must stay sharp. Alpha marks the
// traced pixels.
#import "common.wgsl"
#import "frame.wgsl"
#import "march.wgsl"
#import "atmosphere.wgsl"

@group(2) @binding(0) var vis_id: texture_2d<u32>;
@group(2) @binding(1) var vis_depth: texture_2d<f32>;
@group(2) @binding(2) var vis_motion: texture_2d<f32>;
@group(2) @binding(3) var prev_id: texture_2d<u32>;
@group(2) @binding(4) var surface_prev: texture_2d<f32>;
@group(2) @binding(5) var sky_map: texture_2d<f32>;
@group(2) @binding(6) var sky_ambient: texture_2d<f32>;
@group(2) @binding(7) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(8) var sky_view_lut: texture_2d<f32>;
@group(2) @binding(9) var sky_view_moon: texture_2d<f32>;
@group(2) @binding(10) var lut_sampler: sampler;
@group(2) @binding(11) var out_refraction: texture_storage_2d<rgba16float, write>;
@group(2) @binding(12) var<uniform> params: SvgfParams;
#import "svgf_common.wgsl"
#import "gi_common.wgsl"
#import "surface_detail.wgsl"

const REFRACT_RANGE_M: f32 = 160.0;
// Absorption of clear water per metre (Pope and Fry 1997, rounded): red
// goes within metres, blue carries tens.
const WATER_ABSORPTION: vec3<f32> = vec3<f32>(0.45, 0.07, 0.025);
// What the water itself sends back, per unit of light falling on it: a
// little blue-green scattering by the water and what is suspended in it.
const WATER_SCATTER: vec3<f32> = vec3<f32>(0.006, 0.03, 0.035);
const GLASS_TINT: vec3<f32> = vec3<f32>(0.9, 0.95, 0.93);
const SHADOW_RANGE_M: f32 = 128.0;

// Light leaving a surface seen through glass or water toward the eye. The
// sky map's height field would put everything under a pane or a pool in
// shade, so the sun's visibility is traced (through clear voxels); the
// surface wears its own pattern.
fn shade_behind(at: vec3<f32>, n: vec3<f32>, face: vec3<f32>, voxel: vec3<i32>, word: u32) -> vec3<f32> {
    let m = word & 0xffffu;
    let mat = materials[m];
    let world = vec3<f32>(frame.camera_voxel) + at;
    let world_m = world / VOXELS_PER_METRE;
    let detail = surface_detail(m, mat, voxel, n, world_m, frame.time);
    let sun = normalize(frame.sun_dir);
    var e = vec3<f32>(0.0);
    let nl = dot(detail.normal, sun);
    if nl > 0.0 && dot(face, sun) > 0.0 {
        var shadow = gi_ray(at + face * 0.05, sun, 0.0, SHADOW_RANGE_M);
        shadow.pass_kinds = PASS_CLEAR;
        if march(shadow).kind == HIT_NONE {
            e += frame.sun_illuminance * light_transmittance(world_m.y / 1000.0, sun) * nl;
        }
    }
    e += sky_irradiance(detail.normal) * max(sky_map_sky_visibility(world, n), 0.25);
    return mat.emission + detail.albedo * (1.0 - mat.metallic) * INV_PI * e;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = signal_size();
    let p = vec2<i32>(gid.xy);
    if any(p >= size) {
        return;
    }
    let c = gbuffer(p);
    if (c.id.w >> 30u) == 0u {
        textureStore(out_refraction, p, vec4<f32>(0.0));
        return;
    }
    let m = c.id.w & 0xffffu;
    let mat = materials[m];
    let liquid = mat.kind == KIND_LIQUID;
    if !liquid && mat.kind != KIND_TRANSPARENT {
        textureStore(out_refraction, p, vec4<f32>(0.0));
        return;
    }
    let view = normalize(c.position);
    let world_m = (vec3<f32>(frame.camera_voxel) + frame.camera_frac) / VOXELS_PER_METRE + c.position;
    // The same rippled normal the surface is shaded with.
    let detail = surface_detail(m, mat, vec3<i32>(c.id.xyz), c.normal, world_m, frame.time);
    var n = detail.normal;
    if dot(n, view) > 0.0 {
        n = -n;
    }
    var dir = view;
    if liquid {
        let bent = refract(view, n, 1.0 / mat.ior);
        if dot(bent, bent) > 0.0 {
            dir = normalize(bent);
        }
    }
    // Start just inside the surface and pass the material itself.
    let face = id_face(c.id, c.normal);
    let origin = frame.camera_frac + c.position * VOXELS_PER_METRE - face * 0.05;
    var ray = gi_ray(origin, dir, 0.0, REFRACT_RANGE_M);
    ray.skip = m;
    let hit = trace_scene(ray, ray.t_min);
    var radiance = vec3<f32>(0.0);
    var through_m = REFRACT_RANGE_M;
    if hit.kind == HIT_NONE {
        radiance = sky_radiance(dir);
    } else {
        let at = origin + dir * hit.t;
        let word = hit_id_word(hit, dir);
        let hn = hit_normal(hit, dir);
        radiance = shade_behind(at, hn, id_face(vec4<u32>(vec3<u32>(hit.voxel), word), hn), hit.voxel, word);
        through_m = hit.t / VOXELS_PER_METRE;
    }
    if liquid {
        let transmitted = exp(-WATER_ABSORPTION * through_m);
        // Light the water scatters toward the eye: the sky and the sun
        // falling on it, to the extent the path is not transparent.
        let sun = normalize(frame.sun_dir);
        let altitude_km = world_m.y / 1000.0;
        let falling = sky_irradiance(vec3<f32>(0.0, 1.0, 0.0))
            + frame.sun_illuminance * light_transmittance(altitude_km, sun) * max(sun.y, 0.0);
        let scattered = falling * INV_PI * WATER_SCATTER / (WATER_ABSORPTION + 0.05) * (1.0 - transmitted);
        radiance = radiance * transmitted + scattered;
    } else {
        radiance *= GLASS_TINT;
    }
    textureStore(out_refraction, p, vec4<f32>(radiance, 1.0));
}
