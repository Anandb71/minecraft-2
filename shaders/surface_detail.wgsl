// What each material looks like within its voxels. A voxel is 1/16 m, so
// a block face is 16 x 16 of them, a texture's worth of texels, except that
// they are geometry: the pattern is chosen per voxel from its world
// coordinate (a body's own coordinate on a body, so the pattern rides with
// the fragment). Built materials (brick, planks, cobble, tiles) follow the
// 1 m block grid; natural ones follow noise at several scales.
//
// Material ids match crates/mc2-voxel/src/material.rs; a test in
// mc2-render checks every M_* constant against it.
//
// Expects common.wgsl and voxel_data.wgsl imported before.

const M_GRANITE: u32 = 1u;
const M_BASALT: u32 = 2u;
const M_LIMESTONE: u32 = 3u;
const M_SANDSTONE: u32 = 4u;
const M_SHALE: u32 = 5u;
const M_MARBLE: u32 = 6u;
const M_SLATE: u32 = 7u;
const M_DIRT: u32 = 8u;
const M_GRASS: u32 = 9u;
const M_SAND: u32 = 10u;
const M_GRAVEL: u32 = 11u;
const M_CLAY: u32 = 12u;
const M_SNOW: u32 = 13u;
const M_ICE: u32 = 14u;
const M_WATER: u32 = 15u;
const M_OBSIDIAN: u32 = 17u;
const M_GLASS: u32 = 18u;
const M_OAK_LOG: u32 = 19u;
const M_PINE_LOG: u32 = 20u;
const M_PLANKS: u32 = 21u;
const M_LEAVES: u32 = 22u;
const M_PINE_NEEDLES: u32 = 23u;
const M_COBBLESTONE: u32 = 26u;
const M_STONE_BRICK: u32 = 27u;
const M_COAL_ORE: u32 = 28u;
const M_IRON_ORE: u32 = 29u;
const M_COPPER_ORE: u32 = 30u;
const M_GOLD_ORE: u32 = 31u;
const M_IRON: u32 = 34u;
const M_TNT: u32 = 35u;
const M_WET_DIRT: u32 = 38u;
const M_ROOF_TILE: u32 = 39u;
const M_WHITEWASH: u32 = 40u;
const M_WOOL: u32 = 41u;
const M_STEEL: u32 = 42u;
const M_BEDROCK: u32 = 43u;
const M_RHYOLITE: u32 = 44u;
const M_CONGLOMERATE: u32 = 45u;
const M_CHALK: u32 = 46u;
const M_GNEISS: u32 = 47u;
const M_FARMLAND: u32 = 48u;
const M_WHEAT: u32 = 49u;
const M_TALL_GRASS: u32 = 51u;
const M_FLOWER_RED: u32 = 52u;
const M_FLOWER_YELLOW: u32 = 53u;
const M_FLOWER_BLUE: u32 = 54u;
const M_FLOWER_WHITE: u32 = 55u;
const M_BIRCH_LOG: u32 = 56u;
const M_BIRCH_LEAVES: u32 = 57u;
const M_CACTUS: u32 = 58u;
const M_MOSS: u32 = 59u;
const M_RED_BRICK: u32 = 60u;
const M_CONCRETE: u32 = 61u;
const M_ASPHALT: u32 = 62u;
const M_DARK_PLANKS: u32 = 63u;
const M_THATCH: u32 = 64u;

struct Detail {
    albedo: vec3<f32>,
    roughness: f32,
    // Shading normal: the face normal tilted by the pattern (bevels at
    // mortar joints, the lean of each cobble, ripples on water).
    normal: vec3<f32>,
    // For clear materials, how much of the surface is opaque (frosted edges
    // of a glass block); 0 elsewhere.
    cover: f32,
}

fn voxel_hash(v: vec3<i32>, salt: u32) -> f32 {
    let u = vec3<u32>(v);
    return hash_to_unit(pcg(u.x * 73856093u ^ u.y * 19349663u ^ u.z * 83492791u ^ (salt * 2654435761u)));
}

fn lattice_hash(p: vec3<i32>, salt: u32) -> f32 {
    return voxel_hash(p, salt ^ 0x9e37u);
}

// Smooth value noise in [0, 1].
fn value_noise(p: vec3<f32>, salt: u32) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let a = mix(lattice_hash(i, salt), lattice_hash(i + vec3<i32>(1, 0, 0), salt), w.x);
    let b = mix(lattice_hash(i + vec3<i32>(0, 1, 0), salt), lattice_hash(i + vec3<i32>(1, 1, 0), salt), w.x);
    let c = mix(lattice_hash(i + vec3<i32>(0, 0, 1), salt), lattice_hash(i + vec3<i32>(1, 0, 1), salt), w.x);
    let d = mix(lattice_hash(i + vec3<i32>(0, 1, 1), salt), lattice_hash(i + vec3<i32>(1, 1, 1), salt), w.x);
    return mix(mix(a, b, w.y), mix(c, d, w.y), w.z);
}

fn fbm(p: vec3<f32>, salt: u32) -> f32 {
    return value_noise(p, salt) * 0.55 + value_noise(p * 2.03, salt + 1u) * 0.3 + value_noise(p * 4.11, salt + 2u) * 0.15;
}

// Cellular noise in a plane: distance to the nearest feature, the gap to
// the second nearest (small near cell edges), the nearest cell's hash, and
// the offset from that feature.
struct Cells {
    f1: f32,
    edge: f32,
    id: f32,
    offset: vec2<f32>,
}

fn cells2(p: vec2<f32>, salt: u32) -> Cells {
    let i = vec2<i32>(floor(p));
    var f1 = 1e9;
    var f2 = 1e9;
    var id = 0.0;
    var offset = vec2<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = i + vec2<i32>(x, y);
            let h = vec3<i32>(c, 0);
            let feature = vec2<f32>(c) + vec2<f32>(lattice_hash(h, salt), lattice_hash(h, salt + 7u)) * 0.8 + 0.1;
            let d = length(p - feature);
            if d < f1 {
                f2 = f1;
                f1 = d;
                id = lattice_hash(h, salt + 13u);
                offset = p - feature;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    return Cells(f1, f2 - f1, id, offset);
}

// The face axis of a unit face normal.
fn face_axis(n: vec3<f32>) -> u32 {
    let a = abs(n);
    if a.x >= a.y && a.x >= a.z {
        return 0u;
    }
    if a.y >= a.z {
        return 1u;
    }
    return 2u;
}

// Coordinates in the face's plane: horizontal first, then up the face (or
// along z on a top or bottom face).
fn plane2(v: vec3<i32>, axis: u32) -> vec2<i32> {
    if axis == 0u {
        return v.zy;
    }
    if axis == 1u {
        return v.xz;
    }
    return v.xy;
}

fn plane2f(p: vec3<f32>, axis: u32) -> vec2<f32> {
    if axis == 0u {
        return p.zy;
    }
    if axis == 1u {
        return p.xz;
    }
    return p.xy;
}

// Tilts `n` by a vector in the face's plane coordinates.
fn tilt(n: vec3<f32>, axis: u32, t: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(0.0);
    if axis == 0u {
        d = vec3<f32>(0.0, t.y, t.x);
    } else if axis == 1u {
        d = vec3<f32>(t.x, 0.0, t.y);
    } else {
        d = vec3<f32>(t.x, t.y, 0.0);
    }
    return normalize(n + d);
}

fn jitter(v: vec3<i32>, salt: u32, amount: f32) -> f32 {
    return 1.0 + (voxel_hash(v, salt) - 0.5) * 2.0 * amount;
}

// Speckled stone: a base colour broken by minerals, `a` and `b` of them.
fn speckle(base: vec3<f32>, v: vec3<i32>, a: vec3<f32>, a_share: f32, b: vec3<f32>, b_share: f32) -> vec3<f32> {
    let h = voxel_hash(v, 3u);
    if h < a_share {
        return a;
    }
    if h < a_share + b_share {
        return b;
    }
    return base * jitter(v, 4u, 0.1);
}

// Letters of "TNT" on a 16 x 4 strip, one row per word, bit x for column x.
fn tnt_letter(u: i32, row: i32) -> bool {
    var bits = 0u;
    if row == 0 {
        bits = 29390u;
    } else if row == 1 {
        bits = 8900u;
    } else if row == 2 {
        bits = 9028u;
    } else {
        bits = 8772u;
    }
    return ((bits >> u32(u)) & 1u) != 0u;
}

// One travelling wave of wavelength `1 / freq` metres along `dir`, as a
// slope (deep water: speed sqrt(g / k)).
fn wave(p: vec2<f32>, dir: vec2<f32>, freq: f32, amp: f32, time: f32) -> vec2<f32> {
    let k = freq * TAU;
    let phase = (dot(p, dir) - time * sqrt(9.81 / k)) * k;
    return dir * cos(phase) * amp * k;
}

// Moving ripples on still water as a slope, from the world position in
// metres: four swells and a fine chop.
fn water_ripple(p: vec2<f32>, time: f32) -> vec2<f32> {
    var t = wave(p, vec2<f32>(0.8, 0.6), 0.35, 0.012, time);
    t += wave(p, vec2<f32>(-0.5, 0.86), 0.6, 0.006, time);
    t += wave(p, vec2<f32>(0.2, -0.98), 1.1, 0.0028, time);
    t += wave(p, vec2<f32>(-0.9, -0.3), 1.9, 0.0014, time);
    let q = vec3<f32>(p * 3.0, time * 0.7);
    t += vec2<f32>(value_noise(q, 91u) - 0.5, value_noise(q + 17.0, 92u) - 0.5) * 0.08;
    return t;
}

// Planks: boards four voxels wide with dark seams, grain running along
// each board, nail heads at the ends, the odd knot.
fn planks(d_in: Detail, a: vec3<f32>, voxel: vec3<i32>, b: vec3<i32>, uv: vec2<i32>, axis: u32) -> Detail {
    var d = d_in;
    let pv = vec3<f32>(voxel);
    var along = uv.x;
    var across = uv.y;
    if axis == 1u {
        along = b.x;
        across = b.z;
    }
    let board = vec3<i32>(voxel.x >> 4u, (across >> 2u) + (voxel.y >> 4u) * 8, (voxel.z >> 4u) + i32(axis) * 1000);
    let tone = 0.82 + 0.3 * lattice_hash(board, 81u);
    let grain = 1.0 + 0.1 * sin(f32(along) * 0.8 + lattice_hash(board, 82u) * 20.0 + 2.5 * value_noise(pv * vec3<f32>(0.1, 0.6, 0.1), 83u));
    var c = a * tone * grain * jitter(voxel, 84u, 0.04);
    if (across & 3) == 0 {
        c *= 0.55;
        d.roughness = 0.9;
    } else if ((along & 15) == 1 || (along & 15) == 14) && (across & 3) == 2 {
        c = vec3<f32>(0.12, 0.12, 0.13);
        d.roughness = 0.4;
    }
    if voxel_hash(voxel, 85u) < 0.02 {
        // Knots.
        c *= 0.6;
    }
    d.albedo = c;
    return d;
}

// Masonry in running bond: bricks `w` voxels long and `h` high with a voxel
// of mortar between, each brick its own shade, edges bevelled. Follows the
// world grid so courses run on across blocks.
fn masonry(d_in: Detail, a: vec3<f32>, mortar: vec3<f32>, voxel: vec3<i32>, axis: u32, normal: vec3<f32>, w: i32, h: i32, salt: u32) -> Detail {
    var d = d_in;
    let q = plane2(voxel, axis);
    let row = i32(floor(f32(q.y) / f32(h + 1)));
    let in_v = q.y - row * (h + 1);
    let shift = select(0, (w + 1) / 2, (row & 1) == 1);
    let col = i32(floor(f32(q.x + shift) / f32(w + 1)));
    let in_u = q.x + shift - col * (w + 1);
    let joint = in_u == w || in_v == h;
    let id = vec3<i32>(col, row, i32(axis) * 7919 + i32(salt));
    let tone = 0.78 + 0.36 * lattice_hash(id, salt);
    var c = a * tone * jitter(voxel, salt + 1u, 0.06);
    let hue = lattice_hash(id, salt + 2u);
    if hue < 0.15 {
        c *= vec3<f32>(0.92, 0.96, 0.84);
    } else if hue > 0.85 {
        c *= vec3<f32>(1.06, 0.98, 0.92);
    }
    if joint {
        c = mortar * jitter(voxel, salt + 3u, 0.1);
        d.roughness = 0.95;
    } else {
        var t = vec2<f32>(0.0);
        t.x = select(0.0, 0.3, in_u == 0) - select(0.0, 0.3, in_u == w - 1);
        t.y = select(0.0, 0.3, in_v == 0) - select(0.0, 0.3, in_v == h - 1);
        d.normal = tilt(normal, axis, t);
    }
    d.albedo = c;
    return d;
}

fn surface_detail(m: u32, mat: Material, voxel: vec3<i32>, normal: vec3<f32>, world_m: vec3<f32>, time: f32) -> Detail {
    var d = Detail(mat.albedo, mat.roughness, normal, 0.0);
    let axis = face_axis(normal);
    let top = axis == 1u && normal.y > 0.0;
    let b = voxel & vec3<i32>(15);
    let pv = vec3<f32>(voxel);
    let uv = plane2(b, axis);
    let a = mat.albedo;

    switch m {
        case 9u: {
            // Grass: patches of lush and dry turf a few metres across,
            // clumps within them, single blades lighter or darker.
            let turf = fbm(pv * vec3<f32>(0.025, 0.05, 0.025), 11u);
            let clump = value_noise(pv * 0.18, 12u);
            // Darker than a blade: turf seen between and under them, and
            // seen at a distance where the blades are not generated.
            let deep = vec3<f32>(0.075, 0.23, 0.035);
            let lush = vec3<f32>(0.14, 0.32, 0.06);
            let dry = vec3<f32>(0.28, 0.33, 0.1);
            var c = mix(deep, lush, smoothstep(0.25, 0.55, turf));
            c = mix(c, dry, smoothstep(0.62, 0.85, turf) * 0.7);
            c *= 0.85 + 0.25 * clump;
            let blade = voxel_hash(voxel, 13u);
            if blade < 0.08 {
                c *= 0.72;
            } else if blade > 0.975 {
                c = mix(c, vec3<f32>(0.42, 0.52, 0.16), 0.5);
            }
            if !top {
                // Sides: turf over soil, roots showing.
                c = mix(c, vec3<f32>(0.26, 0.18, 0.11), select(0.0, 0.5, voxel_hash(voxel, 14u) < 0.3));
            }
            d.albedo = c * jitter(voxel, 15u, 0.08);
        }
        case 8u, 38u: {
            // Soil: specks of dark humus and the odd pebble.
            let h = voxel_hash(voxel, 21u);
            var c = a * (0.85 + 0.3 * value_noise(pv * 0.2, 22u));
            if h < 0.1 {
                c *= 0.55;
            } else if h > 0.965 {
                c = vec3<f32>(0.42, 0.40, 0.37) * jitter(voxel, 23u, 0.15);
            }
            d.albedo = c * jitter(voxel, 24u, 0.07);
            if m == M_WET_DIRT {
                d.roughness = 0.35 + 0.3 * voxel_hash(voxel, 25u);
            }
        }
        case 10u: {
            // Sand: fine grains, wind ripples on top.
            var c = a * jitter(voxel, 31u, 0.06);
            if top {
                let ripple = sin(pv.x * 0.45 + pv.z * 0.2 + 3.0 * value_noise(pv * 0.05, 32u));
                c *= 1.0 + 0.05 * ripple;
            }
            let g = voxel_hash(voxel, 33u);
            if g < 0.05 {
                c *= 0.72;
            } else if g > 0.97 {
                c = mix(c, vec3<f32>(0.95, 0.92, 0.85), 0.6);
            }
            d.albedo = c;
        }
        case 11u, 45u: {
            // Gravel and conglomerate: pebbles two voxels across.
            let pebble = voxel >> vec3<u32>(1u);
            let tone = voxel_hash(pebble, 41u);
            var c = mix(vec3<f32>(0.30, 0.29, 0.28), vec3<f32>(0.55, 0.52, 0.47), tone);
            if voxel_hash(pebble, 42u) < 0.2 {
                c *= vec3<f32>(1.1, 0.95, 0.8);
            }
            if m == M_CONGLOMERATE {
                c = mix(a, c, select(0.0, 1.0, voxel_hash(pebble, 43u) < 0.45));
            }
            d.albedo = c * jitter(voxel, 44u, 0.06);
            d.normal = tilt(normal, axis, (vec2<f32>(voxel_hash(pebble, 45u), voxel_hash(pebble, 46u)) - 0.5) * 0.5);
        }
        case 1u: {
            d.albedo = speckle(a, voxel, vec3<f32>(0.62, 0.46, 0.42), 0.16, vec3<f32>(0.08, 0.08, 0.09), 0.1)
                * (0.9 + 0.2 * value_noise(pv * 0.08, 51u));
        }
        case 2u, 17u: {
            var c = a * (0.85 + 0.3 * value_noise(pv * 0.15, 52u)) * jitter(voxel, 53u, 0.08);
            if m == M_OBSIDIAN && voxel_hash(voxel, 54u) < 0.08 {
                c = vec3<f32>(0.12, 0.06, 0.18);
            }
            d.albedo = c;
        }
        case 3u, 46u: {
            // Limestone and chalk: soft blotches, rare fossil flecks.
            var c = a * (0.88 + 0.24 * fbm(pv * 0.06, 55u)) * jitter(voxel, 56u, 0.04);
            if voxel_hash(voxel, 57u) < 0.015 {
                c *= 1.12;
            }
            d.albedo = c;
        }
        case 4u, 5u, 7u, 44u: {
            // Bedded rock: strata across the face, bent a little.
            let bend = 4.0 * value_noise(pv * vec3<f32>(0.03, 0.0, 0.03), 58u);
            var period = 0.55;
            var depth = 0.12;
            if m == M_SHALE || m == M_SLATE {
                period = 1.6;
                depth = 0.1;
            }
            let band = sin((pv.y + bend) * period);
            d.albedo = a * (1.0 + depth * band) * jitter(voxel, 59u, 0.05);
            if m == M_SLATE {
                d.roughness = mat.roughness * (0.8 + 0.4 * voxel_hash(voxel, 60u));
            }
        }
        case 6u: {
            // Marble: grey veins through white.
            let n = fbm(pv * 0.04, 61u);
            let vein = abs(sin((pv.x * 0.7 + pv.y * 0.4 + pv.z * 0.55) * 0.18 + n * 9.0));
            var c = a * jitter(voxel, 62u, 0.03);
            c = mix(vec3<f32>(0.42, 0.42, 0.44), c, smoothstep(0.0, 0.12, vein));
            d.albedo = c;
        }
        case 47u: {
            // Gneiss: wavy light and dark bands.
            let n = fbm(pv * 0.05, 63u);
            let band = sin(pv.y * 0.9 + n * 7.0);
            d.albedo = mix(a * 0.6, vec3<f32>(0.7, 0.66, 0.62), smoothstep(-0.2, 0.4, band)) * jitter(voxel, 64u, 0.05);
        }
        case 26u: {
            // Cobblestone: rounded stones five voxels across in mortar,
            // each its own shade, leaning a little.
            let q = vec2<f32>(plane2(voxel, axis)) / 5.0;
            let c = cells2(q, 71u);
            var stone = mix(vec3<f32>(0.28, 0.27, 0.26), vec3<f32>(0.52, 0.50, 0.47), c.id);
            if fract(c.id * 7.3) < 0.2 {
                stone *= vec3<f32>(1.08, 1.0, 0.9);
            }
            let mortar = 1.0 - smoothstep(0.08, 0.2, c.edge);
            d.albedo = mix(stone * jitter(voxel, 72u, 0.08), vec3<f32>(0.2, 0.19, 0.18), mortar);
            d.roughness = mix(mat.roughness, 0.95, mortar);
            d.normal = tilt(normal, axis, c.offset * 0.9);
        }
        case 27u: {
            // Stone brick: blocks two to a metre, four courses to a metre.
            d = masonry(d, a, vec3<f32>(0.24, 0.23, 0.21), voxel, axis, normal, 7, 3, 73u);
        }
        case 60u: {
            // Clay brick: 25 x 12 cm with dark mortar.
            d = masonry(d, a, vec3<f32>(0.30, 0.28, 0.26), voxel, axis, normal, 4, 1, 201u);
        }
        case 21u, 63u: {
            d = planks(d, a, voxel, b, uv, axis);
        }
        case 19u, 20u: {
            // Logs: furrowed bark on the sides; growth rings on the ends.
            if axis == 1u {
                let r = length(vec2<f32>(b.xz) - 7.5);
                let ring = 0.5 + 0.5 * sin(r * 2.2 + 2.0 * value_noise(pv * 0.3, 91u));
                var c = vec3<f32>(0.62, 0.47, 0.30) * (0.85 + 0.2 * ring);
                if r > 6.2 {
                    c = a * 0.8;
                }
                d.albedo = c * jitter(voxel, 92u, 0.04);
            } else {
                let q = vec3<f32>(pv.x * 0.6, pv.y * 0.09, pv.z * 0.6);
                let furrow = value_noise(q, 93u);
                var c = a * (0.9 + 0.25 * value_noise(pv * 0.3, 94u));
                if furrow > 0.62 {
                    c *= 0.5;
                } else if furrow < 0.2 {
                    c *= 1.15;
                }
                if m == M_OAK_LOG && voxel_hash(voxel, 95u) < 0.03 {
                    // Lichen.
                    c = mix(c, vec3<f32>(0.35, 0.42, 0.25), 0.6);
                }
                d.albedo = c * jitter(voxel, 96u, 0.05);
                d.normal = tilt(normal, axis, vec2<f32>((furrow - 0.5) * 0.6, 0.0));
            }
        }
        case 22u, 23u: {
            // Foliage: every leaf its own green, some yellowing, deeper in
            // the crown darker.
            let h = voxel_hash(voxel, 101u);
            var c = a * (0.7 + 0.6 * value_noise(pv * 0.25, 102u));
            if h < 0.08 {
                c = mix(c, vec3<f32>(0.45, 0.42, 0.10), 0.7);
            } else if h > 0.9 {
                c *= 1.35;
            }
            d.albedo = c * jitter(voxel, 103u, 0.12);
            d.normal = tilt(normal, axis, (vec2<f32>(voxel_hash(voxel, 104u), voxel_hash(voxel, 105u)) - 0.5) * 1.2);
        }
        case 39u: {
            // Roof tiles: overlapping courses, each tile's lower edge in the
            // shadow of the next, the odd weathered or mossy tile.
            let course = uv.y >> 2u;
            let shift = select(0, 2, (course & 1) == 1);
            let tile = vec3<i32>(voxel.x >> 4u, course + (voxel.y >> 4u) * 4, ((uv.x + shift) >> 2u) + (voxel.z >> 4u) * 4 + i32(axis) * 1000);
            let in_v = uv.y & 3;
            var c = a * (0.8 + 0.3 * lattice_hash(tile, 111u)) * (0.72 + 0.1 * f32(in_v));
            let w = lattice_hash(tile, 112u);
            if w < 0.07 {
                c = mix(c, vec3<f32>(0.25, 0.32, 0.14), 0.55);
            } else if w > 0.92 {
                c *= 0.7;
            }
            if ((uv.x + shift) & 3) == 0 {
                c *= 0.7;
            }
            d.albedo = c * jitter(voxel, 113u, 0.04);
            d.normal = tilt(normal, axis, vec2<f32>(0.0, select(0.0, -0.3, in_v == 0)));
        }
        case 40u: {
            // Plaster: faint stains and grime rising from the ground.
            let stain = fbm(pv * 0.04, 121u);
            let grime = clamp(1.0 - f32(b.y) / 6.0, 0.0, 1.0) * 0.12;
            d.albedo = a * (0.93 + 0.1 * stain - grime) * jitter(voxel, 122u, 0.025);
        }
        case 41u: {
            d.albedo = a * (0.9 + 0.2 * value_noise(pv * 0.5, 131u)) * jitter(voxel, 132u, 0.05);
        }
        case 28u, 29u, 30u, 31u: {
            // Ore in a granite-grey host: clusters of the mineral.
            let host = vec3<f32>(0.40, 0.39, 0.38) * (0.9 + 0.2 * value_noise(pv * 0.1, 141u)) * jitter(voxel, 142u, 0.08);
            let cluster = value_noise(pv * 0.45, 143u);
            let mineral = cluster > 0.58 && voxel_hash(voxel, 144u) < 0.8;
            d.albedo = select(host, a * jitter(voxel, 145u, 0.15), mineral);
            if mineral && (m == M_GOLD_ORE || m == M_COPPER_ORE) {
                d.roughness = 0.25;
            }
        }
        case 34u, 42u: {
            // Metal: brushed along the face, a little uneven.
            let streak = value_noise(vec3<f32>(pv.x * 0.05, pv.y * 2.0, pv.z * 0.05), 151u);
            d.albedo = a * (0.94 + 0.08 * streak);
            d.roughness = mat.roughness * (0.8 + 0.5 * streak);
        }
        case 13u: {
            var c = a * jitter(voxel, 161u, 0.02);
            if !top {
                c *= vec3<f32>(0.93, 0.96, 1.0);
            }
            d.albedo = c;
            if voxel_hash(voxel, 162u) < 0.03 {
                // Sparkle: ice crystals facing the right way.
                d.roughness = 0.05;
            }
        }
        case 14u: {
            let c = cells2(vec2<f32>(plane2(voxel, axis)) / 9.0, 163u);
            d.cover = (1.0 - smoothstep(0.02, 0.08, c.edge)) * 0.6;
        }
        case 18u: {
            // Glass: clear, with a frosted rim where the block's faces meet.
            let rim = uv.x == 0 || uv.x == 15 || uv.y == 0 || uv.y == 15;
            d.cover = select(0.0, 0.55, rim);
            d.albedo = vec3<f32>(0.85, 0.9, 0.9);
        }
        case 15u: {
            if top {
                let slope = water_ripple(world_m.xz, time);
                d.normal = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
            }
        }
        case 35u: {
            // TNT: red sticks bound by a paper band lettered TNT, a fuse on
            // top.
            if axis == 1u {
                let r = length(vec2<f32>(b.xz) - 7.5);
                var c = a * select(0.85, 1.0, (b.x & 3) != 0 && (b.z & 3) != 0);
                if r < 1.5 {
                    c = vec3<f32>(0.1, 0.1, 0.1);
                }
                d.albedo = c;
            } else {
                var u = uv.x;
                if (axis == 0u && normal.x > 0.0) || (axis == 2u && normal.z < 0.0) {
                    u = 15 - u;
                }
                var c = a * select(0.8, 1.0, (u & 3) != 0);
                if b.y >= 5 && b.y <= 10 {
                    c = vec3<f32>(0.86, 0.82, 0.68);
                    if b.y >= 6 && b.y <= 9 && tnt_letter(u, 9 - b.y) {
                        c = vec3<f32>(0.08, 0.07, 0.07);
                    }
                }
                d.albedo = c * jitter(voxel, 171u, 0.03);
            }
        }
        case 48u: {
            // Farmland: furrows along x.
            d.albedo = a * select(0.7, 1.05, (b.z & 3) != 0) * jitter(voxel, 181u, 0.08);
        }
        case 49u: {
            d.albedo = mix(vec3<f32>(0.45, 0.5, 0.2), a, clamp(f32(b.y) / 10.0, 0.0, 1.0)) * jitter(voxel, 182u, 0.1);
        }
        case 43u: {
            d.albedo = a * (0.6 + 0.8 * value_noise(pv * 0.3, 191u));
        }
        case 12u: {
            d.albedo = a * (0.9 + 0.2 * value_noise(pv * 0.12, 192u)) * jitter(voxel, 193u, 0.04);
        }
        case 51u: {
            // Tall grass: blades a shade off the turf around them, lighter
            // at the tips.
            let turf = fbm(pv * vec3<f32>(0.025, 0.05, 0.025), 11u);
            var c = mix(vec3<f32>(0.14, 0.34, 0.06), vec3<f32>(0.36, 0.42, 0.14), smoothstep(0.55, 0.85, turf));
            let tip = voxel_hash(voxel, 211u);
            c *= 0.75 + 0.5 * tip;
            d.albedo = c;
            d.normal = tilt(normal, axis, (vec2<f32>(voxel_hash(voxel, 212u), voxel_hash(voxel, 213u)) - 0.5) * 1.5);
        }
        case 52u, 53u, 54u, 55u: {
            d.albedo = a * jitter(voxel, 221u, 0.15);
        }
        case 56u: {
            // Birch: white bark with dark lenticels and black scars.
            var c = a * jitter(voxel, 231u, 0.04);
            if axis != 1u {
                let dash = ((voxel.y % 5) == 0) && voxel_hash(vec3<i32>(voxel.x >> 1u, voxel.y, voxel.z >> 1u), 232u) < 0.55;
                let scar = value_noise(pv * vec3<f32>(0.3, 0.15, 0.3), 233u) > 0.74;
                if dash || scar {
                    c = vec3<f32>(0.08, 0.08, 0.07);
                }
            } else {
                c = vec3<f32>(0.75, 0.62, 0.45) * jitter(voxel, 234u, 0.06);
            }
            d.albedo = c;
        }
        case 57u, 59u: {
            let h = voxel_hash(voxel, 241u);
            var c = a * (0.72 + 0.56 * value_noise(pv * 0.25, 242u));
            if h < 0.06 {
                c = mix(c, vec3<f32>(0.5, 0.48, 0.12), 0.6);
            }
            d.albedo = c * jitter(voxel, 243u, 0.1);
            d.normal = tilt(normal, axis, (vec2<f32>(voxel_hash(voxel, 244u), voxel_hash(voxel, 245u)) - 0.5) * 1.2);
        }
        case 58u: {
            // Cactus: ribs, a spine here and there.
            var c = a * select(0.8, 1.05, (uv.x % 3) != 0);
            if voxel_hash(voxel, 251u) < 0.04 {
                c = vec3<f32>(0.85, 0.82, 0.65);
            }
            d.albedo = c * jitter(voxel, 252u, 0.05);
        }
        case 61u: {
            // Concrete: cast in 1 m forms (seams at block edges), tie holes,
            // blotches of cure.
            var c = a * (0.9 + 0.2 * fbm(pv * 0.05, 261u)) * jitter(voxel, 262u, 0.04);
            if uv.x == 0 || uv.y == 0 {
                c *= 0.82;
            }
            if (uv.x == 3 || uv.x == 12) && (uv.y == 3 || uv.y == 12) && axis != 1u {
                c *= 0.35;
            }
            d.albedo = c;
        }
        case 62u: {
            var c = a * (0.85 + 0.3 * value_noise(pv * 0.1, 271u)) * jitter(voxel, 272u, 0.12);
            if voxel_hash(voxel, 273u) < 0.05 {
                c = vec3<f32>(0.3, 0.3, 0.3);
            }
            d.albedo = c;
        }
        case 64u: {
            // Thatch: straw in bundles running down the roof.
            let streak = value_noise(vec3<f32>(pv.x * 0.9, pv.y * 0.12, pv.z * 0.9), 281u);
            d.albedo = a * (0.75 + 0.45 * streak) * jitter(voxel, 282u, 0.08);
        }
        default: {
            d.albedo = a * jitter(voxel, 199u, 0.06);
        }
    }
    return d;
}
