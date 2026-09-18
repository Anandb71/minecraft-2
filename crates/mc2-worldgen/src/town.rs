//! Towns: a grid of streets and the buildings between them.
//!
//! Asphalt streets twelve metres wide with dashed centre lines, zebra
//! crossings at the junctions and raised concrete pavements round every
//! block, street lights along them. Blocks hold a glass tower on a podium of
//! shops, brick apartment blocks over shopfronts, concrete offices with
//! ribbon windows, or a park with a fountain. Buildings have floors every
//! storey with light panels under the slabs on some, so windows glow at
//! night, a parapet and plant on the roof. Built from the same operations as
//! villages (see `settlement`), facades as patterns rather than one
//! operation a window.

use crate::amplify::Surface;
use crate::noise::hash_unit;
use crate::settlement::{FacadeStyle, Marks, Op, Shape, b, path_op, street_op};
use glam::{Vec2, Vec3};
use mc2_voxel::material::{MaterialId, ids};

const BLOCK_M: f32 = 36.0;
const STREET_M: f32 = 12.0;
const PAVEMENT_M: f32 = 2.0;

struct Rand(u64);

impl Rand {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        hash_unit(self.0 ^ (self.0 >> 29))
    }

    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next()
    }

    fn int(&mut self, a: i32, b: i32) -> i32 {
        a + (self.next() * (b - a + 1) as f32) as i32
    }
}

/// Rounds up to the brick cell grid, so slabs fill whole cells.
fn cell_up(v: f32) -> f32 {
    (v * 2.0).ceil() * 0.5
}

/// Blocks to a side for a town of this seed, and its extent in metres.
pub fn size(seed: u64) -> (i32, f32) {
    let n = 3 + (hash_unit(seed.rotate_left(7)) * 1.99) as i32;
    (n, n as f32 * BLOCK_M + (n + 1) as f32 * STREET_M)
}

struct Town<'a> {
    surface: &'a Surface,
    ops: Vec<Op>,
    claimed: Vec<(Vec2, Vec2)>,
    lit: u32,
}

#[derive(Clone, Copy)]
enum Kind {
    Tower,
    Apartments,
    Offices,
}

impl Town<'_> {
    fn ground(&self, x: f32, z: f32) -> f32 {
        self.surface.sample(x, z).height
    }

    fn path(&mut self, a: Vec2, b: Vec2, half: f32, depth: f32, rise: f32, m: MaterialId) {
        self.ops
            .extend(path_op(self.surface, a, b, half, depth, rise, m));
    }

    fn street_light(&mut self, p: Vec2, toward: Vec2) {
        let g = self.ground(p.x, p.y) + 0.15;
        let at = |dx: f32, y: f32, dz: f32| Vec3::new(p.x + dx, g + y, p.y + dz);
        self.ops
            .push(b(at(-0.12, -0.8, -0.12), at(0.12, 6.5, 0.12), ids::STEEL));
        let arm = toward * 1.4;
        let (lo, hi) = (
            at(arm.x.min(0.0) - 0.06, 6.3, arm.y.min(0.0) - 0.06),
            at(arm.x.max(0.0) + 0.06, 6.42, arm.y.max(0.0) + 0.06),
        );
        self.ops.push(b(lo, hi, ids::STEEL));
        let head = at(arm.x, 6.2, arm.y);
        self.ops.push(b(
            head - Vec3::new(0.25, 0.0, 0.25),
            head + Vec3::new(0.25, 0.14, 0.25),
            ids::STEEL,
        ));
        self.ops.push(b(
            head - Vec3::new(0.2, 0.04, 0.2),
            head + Vec3::new(0.2, 0.02, 0.2),
            ids::GLOWSTONE,
        ));
    }

    /// One building on lot `lo..hi` (x, z), its entrance toward `street`.
    fn building(&mut self, lo: Vec2, hi: Vec2, kind: Kind, street: Vec2, rng: &mut Rand) {
        let corners = [
            self.ground(lo.x, lo.y),
            self.ground(hi.x, lo.y),
            self.ground(lo.x, hi.y),
            self.ground(hi.x, hi.y),
        ];
        let gmin = corners.iter().copied().fold(f32::MAX, f32::min);
        let gmax = corners.iter().copied().fold(f32::MIN, f32::max);
        self.claimed.push((lo, hi));
        let floor = cell_up(gmax + 0.15);
        let (storey, floors, wall, style) = match kind {
            Kind::Tower => (3.5, rng.int(10, 20), ids::STEEL, FacadeStyle::Curtain),
            Kind::Apartments => (3.0, rng.int(4, 7), ids::RED_BRICK, FacadeStyle::Punched),
            Kind::Offices => (3.5, rng.int(3, 6), ids::CONCRETE, FacadeStyle::Ribbon),
        };
        let v3 = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        self.ops.push(b(
            v3(lo.x - 0.2, gmin - 2.5, lo.y - 0.2),
            v3(hi.x + 0.2, floor, hi.y + 0.2),
            ids::CONCRETE,
        ));
        // A tower rises from a podium of shops, set back from its edge.
        let (podium, tower) = match kind {
            Kind::Tower => (3, floors),
            _ => (floors, 0),
        };
        self.block(
            lo,
            hi,
            floor,
            podium,
            storey,
            if matches!(kind, Kind::Tower) {
                ids::CONCRETE
            } else {
                wall
            },
            if matches!(kind, Kind::Tower) {
                FacadeStyle::Ribbon
            } else {
                style
            },
        );
        let mut top = floor + podium as f32 * storey;
        if tower > 0 {
            let inset = Vec2::splat(rng.range(3.0, 5.0).round());
            let (tlo, thi) = (lo + inset, hi - inset);
            self.block(tlo, thi, top, tower, storey, wall, style);
            // Podium roof and its parapet.
            self.parapet(lo, hi, top, ids::CONCRETE);
            top += tower as f32 * storey;
            self.parapet(tlo, thi, top, wall);
            // Plant and a mast.
            let c = (tlo + thi) * 0.5;
            self.ops.push(b(
                v3(c.x - 3.0, top, c.y - 2.0),
                v3(c.x + 3.0, top + 2.5, c.y + 2.0),
                ids::STEEL,
            ));
            self.ops.push(b(
                v3(c.x - 0.15, top, c.y - 0.15),
                v3(c.x + 0.15, top + 12.0, c.y + 0.15),
                ids::STEEL,
            ));
            self.ops.push(b(
                v3(c.x - 0.12, top + 12.0, c.y - 0.12),
                v3(c.x + 0.12, top + 12.2, c.y + 0.12),
                ids::GLOWSTONE,
            ));
        } else {
            self.parapet(lo, hi, top, wall);
            if matches!(kind, Kind::Apartments) {
                // Cornice.
                self.ops.push(b(
                    v3(lo.x - 0.25, top - 0.4, lo.y - 0.25),
                    v3(hi.x + 0.25, top - 0.1, hi.y + 0.25),
                    ids::STONE_BRICK,
                ));
            }
            let c = (lo + hi) * 0.5;
            let w = rng.range(1.5, 3.0);
            self.ops.push(b(
                v3(c.x - w, top, c.y - w),
                v3(c.x + w, top + 2.0, c.y + w),
                ids::STEEL,
            ));
        }
        // Shopfronts on the ground floor of apartment blocks.
        if matches!(kind, Kind::Apartments) {
            self.facades(
                lo,
                hi,
                floor,
                floor + storey,
                storey,
                wall,
                FacadeStyle::Shopfront,
            );
        }
        // Entrance on the street side: a glass door in a steel frame.
        let c = (lo + hi) * 0.5;
        let (dlo, dhi) = if street.x.abs() > street.y.abs() {
            let x = if street.x > 0.0 { hi.x } else { lo.x };
            (
                v3(x - 0.5, floor, c.y - 1.5),
                v3(x + 0.5, floor + 3.0, c.y + 1.5),
            )
        } else {
            let z = if street.y > 0.0 { hi.y } else { lo.y };
            (
                v3(c.x - 1.5, floor, z - 0.5),
                v3(c.x + 1.5, floor + 3.0, z + 0.5),
            )
        };
        self.ops.push(b(dlo, dhi, ids::AIR));
        self.ops
            .push(b(dlo, v3(dhi.x, floor + 0.5, dhi.z), ids::CONCRETE));
    }

    /// Storeys `floor..` of a building on `lo..hi`: shell, hollow, floors,
    /// windowed walls.
    #[allow(clippy::too_many_arguments)]
    fn block(
        &mut self,
        lo: Vec2,
        hi: Vec2,
        floor: f32,
        storeys: i32,
        storey: f32,
        wall: MaterialId,
        style: FacadeStyle,
    ) {
        let top = floor + storeys as f32 * storey;
        let v3 = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        self.ops
            .push(b(v3(lo.x, floor, lo.y), v3(hi.x, top, hi.y), wall));
        self.ops.push(b(
            v3(lo.x + 0.5, floor + 0.5, lo.y + 0.5),
            v3(hi.x - 0.5, top, hi.y - 0.5),
            ids::AIR,
        ));
        self.lit = self.lit.wrapping_add(1);
        self.ops.push(Op::new(
            Shape::Floors {
                lo: v3(lo.x + 0.5, floor, lo.y + 0.5),
                hi: v3(hi.x - 0.5, top, hi.y - 0.5),
                storey,
                thick: 0.5,
                lit: self.lit,
            },
            ids::CONCRETE,
        ));
        self.ops.push(b(
            v3(lo.x, top - 0.5, lo.y),
            v3(hi.x, top, hi.y),
            ids::CONCRETE,
        ));
        self.facades(lo, hi, floor, top, storey, wall, style);
    }

    #[allow(clippy::too_many_arguments)]
    fn facades(
        &mut self,
        lo: Vec2,
        hi: Vec2,
        from: f32,
        to: f32,
        storey: f32,
        wall: MaterialId,
        style: FacadeStyle,
    ) {
        let t = 0.5;
        let v3 = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        let sides = [
            (v3(lo.x, from, lo.y), v3(lo.x + t, to, hi.y), 0, -1.0),
            (v3(hi.x - t, from, lo.y), v3(hi.x, to, hi.y), 0, 1.0),
            (v3(lo.x, from, lo.y), v3(hi.x, to, lo.y + t), 2, -1.0),
            (v3(lo.x, from, hi.y - t), v3(hi.x, to, hi.y), 2, 1.0),
        ];
        for (a, bb, face, out) in sides {
            self.ops.push(Op::new(
                Shape::Facade {
                    lo: a,
                    hi: bb,
                    face,
                    out,
                    style,
                    base: from,
                    storey,
                },
                wall,
            ));
        }
    }

    fn parapet(&mut self, lo: Vec2, hi: Vec2, top: f32, m: MaterialId) {
        let v3 = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        self.ops
            .push(b(v3(lo.x, top, lo.y), v3(hi.x, top + 1.1, hi.y), m));
        self.ops.push(b(
            v3(lo.x + 0.3, top, lo.y + 0.3),
            v3(hi.x - 0.3, top + 1.1, hi.y - 0.3),
            ids::AIR,
        ));
    }

    fn park(&mut self, lo: Vec2, hi: Vec2) {
        let c = (lo + hi) * 0.5;
        self.path(
            Vec2::new(lo.x + 1.0, c.y),
            Vec2::new(hi.x - 1.0, c.y),
            1.2,
            0.25,
            0.0,
            ids::GRAVEL,
        );
        self.path(
            Vec2::new(c.x, lo.y + 1.0),
            Vec2::new(c.x, hi.y - 1.0),
            1.2,
            0.25,
            0.0,
            ids::GRAVEL,
        );
        let g = self.ground(c.x, c.y);
        let base = Vec3::new(c.x, g - 0.5, c.y);
        self.ops.push(Op::new(
            Shape::Cylinder {
                c: base,
                r: 3.4,
                h: 1.1,
            },
            ids::STONE_BRICK,
        ));
        self.ops.push(Op::new(
            Shape::Cylinder {
                c: base + Vec3::Y * 0.5,
                r: 3.0,
                h: 0.6,
            },
            ids::AIR,
        ));
        self.ops.push(Op::new(
            Shape::Cylinder {
                c: base + Vec3::Y * 0.5,
                r: 3.0,
                h: 0.4,
            },
            ids::WATER,
        ));
        self.ops.push(Op::new(
            Shape::Cylinder {
                c: base,
                r: 0.45,
                h: 2.8,
            },
            ids::MARBLE,
        ));
        self.ops.push(Op::new(
            Shape::Cylinder {
                c: base + Vec3::Y * 2.8,
                r: 1.1,
                h: 0.25,
            },
            ids::MARBLE,
        ));
        self.claimed
            .push((c - Vec2::splat(4.0), c + Vec2::splat(4.0)));
    }
}

/// Plans a town centred on `centre` (its ground there).
pub(crate) fn plan(surface: &Surface, centre: Vec3, seed: u64) -> (Vec<Op>, Vec<(Vec2, Vec2)>) {
    let mut rng = Rand(seed ^ 0x70_77);
    let (n, extent) = size(seed);
    let o = Vec2::new(centre.x, centre.z) - Vec2::splat(extent * 0.5);
    let mut town = Town {
        surface,
        ops: Vec::new(),
        claimed: Vec::new(),
        lit: (seed & 0xffff) as u32,
    };
    let pitch = BLOCK_M + STREET_M;
    // Streets, markings painted on: junctions every block, the first half
    // a street in from the town's edge.
    let marks = Marks {
        along0: 0.0,
        pitch,
        junction0: STREET_M * 0.5,
        width: STREET_M,
    };
    for i in 0..=n {
        let x = o.x + i as f32 * pitch + STREET_M * 0.5;
        let (a, b0) = (Vec2::new(x, o.y), Vec2::new(x, o.y + extent));
        town.ops.extend(street_op(
            surface,
            a,
            b0,
            STREET_M * 0.5,
            marks,
            ids::ASPHALT,
        ));
        town.claimed.push((
            Vec2::new(x - STREET_M * 0.5, o.y),
            Vec2::new(x + STREET_M * 0.5, o.y + extent),
        ));
        let z = o.y + i as f32 * pitch + STREET_M * 0.5;
        let (a, b0) = (Vec2::new(o.x, z), Vec2::new(o.x + extent, z));
        town.ops.extend(street_op(
            surface,
            a,
            b0,
            STREET_M * 0.5,
            marks,
            ids::ASPHALT,
        ));
        town.claimed.push((
            Vec2::new(o.x, z - STREET_M * 0.5),
            Vec2::new(o.x + extent, z + STREET_M * 0.5),
        ));
    }
    // Blocks: a pavement round each, then what stands on it.
    for i in 0..n {
        for k in 0..n {
            let blo = o + Vec2::new(i as f32, k as f32) * pitch + Vec2::splat(STREET_M);
            let bhi = blo + Vec2::splat(BLOCK_M);
            let r = PAVEMENT_M * 0.5;
            let (a, bb) = (blo - Vec2::splat(r), bhi + Vec2::splat(r));
            for (p0, p1) in [
                (a, Vec2::new(bb.x, a.y)),
                (Vec2::new(bb.x, a.y), bb),
                (bb, Vec2::new(a.x, bb.y)),
                (Vec2::new(a.x, bb.y), a),
            ] {
                town.path(p0, p1, r, 0.3, 0.15, ids::CONCRETE);
            }
            // Street lights on the pavement, facing the road.
            let mut d = 4.0;
            while d < BLOCK_M {
                town.street_light(Vec2::new(blo.x + d, a.y), -Vec2::Y);
                town.street_light(Vec2::new(blo.x + d, bb.y), Vec2::Y);
                d += 16.0;
            }
            let central = (i as f32 - (n - 1) as f32 * 0.5).abs()
                + (k as f32 - (n - 1) as f32 * 0.5).abs()
                < 1.5;
            let roll = rng.next();
            let (lo, hi) = (blo + Vec2::splat(1.0), bhi - Vec2::splat(1.0));
            let street = if k == 0 { -Vec2::Y } else { Vec2::Y };
            if roll < 0.14 && !central {
                town.park(blo, bhi);
            } else if central && roll < 0.8 || roll < 0.3 {
                town.building(lo, hi, Kind::Tower, street, &mut rng);
            } else {
                // Two or four lots.
                let mid = (lo + hi) * 0.5;
                let lots: Vec<(Vec2, Vec2)> = if rng.next() < 0.5 {
                    vec![
                        (lo, Vec2::new(mid.x - 1.0, hi.y)),
                        (Vec2::new(mid.x + 1.0, lo.y), hi),
                    ]
                } else {
                    vec![
                        (lo, mid - Vec2::splat(1.0)),
                        (Vec2::new(mid.x + 1.0, lo.y), Vec2::new(hi.x, mid.y - 1.0)),
                        (Vec2::new(lo.x, mid.y + 1.0), Vec2::new(mid.x - 1.0, hi.y)),
                        (mid + Vec2::splat(1.0), hi),
                    ]
                };
                for (llo, lhi) in lots {
                    let kind = if rng.next() < 0.6 {
                        Kind::Apartments
                    } else {
                        Kind::Offices
                    };
                    let facing = if (llo.y + lhi.y) * 0.5 < (lo.y + hi.y) * 0.5 {
                        -Vec2::Y
                    } else {
                        Vec2::Y
                    };
                    town.building(llo.floor(), lhi.floor(), kind, facing, &mut rng);
                }
            }
        }
    }
    (town.ops, town.claimed)
}
