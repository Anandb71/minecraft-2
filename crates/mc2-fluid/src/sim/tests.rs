use super::*;

/// A closed box of solid: floor at y = 0, walls around x and z in
/// `0..=size`, open above.
struct BoxTerrain {
    size: IVec3,
    extra: Vec<IVec3>,
}

impl Terrain for BoxTerrain {
    fn solid(&self, c: IVec3) -> bool {
        c.y <= 0
            || c.x <= 0
            || c.z <= 0
            || c.x >= self.size.x
            || c.z >= self.size.z
            || self.extra.contains(&c)
    }
}

fn cells(lo: IVec3, hi: IVec3) -> Vec<IVec3> {
    let mut v = Vec::new();
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                v.push(IVec3::new(x, y, z));
            }
        }
    }
    v
}

fn column_height(w: &FluidWorld, x: i32, z: i32) -> f32 {
    (1..40).map(|y| w.fill(IVec3::new(x, y, z))).sum()
}

#[test]
fn still_water_stays_still_and_keeps_its_mass() {
    let terrain = BoxTerrain {
        size: IVec3::new(9, 0, 9),
        extra: Vec::new(),
    };
    let mut w = FluidWorld::default();
    w.add_water(cells(IVec3::new(1, 1, 1), IVec3::new(8, 4, 8)), &terrain);
    let m0 = w.mass();
    for _ in 0..960 {
        w.step(&terrain);
    }
    let m1 = w.mass() + w.stats.lost_mass;
    assert!((m1 - m0).abs() / m0 < 0.005, "{m0} -> {m1}");
    let h = column_height(&w, 4, 4);
    assert!((h - 4.0).abs() < 0.15, "height {h}");
    assert!(w.velocity(IVec3::new(4, 2, 4)).length() < 1e-3);
    // Still water falls asleep.
    assert_eq!(w.stats.awake, 0, "{:?}", w.stats);
}

#[test]
fn a_column_of_water_spreads_to_one_level() {
    // An 8 x 8 cell basin (4 m); a 4 x 4 column 8 cells tall in one
    // corner must spread to the same depth, two cells, everywhere. (A
    // film thinner than a cell does not keep thinning: at half-metre
    // cells that is the resolution of the water.)
    let terrain = BoxTerrain {
        size: IVec3::new(9, 0, 9),
        extra: Vec::new(),
    };
    let mut w = FluidWorld::default();
    w.add_water(cells(IVec3::new(1, 1, 1), IVec3::new(4, 8, 4)), &terrain);
    let m0 = w.mass();
    // Ten simulated seconds, averaging the last two: the basin still
    // sloshes, as water with this little viscosity does.
    // Compare the corner quadrants, not single columns: a lone
    // interface cell can sit a fraction of a cell proud of the rest.
    let quadrant = |w: &FluidWorld, lo: i32| {
        let mut h = 0.0;
        for z in lo..lo + 4 {
            for x in lo..lo + 4 {
                h += column_height(w, x, z) / 16.0;
            }
        }
        h
    };
    let (mut a, mut b) = (0.0, 0.0);
    for s in 0..2400 {
        w.step(&terrain);
        if s >= 1920 {
            a += quadrant(&w, 1) / 480.0;
            b += quadrant(&w, 5) / 480.0;
        }
    }
    let m1 = w.mass() + w.stats.lost_mass;
    assert!((m1 - m0).abs() / m0 < 0.02, "mass {m0} -> {m1}");
    assert!((a - 2.0).abs() < 0.2, "near quadrant {a}");
    assert!((b - 2.0).abs() < 0.2, "far quadrant {b}");
}
