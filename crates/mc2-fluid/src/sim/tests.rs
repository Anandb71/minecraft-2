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

/// Mean water depth over columns x in `x0..=x1` of a channel 1..=4 wide.
fn mean_depth(w: &FluidWorld, x0: i32, x1: i32) -> f32 {
    let mut h = 0.0;
    let mut n = 0.0;
    for z in 1..=4 {
        for x in x0..=x1 {
            h += column_height(w, x, z);
            n += 1.0;
        }
    }
    h / n
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
fn pressure_pushes_water_through_a_pipe_to_one_level() {
    // A channel 4 cells wide split by a wall with a one-cell pipe along
    // its foot. The left side holds 8 cells of water; the right is dry.
    let mut wall = Vec::new();
    for z in 1..=4 {
        for y in 2..=12 {
            wall.push(IVec3::new(8, y, z));
        }
    }
    let terrain = BoxTerrain {
        size: IVec3::new(17, 0, 5),
        extra: wall,
    };
    let mut w = FluidWorld::default();
    w.add_water(cells(IVec3::new(1, 1, 1), IVec3::new(7, 8, 4)), &terrain);
    let m0 = w.mass();
    let (mut left, mut right) = (0.0, 0.0);
    // Twenty seconds, levels averaged over the last two.
    for s in 0..4800 {
        w.step(&terrain);
        if s >= 4320 {
            left += mean_depth(&w, 1, 7) / 480.0;
            right += mean_depth(&w, 9, 16) / 480.0;
        }
    }
    let m1 = w.mass() + w.stats.lost_mass;
    assert!((m1 - m0).abs() / m0 < 0.02, "mass {m0} -> {m1}");
    // 224 cells over 15 columns and the pipe: about 3.7 each side.
    assert!(right > 3.0, "only {right} came through");
    assert!((left - right).abs() < 0.3, "left {left} right {right}");
}

#[test]
fn a_breached_dam_floods_downstream() {
    let dam: Vec<IVec3> = (1..=4)
        .flat_map(|z| (1..=12).map(move |y| IVec3::new(8, y, z)))
        .collect();
    let mut terrain = BoxTerrain {
        size: IVec3::new(41, 0, 5),
        extra: dam,
    };
    let mut w = FluidWorld::default();
    w.add_water(cells(IVec3::new(1, 1, 1), IVec3::new(7, 10, 4)), &terrain);
    let m0 = w.mass();
    // The dam holds.
    for _ in 0..240 {
        w.step(&terrain);
    }
    assert!(mean_depth(&w, 9, 40) < 0.01);
    // Breach it.
    terrain.extra.clear();
    w.terrain_changed(IVec3::new(8, 1, 1), IVec3::new(8, 12, 4), &terrain);
    let mut reached = None;
    for s in 0..2400 {
        w.step(&terrain);
        if reached.is_none() && mean_depth(&w, 38, 40) > 0.3 {
            reached = Some(s);
        }
    }
    let m1 = w.mass() + w.stats.lost_mass;
    assert!((m1 - m0).abs() / m0 < 0.02, "mass {m0} -> {m1}");
    // A 5 m head runs 16 m of channel in a few seconds.
    let s = reached.expect("the flood never reached the end of the valley");
    assert!(s < 1200, "took {} s", s as f64 * STEP_S);
    assert!(mean_depth(&w, 20, 40) > 1.0);
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
