//! Kinematic plates. Erosion still moves the sediment; this only says where
//! the crust stood before the water arrived.
//!
//! A handful of plates, each with a seeded velocity. Where two meet, the
//! relative motion picks the landform: a ridged range if they collide, a
//! rift if they pull apart, a scarp if they slide.

use crate::noise::{Perlin, hash_unit, hash3};
use glam::Vec2;

const PLATE_M: f32 = 5200.0;

#[derive(Clone, Copy)]
struct Plate {
    pos: Vec2,
    vel: Vec2,
    dist: f32,
}

fn nearest_two(x: f32, z: f32, seed: u64) -> (Plate, Plate) {
    let (cx, cz) = ((x / PLATE_M).floor() as i32, (z / PLATE_M).floor() as i32);
    let far = Plate {
        pos: Vec2::ZERO,
        vel: Vec2::ZERO,
        dist: f32::MAX,
    };
    let mut best = [far, far];
    for dz in -1..=1 {
        for dx in -1..=1 {
            let (gx, gz) = (cx + dx, cz + dz);
            let h = hash3(gx, gz, 0, seed ^ 0x71a7_e011);
            let pos = Vec2::new(
                (gx as f32 + hash_unit(h)) * PLATE_M,
                (gz as f32 + hash_unit(h.rotate_left(21))) * PLATE_M,
            );
            let dist = pos.distance(Vec2::new(x, z));
            let ang = hash_unit(h.rotate_left(40)) * std::f32::consts::TAU;
            let speed = 0.45 + 0.55 * hash_unit(h.rotate_left(11));
            let plate = Plate {
                pos,
                vel: Vec2::new(ang.cos(), ang.sin()) * speed,
                dist,
            };
            if dist < best[0].dist {
                best[1] = best[0];
                best[0] = plate;
            } else if dist < best[1].dist {
                best[1] = plate;
            }
        }
    }
    (best[0], best[1])
}

/// Positive when `b` moves toward `a`. `n` points from `a`'s plate to `b`'s.
pub fn convergence(a: Vec2, b: Vec2, n: Vec2) -> f32 {
    -(b - a).dot(n)
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Metres added to the pre-erosion height. `interior` is 1 on a continent
/// and 0 offshore, so a collision does not raise a range out of the sea.
pub fn uplift(noise: &Perlin, seed: u64, x: f32, z: f32, interior: f32) -> f32 {
    let (a, b) = nearest_two(x, z, seed);
    let n = (b.pos - a.pos).normalize_or_zero();
    if n.length_squared() == 0.0 {
        return 0.0;
    }
    let conv = (convergence(a.vel, b.vel, n) / 1.15).clamp(-1.0, 1.0);
    let edge = (b.dist - a.dist) * 0.5;
    let here = Vec2::new(x, z);
    let along = here.dot(Vec2::new(-n.y, n.x));
    let across = here.dot(n);
    if conv > 0.18 {
        let band = smooth(1.0 - edge / 2200.0);
        let ridges = noise.ridged2(along / 2600.0, across / 720.0, 5, 2.05, 0.48);
        return band * conv * interior * ridges.powf(1.25) * 420.0;
    }
    if conv < -0.18 {
        let band = smooth(1.0 - edge / 1500.0);
        return -band * (-conv) * interior.max(0.35) * 95.0;
    }
    let band = smooth(1.0 - edge / 500.0);
    let step = (across / 220.0).clamp(-1.0, 1.0);
    step * band * 32.0 * interior
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_on_plates_converge_and_parting_ones_rift() {
        let n = Vec2::X;
        let together = convergence(Vec2::new(0.6, 0.0), Vec2::new(-0.4, 0.0), n);
        let apart = convergence(Vec2::new(-0.6, 0.0), Vec2::new(0.4, 0.0), n);
        assert!(together > 0.5, "{together}");
        assert!(apart < -0.5, "{apart}");
    }

    #[test]
    fn uplift_is_the_seed_and_both_landforms_occur() {
        let noise = Perlin::new(4);
        assert_eq!(
            uplift(&noise, 4, 4000.0, 2500.0, 1.0),
            uplift(&Perlin::new(4), 4, 4000.0, 2500.0, 1.0)
        );
        let mut peak = 0.0f32;
        let mut trench = 0.0f32;
        for i in 0..40 {
            for j in 0..40 {
                let u = uplift(
                    &noise,
                    4,
                    400.0 + i as f32 * 400.0,
                    400.0 + j as f32 * 400.0,
                    1.0,
                );
                peak = peak.max(u);
                trench = trench.min(u);
            }
        }
        assert!(peak > 80.0, "no convergent range: {peak}");
        assert!(trench < -20.0, "no rift: {trench}");
    }
}
