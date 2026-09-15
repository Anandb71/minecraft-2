//! Seeded gradient noise, fractal sums, cellular noise and domain warping.
//!
//! Perlin's improved noise (2002): quintic fade, gradients from a seeded
//! permutation. Worley noise returns the distance to the nearest feature
//! point, used for fracture networks and cell patterns. Everything is
//! deterministic per seed and identical on every platform.

#[derive(Clone)]
pub struct Perlin {
    perm: [u8; 512],
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Hash of integer lattice coordinates and a seed, well mixed.
#[inline]
pub fn hash3(x: i32, y: i32, z: i32, seed: u64) -> u64 {
    let mut s = seed
        ^ (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (y as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
        ^ (z as u64).wrapping_mul(0x1656_67b1_9e37_79f9);
    splitmix(&mut s)
}

#[inline]
pub fn hash_unit(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn grad3(h: u8, x: f32, y: f32, z: f32) -> f32 {
    match h & 15 {
        0 | 12 => x + y,
        1 | 14 => -x + y,
        2 => x - y,
        3 => -x - y,
        4 => x + z,
        5 => -x + z,
        6 => x - z,
        7 => -x - z,
        8 => y + z,
        9 | 13 => -y + z,
        10 => y - z,
        _ => -y - z,
    }
}

#[inline]
fn grad2(h: u8, x: f32, y: f32) -> f32 {
    match h & 7 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        3 => -x - y,
        4 => x,
        5 => -x,
        6 => y,
        _ => -y,
    }
}

impl Perlin {
    pub fn new(seed: u64) -> Self {
        let mut p: [u8; 256] = std::array::from_fn(|i| i as u8);
        let mut s = seed;
        for i in (1..256).rev() {
            let j = (splitmix(&mut s) % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for (i, slot) in perm.iter_mut().enumerate() {
            *slot = p[i & 255];
        }
        Self { perm }
    }

    #[inline]
    fn p(&self, i: i32) -> i32 {
        i32::from(self.perm[(i & 511) as usize])
    }

    /// Range roughly [-1, 1].
    pub fn noise2(&self, x: f32, y: f32) -> f32 {
        let (xf, yf) = (x.floor(), y.floor());
        let (xi, yi) = (xf as i32 & 255, yf as i32 & 255);
        let (x, y) = (x - xf, y - yf);
        let (u, v) = (fade(x), fade(y));
        let a = self.p(xi) + yi;
        let b = self.p(xi + 1) + yi;
        let g = |i: i32| self.perm[(i & 511) as usize];
        lerp(
            lerp(grad2(g(a), x, y), grad2(g(b), x - 1.0, y), u),
            lerp(
                grad2(g(a + 1), x, y - 1.0),
                grad2(g(b + 1), x - 1.0, y - 1.0),
                u,
            ),
            v,
        )
    }

    /// Range roughly [-1, 1].
    pub fn noise3(&self, x: f32, y: f32, z: f32) -> f32 {
        let (xf, yf, zf) = (x.floor(), y.floor(), z.floor());
        let (xi, yi, zi) = (xf as i32 & 255, yf as i32 & 255, zf as i32 & 255);
        let (x, y, z) = (x - xf, y - yf, z - zf);
        let (u, v, w) = (fade(x), fade(y), fade(z));
        let a = self.p(xi) + yi;
        let aa = self.p(a) + zi;
        let ab = self.p(a + 1) + zi;
        let b = self.p(xi + 1) + yi;
        let ba = self.p(b) + zi;
        let bb = self.p(b + 1) + zi;
        let g = |i: i32| self.perm[(i & 511) as usize];
        lerp(
            lerp(
                lerp(grad3(g(aa), x, y, z), grad3(g(ba), x - 1.0, y, z), u),
                lerp(
                    grad3(g(ab), x, y - 1.0, z),
                    grad3(g(bb), x - 1.0, y - 1.0, z),
                    u,
                ),
                v,
            ),
            lerp(
                lerp(
                    grad3(g(aa + 1), x, y, z - 1.0),
                    grad3(g(ba + 1), x - 1.0, y, z - 1.0),
                    u,
                ),
                lerp(
                    grad3(g(ab + 1), x, y - 1.0, z - 1.0),
                    grad3(g(bb + 1), x - 1.0, y - 1.0, z - 1.0),
                    u,
                ),
                v,
            ),
            w,
        )
    }

    /// Fractal Brownian motion, normalised to roughly [-1, 1].
    pub fn fbm2(&self, x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
        for o in 0..octaves {
            // Offset octaves so their lattices do not align at the origin.
            let off = o as f32 * 17.13;
            sum += amp * self.noise2(x * freq + off, y * freq - off);
            norm += amp;
            amp *= gain;
            freq *= lacunarity;
        }
        sum / norm
    }

    pub fn fbm3(&self, x: f32, y: f32, z: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
        for o in 0..octaves {
            let off = o as f32 * 17.13;
            sum += amp * self.noise3(x * freq + off, y * freq, z * freq - off);
            norm += amp;
            amp *= gain;
            freq *= lacunarity;
        }
        sum / norm
    }

    /// Ridged multifractal (Musgrave): sharp crests where noise crosses zero,
    /// each octave weighted by the previous one's ridge. Range [0, 1].
    pub fn ridged2(&self, x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let (mut sum, mut amp, mut freq, mut norm, mut weight) = (0.0, 1.0, 1.0, 0.0, 1.0);
        for o in 0..octaves {
            let off = o as f32 * 31.7;
            let n = 1.0 - self.noise2(x * freq + off, y * freq + off).abs();
            let n = n * n * weight;
            weight = (n * 2.0).clamp(0.0, 1.0);
            sum += amp * n;
            norm += amp;
            amp *= gain;
            freq *= lacunarity;
        }
        sum / norm
    }
}

/// Distances to the nearest and second nearest feature point of a 3D
/// Worley lattice (one jittered point per unit cell).
pub fn worley3(x: f32, y: f32, z: f32, seed: u64) -> (f32, f32) {
    let (cx, cy, cz) = (x.floor() as i32, y.floor() as i32, z.floor() as i32);
    let (mut d1, mut d2) = (f32::MAX, f32::MAX);
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (gx, gy, gz) = (cx + dx, cy + dy, cz + dz);
                let h = hash3(gx, gy, gz, seed);
                let px = gx as f32 + hash_unit(h);
                let py = gy as f32 + hash_unit(h.rotate_left(21));
                let pz = gz as f32 + hash_unit(h.rotate_left(42));
                let d = (px - x) * (px - x) + (py - y) * (py - y) + (pz - z) * (pz - z);
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                } else if d < d2 {
                    d2 = d;
                }
            }
        }
    }
    (d1.sqrt(), d2.sqrt())
}

/// 2D Worley: nearest and second nearest distances.
pub fn worley2(x: f32, y: f32, seed: u64) -> (f32, f32) {
    let (cx, cy) = (x.floor() as i32, y.floor() as i32);
    let (mut d1, mut d2) = (f32::MAX, f32::MAX);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (gx, gy) = (cx + dx, cy + dy);
            let h = hash3(gx, gy, 0, seed);
            let px = gx as f32 + hash_unit(h);
            let py = gy as f32 + hash_unit(h.rotate_left(21));
            let d = (px - x) * (px - x) + (py - y) * (py - y);
            if d < d1 {
                d2 = d1;
                d1 = d;
            } else if d < d2 {
                d2 = d;
            }
        }
    }
    (d1.sqrt(), d2.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_deterministic_and_seeded() {
        let a = Perlin::new(1);
        let b = Perlin::new(1);
        let c = Perlin::new(2);
        let p = (12.34, -5.67, 8.9);
        assert_eq!(a.noise3(p.0, p.1, p.2), b.noise3(p.0, p.1, p.2));
        assert_ne!(a.noise3(p.0, p.1, p.2), c.noise3(p.0, p.1, p.2));
    }

    #[test]
    fn noise_is_zero_on_lattice_and_bounded() {
        let n = Perlin::new(42);
        assert_eq!(n.noise3(3.0, -4.0, 7.0), 0.0);
        let mut max: f32 = 0.0;
        for i in 0..20000 {
            let t = i as f32 * 0.137;
            let v = n.noise3(t, t * 0.71 - 3.0, t * 1.37 + 1.0);
            max = max.max(v.abs());
            assert!(v.abs() <= 1.1);
        }
        assert!(max > 0.5, "noise has no amplitude: {max}");
    }

    #[test]
    fn noise_is_continuous() {
        let n = Perlin::new(9);
        let mut prev = n.noise2(0.0, 0.5);
        for i in 1..10000 {
            let v = n.noise2(i as f32 * 0.001, 0.5);
            assert!((v - prev).abs() < 0.01, "jump at {i}");
            prev = v;
        }
    }

    #[test]
    fn ridged_and_worley_ranges() {
        let n = Perlin::new(3);
        for i in 0..1000 {
            let t = i as f32 * 0.31;
            let r = n.ridged2(t, -t, 5, 2.0, 0.5);
            assert!((0.0..=1.0).contains(&r));
            let (d1, d2) = worley3(t, t * 0.5, -t, 7);
            assert!(d1 <= d2 && d1 < 1.8);
        }
    }
}
