//! Hydraulic erosion after Mei, Decaudin and Hu (2007): a virtual-pipe
//! shallow water model moves rain across the terrain; flowing water
//! dissolves rock up to a transport capacity proportional to local tilt and
//! speed, carries it semi-Lagrangian along the velocity field and deposits it
//! where it slows. Thermal erosion collapses slopes steeper than a talus angle.
//!
//! Additions for this world: dissolving rate scales with the hardness of the
//! stratum at the current surface, so soft shale retreats under hard
//! sandstone caps and canyon walls expose their layers; cells below sea level
//! are clamped to the sea surface, so rivers terminate in deltas.
//!
//! Every pass reads the previous state and writes a new one (Jacobi), so rows
//! run in parallel and the result is identical for any thread count.

use crate::grid::Grid2;
use crate::strata::Strata;
use crate::terrain::TerrainParams;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct ErosionParams {
    pub iterations: u32,
    pub dt: f32,
    /// Rain added per iteration, cell units.
    pub rain: f32,
    pub pipe_area: f32,
    pub gravity: f32,
    /// Sediment capacity constant `Kc`.
    pub capacity: f32,
    /// Dissolving constant `Ks` for rock of hardness 1.
    pub dissolve: f32,
    /// Deposition constant `Kd`.
    pub deposit: f32,
    /// Evaporation constant `Ke`.
    pub evaporation: f32,
    pub min_tilt: f32,
    /// Talus slope (rise over run) for thermal erosion.
    pub talus: f32,
    pub thermal_rate: f32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        Self {
            iterations: 600,
            dt: 0.02,
            rain: 0.012,
            pipe_area: 20.0,
            gravity: 9.81,
            capacity: 1.2,
            dissolve: 0.3,
            deposit: 0.3,
            evaporation: 0.5,
            min_tilt: 0.02,
            talus: 1.1,
            thermal_rate: 0.15,
        }
    }
}

/// Cells per unit time; well below the semi-Lagrangian step's reach.
const MAX_SPEED: f32 = 8.0;

pub struct ErosionOutput {
    pub height: Grid2,
    pub flow: Grid2,
    pub sediment: Grid2,
}

/// Maps `f(x, y)` over a `w`-wide grid in parallel rows.
fn par_fill(out: &mut [f32], w: usize, f: impl Fn(usize, usize) -> f32 + Sync) {
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            *v = f(x, y);
        }
    });
}

struct State {
    w: usize,
    h: usize,
    /// Terrain, water, sediment in cell units (metres / cell size).
    b: Vec<f32>,
    d: Vec<f32>,
    s: Vec<f32>,
    /// Outflow flux left, right, down (-y), up (+y).
    fl: Vec<f32>,
    fr: Vec<f32>,
    fd: Vec<f32>,
    fu: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    erodibility: Vec<f32>,
    flow: Vec<f32>,
    sea: f32,
}

impl State {
    #[inline]
    fn i(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }

    #[inline]
    fn clamp_xy(&self, x: i64, y: i64) -> usize {
        let xc = x.clamp(0, self.w as i64 - 1) as usize;
        let yc = y.clamp(0, self.h as i64 - 1) as usize;
        yc * self.w + xc
    }

    fn sample(&self, field: &[f32], x: f32, y: f32) -> f32 {
        let (xf, yf) = (x.floor(), y.floor());
        let (tx, ty) = (x - xf, y - yf);
        let (xi, yi) = (xf as i64, yf as i64);
        let g = |a: i64, b: i64| field[self.clamp_xy(a, b)];
        let top = g(xi, yi) + (g(xi + 1, yi) - g(xi, yi)) * tx;
        let bot = g(xi, yi + 1) + (g(xi + 1, yi + 1) - g(xi, yi + 1)) * tx;
        top + (bot - top) * ty
    }
}

pub fn erode(
    height: Grid2,
    strata: &Strata,
    params: &TerrainParams,
    progress: &mut dyn FnMut(&str, f32),
) -> ErosionOutput {
    let p = params.erosion;
    let (w, h) = (height.w, height.h);
    let cell = params.cell_m;
    let n = w * h;
    let mut st = State {
        w,
        h,
        b: height.data.iter().map(|v| v / cell).collect(),
        d: vec![0.0; n],
        s: vec![0.0; n],
        fl: vec![0.0; n],
        fr: vec![0.0; n],
        fd: vec![0.0; n],
        fu: vec![0.0; n],
        u: vec![0.0; n],
        v: vec![0.0; n],
        erodibility: vec![1.0; n],
        flow: vec![0.0; n],
        sea: params.sea_level / cell,
    };
    let original: Vec<f32> = st.b.clone();
    let mut scratch = vec![0.0f32; n];
    let mut scratch2 = vec![0.0f32; n];
    let mut spare = Spare::new(n);

    for it in 0..p.iterations {
        if it % 25 == 0 {
            progress("erosion", it as f32 / p.iterations as f32);
            // Surface rock hardness changes as layers are stripped.
            let b = &st.b;
            par_fill(&mut st.erodibility, w, |x, y| {
                let wx = (x as f32 + 0.5) * cell;
                let wz = (y as f32 + 0.5) * cell;
                let hardness = strata.hardness(wx, b[y * w + x] * cell - 1.0, wz);
                (2.0 / (1.0 + hardness)).clamp(0.05, 2.0)
            });
        }
        step(&mut st, &p, &mut scratch, &mut scratch2, &mut spare);
        if it % 4 == 0 {
            thermal(&mut st, &p, &mut scratch);
        }
    }
    progress("erosion", 1.0);

    let height = Grid2 {
        w,
        h,
        data: st.b.iter().map(|v| (v * cell).max(2.0)).collect(),
    };
    let flow = Grid2 {
        w,
        h,
        data: st.flow.iter().map(|f| f / p.iterations as f32).collect(),
    };
    let sediment = Grid2 {
        w,
        h,
        data: st
            .b
            .iter()
            .zip(&original)
            .map(|(now, was)| ((now - was) * cell).max(0.0))
            .collect(),
    };
    ErosionOutput {
        height,
        flow,
        sediment,
    }
}

/// Second copies of every field a pass rewrites, swapped in after the pass.
struct Spare {
    fl: Vec<f32>,
    fr: Vec<f32>,
    fd: Vec<f32>,
    fu: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    b: Vec<f32>,
}

impl Spare {
    fn new(n: usize) -> Self {
        Self {
            fl: vec![0.0; n],
            fr: vec![0.0; n],
            fd: vec![0.0; n],
            fu: vec![0.0; n],
            u: vec![0.0; n],
            v: vec![0.0; n],
            b: vec![0.0; n],
        }
    }
}

fn step(
    st: &mut State,
    p: &ErosionParams,
    scratch: &mut [f32],
    scratch2: &mut [f32],
    spare: &mut Spare,
) {
    let (w, h) = (st.w, st.h);

    // 1. Rain.
    st.d.par_iter_mut().for_each(|d| *d += p.rain);

    // 2. Outflow flux. Each direction reads last iteration's flux.
    let k = p.dt * p.pipe_area * p.gravity;
    {
        let (b, d) = (&st.b, &st.d);
        let surface = |i: usize| b[i] + d[i];
        let flux = |old: &[f32], out: &mut [f32], dx: i64, dy: i64| {
            out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                for (x, f) in row.iter_mut().enumerate() {
                    let i = y * w + x;
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        *f = 0.0;
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    *f = (old[i] + k * (surface(i) - surface(j))).max(0.0);
                }
            });
        };
        let (nl, nr, nd, nu) = (&mut spare.fl, &mut spare.fr, &mut spare.fd, &mut spare.fu);
        flux(&st.fl, nl, -1, 0);
        flux(&st.fr, nr, 1, 0);
        flux(&st.fd, nd, 0, -1);
        flux(&st.fu, nu, 0, 1);
        // Scale so a cell never sends more water than it holds.
        let dt = p.dt;
        (
            nl.par_iter_mut(),
            nr.par_iter_mut(),
            nd.par_iter_mut(),
            nu.par_iter_mut(),
            d.par_iter(),
        )
            .into_par_iter()
            .for_each(|(l, r, dn, up, &water)| {
                let total = (*l + *r + *dn + *up) * dt;
                if total > water && total > 0.0 {
                    let s = water / total;
                    *l *= s;
                    *r *= s;
                    *dn *= s;
                    *up *= s;
                }
            });
        std::mem::swap(&mut st.fl, &mut spare.fl);
        std::mem::swap(&mut st.fr, &mut spare.fr);
        std::mem::swap(&mut st.fd, &mut spare.fd);
        std::mem::swap(&mut st.fu, &mut spare.fu);
    }

    // 3. Water surface and velocity.
    {
        let s = &*st;
        par_fill(scratch, w, |x, y| {
            let i = s.i(x, y);
            let mut inflow = 0.0;
            if x > 0 {
                inflow += s.fr[i - 1];
            }
            if x + 1 < w {
                inflow += s.fl[i + 1];
            }
            if y > 0 {
                inflow += s.fu[i - w];
            }
            if y + 1 < h {
                inflow += s.fd[i + w];
            }
            let outflow = s.fl[i] + s.fr[i] + s.fd[i] + s.fu[i];
            (s.d[i] + p.dt * (inflow - outflow)).max(0.0)
        });
        let d_new = &*scratch;
        spare
            .u
            .par_chunks_mut(w)
            .zip(spare.v.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (urow, vrow))| {
                for x in 0..w {
                    let i = s.i(x, y);
                    let left_in = if x > 0 { s.fr[i - 1] } else { 0.0 };
                    let right_in = if x + 1 < w { s.fl[i + 1] } else { 0.0 };
                    let down_in = if y > 0 { s.fu[i - w] } else { 0.0 };
                    let up_in = if y + 1 < h { s.fd[i + w] } else { 0.0 };
                    let wx = (left_in - s.fl[i] + s.fr[i] - right_in) * 0.5;
                    let wy = (down_in - s.fd[i] + s.fu[i] - up_in) * 0.5;
                    // A film of water would otherwise report enormous speeds
                    // and cut grid-aligned grooves.
                    let depth = ((s.d[i] + d_new[i]) * 0.5).max(0.02);
                    let (ux, vy) = (wx / depth, wy / depth);
                    let speed = (ux * ux + vy * vy).sqrt();
                    let limit = if speed > MAX_SPEED {
                        MAX_SPEED / speed
                    } else {
                        1.0
                    };
                    urow[x] = ux * limit;
                    vrow[x] = vy * limit;
                }
            });
        st.d.copy_from_slice(scratch);
        std::mem::swap(&mut st.u, &mut spare.u);
        std::mem::swap(&mut st.v, &mut spare.v);
    }

    // 4. Erosion and deposition against transport capacity.
    {
        let s = &*st;
        spare
            .b
            .par_chunks_mut(w)
            .zip(scratch.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (brow, srow))| {
                for x in 0..w {
                    let i = s.i(x, y);
                    let (xi, yi) = (x as i64, y as i64);
                    let gx = (s.b[s.clamp_xy(xi + 1, yi)] - s.b[s.clamp_xy(xi - 1, yi)]) * 0.5;
                    let gy = (s.b[s.clamp_xy(xi, yi + 1)] - s.b[s.clamp_xy(xi, yi - 1)]) * 0.5;
                    let g2 = gx * gx + gy * gy;
                    let sin_tilt = (g2 / (1.0 + g2)).sqrt().max(p.min_tilt);
                    let speed = (s.u[i] * s.u[i] + s.v[i] * s.v[i]).sqrt();
                    let capacity = p.capacity * sin_tilt * speed * s.d[i].clamp(0.0, 1.0);
                    let sed = s.s[i];
                    // The sea bed only receives sediment; rivers end in deltas.
                    if capacity > sed && s.b[i] >= s.sea {
                        let amount =
                            (p.dissolve * s.erodibility[i] * (capacity - sed) * p.dt).min(0.05);
                        brow[x] = s.b[i] - amount;
                        srow[x] = sed + amount;
                    } else {
                        let amount = p.deposit * (sed - capacity) * p.dt;
                        brow[x] = s.b[i] + amount;
                        srow[x] = sed - amount;
                    }
                }
            });
        std::mem::swap(&mut st.b, &mut spare.b);
        st.s.copy_from_slice(scratch);
    }

    // 5. Sediment transport, semi-Lagrangian.
    {
        let s = &*st;
        par_fill(scratch2, w, |x, y| {
            let i = s.i(x, y);
            s.sample(&s.s, x as f32 - s.u[i] * p.dt, y as f32 - s.v[i] * p.dt)
        });
        st.s.copy_from_slice(scratch2);
    }

    // 6. Evaporation, the sea as a fixed reservoir, and discharge statistics.
    let sea = st.sea;
    let evap = 1.0 - p.evaporation * p.dt;
    (
        st.d.par_iter_mut(),
        st.b.par_iter(),
        st.flow.par_iter_mut(),
        st.u.par_iter(),
        st.v.par_iter(),
    )
        .into_par_iter()
        .for_each(|(d, &b, flow, &u, &v)| {
            *d *= evap;
            if b < sea {
                *d = sea - b;
            } else {
                *flow += (u * u + v * v).sqrt() * *d;
            }
        });
}

fn thermal(st: &mut State, p: &ErosionParams, scratch: &mut [f32]) {
    let (w, h) = (st.w, st.h);
    let s = &*st;
    // Net change per cell: material slides to lower neighbours past the talus
    // slope, and arrives from higher ones. Symmetric, so mass is conserved.
    par_fill(scratch, w, |x, y| {
        let i = s.i(x, y);
        let mut delta = 0.0;
        for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
            if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            let talus = p.talus * (2.0 / (s.erodibility[i] + s.erodibility[j])).sqrt();
            let diff = s.b[i] - s.b[j];
            if diff > talus {
                delta -= (diff - talus) * p.thermal_rate * 0.25;
            } else if -diff > talus {
                delta += (-diff - talus) * p.thermal_rate * 0.25;
            }
        }
        delta
    });
    st.b.par_iter_mut()
        .zip(scratch.par_iter())
        .for_each(|(b, d)| *b += d);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainParams;

    fn small_params(iterations: u32) -> TerrainParams {
        TerrainParams {
            size: 96,
            cell_m: 16.0,
            sea_level: 20.0,
            erosion: ErosionParams {
                iterations,
                ..Default::default()
            },
        }
    }

    fn cone(size: usize) -> Grid2 {
        Grid2::from_fn(size, size, |x, y| {
            let dx = x as f32 - size as f32 / 2.0;
            let dy = y as f32 - size as f32 / 2.0;
            (300.0 - (dx * dx + dy * dy).sqrt() * 6.0).max(30.0)
                + ((x * 7 + y * 13) % 5) as f32 * 0.5
        })
    }

    #[test]
    fn erosion_is_deterministic() {
        let strata = Strata::new(1);
        let p = small_params(40);
        let a = erode(cone(96), &strata, &p, &mut |_, _| {});
        let b = erode(cone(96), &strata, &p, &mut |_, _| {});
        assert_eq!(a.height, b.height);
    }

    #[test]
    fn erosion_lowers_peaks_and_fills_valleys_without_blowing_up() {
        let strata = Strata::new(1);
        let p = small_params(200);
        let before = cone(96);
        let (lo0, hi0) = before.min_max();
        let out = erode(before.clone(), &strata, &p, &mut |_, _| {});
        let (lo, hi) = out.height.min_max();
        let high_mean = |g: &Grid2| {
            let v: Vec<f32> = g
                .data
                .iter()
                .zip(&before.data)
                .filter(|(_, b)| **b > 200.0)
                .map(|(a, _)| *a)
                .collect();
            v.iter().sum::<f32>() / v.len() as f32
        };
        assert!(
            high_mean(&out.height) < high_mean(&before),
            "upland did not erode"
        );
        assert!(hi < hi0 + 5.0, "peak {hi} vs {hi0}");
        assert!(lo >= lo0 - 50.0 && hi.is_finite());
        let moved: f32 = out
            .height
            .data
            .iter()
            .zip(&before.data)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(moved > 100.0, "erosion did nothing: {moved}");
        assert!(out.flow.data.iter().all(|f| f.is_finite() && *f >= 0.0));
    }

    #[test]
    fn thermal_erosion_conserves_mass() {
        let mut p = small_params(0);
        p.erosion.thermal_rate = 0.5;
        let g = cone(48);
        let mut st = State {
            w: 48,
            h: 48,
            b: g.data.iter().map(|v| v / 16.0).collect(),
            d: vec![0.0; 48 * 48],
            s: vec![0.0; 48 * 48],
            fl: vec![0.0; 48 * 48],
            fr: vec![0.0; 48 * 48],
            fd: vec![0.0; 48 * 48],
            fu: vec![0.0; 48 * 48],
            u: vec![0.0; 48 * 48],
            v: vec![0.0; 48 * 48],
            erodibility: vec![1.0; 48 * 48],
            flow: vec![0.0; 48 * 48],
            sea: 0.0,
        };
        let before: f64 = st.b.iter().map(|&v| f64::from(v)).sum();
        let mut scratch = vec![0.0; 48 * 48];
        p.erosion.talus = 0.1;
        for _ in 0..20 {
            thermal(&mut st, &p.erosion, &mut scratch);
        }
        let after: f64 = st.b.iter().map(|&v| f64::from(v)).sum();
        assert!(
            (before - after).abs() < 1e-2 * before.abs().max(1.0),
            "{before} vs {after}"
        );
    }
}
