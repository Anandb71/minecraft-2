//! The D3Q19 lattice in lattice units (cell size 1, time step 1): the 19
//! velocities, their weights, the equilibrium, and Guo forcing.

use glam::Vec3;

pub const Q: usize = 19;

/// Lattice velocities: rest, the 6 faces, then the 12 edges, each followed
/// by its opposite.
pub const C: [[i32; 3]; Q] = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, -1, -1],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, -1],
    [0, -1, 1],
];

pub const W: [f32; Q] = [
    1.0 / 3.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// Index of the opposite velocity.
#[inline]
pub const fn opp(i: usize) -> usize {
    if i == 0 {
        0
    } else if i % 2 == 1 {
        i + 1
    } else {
        i - 1
    }
}

#[inline]
pub fn c(i: usize) -> Vec3 {
    Vec3::new(C[i][0] as f32, C[i][1] as f32, C[i][2] as f32)
}

/// Equilibrium population `i` at density `rho` and velocity `u`
/// (Maxwell-Boltzmann to second order, c_s^2 = 1/3).
#[inline]
pub fn equilibrium(i: usize, rho: f32, u: Vec3) -> f32 {
    let cu = c(i).dot(u);
    W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u.dot(u))
}

/// Guo et al. forcing term for population `i`, relaxation time `tau`,
/// velocity `u` and force density `force`.
#[inline]
pub fn guo(i: usize, tau: f32, u: Vec3, force: Vec3) -> f32 {
    let ci = c(i);
    (1.0 - 0.5 / tau) * W[i] * (3.0 * (ci - u) + 9.0 * ci.dot(u) * ci).dot(force)
}

/// Density and velocity of a cell's populations under force density
/// `force` (the velocity carries half the force, as Guo prescribes).
pub fn moments(f: &[f32; Q], force: Vec3) -> (f32, Vec3) {
    let mut rho = 0.0;
    let mut j = Vec3::ZERO;
    for (i, &fi) in f.iter().enumerate() {
        rho += fi;
        j += c(i) * fi;
    }
    let u = if rho > 1e-6 {
        (j + 0.5 * force) / rho
    } else {
        Vec3::ZERO
    };
    (rho, u)
}

/// Relaxation time with a Smagorinsky subgrid term: the strain carried by
/// the non-equilibrium part of the populations raises the local viscosity,
/// which keeps nearly inviscid water stable.
pub fn les_tau(f: &[f32; Q], rho: f32, u: Vec3, tau0: f32, smagorinsky: f32) -> f32 {
    let mut pi = [0.0f32; 6];
    for (i, &fi) in f.iter().enumerate() {
        let neq = fi - equilibrium(i, rho, u);
        let ci = c(i);
        pi[0] += ci.x * ci.x * neq;
        pi[1] += ci.y * ci.y * neq;
        pi[2] += ci.z * ci.z * neq;
        pi[3] += ci.x * ci.y * neq;
        pi[4] += ci.x * ci.z * neq;
        pi[5] += ci.y * ci.z * neq;
    }
    let q = (pi[0] * pi[0]
        + pi[1] * pi[1]
        + pi[2] * pi[2]
        + 2.0 * (pi[3] * pi[3] + pi[4] * pi[4] + pi[5] * pi[5]))
        .sqrt();
    let c2 = smagorinsky * smagorinsky;
    0.5 * (tau0 + (tau0 * tau0 + 18.0 * std::f32::consts::SQRT_2 * c2 * q / rho.max(1e-6)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_is_symmetric_and_normalised() {
        let sum: f32 = W.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for i in 0..Q {
            let (a, b) = (c(i), c(opp(i)));
            assert_eq!(a, -b, "{i}");
            assert_eq!(W[i], W[opp(i)]);
        }
        // Second moment of the weights is c_s^2 = 1/3 per axis.
        let mut m = Vec3::ZERO;
        for (i, w) in W.iter().enumerate() {
            m += w * c(i) * c(i);
        }
        assert!((m - Vec3::splat(1.0 / 3.0)).length() < 1e-6);
    }

    #[test]
    fn equilibrium_moments_recover_density_and_velocity() {
        let (rho, u) = (1.07, Vec3::new(0.03, -0.02, 0.05));
        let mut f = [0.0; Q];
        for (i, fi) in f.iter_mut().enumerate() {
            *fi = equilibrium(i, rho, u);
        }
        let (r, v) = moments(&f, Vec3::ZERO);
        assert!((r - rho).abs() < 1e-5);
        assert!((v - u).length() < 1e-5);
        // At equilibrium the subgrid term adds nothing.
        assert!((les_tau(&f, rho, u, 0.51, 0.127) - 0.51).abs() < 1e-5);
    }

    #[test]
    fn guo_forcing_adds_momentum_but_no_mass() {
        let force = Vec3::new(0.0, -1e-3, 0.0);
        let u = Vec3::new(0.01, 0.0, 0.0);
        let (mut mass, mut momentum) = (0.0, Vec3::ZERO);
        for i in 0..Q {
            let g = guo(i, 0.6, u, force);
            mass += g;
            momentum += c(i) * g;
        }
        assert!(mass.abs() < 1e-8);
        let expect = (1.0 - 0.5 / 0.6) * force;
        assert!((momentum - expect).length() < 1e-8, "{momentum}");
    }
}
