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
}
