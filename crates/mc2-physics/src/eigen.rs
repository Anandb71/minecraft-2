//! Symmetric 3x3 eigen decomposition, for principal axes of inertia.

use glam::{Mat3, Quat, Vec3};

/// Eigen decomposition of a symmetric 3x3 matrix by cyclic Jacobi
/// rotations: eigenvalues and the rotation whose columns are eigenvectors.
pub fn eigen_symmetric(m: [[f64; 3]; 3]) -> (Vec3, Quat) {
    let mut a = m;
    let mut v = [[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-14 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-18 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // A <- J^T A J
            for row in a.iter_mut() {
                let (akp, akq) = (row[p], row[q]);
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            let (ap, aq) = (a[p], a[q]);
            for k in 0..3 {
                a[p][k] = c * ap[k] - s * aq[k];
                a[q][k] = s * ap[k] + c * aq[k];
            }
            for row in v.iter_mut() {
                let vp = row[p];
                let vq = row[q];
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let col = |j: usize| Vec3::new(v[0][j] as f32, v[1][j] as f32, v[2][j] as f32);
    let (c0, c1) = (col(0), col(1));
    // A right-handed basis for the quaternion.
    let c2 = c0.cross(c1);
    let rot = Quat::from_mat3(&Mat3::from_cols(c0, c1, c2)).normalize();
    (
        Vec3::new(a[0][0] as f32, a[1][1] as f32, a[2][2] as f32),
        rot,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_rotated_diagonal_matrix() {
        let rot = Mat3::from_quat(Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.8, 1.1));
        let m = rot * Mat3::from_diagonal(Vec3::new(1.0, 4.0, 9.0)) * rot.transpose();
        let rows = [0, 1, 2].map(|i| [0, 1, 2].map(|j| f64::from(m.col(j)[i])));
        let (values, q) = eigen_symmetric(rows);
        let mut sorted = values.to_array();
        sorted.sort_by(f32::total_cmp);
        assert!((Vec3::from(sorted) - Vec3::new(1.0, 4.0, 9.0)).length() < 1e-3);
        // Each eigenvector satisfies M v = lambda v.
        for (i, lambda) in values.to_array().into_iter().enumerate() {
            let v = q * Vec3::AXES[i];
            assert!((m * v - v * lambda).length() < 1e-3, "{i}");
        }
    }
}
