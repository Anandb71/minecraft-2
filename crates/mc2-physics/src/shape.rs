//! Rigid body shapes: a dense voxel grid with its mass properties and a
//! small set of collision sample points.

use crate::eigen::eigen_symmetric;
use glam::{IVec3, Quat, Vec3};
use mc2_voxel::material::MaterialId;

/// Largest body extent per axis, voxels (4 m).
pub const MAX_EXTENT: i32 = 64;
/// Voxel edge length, metres.
pub const VOXEL_M: f32 = 1.0 / 16.0;
/// At most this many collision samples per body.
pub const MAX_SAMPLES: usize = 64;

#[derive(Clone, Debug)]
pub struct BodyShape {
    /// Grid size in voxels.
    pub size: IVec3,
    /// Materials, index `x + size.x * (y + size.y * z)`.
    pub voxels: Vec<MaterialId>,
    pub solid_count: u32,
    /// Kilograms.
    pub mass: f32,
    /// Centre of mass in grid coordinates (voxels, continuous).
    pub com: Vec3,
    /// Principal moments of inertia, kg m^2.
    pub inertia: Vec3,
    /// Rotation taking principal-frame vectors to grid-frame vectors.
    pub principal: Quat,
    /// Collision samples: centres of surface voxels, metres, in the
    /// principal frame relative to the centre of mass.
    pub samples: Vec<Vec3>,
    /// Bounding sphere radius around the centre of mass, metres.
    pub radius: f32,
}

impl BodyShape {
    /// Builds a shape; `None` when no voxel is solid or it is too large.
    pub fn from_voxels(size: IVec3, voxels: Vec<MaterialId>) -> Option<Self> {
        if size.min_element() <= 0
            || size.max_element() > MAX_EXTENT
            || voxels.len() != (size.x * size.y * size.z) as usize
        {
            return None;
        }
        let idx = |p: IVec3| (p.x + size.x * (p.y + size.y * p.z)) as usize;
        let volume = VOXEL_M * VOXEL_M * VOXEL_M;
        let mut mass = 0.0f64;
        let mut first = glam::DVec3::ZERO;
        let mut count = 0u32;
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let m = voxels[idx(IVec3::new(x, y, z))];
                    if !m.is_solid() {
                        continue;
                    }
                    let dm = f64::from(m.get().density * volume);
                    mass += dm;
                    first += (IVec3::new(x, y, z).as_dvec3() + 0.5) * dm;
                    count += 1;
                }
            }
        }
        if count == 0 || mass <= 0.0 {
            return None;
        }
        let com = (first / mass).as_vec3();

        // Inertia tensor about the centre of mass, each voxel a small cube.
        let mut t = [[0.0f64; 3]; 3];
        let cube = f64::from(VOXEL_M * VOXEL_M) / 6.0;
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let m = voxels[idx(IVec3::new(x, y, z))];
                    if !m.is_solid() {
                        continue;
                    }
                    let dm = f64::from(m.get().density * volume);
                    let r = ((IVec3::new(x, y, z).as_vec3() + 0.5 - com) * VOXEL_M).as_dvec3();
                    let rr = r.dot(r);
                    let ra = r.to_array();
                    for (i, row) in t.iter_mut().enumerate() {
                        for (j, cell) in row.iter_mut().enumerate() {
                            let delta = if i == j { rr + cube } else { 0.0 };
                            *cell += dm * (delta - ra[i] * ra[j]);
                        }
                    }
                }
            }
        }
        let (inertia, principal) = eigen_symmetric(t);

        // Collision samples: per cell of a 4x4x4 partition of the grid, the
        // surface voxel farthest from the centre of mass, so extremities
        // always collide.
        let solid = |p: IVec3| {
            p.cmpge(IVec3::ZERO).all() && p.cmplt(size).all() && voxels[idx(p)].is_solid()
        };
        let mut best: Vec<Option<(f32, Vec3)>> = vec![None; 64];
        let mut radius = 0.0f32;
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let p = IVec3::new(x, y, z);
                    if !solid(p) {
                        continue;
                    }
                    let centre = (p.as_vec3() + 0.5 - com) * VOXEL_M;
                    let d = centre.length();
                    radius = radius.max(d + VOXEL_M * 0.87);
                    let exposed = [
                        IVec3::X,
                        IVec3::NEG_X,
                        IVec3::Y,
                        IVec3::NEG_Y,
                        IVec3::Z,
                        IVec3::NEG_Z,
                    ]
                    .iter()
                    .any(|&o| !solid(p + o));
                    if !exposed {
                        continue;
                    }
                    let cell = (p * 4 / size).min(IVec3::splat(3));
                    let k = (cell.x + 4 * (cell.y + 4 * cell.z)) as usize;
                    if best[k].is_none_or(|(bd, _)| d > bd) {
                        best[k] = Some((d, centre));
                    }
                }
            }
        }
        let to_principal = principal.inverse();
        let samples: Vec<Vec3> = best
            .into_iter()
            .flatten()
            .map(|(_, c)| to_principal * c)
            .take(MAX_SAMPLES)
            .collect();
        Some(Self {
            size,
            voxels,
            solid_count: count,
            mass: mass as f32,
            com,
            inertia: inertia.max(Vec3::splat(1e-6)),
            principal,
            samples,
            radius,
        })
    }

    pub fn get(&self, p: IVec3) -> MaterialId {
        if p.cmplt(IVec3::ZERO).any() || p.cmpge(self.size).any() {
            return MaterialId(0);
        }
        self.voxels[(p.x + self.size.x * (p.y + self.size.y * p.z)) as usize]
    }

    /// Grid coordinates (voxels) of a principal-frame point (metres).
    pub fn grid_of(&self, p: Vec3) -> Vec3 {
        self.principal * p / VOXEL_M + self.com
    }

    /// Principal-frame vector of a grid-frame vector.
    pub fn to_principal(&self, v: Vec3) -> Vec3 {
        self.principal.inverse() * v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;

    fn slab(size: IVec3, m: MaterialId) -> BodyShape {
        let n = (size.x * size.y * size.z) as usize;
        BodyShape::from_voxels(size, vec![m; n]).expect("shape")
    }

    #[test]
    fn cube_mass_properties_match_formulas() {
        // 16^3 voxels of granite: a 1 m cube of 2700 kg.
        let s = slab(IVec3::splat(16), ids::GRANITE);
        assert!((s.mass - 2700.0).abs() < 1.0, "{}", s.mass);
        assert!((s.com - Vec3::splat(8.0)).length() < 1e-4);
        let expect = 2700.0 / 6.0;
        for i in s.inertia.to_array() {
            assert!((i - expect).abs() / expect < 1e-3, "{i} vs {expect}");
        }
        assert!(!s.samples.is_empty() && s.samples.len() <= MAX_SAMPLES);
        assert!((s.radius - 0.866).abs() < 0.07);
    }

    #[test]
    fn rod_principal_axes_follow_the_long_axis() {
        // A 2 m x 0.25 m x 0.25 m rod along x: tiny moment about x.
        let s = slab(IVec3::new(32, 4, 4), ids::PLANKS);
        let (small_axis, _) = s
            .inertia
            .to_array()
            .into_iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let axis = s.principal * Vec3::AXES[small_axis];
        assert!(axis.x.abs() > 0.999, "{axis}");
        let long = s.mass * 4.0 / 12.0;
        let max = s.inertia.max_element();
        assert!((max - long).abs() / long < 0.03, "{max} vs {long}");
    }

    #[test]
    fn empty_and_oversized_shapes_are_rejected() {
        assert!(BodyShape::from_voxels(IVec3::splat(2), vec![ids::AIR; 8]).is_none());
        assert!(BodyShape::from_voxels(IVec3::new(65, 1, 1), vec![ids::GRANITE; 65]).is_none());
    }
}
