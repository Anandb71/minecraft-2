//! Explosions that turn terrain into debris.

use crate::shape::{BodyShape, VOXEL_M};
use crate::world::PhysicsWorld;
use glam::{DVec3, IVec3, Vec3};
use mc2_voxel::material::{MaterialId, ids};
use mc2_voxel::world::VoxelWorld;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExplosionReport {
    pub removed_voxels: u64,
    pub debris: usize,
    /// Existing bodies pushed by the blast.
    pub pushed: usize,
}

/// Small deterministic generator for debris layout.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Blasts a sphere of `radius` metres out of the world. Solid voxels inside
/// are removed; material from the outer shell is thrown out as up to
/// `max_debris` fragments of a few decimetres, and bodies within twice the
/// radius are pushed away. Materials weaker than the blast (by compressive
/// strength) break further out.
pub fn explode(
    world: &mut VoxelWorld,
    physics: &mut PhysicsWorld,
    centre: DVec3,
    radius: f32,
    max_debris: usize,
    seed: u64,
) -> ExplosionReport {
    let mut report = ExplosionReport::default();
    let mut rng = Lcg(seed ^ 0x9e37_79b9_7f4a_7c15);
    let c = centre * 16.0;
    let r_vox = f64::from(radius) * 16.0;
    let lo = (c - r_vox * 1.3).floor().as_ivec3();
    let hi = (c + r_vox * 1.3).ceil().as_ivec3();
    // Where each removed voxel was, and what it was made of.
    let mut removed: Vec<(IVec3, MaterialId)> = Vec::new();
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let v = IVec3::new(x, y, z);
                let m = world.voxel(v);
                if !m.is_solid() || m == ids::BEDROCK {
                    continue;
                }
                let d = (v.as_dvec3() + 0.5 - c).length() / r_vox;
                // Weak materials crumble up to 30% beyond the radius.
                let weakness = (1.0 - (m.get().compressive / 250.0).min(1.0)) as f64;
                let reach = 1.0 + 0.3 * weakness;
                if d > reach {
                    continue;
                }
                removed.push((v, m));
                world.set_voxel(v, ids::AIR);
            }
        }
    }
    report.removed_voxels = removed.len() as u64;
    // Bodies present before the blast; debris gets its own velocities.
    let existing = physics.bodies.len();

    // Debris: fragments seeded on the outer shell of what was removed.
    if !removed.is_empty() && max_debris > 0 {
        let shell: Vec<&(IVec3, MaterialId)> = removed
            .iter()
            .filter(|(v, _)| (v.as_dvec3() + 0.5 - c).length() > r_vox * 0.55)
            .collect();
        let mut taken = mc2_core::FxHashSet::default();
        let wanted = max_debris.min(shell.len() / 8);
        for _ in 0..wanted * 4 {
            if report.debris >= wanted {
                break;
            }
            let &&(seed_v, _) = &shell[(rng.next() * shell.len() as f32) as usize % shell.len()];
            if taken.contains(&seed_v) {
                continue;
            }
            // A fragment 3 to 7 voxels a side around the seed.
            let size = IVec3::new(
                3 + (rng.next() * 5.0) as i32,
                3 + (rng.next() * 5.0) as i32,
                3 + (rng.next() * 5.0) as i32,
            );
            let origin = seed_v - size / 2;
            let mut voxels = vec![MaterialId(0); (size.x * size.y * size.z) as usize];
            let mut any = false;
            for &(v, m) in &removed {
                let l = v - origin;
                if l.cmpge(IVec3::ZERO).all() && l.cmplt(size).all() && !taken.contains(&v) {
                    // Knock corners off so fragments are not boxes.
                    let corner = l.cmpeq(IVec3::ZERO) | l.cmpeq(size - 1);
                    if corner.x as u8 + corner.y as u8 + corner.z as u8 >= 2 && rng.next() < 0.6 {
                        continue;
                    }
                    voxels[(l.x + size.x * (l.y + size.y * l.z)) as usize] = m;
                    taken.insert(v);
                    any = true;
                }
            }
            if !any {
                continue;
            }
            let Some(shape) = BodyShape::from_voxels(size, voxels) else {
                continue;
            };
            let shape = Arc::new(shape);
            let com_world = (origin.as_dvec3() + shape.com.as_dvec3()) * f64::from(VOXEL_M);
            let rot = shape.principal;
            let id = physics.spawn(shape, com_world, rot);
            let out = (com_world - centre).as_vec3();
            // Ground throws material up and out, never down into itself.
            let away = out.normalize_or(Vec3::Y);
            let dir = Vec3::new(away.x, away.y.abs(), away.z) + Vec3::Y * 0.6;
            let falloff = (1.0 - out.length() / (radius * 1.3)).clamp(0.2, 1.0);
            if let Some(b) = physics.body_mut(id) {
                b.vel = dir.normalize() * (6.0 + 10.0 * falloff) * (0.7 + 0.6 * rng.next());
                b.ang_vel = Vec3::new(rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5) * 20.0;
            }
            report.debris += 1;
        }
    }

    // Push bodies already in the world.
    let reach = f64::from(radius) * 2.0;
    for b in &mut physics.bodies[..existing] {
        let d = b.pos - centre;
        let dist = d.length();
        if dist > reach {
            continue;
        }
        let strength = (1.0 - dist / reach) as f32;
        let dir = d.as_vec3().normalize_or(Vec3::Y);
        b.wake();
        // An impulse of 2000 N s at the centre, tapering to nothing.
        b.vel += dir * (2000.0 * strength * b.inv_mass).min(30.0);
        report.pushed += 1;
    }
    physics.wake_region(DVec3::splat(-reach) + centre, DVec3::splat(reach) + centre);
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explosion_carves_a_crater_and_throws_debris() {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(255, 63, 255), ids::DIRT);
        let mut p = PhysicsWorld::new();
        let centre = DVec3::new(8.0, 4.0, 8.0);
        let report = explode(&mut w, &mut p, centre, 1.5, 24, 7);
        // Dirt is weak: the crater reaches past the nominal radius.
        let nominal = 4.0 / 3.0 * std::f64::consts::PI * 24.0f64.powi(3);
        assert!(report.removed_voxels as f64 > nominal, "{report:?}");
        assert!(report.debris > 8, "{report:?}");
        assert!(w.voxel((centre * 16.0).as_ivec3()).is_air());
        // Debris flies outward and up.
        let rising = p.bodies.iter().filter(|b| b.vel.y > 0.0).count();
        assert!(rising > p.bodies.len() / 2);
        // Bedrock is untouched even inside the blast.
        let mut w2 = VoxelWorld::new();
        w2.fill_box(IVec3::ZERO, IVec3::splat(63), ids::BEDROCK);
        let r2 = explode(
            &mut w2,
            &mut PhysicsWorld::new(),
            DVec3::splat(2.0),
            1.0,
            8,
            1,
        );
        assert_eq!(r2.removed_voxels, 0);
    }
}
