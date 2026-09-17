//! Explosions that turn terrain into debris, and baking settled debris back
//! into the voxel world.

use crate::body::{Body, BodyId};
use crate::shape::{BodyShape, VOXEL_M};
use crate::world::PhysicsWorld;
use glam::{DVec3, IVec3, Vec3};
use mc2_voxel::coords::{BRICK_SHIFT, CHUNK_SHIFT, ChunkPos};
use mc2_voxel::material::{MaterialId, ids};
use mc2_voxel::tree::Cell;
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

/// What a blast removed: materials over the blast box, 0 where nothing was.
struct Carved {
    lo: IVec3,
    dims: IVec3,
    materials: Vec<u16>,
    count: u64,
}

impl Carved {
    fn index(&self, v: IVec3) -> Option<usize> {
        let l = v - self.lo;
        if l.cmplt(IVec3::ZERO).any() || l.cmpge(self.dims).any() {
            return None;
        }
        Some((l.x + self.dims.x * (l.y + self.dims.y * l.z)) as usize)
    }

    fn get(&self, v: IVec3) -> MaterialId {
        self.index(v)
            .map_or(MaterialId(0), |i| MaterialId(self.materials[i]))
    }

    /// The material removed at `v`, which no other fragment may take.
    fn take(&mut self, v: IVec3) -> MaterialId {
        match self.index(v) {
            Some(i) => MaterialId(std::mem::take(&mut self.materials[i])),
            None => MaterialId(0),
        }
    }
}

/// How far out a material breaks, as a multiple of the blast radius: weak
/// materials (by compressive strength) crumble up to 30% farther.
fn reach_of(m: MaterialId) -> f64 {
    let weakness = f64::from(1.0 - (m.get().compressive / 250.0).min(1.0));
    1.0 + 0.3 * weakness
}

/// Removes breakable solid voxels within reach of `c` (voxel units),
/// brick cell by brick cell: empty cells cost one lookup, and uniform
/// cells wholly inside reach are cleared without touching their voxels.
fn carve(world: &mut VoxelWorld, c: DVec3, r_vox: f64) -> Carved {
    let outer = r_vox * 1.3;
    let lo = (c - outer).floor().as_ivec3();
    let hi = (c + outer).ceil().as_ivec3();
    let dims = hi - lo + 1;
    let mut carved = Carved {
        lo,
        dims,
        materials: vec![0; dims.element_product() as usize],
        count: 0,
    };
    let breakable = |m: MaterialId| m.is_solid() && m != ids::BEDROCK;
    for cz in (lo.z >> 3)..=(hi.z >> 3) {
        for cy in (lo.y >> 3)..=(hi.y >> 3) {
            for cx in (lo.x >> 3)..=(hi.x >> 3) {
                let cell = IVec3::new(cx, cy, cz);
                let min = (cell * 8).as_dvec3();
                let max = min + 8.0;
                // Nearest and farthest voxel centres of the cell.
                let near = (c.clamp(min + 0.5, max - 0.5) - c).length();
                if near > outer {
                    continue;
                }
                let far = (c - (min + 0.5))
                    .abs()
                    .max((c - (max - 0.5)).abs())
                    .length();
                let pos = ChunkPos::of_voxel(cell * 8);
                let local = cell - (pos.0 << (CHUNK_SHIFT - BRICK_SHIFT));
                let Some(tree) = world.chunk(pos) else {
                    continue;
                };
                match tree.cell(local) {
                    Cell::Empty => continue,
                    Cell::Uniform(m) if !breakable(m) => continue,
                    Cell::Uniform(m) if far <= r_vox * reach_of(m) => {
                        // The whole cell goes.
                        let origin = cell * 8;
                        for i in 0..512 {
                            let v = origin + IVec3::new(i & 7, (i >> 6) & 7, (i >> 3) & 7);
                            if let Some(k) = carved.index(v) {
                                carved.materials[k] = m.0;
                            }
                        }
                        carved.count += 512;
                        if let Some(tree) = world.chunk_mut(pos) {
                            tree.set_cell(local, Cell::Empty);
                        }
                        continue;
                    }
                    _ => {}
                }
                let Some(tree) = world.chunk_mut(pos) else {
                    continue;
                };
                let origin = cell * 8;
                tree.edit_brick(local, |brick| {
                    for i in 0..512 {
                        let l = IVec3::new(i & 7, (i >> 6) & 7, (i >> 3) & 7);
                        let m = brick.get(l);
                        if !breakable(m) {
                            continue;
                        }
                        let v = origin + l;
                        let reach = r_vox * reach_of(m);
                        if (v.as_dvec3() + 0.5 - c).length_squared() > reach * reach {
                            continue;
                        }
                        brick.set(l, MaterialId(0));
                        if let Some(k) = carved.index(v) {
                            carved.materials[k] = m.0;
                        }
                        carved.count += 1;
                    }
                });
            }
        }
    }
    carved
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
    let carved = carve(world, c, r_vox);
    report.removed_voxels = carved.count;
    // Bodies present before the blast; debris gets its own velocities.
    let existing = physics.bodies.len();

    // Debris: fragments seeded on the outer shell of what was removed.
    let mut carved = carved;
    let wanted = max_debris.min((carved.count / 8) as usize);
    let mut attempts = 0;
    while report.debris < wanted && attempts < wanted * 8 {
        attempts += 1;
        // A random point 60-105% of the radius out.
        let z = f64::from(rng.next()) * 2.0 - 1.0;
        let phi = f64::from(rng.next()) * std::f64::consts::TAU;
        let s = (1.0 - z * z).sqrt();
        let dir = DVec3::new(s * phi.cos(), z, s * phi.sin());
        let at = c + dir * r_vox * (0.6 + 0.45 * f64::from(rng.next()));
        let seed_v = at.floor().as_ivec3();
        if carved.get(seed_v).is_air() {
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
        for lz in 0..size.z {
            for ly in 0..size.y {
                for lx in 0..size.x {
                    let l = IVec3::new(lx, ly, lz);
                    // Knock corners off so fragments are not boxes.
                    let corner = l.cmpeq(IVec3::ZERO) | l.cmpeq(size - 1);
                    if corner.x as u8 + corner.y as u8 + corner.z as u8 >= 2 && rng.next() < 0.6 {
                        continue;
                    }
                    let m = carved.take(origin + l);
                    if !m.is_air() {
                        voxels[(lx + size.x * (ly + size.y * lz)) as usize] = m;
                        any = true;
                    }
                }
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

/// Writes a body into the world at its current pose: every empty world
/// voxel whose centre lies inside a solid body voxel takes its material.
/// Returns the number of voxels written.
pub fn bake(world: &mut VoxelWorld, body: &Body) -> u64 {
    let r = f64::from(body.shape.radius) * 16.0;
    let c = body.pos * 16.0;
    let lo = (c - r).floor().as_ivec3();
    let hi = (c + r).ceil().as_ivec3();
    let mut written = 0;
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let v = IVec3::new(x, y, z);
                let centre = (v.as_dvec3() + 0.5) * f64::from(VOXEL_M);
                let m = body.material_at(centre);
                if !m.is_solid() || !world.voxel(v).is_air() {
                    continue;
                }
                world.set_voxel(v, m);
                written += 1;
            }
        }
    }
    written
}

/// Bakes bodies that have slept for `after_s` seconds (or the oldest
/// sleepers while more than `max_bodies` exist) into the world. Bodies for
/// which `keep` is true stay bodies.
pub fn bake_settled(
    world: &mut VoxelWorld,
    physics: &mut PhysicsWorld,
    after_s: f32,
    max_bodies: usize,
    keep: impl Fn(BodyId) -> bool,
) -> Vec<BodyId> {
    let mut sleepers: Vec<(f32, BodyId)> = physics
        .bodies
        .iter()
        .filter(|b| b.asleep && !keep(b.id))
        .map(|b| (b.still_time, b.id))
        .collect();
    sleepers.sort_by(|a, b| b.0.total_cmp(&a.0));
    let excess = physics.bodies.len().saturating_sub(max_bodies);
    let mut baked = Vec::new();
    for (i, (still, id)) in sleepers.into_iter().enumerate() {
        if still < after_s && i >= excess {
            continue;
        }
        if let Some(b) = physics.remove(id) {
            bake(world, &b);
            baked.push(id);
        }
    }
    baked
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

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

    #[test]
    fn baking_writes_a_body_back_into_the_world() {
        let mut w = VoxelWorld::new();
        let mut p = PhysicsWorld::new();
        let shape = Arc::new(
            BodyShape::from_voxels(IVec3::splat(4), vec![ids::GRANITE; 64]).expect("shape"),
        );
        let id = p.spawn(shape, DVec3::new(4.0, 4.0, 4.0), Quat::IDENTITY);
        let body = p.remove(id).unwrap();
        let written = bake(&mut w, &body);
        assert_eq!(written, 64);
        assert_eq!(w.voxel(IVec3::splat(64)), ids::GRANITE);
    }

    #[test]
    fn only_settled_bodies_not_kept_are_baked() {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(255, 31, 255), ids::GRANITE);
        let mut p = PhysicsWorld::new();
        let shape = Arc::new(
            BodyShape::from_voxels(IVec3::splat(4), vec![ids::PLANKS; 64]).expect("shape"),
        );
        let a = p.spawn(shape.clone(), DVec3::new(4.0, 2.2, 4.0), Quat::IDENTITY);
        let kept = p.spawn(shape.clone(), DVec3::new(8.0, 2.2, 8.0), Quat::IDENTITY);
        let flying = p.spawn(shape, DVec3::new(12.0, 30.0, 12.0), Quat::IDENTITY);
        for _ in 0..240 {
            p.step(1.0 / 120.0, &w);
        }
        let baked = bake_settled(&mut w, &mut p, 1.0, 100, |id| id == kept);
        assert_eq!(baked, vec![a]);
        assert!(p.body(kept).is_some() && p.body(flying).is_some());
        let inside = (DVec3::new(4.0, 2.1, 4.0) * 16.0).floor().as_ivec3();
        assert_eq!(w.voxel(inside), ids::PLANKS);
    }
}
