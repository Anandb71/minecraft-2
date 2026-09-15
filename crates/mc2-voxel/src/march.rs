//! CPU reference marcher: hierarchical DDA over chunks, 64-tree levels,
//! brick subblocks and voxels.
//!
//! Each level runs an Amanatides and Woo DDA over its 4^3 (or 2^3 for brick
//! subblocks) child grid between the parent cell's entry and exit times, and
//! descends only into children whose mask bit is set (Museth's hierarchical
//! DDA). Robustness follows the 64-tree guide: a child DDA locates its first
//! cell by clamping the entry point into the child's bounds rather than
//! biasing the intersection distance, and boundary times come from cell
//! planes, never from accumulated positions.
//!
//! This is the ground truth the GPU marcher is tested against, and the ray
//! query used by gameplay, physics, NPC perception and audio.

use crate::brick::subblock_bit;
use crate::coords::{self, CHUNK_VOXELS, ChunkPos};
use crate::material::MaterialId;
use crate::tree::{Cell, ChunkTree, child_index};
use crate::world::VoxelWorld;
use glam::{DVec3, IVec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Distance along the normalised direction, metres.
    pub t: f64,
    pub voxel: IVec3,
    /// Outward normal of the face the ray entered through.
    pub normal: IVec3,
    pub material: MaterialId,
    /// DDA steps taken across every level.
    pub iterations: u32,
}

#[derive(Clone, Copy)]
struct Grid {
    min: DVec3,
    size: f64,
    dims: IVec3,
}

fn grid(min: IVec3, size: i32, n: i32) -> Grid {
    Grid {
        min: min.as_dvec3(),
        size: f64::from(size),
        dims: IVec3::splat(n),
    }
}

struct Ray<F> {
    o: DVec3,
    d: DVec3,
    inv: DVec3,
    step: IVec3,
    accept: F,
    iterations: u32,
}

impl<F: Fn(MaterialId) -> bool> Ray<F> {
    /// DDA over a `dims` grid of `size`-voxel cells anchored at `min`, for
    /// `t` in `[t0, t1]` (voxel units). `visit` gets the cell, its entry and
    /// exit times and the axis crossed to enter it.
    fn dda<R>(
        &mut self,
        g: Grid,
        t0: f64,
        t1: f64,
        entry_axis: usize,
        mut visit: impl FnMut(&mut Self, IVec3, f64, f64, usize) -> Option<R>,
    ) -> Option<R> {
        let (min, size, dims) = (g.min, g.size, g.dims);
        let p = self.o + self.d * t0;
        let mut cell = ((p - min) / size)
            .floor()
            .as_ivec3()
            .clamp(IVec3::ZERO, dims - 1);
        let mut tmax = DVec3::splat(f64::INFINITY);
        let mut tdelta = DVec3::splat(f64::INFINITY);
        for a in 0..3 {
            if self.step[a] != 0 {
                let boundary = min[a] + f64::from(cell[a] + i32::from(self.step[a] > 0)) * size;
                tmax[a] = (boundary - self.o[a]) * self.inv[a];
                tdelta[a] = size * self.inv[a].abs();
            }
        }
        let mut t = t0;
        let mut axis = entry_axis;
        loop {
            self.iterations += 1;
            let next = tmax.min_element();
            if let Some(r) = visit(self, cell, t, next.min(t1), axis) {
                return Some(r);
            }
            if next >= t1 {
                return None;
            }
            axis = if tmax.x <= tmax.y && tmax.x <= tmax.z {
                0
            } else if tmax.y <= tmax.z {
                1
            } else {
                2
            };
            cell[axis] += self.step[axis];
            if cell[axis] < 0 || cell[axis] >= dims[axis] {
                return None;
            }
            t = tmax[axis];
            tmax[axis] += tdelta[axis];
        }
    }

    fn hit(&self, voxel: IVec3, t: f64, axis: usize, material: MaterialId) -> RayHit {
        let mut normal = IVec3::ZERO;
        normal[axis] = -self.step[axis];
        RayHit {
            t: t / f64::from(coords::VOXELS_PER_BLOCK),
            voxel,
            normal,
            material,
            iterations: self.iterations,
        }
    }

    fn chunk(
        &mut self,
        tree: &ChunkTree,
        origin: IVec3,
        t0: f64,
        t1: f64,
        axis: usize,
    ) -> Option<RayHit> {
        let root = &tree.root;
        self.dda(grid(origin, 128, 4), t0, t1, axis, |r, c, a0, a1, ax| {
            let l2 = root.get(child_index(c))?;
            let l2_min = origin + c * 128;
            r.dda(grid(l2_min, 32, 4), a0, a1, ax, |r, c, b0, b1, bx| {
                let l1 = l2.get(child_index(c))?;
                let l1_min = l2_min + c * 32;
                r.dda(grid(l1_min, 8, 4), b0, b1, bx, |r, c, c0, c1, cx| {
                    let cell = *l1.get(child_index(c))?;
                    let brick_min = l1_min + c * 8;
                    r.cell(tree, cell, brick_min, c0, c1, cx)
                })
            })
        })
    }

    fn cell(
        &mut self,
        tree: &ChunkTree,
        cell: Cell,
        min: IVec3,
        t0: f64,
        t1: f64,
        axis: usize,
    ) -> Option<RayHit> {
        match cell {
            Cell::Empty => None,
            Cell::Uniform(m) => {
                if !(self.accept)(m) {
                    return None;
                }
                let p = self.o + self.d * t0;
                let v = p.floor().as_ivec3().clamp(min, min + 7);
                Some(self.hit(v, t0, axis, m))
            }
            Cell::Brick(i) => {
                let brick = tree.brick(i);
                let occ = brick.occupancy();
                self.dda(grid(min, 4, 2), t0, t1, axis, |r, s, s0, s1, sx| {
                    let sub_min = min + s * 4;
                    let (si, _) = subblock_bit(s * 4);
                    if occ[si] == 0 {
                        return None;
                    }
                    r.dda(grid(sub_min, 1, 4), s0, s1, sx, |r, v, v0, _, vx| {
                        let local = s * 4 + v;
                        if !brick.is_occupied(local) {
                            return None;
                        }
                        let m = brick.get(local);
                        if !(r.accept)(m) {
                            return None;
                        }
                        Some(r.hit(min + local, v0, vx, m))
                    })
                })
            }
        }
    }
}

/// Casts a ray in metres. `dir` need not be normalised. Hits any non-air voxel.
pub fn raycast(world: &VoxelWorld, origin: DVec3, dir: DVec3, max_t: f64) -> Option<RayHit> {
    raycast_filtered(world, origin, dir, max_t, |m| !m.is_air())
}

/// Casts a ray, reporting the first voxel whose material passes `accept`.
pub fn raycast_filtered(
    world: &VoxelWorld,
    origin: DVec3,
    dir: DVec3,
    max_t: f64,
    accept: impl Fn(MaterialId) -> bool,
) -> Option<RayHit> {
    let len = dir.length();
    if len == 0.0 || !len.is_finite() {
        return None;
    }
    let scale = f64::from(coords::VOXELS_PER_BLOCK);
    let d = dir / len;
    let o = origin * scale;
    let inv = DVec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
    let step = IVec3::new(
        d.x.signum() as i32,
        d.y.signum() as i32,
        d.z.signum() as i32,
    ) * IVec3::new(
        i32::from(d.x != 0.0),
        i32::from(d.y != 0.0),
        i32::from(d.z != 0.0),
    );

    // Clip against the world box.
    let world_max = DVec3::new(
        f64::from(coords::WORLD_SECTORS_XZ * coords::SECTOR_VOXELS),
        f64::from(coords::WORLD_SECTORS_Y * coords::SECTOR_VOXELS),
        f64::from(coords::WORLD_SECTORS_XZ * coords::SECTOR_VOXELS),
    );
    let mut t0 = 0.0f64;
    let mut t1 = max_t * scale;
    let mut entry_axis = d.abs().max_position();
    for a in 0..3 {
        if d[a] == 0.0 {
            if o[a] < 0.0 || o[a] >= world_max[a] {
                return None;
            }
            continue;
        }
        let (mut near, mut far) = ((0.0 - o[a]) * inv[a], (world_max[a] - o[a]) * inv[a]);
        if near > far {
            std::mem::swap(&mut near, &mut far);
        }
        if near > t0 {
            t0 = near;
            entry_axis = a;
        }
        t1 = t1.min(far);
    }
    if t0 > t1 {
        return None;
    }

    let mut ray = Ray {
        o,
        d,
        inv,
        step,
        accept,
        iterations: 0,
    };
    let chunk_dims = (world_max / f64::from(CHUNK_VOXELS)).as_ivec3();
    ray.dda(
        Grid {
            min: DVec3::ZERO,
            size: f64::from(CHUNK_VOXELS),
            dims: chunk_dims,
        },
        t0,
        t1,
        entry_axis,
        |r, c, c0, c1, axis| {
            let tree = world.chunk(ChunkPos(c))?;
            if tree.is_empty() {
                return None;
            }
            r.chunk(tree, c * CHUNK_VOXELS, c0, c1, axis)
        },
    )
}

/// Voxel-by-voxel Amanatides and Woo over `VoxelWorld::voxel`. Slow; exists
/// to validate the hierarchical marcher.
pub fn raycast_brute(world: &VoxelWorld, origin: DVec3, dir: DVec3, max_t: f64) -> Option<RayHit> {
    let scale = f64::from(coords::VOXELS_PER_BLOCK);
    let d = dir.normalize();
    let o = origin * scale;
    let mut v = o.floor().as_ivec3();
    let step = IVec3::new(
        d.x.signum() as i32,
        d.y.signum() as i32,
        d.z.signum() as i32,
    );
    let inv = DVec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
    let mut tmax = DVec3::ZERO;
    let mut tdelta = DVec3::ZERO;
    for a in 0..3 {
        if d[a] == 0.0 {
            tmax[a] = f64::INFINITY;
            tdelta[a] = f64::INFINITY;
        } else {
            let boundary = f64::from(v[a] + i32::from(d[a] > 0.0));
            tmax[a] = (boundary - o[a]) * inv[a];
            tdelta[a] = inv[a].abs();
        }
    }
    let mut t = 0.0;
    let mut axis = d.abs().max_position();
    let mut iterations = 0;
    while t <= max_t * scale {
        iterations += 1;
        let m = world.voxel(v);
        if !m.is_air() {
            let mut normal = IVec3::ZERO;
            normal[axis] = -step[axis];
            return Some(RayHit {
                t: t / scale,
                voxel: v,
                normal,
                material: m,
                iterations,
            });
        }
        axis = tmax.min_position();
        v[axis] += step[axis];
        t = tmax[axis];
        tmax[axis] += tdelta[axis];
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::ids;

    /// Deterministic xorshift so the test world and rays are reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn test_world(rng: &mut Rng) -> VoxelWorld {
        let mut w = VoxelWorld::new();
        // Rolling terrain across four chunks with a strata boundary.
        for x in 0..1024 {
            for z in 0..600 {
                let h = 200.0 + 40.0 * (x as f64 * 0.013).sin() + 25.0 * (z as f64 * 0.021).cos();
                let top = h as i32;
                let mat = if (x / 16 + z / 16) % 5 == 0 {
                    ids::SANDSTONE
                } else {
                    ids::GRANITE
                };
                for y in (top - 6).max(150)..=top {
                    w.set_voxel(
                        IVec3::new(x, y, z),
                        if y > top - 2 { ids::GRASS } else { mat },
                    );
                }
            }
        }
        w.fill_box(
            IVec3::new(0, 100, 0),
            IVec3::new(1023, 149, 599),
            ids::BASALT,
        );
        // Carved caves and floating debris.
        for _ in 0..40 {
            let c = DVec3::new(
                rng.next() * 1024.0,
                120.0 + rng.next() * 120.0,
                rng.next() * 600.0,
            );
            w.fill_sphere(c, 4.0 + rng.next() * 20.0, ids::AIR);
        }
        for _ in 0..200 {
            let v = IVec3::new(
                (rng.next() * 1024.0) as i32,
                240 + (rng.next() * 60.0) as i32,
                (rng.next() * 600.0) as i32,
            );
            w.set_voxel(v, ids::IRON_ORE);
        }
        w
    }

    #[test]
    fn hierarchical_matches_brute_force() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let w = test_world(&mut rng);
        let mut hits = 0;
        for i in 0..3000 {
            let o = DVec3::new(
                rng.next() * 64.0,
                6.0 + rng.next() * 16.0,
                rng.next() * 37.5,
            );
            let mut d = DVec3::new(rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5);
            // A share of axis-aligned and planar rays, the classic failure cases.
            match i % 7 {
                0 => d = DVec3::new(0.0, -1.0, 0.0),
                1 => d.y = 0.0,
                2 => d.x = 0.0,
                _ => {}
            }
            if d.length() < 1e-3 {
                continue;
            }
            let a = raycast(&w, o, d, 40.0);
            let b = raycast_brute(&w, o, d, 40.0);
            match (a, b) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    hits += 1;
                    assert_eq!(a.voxel, b.voxel, "ray {i} o={o} d={d}");
                    assert!((a.t - b.t).abs() < 1e-9, "ray {i}: t {} vs {}", a.t, b.t);
                    assert_eq!(a.material, b.material);
                }
                (a, b) => {
                    // The brute force walks past max_t by at most one voxel.
                    let edge = a.or(b).is_some_and(|h| h.t > 39.9);
                    assert!(edge, "ray {i} o={o} d={d}: {a:?} vs {b:?}");
                }
            }
        }
        assert!(hits > 1500, "only {hits} hits, test world too sparse");
    }

    #[test]
    fn normals_face_the_ray() {
        let mut w = VoxelWorld::new();
        w.fill_box(
            IVec3::new(160, 160, 160),
            IVec3::new(175, 175, 175),
            ids::GRANITE,
        );
        let c = DVec3::splat(10.5);
        for (d, n) in [
            (DVec3::X, IVec3::NEG_X),
            (DVec3::NEG_X, IVec3::X),
            (DVec3::Y, IVec3::NEG_Y),
            (DVec3::NEG_Y, IVec3::Y),
            (DVec3::Z, IVec3::NEG_Z),
            (DVec3::NEG_Z, IVec3::Z),
        ] {
            let o = c - d * 5.0;
            let h = raycast(&w, o, d, 20.0).expect("hit");
            assert_eq!(h.normal, n, "dir {d}");
            assert!((h.t - 4.5).abs() < 1e-9, "dir {d} t {}", h.t);
        }
    }

    #[test]
    fn filter_passes_through_rejected_materials() {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::new(0, 0, 16), IVec3::new(63, 63, 31), ids::GLASS);
        w.fill_box(IVec3::new(0, 0, 48), IVec3::new(63, 63, 63), ids::GRANITE);
        let o = DVec3::new(2.0, 2.0, 0.1);
        let first = raycast(&w, o, DVec3::Z, 10.0).unwrap();
        assert_eq!(first.material, ids::GLASS);
        let opaque = raycast_filtered(&w, o, DVec3::Z, 10.0, |m| m == ids::GRANITE).unwrap();
        assert_eq!(opaque.material, ids::GRANITE);
        assert!((opaque.t - 2.9).abs() < 1e-9, "t {}", opaque.t);
    }
}
