//! Turning blocks into falling bodies: their solid voxels are cut out of
//! the world, a brick cell at a time, and grouped into pieces of at most
//! `piece` voxels a side.

use glam::{DVec3, IVec3, Vec3};
use mc2_core::FxHashMap;
use mc2_physics::BodyShape;
use mc2_physics::explode::Debris;
use mc2_physics::shape::VOXEL_M;
use mc2_voxel::coords::{BRICK_SHIFT, BlockPos, CHUNK_SHIFT, ChunkPos};
use mc2_voxel::material::MaterialId;
use mc2_voxel::tree::Cell;
use mc2_voxel::world::VoxelWorld;
use std::sync::Arc;

/// Cuts `blocks` out of the world (everything in them, solid or not) and
/// returns their solid voxels as bodies at rest, one per `piece`^3 cell of
/// the voxel grid that holds any. `piece` must be 8, 16 or a multiple of
/// 16, and at most the largest body extent, so that a brick cell never
/// straddles two pieces.
pub fn cut(world: &mut VoxelWorld, blocks: &[BlockPos], piece: i32) -> Vec<Debris> {
    assert!(piece == 8 || (piece % 16 == 0 && piece > 0));
    let n = piece as usize;
    let mut pieces: FxHashMap<IVec3, Vec<MaterialId>> = FxHashMap::default();
    for block in blocks {
        let origin = block.origin();
        let pos = ChunkPos::of_voxel(origin);
        for i in 0..8 {
            let offset = IVec3::new(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            let cell = (origin >> BRICK_SHIFT) + offset;
            let local = cell - (pos.0 << (CHUNK_SHIFT - BRICK_SHIFT));
            let Some(tree) = world.chunk(pos) else {
                continue;
            };
            let content = tree.cell(local);
            let brick = match content {
                Cell::Empty => continue,
                Cell::Uniform(_) => None,
                Cell::Brick(b) => Some(tree.brick(b).clone()),
            };
            let cell_origin = cell << BRICK_SHIFT;
            // A brick cell lies in one piece: pieces are at least 8 voxels
            // a side and aligned, so the grid is found once, not per voxel.
            let key = cell_origin.div_euclid(IVec3::splat(piece));
            let grid = pieces
                .entry(key)
                .or_insert_with(|| vec![MaterialId(0); n * n * n]);
            let base = cell_origin - key * piece;
            for v in 0..512 {
                let l = IVec3::new(v & 7, (v >> 6) & 7, (v >> 3) & 7);
                let m = match (&content, &brick) {
                    (Cell::Uniform(m), _) => *m,
                    (_, Some(b)) => b.get(l),
                    _ => MaterialId(0),
                };
                if !m.is_solid() {
                    continue;
                }
                let g = base + l;
                grid[g.x as usize + n * (g.y as usize + n * g.z as usize)] = m;
            }
            if let Some(tree) = world.chunk_mut(pos) {
                tree.set_cell(local, Cell::Empty);
            }
        }
    }
    let mut keys: Vec<IVec3> = pieces.keys().copied().collect();
    // Lowest pieces first, so callers spawning over several frames start
    // with the ones everything else rests on.
    keys.sort_unstable_by_key(|k| (k.y, k.x, k.z));
    keys.into_iter()
        .filter_map(|key| {
            let grid = pieces.remove(&key)?;
            let shape = Arc::new(BodyShape::from_voxels(IVec3::splat(piece), grid)?);
            let origin = (key * piece).as_dvec3();
            Some(Debris {
                pos: (origin + shape.com.as_dvec3()) * f64::from(VOXEL_M),
                rot: shape.principal,
                shape,
                vel: Vec3::ZERO,
                ang_vel: Vec3::ZERO,
            })
        })
        .collect()
}

/// World position of a body's grid origin, for checking a cut body lines
/// up with where its voxels were.
pub fn grid_origin(d: &Debris) -> DVec3 {
    let grid_rot = d.rot * d.shape.principal.inverse();
    d.pos - (grid_rot * (d.shape.com * VOXEL_M)).as_dvec3()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;

    #[test]
    fn cut_pieces_hold_exactly_the_solid_voxels() {
        let mut w = VoxelWorld::new();
        let a = BlockPos(IVec3::new(10, 4, 10));
        let b = BlockPos(IVec3::new(11, 4, 10));
        w.fill_box(a.origin(), a.origin() + 15, ids::GRANITE);
        w.fill_box(b.origin(), b.origin() + 15, ids::PLANKS);
        // Water in one corner is not solid and simply goes.
        w.fill_box(b.origin(), b.origin() + 2, ids::WATER);
        let before = w.voxel(a.origin() + 3);
        assert_eq!(before, ids::GRANITE);
        let pieces = cut(&mut w, &[a, b], 32);
        // Both blocks lie in one 2 m piece.
        assert_eq!(pieces.len(), 1);
        let p = &pieces[0];
        assert_eq!(p.shape.solid_count as usize, 4096 * 2 - 27);
        assert!(w.voxel(a.origin() + 3).is_air());
        assert!(w.voxel(b.origin()).is_air());
        // The piece sits where the blocks were.
        let origin = grid_origin(p);
        assert!(
            (origin - DVec3::new(10.0, 4.0, 10.0)).length() < 1e-4,
            "{origin}"
        );

        // Half-metre crumbs: eight per block.
        let c = BlockPos(IVec3::new(20, 4, 20));
        w.fill_box(c.origin(), c.origin() + 15, ids::SANDSTONE);
        let crumbs = cut(&mut w, &[c], 8);
        assert_eq!(crumbs.len(), 8);
        assert!(crumbs.iter().all(|d| d.shape.solid_count == 512));
        // Lowest first.
        assert!(crumbs.windows(2).all(|w| w[0].pos.y <= w[1].pos.y + 1e-9));
    }
}
