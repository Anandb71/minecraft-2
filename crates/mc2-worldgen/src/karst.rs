//! Karst caves: water dissolving carbonate rock along bedding and joints.
//!
//! Noise-worm tunnels were rejected in step 4. These passages follow two
//! vertical joint sets and a bedding plane, so they open in limestone, chalk
//! and marble, stay mostly horizontal, and meet in rooms. Lower passages
//! flood to a water table. Sinkholes punch down where a river sits over a
//! shaft.

use crate::amplify::Surface;
use crate::chunkgen::Lod;
use crate::noise::{Perlin, hash_unit, hash3};
use crate::strata::Strata;
use glam::{IVec3, Vec3};
use mc2_voxel::brick::Brick;
use mc2_voxel::coords::{CHUNK_VOXELS, ChunkPos};
use mc2_voxel::material::{MaterialId, ids};
use mc2_voxel::tree::{Cell, ChunkTree};

const VOXEL_M: f32 = 1.0 / 16.0;
/// Joint half-width in noise units. Wider is more Swiss cheese.
const JOINT: f32 = 0.13;
const OPEN_AT: f32 = 0.46;
const ROOF_M: f32 = 3.0;
/// On a steep face the slope itself is the entrance, so the roof thins.
const HILLSIDE: f32 = 0.42;
const HILLSIDE_ROOF_M: f32 = 0.5;
const FLOOR_M: f32 = 88.0;

fn roof_clearance(slope: f32) -> f32 {
    if slope > HILLSIDE {
        HILLSIDE_ROOF_M
    } else {
        ROOF_M
    }
}

struct At {
    surface: f32,
    sea: f32,
    slope: f32,
}

struct DripAt {
    roof: Option<f32>,
    floor: Option<f32>,
    sea: f32,
    surface: f32,
}

pub struct Karst {
    noise: Perlin,
    seed: u64,
}

pub fn soluble(m: MaterialId) -> bool {
    m == ids::LIMESTONE || m == ids::CHALK || m == ids::MARBLE
}

impl Karst {
    pub fn new(seed: u64) -> Self {
        Self {
            noise: Perlin::new(seed ^ 0x0ca7_e501),
            seed,
        }
    }

    fn joints(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let n = &self.noise;
        let bed = n
            .noise3(x * (1.0 / 46.0), y * (1.0 / 6.2), z * (1.0 / 46.0))
            .abs();
        let jx = n
            .noise3(x * (1.0 / 8.4), y * (1.0 / 40.0), z * (1.0 / 54.0) + 19.0)
            .abs();
        let jz = n
            .noise3(x * (1.0 / 54.0) + 11.0, y * (1.0 / 40.0), z * (1.0 / 8.4))
            .abs();
        (bed, jx, jz)
    }

    /// 0 closed, 1 fully open. Independent of the host rock. `slope` is
    /// rise over run of the ground above this column: steep ground cuts
    /// the roof so a joint can daylight.
    pub fn openness(&self, x: f32, y: f32, z: f32, surface: f32, slope: f32) -> f32 {
        let depth = surface - y;
        if !(roof_clearance(slope)..=FLOOR_M).contains(&depth) || y < 8.0 {
            return 0.0;
        }
        let (bed, jx, jz) = self.joints(x, y, z);
        let corridor = (1.0 - (jx / JOINT).clamp(0.0, 1.0)) * (1.0 - (jz / JOINT).clamp(0.0, 1.0));
        let chamber = corridor * (1.0 - (bed / (JOINT + 0.05)).clamp(0.0, 1.0));
        let room = self
            .noise
            .fbm3(x / 72.0, y / 30.0, z / 72.0, 3, 2.0, 0.5)
            .max(0.0);
        (corridor * 0.9 + chamber * 0.5 + room * 0.18).min(1.0)
    }

    fn sinkhole_shaft(&self, p: Vec3, surface: f32, sea: f32, river: f32) -> bool {
        if river < 0.62 || surface <= sea + 4.0 {
            return false;
        }
        if p.y < 8.0 || p.y > surface || p.y < surface - 22.0 {
            return false;
        }
        let (_, jx, jz) = self.joints(p.x, surface - 8.0, p.z);
        jx <= 0.11 && jz <= 0.11
    }

    /// What a soluble voxel becomes, or `None` to leave it.
    pub fn dissolve(
        &self,
        host: MaterialId,
        p: Vec3,
        surface: f32,
        sea: f32,
        river: f32,
        slope: f32,
    ) -> Option<MaterialId> {
        if !soluble(host) {
            return None;
        }
        let joint = self.openness(p.x, p.y, p.z, surface, slope) >= OPEN_AT;
        if !joint && !self.sinkhole_shaft(p, surface, sea, river) {
            return None;
        }
        let water_table = sea.max(surface - 26.0);
        Some(if p.y < water_table {
            ids::WATER
        } else {
            ids::AIR
        })
    }

    /// Length of the drip on this 2-voxel column, metres, if it grows one.
    fn drip_length(&self, vx: i32, vz: i32) -> Option<f32> {
        let h = hash3(vx >> 1, 0, vz >> 1, self.seed ^ 0xD51B);
        if hash_unit(h) > 0.2 {
            return None;
        }
        Some(0.35 + hash_unit(h.rotate_left(17)) * 2.05)
    }

    /// Limestone where a stalactite or stalagmite occupies this air voxel.
    /// `roof` and `floor` are the contact planes in metres, if known.
    pub fn drip(
        &self,
        p: Vec3,
        roof: Option<f32>,
        floor: Option<f32>,
        sea: f32,
        surface: f32,
    ) -> Option<MaterialId> {
        let water_table = sea.max(surface - 26.0);
        if p.y < water_table {
            return None;
        }
        let vx = (p.x / VOXEL_M).floor() as i32;
        let vz = (p.z / VOXEL_M).floor() as i32;
        let len = self.drip_length(vx, vz)?;
        let off_axis = (vx & 1) != 0 || (vz & 1) != 0;
        let fits = |along: f32, reach: f32| {
            along >= 0.0 && along <= reach && (!off_axis || along <= reach * 0.42)
        };
        if let Some(roof) = roof
            && fits(roof - p.y, len)
        {
            return Some(ids::LIMESTONE);
        }
        if let Some(floor) = floor
            && fits(p.y - floor, len * 0.7)
        {
            return Some(ids::LIMESTONE);
        }
        None
    }

    /// Dress one voxel column inside a brick. `y_base` is the world voxel y
    /// of `mats[0]`.
    pub fn dress_column(
        &self,
        mats: &mut [MaterialId; 8],
        y_base: i32,
        vx: i32,
        vz: i32,
        surface: f32,
        sea: f32,
    ) {
        if self.drip_length(vx, vz).is_none() {
            return;
        }
        let x = vx as f32 * VOXEL_M + VOXEL_M * 0.5;
        let z = vz as f32 * VOXEL_M + VOXEL_M * 0.5;
        for ly in 0..8i32 {
            if !mats[ly as usize].is_air() {
                continue;
            }
            let mut roof = None;
            for up in (ly + 1)..8 {
                if mats[up as usize].is_air() {
                    continue;
                }
                if soluble(mats[up as usize]) {
                    roof = Some((y_base + up) as f32 * VOXEL_M);
                }
                break;
            }
            let mut floor = None;
            for down in (0..ly).rev() {
                if mats[down as usize].is_air() {
                    continue;
                }
                if soluble(mats[down as usize]) {
                    floor = Some((y_base + down + 1) as f32 * VOXEL_M);
                }
                break;
            }
            if roof.is_none() && floor.is_none() {
                continue;
            }
            let y = (y_base + ly) as f32 * VOXEL_M + VOXEL_M * 0.5;
            if let Some(m) = self.drip(Vec3::new(x, y, z), roof, floor, sea, surface) {
                mats[ly as usize] = m;
            }
        }
    }

    /// Carves passages into an already-generated tree. Node8 skips: caves
    /// thinner than 8 m would alias into Swiss cheese at that lod.
    pub fn carve(
        &self,
        tree: &mut ChunkTree,
        pos: ChunkPos,
        surface: &Surface,
        strata: &Strata,
        lod: Lod,
    ) {
        if lod == Lod::Node8 {
            return;
        }
        let step = match lod {
            Lod::Node2 => 64,
            _ => 32,
        };
        let origin = pos.origin();
        let sea = surface.terrain.params.sea_level;
        let n = CHUNK_VOXELS / step;
        let mut opened = false;
        for cz in 0..n {
            for cy in 0..n {
                for cx in 0..n {
                    let centre = origin + IVec3::new(cx, cy, cz) * step + step / 2;
                    let p = centre.as_vec3() * VOXEL_M;
                    let sample = surface.sample(p.x, p.z);
                    let host = strata.layer(p.x, p.y, p.z).material;
                    let open = self.openness(p.x, p.y, p.z, sample.height, sample.slope);
                    if open < OPEN_AT || !soluble(host) {
                        continue;
                    }
                    let radius_m = 1.15 + 2.6 * open;
                    if lod == Lod::Node2 && radius_m < 3.2 {
                        continue;
                    }
                    opened |= self.carve_ellipsoid(
                        tree,
                        origin,
                        centre,
                        radius_m,
                        At {
                            surface: sample.height,
                            sea,
                            slope: sample.slope,
                        },
                    );
                }
            }
        }
        opened |= self.carve_sinkholes(tree, origin, surface, sea, lod);
        if opened && lod == Lod::Full {
            self.decorate(tree, origin, surface, sea);
        }
    }

    fn carve_sinkholes(
        &self,
        tree: &mut ChunkTree,
        origin: IVec3,
        surface: &Surface,
        sea: f32,
        lod: Lod,
    ) -> bool {
        if lod == Lod::Node2 {
            return false;
        }
        let step = 32;
        let n = CHUNK_VOXELS / step;
        let mut opened = false;
        for cz in 0..n {
            for cx in 0..n {
                let x = (origin.x + cx * step + step / 2) as f32 * VOXEL_M;
                let z = (origin.z + cz * step + step / 2) as f32 * VOXEL_M;
                let sample = surface.sample(x, z);
                if sample.river < 0.62 || sample.height <= sea + 4.0 {
                    continue;
                }
                let (_, jx, jz) = self.joints(x, sample.height - 8.0, z);
                if jx > 0.11 || jz > 0.11 {
                    continue;
                }
                let top = (sample.height * 16.0) as i32;
                let bottom = ((sample.height - 22.0).max(8.0) * 16.0) as i32;
                for y in bottom..=top {
                    let centre = IVec3::new(
                        origin.x + cx * step + step / 2,
                        y,
                        origin.z + cz * step + step / 2,
                    );
                    opened |= self.carve_ellipsoid(
                        tree,
                        origin,
                        centre,
                        1.35,
                        At {
                            surface: sample.height,
                            sea,
                            slope: sample.slope,
                        },
                    );
                }
            }
        }
        opened
    }

    fn carve_ellipsoid(
        &self,
        tree: &mut ChunkTree,
        origin: IVec3,
        centre: IVec3,
        radius_m: f32,
        at: At,
    ) -> bool {
        let At {
            surface,
            sea,
            slope,
        } = at;
        let rx = radius_m * 16.0;
        let ry = radius_m * 16.0 * 0.62;
        let rz = radius_m * 16.0;
        let lo: IVec3 =
            (centre - IVec3::new(rx as i32, ry as i32, rz as i32) - origin).max(IVec3::ZERO);
        let hi: IVec3 = (centre + IVec3::new(rx as i32, ry as i32, rz as i32) - origin)
            .min(IVec3::splat(CHUNK_VOXELS - 1));
        if lo.cmpgt(hi).any() {
            return false;
        }
        let cell_lo: IVec3 = lo >> 3;
        let cell_hi: IVec3 = hi >> 3;
        let mut opened = false;
        for cz in cell_lo.z..=cell_hi.z {
            for cy in cell_lo.y..=cell_hi.y {
                for cx in cell_lo.x..=cell_hi.x {
                    let local = IVec3::new(cx, cy, cz) * 8 + 4;
                    let world = local + origin;
                    let d = (world - centre).as_vec3();
                    if (d.x / rx).powi(2) + (d.y / ry).powi(2) + (d.z / rz).powi(2) > 1.0 {
                        continue;
                    }
                    let cell = IVec3::new(cx, cy, cz);
                    // Palette bricks already have voxel-accurate walls from
                    // `brick_at`; smashing the cell would flatten them.
                    if matches!(tree.cell(cell), Cell::Brick(_)) {
                        continue;
                    }
                    let here = tree.voxel(local);
                    let p = world.as_vec3() * VOXEL_M;
                    let Some(fill) = self.dissolve(here, p, surface, sea, 0.0, slope) else {
                        continue;
                    };
                    if fill.is_air() {
                        tree.set_cell(cell, Cell::Empty);
                    } else {
                        tree.set_cell(cell, Cell::Uniform(fill));
                    }
                    opened = true;
                }
            }
        }
        opened
    }

    fn carbonate_ceiling(&self, tree: &ChunkTree, cell: IVec3) -> bool {
        if cell.cmplt(IVec3::ZERO).any() || cell.cmpge(IVec3::splat(64)).any() {
            return false;
        }
        match tree.cell(cell) {
            Cell::Uniform(m) => soluble(m),
            Cell::Brick(_) => soluble(tree.voxel(cell * 8 + IVec3::new(4, 0, 4))),
            _ => false,
        }
    }

    fn carbonate_floor(&self, tree: &ChunkTree, cell: IVec3) -> bool {
        if cell.cmplt(IVec3::ZERO).any() || cell.cmpge(IVec3::splat(64)).any() {
            return false;
        }
        match tree.cell(cell) {
            Cell::Uniform(m) => soluble(m),
            Cell::Brick(_) => soluble(tree.voxel(cell * 8 + IVec3::new(4, 7, 4))),
            _ => false,
        }
    }

    fn paint_drip(&self, tree: &mut ChunkTree, origin: IVec3, cell: IVec3, at: DripAt) {
        let mut brick = match tree.cell(cell) {
            Cell::Empty => Brick::empty(),
            Cell::Brick(i) => tree.brick(i).clone(),
            _ => return,
        };
        let mut any = false;
        for ly in 0..8 {
            for lz in 0..8 {
                for lx in 0..8 {
                    let v = IVec3::new(lx, ly, lz);
                    if !brick.get(v).is_air() {
                        continue;
                    }
                    let world = origin + cell * 8 + v;
                    let p = (world.as_vec3() + 0.5) * VOXEL_M;
                    if let Some(m) = self.drip(p, at.roof, at.floor, at.sea, at.surface) {
                        brick.set(v, m);
                        any = true;
                    }
                }
            }
        }
        if any {
            tree.set_brick(cell, brick);
        }
    }

    /// Stalactites from ceilings and stalagmites from floors, full detail
    /// only. Coarser levels stay open cells: a drip is thinner than them.
    fn decorate(&self, tree: &mut ChunkTree, origin: IVec3, surface: &Surface, sea: f32) {
        for cz in 0..64 {
            for cx in 0..64 {
                let x = (origin.x + cx * 8 + 4) as f32 * VOXEL_M;
                let z = (origin.z + cz * 8 + 4) as f32 * VOXEL_M;
                let mut height = None;
                for cy in (0..64).rev() {
                    let cell = IVec3::new(cx, cy, cz);
                    if !matches!(tree.cell(cell), Cell::Empty) {
                        continue;
                    }
                    let above = cell + IVec3::Y;
                    if !self.carbonate_ceiling(tree, above) {
                        continue;
                    }
                    let roof = (origin.y + above.y * 8) as f32 * VOXEL_M;
                    let ground = *height.get_or_insert_with(|| surface.sample(x, z).height);
                    for dy in 0..6 {
                        let c = cell - IVec3::Y * dy;
                        if c.y < 0 || (dy > 0 && !matches!(tree.cell(c), Cell::Empty)) {
                            break;
                        }
                        self.paint_drip(
                            tree,
                            origin,
                            c,
                            DripAt {
                                roof: Some(roof),
                                floor: None,
                                sea,
                                surface: ground,
                            },
                        );
                    }
                }
                for cy in 0..64 {
                    let cell = IVec3::new(cx, cy, cz);
                    let below = cell - IVec3::Y;
                    if !self.carbonate_floor(tree, below) {
                        continue;
                    }
                    if !matches!(tree.cell(cell), Cell::Empty | Cell::Brick(_)) {
                        continue;
                    }
                    let floor = (origin.y + cell.y * 8) as f32 * VOXEL_M;
                    let ground = *height.get_or_insert_with(|| surface.sample(x, z).height);
                    for dy in 0..4 {
                        let c = cell + IVec3::Y * dy;
                        if c.y >= 64 {
                            break;
                        }
                        match tree.cell(c) {
                            Cell::Empty | Cell::Brick(_) => self.paint_drip(
                                tree,
                                origin,
                                c,
                                DripAt {
                                    roof: None,
                                    floor: Some(floor),
                                    sea,
                                    surface: ground,
                                },
                            ),
                            _ => break,
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_carbonates_dissolve() {
        let k = Karst::new(1);
        assert!(
            k.dissolve(
                ids::GRANITE,
                Vec3::new(40.0, 50.0, 40.0),
                90.0,
                96.0,
                0.0,
                0.0,
            )
            .is_none()
        );
        assert!(soluble(ids::LIMESTONE) && soluble(ids::CHALK) && soluble(ids::MARBLE));
        assert!(!soluble(ids::SANDSTONE));
    }

    #[test]
    fn same_seed_same_openness() {
        let a = Karst::new(9);
        let b = Karst::new(9);
        let u = a.openness(120.0, 44.0, 80.0, 92.0, 0.0);
        assert_eq!(u, b.openness(120.0, 44.0, 80.0, 92.0, 0.0));
        assert_ne!(u, Karst::new(10).openness(120.0, 44.0, 80.0, 92.0, 0.0));
        assert!((0.0..=1.0).contains(&u));
    }

    #[test]
    fn limestone_opens_along_joints_not_everywhere() {
        let k = Karst::new(7);
        let mut hits = 0;
        let mut n = 0;
        for x in (0..360).step_by(4) {
            for y in 24..70 {
                for z in (0..360).step_by(4) {
                    n += 1;
                    if k.dissolve(
                        ids::LIMESTONE,
                        Vec3::new(x as f32, y as f32, z as f32),
                        92.0,
                        96.0,
                        0.0,
                        0.0,
                    )
                    .is_some()
                    {
                        hits += 1;
                    }
                }
            }
        }
        assert!(hits > 30, "karst never opened: {hits} / {n}");
        assert!(hits * 12 < n, "too much was dissolved: {hits} / {n}");
    }

    #[test]
    fn flat_ground_keeps_its_roof_and_a_hillside_does_not() {
        let k = Karst::new(4);
        let mut flat = 0;
        let mut steep = 0;
        for x in (0..180).step_by(3) {
            for z in (0..180).step_by(3) {
                let p = (x as f32, 90.0, z as f32, 91.5);
                if k.openness(p.0, p.1, p.2, p.3, 0.0) >= OPEN_AT {
                    flat += 1;
                }
                if k.openness(p.0, p.1, p.2, p.3, 0.8) >= OPEN_AT {
                    steep += 1;
                }
            }
        }
        assert_eq!(flat, 0, "a metre of soil should still seal flat ground");
        assert!(steep > 8, "a hillside should cut into joints: {steep}");
    }

    #[test]
    fn stalactite_hangs_under_a_dry_roof_and_stops() {
        let k = Karst::new(3);
        let surface = 140.0;
        let sea = 96.0;
        let roof = 130.0;
        let mut hung = None;
        for x in 0..48 {
            for z in 0..48 {
                let p = Vec3::new(
                    (x as f32 + 0.5) * VOXEL_M,
                    roof - VOXEL_M * 0.5,
                    (z as f32 + 0.5) * VOXEL_M,
                );
                if k.drip(p, Some(roof), None, sea, surface).is_some() {
                    hung = Some(p);
                    break;
                }
            }
            if hung.is_some() {
                break;
            }
        }
        let p = hung.expect("a column should grow a stalactite");
        let tip = Vec3::new(p.x, roof - 3.2, p.z);
        assert!(k.drip(tip, Some(roof), None, sea, surface).is_none());
        let flooded = Vec3::new(p.x, 40.0, p.z);
        assert!(
            k.drip(flooded, Some(42.0), None, sea, surface).is_none(),
            "drips do not grow under the water table"
        );
    }
}
