//! Sky map: the altitude of the highest solid 0.5 m cell in every column
//! within 224 m of the camera, as a toroidally addressed GPU texture.
//!
//! Indirect rays that hit surfaces off screen cannot reuse the lit frame, and
//! a shadow ray and a sky ray per hit would triple their cost. Against this
//! height field a hit is sky-lit when nothing stands above its column, and
//! sun-lit when a short 2D march toward the sun clears the heights: caves
//! and overhangs stay dark, open ground stays lit, and the lookups are
//! texture reads. Weather and audio will ask the same question later.

use glam::{DVec3, IVec2, IVec3};
use mc2_core::FxHashMap;
use mc2_voxel::coords::{BRICK_VOXELS, CHUNK_VOXELS, ChunkPos};
use mc2_voxel::tree::{Cell, ChunkTree};

/// Texels per side; each is one 0.5 m column.
pub const SIZE: u32 = 1024;
const CELLS_PER_CHUNK: i32 = CHUNK_VOXELS / BRICK_VOXELS;
/// Columns this many chunks from the camera column are drawn: 15 chunks
/// (960 cells) fit the 1024-texel torus without aliasing.
const RADIUS_CHUNKS: i32 = 7;
/// Height of a column with no loaded geometry: open to the sky.
pub const UNKNOWN: f32 = -1.0e4;

/// Top of the highest occupied cell per column of one chunk, as cells above
/// the chunk floor (1..=64), or 0 where the column is empty.
pub fn chunk_tops(tree: &ChunkTree) -> Box<[u8; 4096]> {
    let mut tops = Box::new([0u8; 4096]);
    let n = CELLS_PER_CHUNK;
    let mut raise = |x: i32, z: i32, top: i32| {
        let i = (x + z * n) as usize;
        tops[i] = tops[i].max(top as u8);
    };
    for (c, cell) in tree.cells() {
        if !matches!(cell, Cell::Empty) {
            raise(c.x, c.z, c.y + 1);
        }
    }
    for node in tree.uniform_nodes() {
        if node.material.is_air() {
            continue;
        }
        let top = node.min_cell.y + node.cells;
        for z in 0..node.cells {
            for x in 0..node.cells {
                raise(node.min_cell.x + x, node.min_cell.z + z, top);
            }
        }
    }
    tops
}

/// A chunk column's loaded chunks: chunk y and that chunk's column tops.
type Column = Vec<(i32, Box<[u8; 4096]>)>;

pub struct SkyMap {
    columns: FxHashMap<IVec2, Column>,
    centre: Option<IVec2>,
    dirty: Vec<IVec2>,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl SkyMap {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky map"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self {
            columns: FxHashMap::default(),
            centre: None,
            dirty: Vec::new(),
            texture,
            view,
        }
    }

    /// Records a chunk's new contents (or its removal).
    pub fn update_chunk(&mut self, pos: ChunkPos, tree: Option<&ChunkTree>) {
        let key = IVec2::new(pos.0.x, pos.0.z);
        let column = self.columns.entry(key).or_default();
        column.retain(|(y, _)| *y != pos.0.y);
        if let Some(tree) = tree
            && !tree.is_empty()
        {
            column.push((pos.0.y, chunk_tops(tree)));
        }
        if column.is_empty() {
            self.columns.remove(&key);
        }
        self.dirty.push(key);
    }

    /// Heights (metres) of one chunk column's 64x64 cells.
    fn column_heights(&self, key: IVec2) -> Vec<f32> {
        let mut out = vec![UNKNOWN; 4096];
        if let Some(column) = self.columns.get(&key) {
            for (y, tops) in column {
                let floor_cells = y * CELLS_PER_CHUNK;
                for (h, &t) in out.iter_mut().zip(tops.iter()) {
                    if t > 0 {
                        let m = (floor_cells + i32::from(t)) as f32 * 0.5;
                        *h = h.max(m);
                    }
                }
            }
        }
        out
    }

    /// Uploads changed columns near the camera and columns that entered the
    /// window since the last call.
    pub fn upload(&mut self, queue: &wgpu::Queue, camera_m: DVec3) {
        let chunk_m = f64::from(CHUNK_VOXELS) / 16.0;
        let centre = IVec2::new(
            (camera_m.x / chunk_m).floor() as i32,
            (camera_m.z / chunk_m).floor() as i32,
        );
        let inside = |c: IVec2, k: IVec2| (k - c).abs().max_element() <= RADIUS_CHUNKS;
        let mut write: Vec<IVec2> = Vec::new();
        if self.centre != Some(centre) {
            let old = self.centre;
            for z in -RADIUS_CHUNKS..=RADIUS_CHUNKS {
                for x in -RADIUS_CHUNKS..=RADIUS_CHUNKS {
                    let k = centre + IVec2::new(x, z);
                    if old.is_none_or(|o| !inside(o, k)) {
                        write.push(k);
                    }
                }
            }
            // Forget columns far outside the window.
            self.columns
                .retain(|k, _| (*k - centre).abs().max_element() <= RADIUS_CHUNKS * 2);
            self.centre = Some(centre);
        }
        for k in std::mem::take(&mut self.dirty) {
            if inside(centre, k) {
                write.push(k);
            }
        }
        write.sort_unstable_by_key(|k| (k.x, k.y));
        write.dedup();
        for k in write {
            let heights = self.column_heights(k);
            // Toroidal: world cell c lives at texel c mod SIZE.
            let origin = k * CELLS_PER_CHUNK;
            let size = SIZE as i32;
            let tx = origin.x.rem_euclid(size);
            let tz = origin.y.rem_euclid(size);
            // A chunk column may straddle the wrap; split into up to four.
            let n = CELLS_PER_CHUNK;
            let spans = |t: i32| {
                if t + n <= size {
                    vec![(t, 0, n)]
                } else {
                    vec![(t, 0, size - t), (0, size - t, n - (size - t))]
                }
            };
            for (dst_z, src_z, len_z) in spans(tz) {
                for (dst_x, src_x, len_x) in spans(tx) {
                    let mut block = Vec::with_capacity((len_x * len_z) as usize);
                    for z in 0..len_z {
                        for x in 0..len_x {
                            block.push(heights[((src_x + x) + (src_z + z) * n) as usize]);
                        }
                    }
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: dst_x as u32,
                                y: dst_z as u32,
                                z: 0,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        bytemuck::cast_slice(&block),
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * len_x as u32),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width: len_x as u32,
                            height: len_z as u32,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
        }
    }
}

/// World cell of a voxel position, for tests and CPU queries.
pub fn cell_of(voxel: IVec3) -> IVec2 {
    IVec2::new(
        voxel.x.div_euclid(BRICK_VOXELS),
        voxel.z.div_euclid(BRICK_VOXELS),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;

    #[test]
    fn tops_cover_cells_bricks_and_nodes() {
        let mut t = ChunkTree::new();
        // A single voxel at cell (1, 5, 2): top is 6 cells.
        t.set_voxel(IVec3::new(8 + 3, 40 + 1, 16 + 2), ids::GRANITE);
        // A uniform 2 m node spanning cells x 4..8, y 0..4, z 4..8.
        t.set_node(1, IVec3::new(1, 0, 1), ids::GRANITE);
        let tops = chunk_tops(&t);
        assert_eq!(tops[(1 + 2 * 64) as usize], 6);
        assert_eq!(tops[(5 + 6 * 64) as usize], 4);
        assert_eq!(tops[0], 0);
        assert_eq!(cell_of(IVec3::new(-1, 0, 17)), IVec2::new(-1, 2));
    }
}
