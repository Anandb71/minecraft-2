//! Emissive voxel clusters for ReSTIR.
//!
//! Every brick cell (or uniform node) containing emissive material becomes
//! one light: the centroid of its emissive voxels with a radius covering them.
//! Lights live in stable slots so a reservoir kept from last frame still
//! refers to the same emitter. The source distribution is an alias table
//! over slot power (emission luminance times emitting area), softened by
//! distance to the camera so a lava field a kilometre away does not starve
//! the torches in the room.

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, IVec3, Vec3};
use mc2_core::FxHashMap;
use mc2_gpu::{bind, bind_group, layout};
use mc2_voxel::brick::VOXELS;
use mc2_voxel::coords::{BRICK_VOXELS, ChunkPos};
use mc2_voxel::material::MaterialId;
use mc2_voxel::tree::{Cell, ChunkTree};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct GpuLight {
    pub voxel: [i32; 3],
    pub count: u32,
    pub emission: [f32; 3],
    pub radius_voxels: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuAlias {
    pub probability: f32,
    pub alias: u32,
    pub pdf: f32,
    pub _pad: f32,
}

/// Emitters of one chunk.
pub fn chunk_emitters(pos: ChunkPos, tree: &ChunkTree) -> Vec<GpuLight> {
    let origin = pos.origin();
    let mut out = Vec::new();
    let emission = |m: MaterialId| Vec3::from_array(m.get().emission);
    for (cell, c) in tree.cells() {
        let min = origin + cell * BRICK_VOXELS;
        match c {
            Cell::Uniform(m) if m.is_emissive() => out.push(GpuLight {
                voxel: (min + 4).to_array(),
                count: 512,
                emission: emission(m).to_array(),
                radius_voxels: 4.0,
            }),
            Cell::Brick(slot) => {
                let brick = tree.brick(slot);
                for (m, _) in brick.materials() {
                    if !m.is_emissive() {
                        continue;
                    }
                    let mut sum = Vec3::ZERO;
                    let mut count = 0u32;
                    let mut points = Vec::new();
                    for i in 0..VOXELS as i32 {
                        let l = IVec3::new(i & 7, (i >> 6) & 7, (i >> 3) & 7);
                        if brick.get(l) == m {
                            let p = l.as_vec3() + 0.5;
                            sum += p;
                            count += 1;
                            points.push(p);
                        }
                    }
                    let centre = sum / count.max(1) as f32;
                    let radius = points
                        .iter()
                        .map(|p| p.distance(centre))
                        .fold(0.5f32, f32::max);
                    let voxel = min + centre.floor().as_ivec3();
                    out.push(GpuLight {
                        voxel: voxel.to_array(),
                        count,
                        emission: emission(m).to_array(),
                        radius_voxels: radius + 0.5,
                    });
                }
            }
            _ => {}
        }
    }
    for node in tree.uniform_nodes() {
        if !node.material.is_emissive() {
            continue;
        }
        let half = node.cells * BRICK_VOXELS / 2;
        out.push(GpuLight {
            voxel: (origin + node.min_cell * BRICK_VOXELS + half).to_array(),
            count: (node.cells * node.cells * node.cells) as u32 * 512,
            emission: emission(node.material).to_array(),
            radius_voxels: half as f32,
        });
    }
    out
}

/// Vose's alias method: O(n) construction, O(1) sampling.
pub fn build_alias(weights: &[f32]) -> Vec<GpuAlias> {
    let n = weights.len();
    let total: f64 = weights.iter().map(|&w| f64::from(w.max(0.0))).sum();
    if n == 0 || total <= 0.0 {
        return vec![
            GpuAlias {
                probability: 1.0,
                alias: 0,
                pdf: 0.0,
                _pad: 0.0,
            };
            n.max(1)
        ];
    }
    let mut table: Vec<GpuAlias> = weights
        .iter()
        .map(|&w| GpuAlias {
            probability: 0.0,
            alias: 0,
            pdf: (f64::from(w.max(0.0)) / total) as f32,
            _pad: 0.0,
        })
        .collect();
    let mut scaled: Vec<f64> = weights
        .iter()
        .map(|&w| f64::from(w.max(0.0)) / total * n as f64)
        .collect();
    let (mut small, mut large): (Vec<usize>, Vec<usize>) = (0..n).partition(|&i| scaled[i] < 1.0);
    // Check both before popping: a tuple pattern would pop from `small` and
    // lose that entry when `large` is already empty.
    while let (Some(&s), Some(&l)) = (small.last(), large.last()) {
        small.pop();
        table[s].probability = scaled[s] as f32;
        table[s].alias = l as u32;
        scaled[l] -= 1.0 - scaled[s];
        if scaled[l] < 1.0 {
            large.pop();
            small.push(l);
        }
    }
    for i in small.into_iter().chain(large) {
        table[i].probability = 1.0;
        table[i].alias = i as u32;
    }
    table
}

pub struct LightRegistry {
    slots: Vec<GpuLight>,
    free: Vec<u32>,
    by_chunk: FxHashMap<ChunkPos, Vec<u32>>,
    dirty: bool,
    built_at: Option<DVec3>,
    /// Entries in the uploaded alias table, or 0 when no light has weight.
    sampled: u32,
    lights: wgpu::Buffer,
    alias: wgpu::Buffer,
    capacity: usize,
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    /// Lights beyond this distance get no candidates.
    pub radius_m: f64,
}

impl LightRegistry {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = layout(
            device,
            "lights",
            wgpu::ShaderStages::COMPUTE,
            &[bind::storage(true), bind::storage(true)],
        );
        let capacity = 1024;
        let (lights, alias, bind_group) = Self::buffers(device, &layout, capacity);
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            by_chunk: FxHashMap::default(),
            dirty: true,
            built_at: None,
            sampled: 0,
            lights,
            alias,
            capacity,
            layout,
            bind_group,
            radius_m: 192.0,
        }
    }

    fn buffers(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        capacity: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup) {
        let make = |label: &str, stride: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (capacity * stride) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let lights = make("lights", std::mem::size_of::<GpuLight>());
        let alias = make("light alias table", std::mem::size_of::<GpuAlias>());
        let bg = bind_group(
            device,
            "lights",
            layout,
            &[lights.as_entire_binding(), alias.as_entire_binding()],
        );
        (lights, alias, bg)
    }

    pub fn light_count(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    /// Length of the alias table shaders must sample, 0 when no light is in
    /// range. The GPU buffer is larger than the table, so shaders cannot use
    /// its array length.
    pub fn sampled_len(&self) -> u32 {
        self.sampled
    }

    /// Whether a changed chunk needs its emitters found again: it had some,
    /// or its change may have brought some.
    pub fn needs_scan(&self, pos: ChunkPos, maybe_lit: bool) -> bool {
        maybe_lit || self.by_chunk.contains_key(&pos)
    }

    /// Replaces a chunk's emitters; `None` removes the chunk.
    pub fn update_chunk(&mut self, pos: ChunkPos, tree: Option<&ChunkTree>) {
        if let Some(old) = self.by_chunk.remove(&pos) {
            for slot in old {
                self.slots[slot as usize] = GpuLight::zeroed();
                self.free.push(slot);
            }
            self.dirty = true;
        }
        let Some(tree) = tree else {
            return;
        };
        let emitters = chunk_emitters(pos, tree);
        if emitters.is_empty() {
            return;
        }
        let mut slots = Vec::with_capacity(emitters.len());
        for e in emitters {
            let slot = match self.free.pop() {
                Some(s) => {
                    self.slots[s as usize] = e;
                    s
                }
                None => {
                    self.slots.push(e);
                    self.slots.len() as u32 - 1
                }
            };
            slots.push(slot);
        }
        self.by_chunk.insert(pos, slots);
        self.dirty = true;
    }

    /// Rebuilds the alias table when lights changed or the camera moved.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, camera_m: DVec3) {
        let moved = self.built_at.is_none_or(|p| p.distance(camera_m) > 8.0);
        if !self.dirty && !moved {
            return;
        }
        self.dirty = false;
        self.built_at = Some(camera_m);
        let n = self.slots.len().max(1);
        if n > self.capacity {
            self.capacity = n.next_power_of_two();
            let (l, a, bg) = Self::buffers(device, &self.layout, self.capacity);
            self.lights = l;
            self.alias = a;
            self.bind_group = bg;
        }
        let cam_v = camera_m * 16.0;
        let radius_v = self.radius_m * 16.0;
        let weights: Vec<f32> = self
            .slots
            .iter()
            .map(|l| {
                if l.count == 0 {
                    return 0.0;
                }
                let e = Vec3::from_array(l.emission);
                let lum = e.dot(Vec3::new(0.2126, 0.7152, 0.0722));
                let area = (l.count as f32).powf(2.0 / 3.0) * 6.0 / 256.0;
                let d = (IVec3::from_array(l.voxel).as_dvec3() - cam_v).length();
                if d > radius_v {
                    return 0.0;
                }
                let d_m = (d / 16.0) as f32;
                lum * area / (d_m * d_m).max(16.0)
            })
            .collect();
        self.sampled = if weights.iter().any(|&w| w > 0.0) {
            weights.len() as u32
        } else {
            0
        };
        let mut lights = self.slots.clone();
        if lights.is_empty() {
            lights.push(GpuLight::zeroed());
        }
        let table = build_alias(if weights.is_empty() { &[0.0] } else { &weights });
        queue.write_buffer(&self.lights, 0, bytemuck::cast_slice(&lights));
        queue.write_buffer(&self.alias, 0, bytemuck::cast_slice(&table));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc2_voxel::material::ids;

    #[test]
    fn alias_table_reproduces_weights() {
        let weights = [1.0f32, 3.0, 0.0, 6.0];
        let table = build_alias(&weights);
        let mut freq = [0.0f64; 4];
        let n = table.len();
        let bins = 40_000;
        for k in 0..bins {
            // Deterministic stratified sampling over (bin, threshold).
            let u = (k as f64 + 0.5) / bins as f64;
            let bin = ((u * n as f64) as usize).min(n - 1);
            let frac = u * n as f64 - bin as f64;
            let slot = if frac < f64::from(table[bin].probability) {
                bin
            } else {
                table[bin].alias as usize
            };
            freq[slot] += 1.0 / bins as f64;
        }
        for (i, w) in weights.iter().enumerate() {
            let expect = f64::from(*w) / 10.0;
            assert!(
                (freq[i] - expect).abs() < 0.01,
                "slot {i}: {} vs {expect}",
                freq[i]
            );
            assert!((f64::from(table[i].pdf) - expect).abs() < 1e-6);
        }
    }

    #[test]
    fn emitters_from_cells_bricks_and_nodes() {
        let mut t = ChunkTree::new();
        // A torch flame: 4x3x4 emissive voxels in one brick.
        for x in 2..6 {
            for y in 1..4 {
                for z in 2..6 {
                    t.set_voxel(IVec3::new(x, y, z), ids::TORCH_FLAME);
                }
            }
        }
        t.set_cell(IVec3::new(10, 0, 0), Cell::Uniform(ids::LAVA));
        t.set_node(1, IVec3::new(4, 0, 4), ids::LAVA);
        let e = chunk_emitters(ChunkPos(IVec3::ZERO), &t);
        assert_eq!(e.len(), 3);
        let flame = e.iter().find(|l| l.count == 48).expect("flame cluster");
        assert_eq!(flame.voxel, [4, 2, 4]);
        assert!(flame.radius_voxels > 2.0 && flame.radius_voxels < 4.0);
        assert!(e.iter().any(|l| l.count == 512 * 64));
    }
}
