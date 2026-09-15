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

#[cfg(test)]
mod tests {
    use super::*;

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
}
