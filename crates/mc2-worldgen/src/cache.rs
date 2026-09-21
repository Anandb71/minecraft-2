//! On-disk cache of the coarse terrain, so erosion runs once per world.
//!
//! Format (little endian): 8-byte magic, u32 version, u64 seed, u32 size,
//! f32 cell size, f32 sea level, u64 parameter fingerprint, then the height,
//! flow and sediment grids as raw f32, then a u64 FNV-1a checksum of
//! everything before it. Written to a temporary file and renamed, so a crash
//! mid-write never leaves a truncated cache that looks valid.

use crate::grid::Grid2;
use crate::terrain::{CoarseTerrain, TerrainParams};
use std::io::{self, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"MC2TERR\0";
/// 2: kinematic plate uplift replaced the noise mountain belt.
const VERSION: u32 = 2;
const HEADER: usize = 40;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Changes whenever a parameter that affects the result changes.
pub fn fingerprint(p: &TerrainParams) -> u64 {
    let e = p.erosion;
    let text = format!(
        "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
        p.size,
        p.cell_m,
        p.sea_level,
        e.iterations,
        e.dt,
        e.rain,
        e.pipe_area,
        e.gravity,
        e.capacity,
        e.dissolve,
        e.deposit,
        e.evaporation,
        e.min_tilt,
        e.talus,
        e.thermal_rate
    );
    fnv1a(text.as_bytes())
}

pub fn save(terrain: &CoarseTerrain, path: &Path) -> io::Result<()> {
    let p = &terrain.params;
    let mut buf = Vec::with_capacity(48 + p.size * p.size * 12);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&terrain.seed.to_le_bytes());
    buf.extend_from_slice(&(p.size as u32).to_le_bytes());
    buf.extend_from_slice(&p.cell_m.to_le_bytes());
    buf.extend_from_slice(&p.sea_level.to_le_bytes());
    buf.extend_from_slice(&fingerprint(p).to_le_bytes());
    for grid in [&terrain.height, &terrain.flow, &terrain.sediment] {
        for v in &grid.data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    let sum = fnv1a(&buf);
    buf.extend_from_slice(&sum.to_le_bytes());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    std::fs::rename(tmp, path)
}

/// Loads a cache matching `seed` and `params`; `Ok(None)` when absent or stale.
pub fn load(path: &Path, seed: u64, params: &TerrainParams) -> io::Result<Option<CoarseTerrain>> {
    let mut buf = Vec::new();
    match std::fs::File::open(path) {
        Ok(mut f) => f.read_to_end(&mut buf)?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let n = params.size * params.size;
    let expected = HEADER + n * 12 + 8;
    if buf.len() != expected || &buf[..8] != MAGIC {
        return Ok(None);
    }
    let (body, tail) = buf.split_at(buf.len() - 8);
    if fnv1a(body) != u64::from_le_bytes(tail.try_into().expect("8 bytes")) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "terrain cache checksum mismatch",
        ));
    }
    let u32_at = |o: usize| u32::from_le_bytes(body[o..o + 4].try_into().expect("4 bytes"));
    let u64_at = |o: usize| u64::from_le_bytes(body[o..o + 8].try_into().expect("8 bytes"));
    if u32_at(8) != VERSION || u64_at(12) != seed || u64_at(32) != fingerprint(params) {
        return Ok(None);
    }
    let grid = |k: usize| Grid2 {
        w: params.size,
        h: params.size,
        data: body[HEADER + k * n * 4..HEADER + (k + 1) * n * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    };
    Ok(Some(CoarseTerrain {
        seed,
        params: *params,
        height: grid(0),
        flow: grid(1),
        sediment: grid(2),
    }))
}

/// Loads the cached terrain or generates and caches it.
pub fn load_or_generate(
    path: &Path,
    seed: u64,
    params: TerrainParams,
    progress: &mut dyn FnMut(&str, f32),
) -> io::Result<CoarseTerrain> {
    match load(path, seed, &params) {
        Ok(Some(t)) => return Ok(t),
        Ok(None) => {}
        Err(e) => log_invalid(path, &e),
    }
    let terrain = CoarseTerrain::generate(seed, params, progress);
    save(&terrain, path)?;
    Ok(terrain)
}

fn log_invalid(path: &Path, e: &io::Error) {
    eprintln!("regenerating terrain: {} unusable ({e})", path.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erosion::ErosionParams;

    fn params() -> TerrainParams {
        TerrainParams {
            size: 32,
            cell_m: 512.0,
            sea_level: 96.0,
            erosion: ErosionParams {
                iterations: 10,
                ..Default::default()
            },
        }
    }

    #[test]
    fn roundtrip_and_staleness() {
        let dir = std::env::temp_dir().join("mc2_terrain_cache_test");
        let path = dir.join("terrain.bin");
        let _ = std::fs::remove_file(&path);
        let t = load_or_generate(&path, 9, params(), &mut |_, _| {}).unwrap();
        let back = load(&path, 9, &params()).unwrap().expect("cache hit");
        assert_eq!(back.height, t.height);
        assert_eq!(back.sediment, t.sediment);
        assert!(
            load(&path, 10, &params()).unwrap().is_none(),
            "other seed is stale"
        );
        let mut changed = params();
        changed.erosion.rain *= 2.0;
        assert!(
            load(&path, 9, &changed).unwrap().is_none(),
            "other params are stale"
        );
    }

    #[test]
    fn corruption_is_detected() {
        let dir = std::env::temp_dir().join("mc2_terrain_cache_corrupt");
        let path = dir.join("terrain.bin");
        let t = CoarseTerrain::generate(3, params(), &mut |_, _| {});
        save(&t, &path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[100] ^= 0xff;
        std::fs::write(&path, bytes).unwrap();
        assert!(load(&path, 3, &params()).is_err());
    }
}
