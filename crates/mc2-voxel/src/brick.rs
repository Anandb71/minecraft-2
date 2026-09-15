//! 8^3 voxel brick with palette compression and per-subblock occupancy.
//!
//! Voxels index palette entries; entry 0 is always air. A narrow brick packs
//! indices into 4 bits (up to 16 materials, air included), which covers
//! almost every brick. A brick that needs more is promoted to 8-bit wide
//! storage rather than being quantised. Occupancy is one 64-bit mask per
//! 4^3 subblock, so traversal skips an empty quarter-metre with a single
//! comparison and finds the next solid voxel with a bit scan.
//!
//! Layout conventions shared with the GPU:
//! - voxel index `x + z*8 + y*64`
//! - subblock index `(x>>2) + (z>>2)*2 + (y>>2)*4`
//! - bit within a subblock mask `(x&3) + (z&3)*4 + (y&3)*16`

use crate::material::MaterialId;
use glam::IVec3;

pub const VOXELS: usize = 512;
pub const NARROW_CAP: usize = 16;

/// Damage, wetness, temperature band and burn progress, two bits each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VoxelState(pub u8);

impl VoxelState {
    const fn field(self, shift: u8) -> u8 {
        (self.0 >> shift) & 3
    }

    const fn with_field(self, shift: u8, v: u8) -> Self {
        Self((self.0 & !(3 << shift)) | ((v & 3) << shift))
    }

    pub const fn damage(self) -> u8 {
        self.field(0)
    }
    pub const fn wetness(self) -> u8 {
        self.field(2)
    }
    pub const fn temperature(self) -> u8 {
        self.field(4)
    }
    pub const fn burn(self) -> u8 {
        self.field(6)
    }
    pub const fn with_damage(self, v: u8) -> Self {
        self.with_field(0, v)
    }
    pub const fn with_wetness(self, v: u8) -> Self {
        self.with_field(2, v)
    }
    pub const fn with_temperature(self, v: u8) -> Self {
        self.with_field(4, v)
    }
    pub const fn with_burn(self, v: u8) -> Self {
        self.with_field(6, v)
    }
}

#[inline]
pub fn voxel_index(l: IVec3) -> usize {
    debug_assert!(l.cmpge(IVec3::ZERO).all() && l.cmplt(IVec3::splat(8)).all());
    (l.x + l.z * 8 + l.y * 64) as usize
}

#[inline]
pub fn subblock_bit(l: IVec3) -> (usize, u32) {
    let s = ((l.x >> 2) + ((l.z >> 2) << 1) + ((l.y >> 2) << 2)) as usize;
    let b = ((l.x & 3) + ((l.z & 3) << 2) + ((l.y & 3) << 4)) as u32;
    (s, b)
}

#[derive(Clone, Debug)]
struct Wide {
    palette: Vec<MaterialId>,
    counts: Vec<u16>,
    indices: Box<[u8; VOXELS]>,
}

// The narrow case is the common one and stays inline to avoid a heap
// allocation per brick; only the rare wide case is boxed.
#[expect(clippy::large_enum_variant, reason = "narrow bricks must not allocate")]
#[derive(Clone, Debug)]
enum Storage {
    Narrow {
        palette: [MaterialId; NARROW_CAP],
        counts: [u16; NARROW_CAP],
        len: u8,
        nibbles: [u8; VOXELS / 2],
    },
    Wide(Box<Wide>),
}

#[derive(Clone, Debug)]
pub struct Brick {
    storage: Storage,
    occupancy: [u64; 8],
    state: Option<Box<[VoxelState; VOXELS]>>,
}

impl Brick {
    pub fn empty() -> Self {
        let mut counts = [0; NARROW_CAP];
        counts[0] = VOXELS as u16;
        Self {
            storage: Storage::Narrow {
                palette: [MaterialId(0); NARROW_CAP],
                counts,
                len: 1,
                nibbles: [0; VOXELS / 2],
            },
            occupancy: [0; 8],
            state: None,
        }
    }

    pub fn filled(m: MaterialId) -> Self {
        let mut b = Self::empty();
        if m.is_air() {
            return b;
        }
        if let Storage::Narrow {
            palette,
            counts,
            len,
            nibbles,
        } = &mut b.storage
        {
            palette[1] = m;
            counts[0] = 0;
            counts[1] = VOXELS as u16;
            *len = 2;
            nibbles.fill(0x11);
        }
        b.occupancy = [u64::MAX; 8];
        b
    }

    #[inline]
    fn palette_index(&self, i: usize) -> usize {
        match &self.storage {
            Storage::Narrow { nibbles, .. } => ((nibbles[i >> 1] >> ((i & 1) * 4)) & 0xf) as usize,
            Storage::Wide(w) => w.indices[i] as usize,
        }
    }

    #[inline]
    pub fn get(&self, l: IVec3) -> MaterialId {
        let i = voxel_index(l);
        let p = self.palette_index(i);
        match &self.storage {
            Storage::Narrow { palette, .. } => palette[p],
            Storage::Wide(w) => w.palette[p],
        }
    }

    pub fn state(&self, l: IVec3) -> VoxelState {
        self.state
            .as_ref()
            .map_or(VoxelState::default(), |s| s[voxel_index(l)])
    }

    pub fn set_state(&mut self, l: IVec3, st: VoxelState) {
        if st == VoxelState::default() && self.state.is_none() {
            return;
        }
        let s = self
            .state
            .get_or_insert_with(|| Box::new([VoxelState::default(); VOXELS]));
        s[voxel_index(l)] = st;
    }

    /// Frees the state array when every voxel is back to the default state.
    pub fn compact_state(&mut self) {
        if self
            .state
            .as_ref()
            .is_some_and(|s| s.iter().all(|v| *v == VoxelState::default()))
        {
            self.state = None;
        }
    }

    pub fn has_state(&self) -> bool {
        self.state.is_some()
    }

    /// Palette slot for `m`, inserting (and compacting or widening) if needed.
    fn slot_for(&mut self, m: MaterialId) -> usize {
        if m.is_air() {
            return 0;
        }
        match &mut self.storage {
            Storage::Narrow {
                palette,
                counts,
                len,
                ..
            } => {
                let n = *len as usize;
                if let Some(p) = (1..n).find(|&p| palette[p] == m) {
                    return p;
                }
                // Reuse an entry whose voxels have all been overwritten.
                if let Some(p) = (1..n).find(|&p| counts[p] == 0) {
                    palette[p] = m;
                    return p;
                }
                if n < NARROW_CAP {
                    palette[n] = m;
                    counts[n] = 0;
                    *len += 1;
                    return n;
                }
                self.widen();
                self.slot_for(m)
            }
            Storage::Wide(w) => {
                if let Some(p) = w.palette.iter().position(|&x| x == m) {
                    return p;
                }
                if let Some(p) = (1..w.palette.len()).find(|&p| w.counts[p] == 0) {
                    w.palette[p] = m;
                    return p;
                }
                assert!(
                    w.palette.len() < 256,
                    "brick cannot hold more than 256 materials"
                );
                w.palette.push(m);
                w.counts.push(0);
                w.palette.len() - 1
            }
        }
    }

    fn widen(&mut self) {
        let Storage::Narrow {
            palette,
            counts,
            len,
            ..
        } = &self.storage
        else {
            return;
        };
        let n = *len as usize;
        let mut indices = Box::new([0u8; VOXELS]);
        for (i, idx) in indices.iter_mut().enumerate() {
            *idx = self.palette_index(i) as u8;
        }
        self.storage = Storage::Wide(Box::new(Wide {
            palette: palette[..n].to_vec(),
            counts: counts[..n].to_vec(),
            indices,
        }));
    }

    pub fn is_wide(&self) -> bool {
        matches!(self.storage, Storage::Wide(_))
    }

    /// Sets a voxel; returns the previous material.
    pub fn set(&mut self, l: IVec3, m: MaterialId) -> MaterialId {
        let i = voxel_index(l);
        let old_slot = self.palette_index(i);
        let old = match &self.storage {
            Storage::Narrow { palette, .. } => palette[old_slot],
            Storage::Wide(w) => w.palette[old_slot],
        };
        if old == m {
            return old;
        }
        let slot = self.slot_for(m);
        match &mut self.storage {
            Storage::Narrow {
                counts, nibbles, ..
            } => {
                counts[old_slot] -= 1;
                counts[slot] += 1;
                let shift = (i & 1) * 4;
                let byte = &mut nibbles[i >> 1];
                *byte = (*byte & !(0xf << shift)) | ((slot as u8) << shift);
            }
            Storage::Wide(w) => {
                w.counts[old_slot] -= 1;
                w.counts[slot] += 1;
                w.indices[i] = slot as u8;
            }
        }
        let (s, b) = subblock_bit(l);
        if m.is_air() {
            self.occupancy[s] &= !(1u64 << b);
        } else {
            self.occupancy[s] |= 1u64 << b;
        }
        old
    }

    pub fn occupancy(&self) -> &[u64; 8] {
        &self.occupancy
    }

    #[inline]
    pub fn is_occupied(&self, l: IVec3) -> bool {
        let (s, b) = subblock_bit(l);
        self.occupancy[s] >> b & 1 != 0
    }

    pub fn solid_count(&self) -> u32 {
        self.occupancy.iter().map(|m| m.count_ones()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.occupancy.iter().all(|&m| m == 0)
    }

    /// The single material filling every voxel, if any, with no voxel state.
    pub fn uniform(&self) -> Option<MaterialId> {
        if self.state.is_some() {
            return None;
        }
        self.materials()
            .find(|&(_, c)| c as usize == VOXELS)
            .map(|(m, _)| m)
    }

    /// Non-empty palette entries with voxel counts (air included).
    pub fn materials(&self) -> impl Iterator<Item = (MaterialId, u16)> + '_ {
        let (palette, counts): (&[MaterialId], &[u16]) = match &self.storage {
            Storage::Narrow {
                palette,
                counts,
                len,
                ..
            } => (&palette[..*len as usize], &counts[..*len as usize]),
            Storage::Wide(w) => (&w.palette, &w.counts),
        };
        palette
            .iter()
            .zip(counts)
            .filter(|&(_, &c)| c > 0)
            .map(|(&m, &c)| (m, c))
    }

    /// Most common non-air material, used for LOD filtering.
    pub fn dominant(&self) -> Option<(MaterialId, u16)> {
        self.materials()
            .filter(|(m, _)| !m.is_air())
            .max_by_key(|&(_, c)| c)
    }

    /// Raw palette (padded to 16 for narrow bricks) and packed indices, for upload.
    pub fn raw(&self) -> BrickRaw<'_> {
        match &self.storage {
            Storage::Narrow {
                palette, nibbles, ..
            } => BrickRaw::Narrow { palette, nibbles },
            Storage::Wide(w) => BrickRaw::Wide {
                palette: &w.palette,
                indices: &w.indices,
            },
        }
    }

    pub fn raw_state(&self) -> Option<&[VoxelState; VOXELS]> {
        self.state.as_deref()
    }

    /// Approximate heap plus inline bytes.
    pub fn memory_bytes(&self) -> usize {
        let inline = std::mem::size_of::<Self>();
        let wide = match &self.storage {
            Storage::Narrow { .. } => 0,
            Storage::Wide(w) => std::mem::size_of::<Wide>() + VOXELS + w.palette.capacity() * 4,
        };
        inline + wide + self.state.as_ref().map_or(0, |_| VOXELS)
    }
}

pub enum BrickRaw<'a> {
    Narrow {
        palette: &'a [MaterialId; NARROW_CAP],
        nibbles: &'a [u8; VOXELS / 2],
    },
    Wide {
        palette: &'a [MaterialId],
        indices: &'a [u8; VOXELS],
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::ids;

    fn each_voxel() -> impl Iterator<Item = IVec3> {
        (0..512).map(|i| IVec3::new(i & 7, (i >> 6) & 7, (i >> 3) & 7))
    }

    #[test]
    fn set_get_roundtrip_and_occupancy() {
        let mut b = Brick::empty();
        assert!(b.is_empty());
        let p = IVec3::new(5, 6, 1);
        assert_eq!(b.set(p, ids::GRANITE), ids::AIR);
        assert_eq!(b.get(p), ids::GRANITE);
        assert!(b.is_occupied(p));
        assert_eq!(b.solid_count(), 1);
        assert_eq!(b.set(p, ids::AIR), ids::GRANITE);
        assert!(b.is_empty());
    }

    #[test]
    fn filled_brick_is_uniform_until_touched() {
        let mut b = Brick::filled(ids::BASALT);
        assert_eq!(b.uniform(), Some(ids::BASALT));
        assert_eq!(b.solid_count(), 512);
        b.set(IVec3::ZERO, ids::IRON_ORE);
        assert_eq!(b.uniform(), None);
        assert_eq!(b.dominant(), Some((ids::BASALT, 511)));
        b.set(IVec3::ZERO, ids::BASALT);
        assert_eq!(b.uniform(), Some(ids::BASALT));
    }

    #[test]
    fn widens_past_sixteen_materials_and_keeps_contents() {
        let mut b = Brick::empty();
        let voxels: Vec<IVec3> = each_voxel().collect();
        for (i, v) in voxels.iter().enumerate() {
            b.set(*v, MaterialId(1 + (i % 40) as u16));
        }
        assert!(b.is_wide());
        for (i, v) in voxels.iter().enumerate() {
            assert_eq!(b.get(*v), MaterialId(1 + (i % 40) as u16));
        }
        assert_eq!(b.solid_count(), 512);
    }

    #[test]
    fn narrow_palette_recycles_unused_slots() {
        let mut b = Brick::empty();
        let p = IVec3::new(1, 2, 3);
        for m in 1..200u16 {
            b.set(p, MaterialId(m));
        }
        assert!(!b.is_wide(), "one voxel cycling materials must not widen");
        assert_eq!(b.get(p), MaterialId(199));
    }

    #[test]
    fn voxel_state_packs_four_fields() {
        let s = VoxelState::default()
            .with_damage(3)
            .with_wetness(1)
            .with_temperature(2)
            .with_burn(3);
        assert_eq!(
            (s.damage(), s.wetness(), s.temperature(), s.burn()),
            (3, 1, 2, 3)
        );
        let mut b = Brick::filled(ids::OAK_LOG);
        b.set_state(IVec3::ONE, s);
        assert_eq!(b.uniform(), None);
        b.set_state(IVec3::ONE, VoxelState::default());
        b.compact_state();
        assert_eq!(b.uniform(), Some(ids::OAK_LOG));
    }

    #[test]
    fn subblock_layout_matches_voxel_layout() {
        let mut seen = [0u64; 8];
        for v in each_voxel() {
            let (s, bit) = subblock_bit(v);
            assert_eq!(seen[s] >> bit & 1, 0);
            seen[s] |= 1 << bit;
        }
        assert!(seen.iter().all(|&m| m == u64::MAX));
    }
}
