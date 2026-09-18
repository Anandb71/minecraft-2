//! The tile table the CPU keeps: allocating tiles where water goes, freeing
//! them when it has long gone, and the edits sent between steps.

use super::{
    ALLOC_PER_FRAME, COOL_FRAMES, EDIT_OPEN, EDIT_SOLID, EDIT_WATER, FREE_STILL, FluidGpu, NONE,
    SOLID, TileReport,
};
use glam::IVec3;
use mc2_fluid::{TILE, TILE_CELLS, Terrain, index_of, local_of, near_slot, need_of, slot_offset};

/// The tile holding `cell` and the cell's index within it.
pub(super) fn tile_of(cell: IVec3) -> (IVec3, u32) {
    let t = IVec3::splat(TILE);
    (cell.div_euclid(t), index_of(cell.rem_euclid(t)) as u32)
}

impl FluidGpu {
    /// The slot holding tile `pos`, allocating it (gas and solid from the
    /// terrain) if needed; None when the pool is full.
    pub(super) fn allocate(&mut self, pos: IVec3, terrain: &dyn Terrain) -> Option<u32> {
        if let Some(&s) = self.index.get(&pos) {
            return Some(s);
        }
        let s = match self.free.pop() {
            Some(s) => s,
            None if self.slot_count < self.max_slots => {
                self.slot_count += 1;
                self.slot_count - 1
            }
            None => return None,
        };
        let su = s as usize;
        self.slots[su] = Some(pos);
        self.index.insert(pos, s);
        self.water[su] = false;
        self.still[su] = 0;
        self.need[su] = 0;
        // Link both ways: this slot's whole row, and each neighbour's entry
        // pointing back.
        for slot in 0..27 {
            let d = slot_offset(slot);
            let o = if d == IVec3::ZERO {
                Some(s)
            } else {
                self.index.get(&(pos + d)).copied()
            };
            self.near[su * 27 + slot] = o.unwrap_or(NONE);
            if let Some(o) = o {
                self.near[o as usize * 27 + near_slot(-d)] = s;
                self.dirty_rows.insert(o);
            }
        }
        self.dirty_rows.insert(s);
        let kinds: Vec<u32> = (0..TILE_CELLS)
            .map(|i| {
                if terrain.solid(pos * TILE + local_of(i)) {
                    SOLID
                } else {
                    0
                }
            })
            .collect();
        self.fresh.push((s, kinds));
        // Asleep but due to wake: it steps from the next wake or activity.
        self.states.push((
            s,
            TileReport {
                alloc: 1,
                ..TileReport::default()
            },
        ));
        Some(s)
    }

    fn release(&mut self, s: u32) {
        let su = s as usize;
        let Some(pos) = self.slots[su].take() else {
            return;
        };
        self.index.remove(&pos);
        for slot in 0..27 {
            let o = self.near[su * 27 + slot];
            if o != NONE && o != s {
                self.near[o as usize * 27 + near_slot(-slot_offset(slot))] = NONE;
                self.dirty_rows.insert(o);
            }
        }
        self.water[su] = false;
        self.need[su] = 0;
        self.states.push((s, TileReport::default()));
        self.cooling.push((s, self.frame));
    }

    /// Marks slot `s` and its neighbours to have their surface closed and
    /// to wake.
    fn touch(&mut self, s: u32) {
        for slot in 0..27 {
            let o = self.near[s as usize * 27 + slot];
            if o != NONE {
                self.touched.insert(o);
                self.woken.insert(o);
            }
        }
    }

    /// Fills cells with still water at hydrostatic pressure (counting the
    /// water added above each cell). Cells in solid are skipped.
    pub fn add_water(&mut self, cells: &[IVec3], terrain: &dyn Terrain) {
        let set: mc2_core::FxHashSet<IVec3> = cells.iter().copied().collect();
        let mut needs: Vec<IVec3> = Vec::new();
        for &c in cells {
            if terrain.solid(c) {
                continue;
            }
            let (tp, i) = tile_of(c);
            let Some(s) = self.allocate(tp, terrain) else {
                continue;
            };
            let mut above = 0.0;
            let mut n = c + IVec3::Y;
            while set.contains(&n) {
                above += 1.0;
                n += IVec3::Y;
            }
            let rho = self.params.hydrostatic_rho(above);
            self.edits
                .push([s * TILE_CELLS as u32 + i, EDIT_WATER, rho.to_bits(), 0]);
            self.touch(s);
            let bits = need_of(local_of(i as usize));
            for slot in 0..27 {
                if bits & (1 << slot) != 0 {
                    needs.push(tp + slot_offset(slot));
                }
            }
        }
        // The tiles the new water needs, now rather than a report later.
        needs.sort_unstable_by_key(|p| (p.x, p.y, p.z));
        needs.dedup();
        for pos in needs {
            if let Some(s) = self.allocate(pos, terrain) {
                self.woken.insert(s);
            }
        }
    }

    /// Terrain changed in the cells `lo..=hi`: close or open them where a
    /// tile exists, then close the surface and wake the tiles around.
    pub fn terrain_changed(&mut self, lo: IVec3, hi: IVec3, terrain: &dyn Terrain) {
        let t = IVec3::splat(TILE);
        let (tlo, thi) = (lo.div_euclid(t), hi.div_euclid(t));
        for z in tlo.z..=thi.z {
            for y in tlo.y..=thi.y {
                for x in tlo.x..=thi.x {
                    let tp = IVec3::new(x, y, z);
                    let Some(&s) = self.index.get(&tp) else {
                        continue;
                    };
                    let (a, b) = ((lo - tp * t).max(IVec3::ZERO), (hi - tp * t).min(t - 1));
                    for cz in a.z..=b.z {
                        for cy in a.y..=b.y {
                            for cx in a.x..=b.x {
                                let l = IVec3::new(cx, cy, cz);
                                let op = if terrain.solid(tp * t + l) {
                                    EDIT_SOLID
                                } else {
                                    EDIT_OPEN
                                };
                                self.edits.push([
                                    s * TILE_CELLS as u32 + index_of(l) as u32,
                                    op,
                                    0,
                                    0,
                                ]);
                            }
                        }
                    }
                    self.touch(s);
                }
            }
        }
    }

    /// Acts on the last report: allocates the tiles water asks for and
    /// frees tiles long asleep, dry, and with no wet neighbour.
    pub fn maintain(&mut self, terrain: &dyn Terrain) {
        self.frame += 1;
        let frame = self.frame;
        let (cooled, cooling): (Vec<_>, Vec<_>) = self
            .cooling
            .iter()
            .partition(|&&(_, at)| frame >= at + COOL_FRAMES);
        self.cooling = cooling;
        self.free.extend(cooled.into_iter().map(|(s, _)| s));

        let mut wanted = Vec::new();
        for s in 0..self.slot_count as usize {
            let Some(pos) = self.slots[s] else { continue };
            let need = self.need[s];
            for slot in 0..27 {
                if need & (1 << slot) != 0 && self.near[s * 27 + slot] == NONE {
                    wanted.push(pos + slot_offset(slot));
                }
            }
        }
        wanted.sort_unstable_by_key(|p| (p.x, p.y, p.z));
        wanted.dedup();
        for pos in wanted.into_iter().take(ALLOC_PER_FRAME) {
            if let Some(s) = self.allocate(pos, terrain) {
                self.woken.insert(s);
            }
        }

        let mut dry = Vec::new();
        for s in 0..self.slot_count as usize {
            if self.slots[s].is_none() || self.water[s] || self.still[s] <= FREE_STILL {
                continue;
            }
            let wet_near = (0..27).any(|slot| {
                let o = self.near[s * 27 + slot];
                o != NONE && self.water[o as usize]
            });
            if !wet_near {
                dry.push(s as u32);
            }
        }
        for s in dry {
            self.release(s);
        }
    }
}
