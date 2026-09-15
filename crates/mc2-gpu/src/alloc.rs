//! Segregated free-list allocator for word ranges inside fixed GPU buffers.
//!
//! GPU buffers cannot grow in place, so pools are sized up front and carved
//! into ranges. Requests round up to the smallest size class that fits;
//! freed ranges go back on that class's free list. Exact classes for the
//! hot sizes (bricks, state blocks) waste nothing; power-of-two classes
//! cover variable node blocks with at most 2x slack.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    pub offset: u32,
    pub words: u32,
}

pub struct RangeAllocator {
    classes: Vec<u32>,
    free: Vec<Vec<u32>>,
    high_water: u32,
    capacity: u32,
    used: u64,
}

impl RangeAllocator {
    /// `exact` sizes are added to power-of-two classes from 4 to 2^24 words.
    /// Offset 0 is reserved so that 0 can mean "none" in shader data.
    pub fn new(capacity: u32, exact: &[u32]) -> Self {
        let mut classes: Vec<u32> = (2..=24)
            .map(|p| 1u32 << p)
            .chain(exact.iter().copied())
            .collect();
        classes.sort_unstable();
        classes.dedup();
        Self {
            free: vec![Vec::new(); classes.len()],
            classes,
            high_water: 4,
            capacity,
            used: 0,
        }
    }

    fn class_of(&self, words: u32) -> Option<usize> {
        self.classes.iter().position(|&c| c >= words)
    }

    pub fn alloc(&mut self, words: u32) -> Option<Range> {
        let class = self.class_of(words.max(1))?;
        let size = self.classes[class];
        let offset = match self.free[class].pop() {
            Some(o) => o,
            None => {
                let end = self.high_water.checked_add(size)?;
                if end > self.capacity {
                    return None;
                }
                let o = self.high_water;
                self.high_water = end;
                o
            }
        };
        self.used += u64::from(size);
        Some(Range {
            offset,
            words: size,
        })
    }

    pub fn free(&mut self, r: Range) {
        let class = self
            .classes
            .iter()
            .position(|&c| c == r.words)
            .expect("range was not produced by this allocator");
        self.used -= u64::from(r.words);
        self.free[class].push(r.offset);
    }

    pub fn used_words(&self) -> u64 {
        self.used
    }

    pub fn high_water(&self) -> u32 {
        self.high_water
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_classes_and_reuse() {
        let mut a = RangeAllocator::new(10_000, &[89, 273]);
        let b = a.alloc(89).unwrap();
        assert_eq!(b.words, 89);
        assert_eq!(b.offset, 4);
        let n = a.alloc(20).unwrap();
        assert_eq!(n.words, 32);
        a.free(b);
        let b2 = a.alloc(80).unwrap();
        assert_eq!(b2, b, "freed brick slot is reused");
        assert_eq!(a.used_words(), 89 + 32);
    }

    #[test]
    fn capacity_is_enforced() {
        let mut a = RangeAllocator::new(1000, &[]);
        assert!(a.alloc(512).is_some());
        assert!(a.alloc(512).is_none());
        assert!(a.alloc(256).is_some());
    }
}
