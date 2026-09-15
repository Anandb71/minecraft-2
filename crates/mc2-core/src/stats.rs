//! Rolling timing statistics. Every profiler row on the HUD is one of these.

/// Fixed-capacity ring of samples with mean and 99th percentile queries.
///
/// The percentile is computed on demand from a sorted copy; with a 256 sample
/// window that costs well under a microsecond per row and keeps inserts O(1).
#[derive(Clone, Debug)]
pub struct RollingStats {
    samples: Vec<f32>,
    head: usize,
    filled: usize,
}

impl RollingStats {
    pub const DEFAULT_WINDOW: usize = 256;

    pub fn new(window: usize) -> Self {
        assert!(window > 0, "rolling window must hold at least one sample");
        Self {
            samples: vec![0.0; window],
            head: 0,
            filled: 0,
        }
    }

    pub fn push(&mut self, value: f32) {
        self.samples[self.head] = value;
        self.head = (self.head + 1) % self.samples.len();
        self.filled = (self.filled + 1).min(self.samples.len());
    }

    pub fn len(&self) -> usize {
        self.filled
    }

    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// Most recently pushed sample, or zero when empty.
    pub fn last(&self) -> f32 {
        if self.filled == 0 {
            return 0.0;
        }
        let idx = (self.head + self.samples.len() - 1) % self.samples.len();
        self.samples[idx]
    }

    fn window(&self) -> &[f32] {
        if self.filled == self.samples.len() {
            &self.samples
        } else {
            &self.samples[..self.filled]
        }
    }

    pub fn mean(&self) -> f32 {
        if self.filled == 0 {
            return 0.0;
        }
        self.window().iter().sum::<f32>() / self.filled as f32
    }

    pub fn max(&self) -> f32 {
        self.window().iter().copied().fold(0.0, f32::max)
    }

    /// Nearest-rank percentile in `[0, 1]`.
    pub fn percentile(&self, p: f32) -> f32 {
        if self.filled == 0 {
            return 0.0;
        }
        let mut sorted = self.window().to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let rank = (p.clamp(0.0, 1.0) * self.filled as f32).ceil() as usize;
        sorted[rank.saturating_sub(1).min(self.filled - 1)]
    }

    pub fn p99(&self) -> f32 {
        self.percentile(0.99)
    }
}

impl Default for RollingStats {
    fn default() -> Self {
        Self::new(Self::DEFAULT_WINDOW)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stats_are_zero() {
        let s = RollingStats::new(8);
        assert_eq!(s.mean(), 0.0);
        assert_eq!(s.p99(), 0.0);
        assert_eq!(s.last(), 0.0);
    }

    #[test]
    fn mean_and_last_track_window() {
        let mut s = RollingStats::new(4);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            s.push(v);
        }
        assert_eq!(s.len(), 4);
        assert_eq!(s.last(), 5.0);
        assert!((s.mean() - 3.5).abs() < 1e-6);
    }

    #[test]
    fn p99_catches_single_spike() {
        let mut s = RollingStats::new(100);
        for _ in 0..99 {
            s.push(2.0);
        }
        s.push(40.0);
        assert_eq!(s.p99(), 2.0);
        s.push(40.0);
        assert_eq!(s.p99(), 40.0);
        assert_eq!(s.percentile(0.5), 2.0);
    }
}
