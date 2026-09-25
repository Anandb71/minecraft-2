//! Signal processing: noise, filters, delay lines and a reverb.
//!
//! Everything runs a sample at a time at `RATE`, allocation-free once built,
//! so it can sit inside an audio callback.

/// Samples a second.
pub const RATE: f32 = 48_000.0;

/// White noise from a xorshift, -1..1.
#[derive(Clone, Copy, Debug)]
pub struct Noise(u32);

impl Noise {
    pub fn new(seed: u32) -> Noise {
        Noise(seed | 1)
    }

    pub fn sample(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// A one-pole low-pass filter.
#[derive(Clone, Copy, Debug)]
pub struct LowPass {
    a: f32,
    z: f32,
}

impl LowPass {
    pub fn new(cutoff: f32) -> LowPass {
        let mut f = LowPass { a: 0.0, z: 0.0 };
        f.set(cutoff);
        f
    }

    /// Sets the cutoff, Hz.
    pub fn set(&mut self, cutoff: f32) {
        let c = cutoff.clamp(10.0, RATE * 0.45);
        self.a = (-std::f32::consts::TAU * c / RATE).exp();
    }

    pub fn run(&mut self, x: f32) -> f32 {
        self.z = (1.0 - self.a) * x + self.a * self.z;
        self.z
    }
}

/// A delay line of up to `len` samples.
#[derive(Clone, Debug)]
pub struct Delay {
    buf: Vec<f32>,
    at: usize,
}

impl Delay {
    pub fn new(len: usize) -> Delay {
        Delay {
            buf: vec![0.0; len.max(1)],
            at: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// The sample written `back` samples ago (1..=len).
    pub fn read(&self, back: usize) -> f32 {
        let n = self.buf.len();
        self.buf[(self.at + n - back.clamp(1, n)) % n]
    }

    pub fn write(&mut self, x: f32) {
        self.buf[self.at] = x;
        self.at = (self.at + 1) % self.buf.len();
    }
}

/// Lengths of the reverb's eight lines at size 1, samples: mutually
/// prime, spread over 30..80 ms so the echoes never line up.
const LINES: [usize; 8] = [1433, 1601, 1867, 2053, 2251, 2399, 2687, 2857];
/// The longest a line may be stretched to, as a multiple of its length.
const MAX_SIZE: f32 = 3.0;

/// A feedback delay network reverb: eight lines mixed through a
/// Householder matrix, each line damped so highs die first, its gain set
/// so the tail falls 60 dB in `rt60` seconds.
#[derive(Clone, Debug)]
pub struct Reverb {
    lines: Vec<Delay>,
    lengths: [usize; 8],
    gains: [f32; 8],
    damp: [LowPass; 8],
}

impl Default for Reverb {
    fn default() -> Self {
        Self::new()
    }
}

impl Reverb {
    pub fn new() -> Reverb {
        let mut r = Reverb {
            lines: LINES
                .iter()
                .map(|&n| Delay::new((n as f32 * MAX_SIZE) as usize + 1))
                .collect(),
            lengths: LINES,
            gains: [0.0; 8],
            damp: [LowPass::new(8000.0); 8],
        };
        r.set(1.0, 1.0, 8000.0);
        r
    }

    /// Sets the decay time (seconds to fall 60 dB), the room's size (a
    /// stretch of the lines, 0.3..3) and the damping cutoff, Hz.
    #[allow(clippy::needless_range_loop)]
    pub fn set(&mut self, rt60: f32, size: f32, damping: f32) {
        let size = size.clamp(0.3, MAX_SIZE);
        let rt60 = rt60.max(0.05);
        for i in 0..8 {
            self.lengths[i] = ((LINES[i] as f32 * size) as usize).clamp(1, self.lines[i].len());
            let seconds = self.lengths[i] as f32 / RATE;
            // Each trip round a line loses its share of 60 dB.
            self.gains[i] = 10f32.powf(-3.0 * seconds / rt60);
            self.damp[i].set(damping);
        }
    }

    /// One stereo sample in, one out (the wet signal only).
    #[allow(clippy::needless_range_loop)]
    pub fn run(&mut self, left: f32, right: f32) -> (f32, f32) {
        let mut outs = [0.0f32; 8];
        for (i, o) in outs.iter_mut().enumerate() {
            *o = self.lines[i].read(self.lengths[i]);
        }
        // Householder: x - (2/n) * sum(x), energy preserving.
        let sum: f32 = outs.iter().sum::<f32>() * (2.0 / 8.0);
        for i in 0..8 {
            let input = if i % 2 == 0 { left } else { right };
            let fed = (outs[i] - sum) * self.gains[i];
            let v = self.damp[i].run(fed) + input;
            self.lines[i].write(v);
        }
        let l = outs[0] + outs[2] + outs[4] + outs[6];
        let r = outs[1] + outs[3] + outs[5] + outs[7];
        (l * 0.25, r * 0.25)
    }
}

/// Constant-power pan: -1 left, 1 right.
pub fn pan(x: f32, p: f32) -> (f32, f32) {
    let a = (p.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    (x * a.cos(), x * a.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seconds for the energy of the reverb's answer to an impulse to fall
    /// 60 dB below its peak, from 50 ms windows.
    fn measured_rt60(rt60: f32, size: f32) -> f32 {
        let mut r = Reverb::new();
        r.set(rt60, size, 18_000.0);
        let window = (RATE * 0.05) as usize;
        let mut energies = Vec::new();
        let mut e = 0.0f64;
        for n in 0..(RATE * rt60 * 2.0) as usize {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let (l, rr) = r.run(x, x);
            e += f64::from(l * l + rr * rr);
            if n % window == window - 1 {
                energies.push(e);
                e = 0.0;
            }
        }
        // From the peak, which comes a line's length after the impulse.
        let (at, peak) = energies
            .iter()
            .copied()
            .enumerate()
            .fold((0, 0.0), |m, (i, e)| if e > m.1 { (i, e) } else { m });
        let below = energies[at..]
            .iter()
            .position(|&e| e < peak * 1e-6)
            .map_or(energies.len(), |i| at + i);
        below as f32 * 0.05
    }

    #[test]
    fn the_reverb_dies_away_as_asked() {
        for (rt60, size) in [(0.5, 0.6), (1.5, 1.0), (3.0, 2.0)] {
            let got = measured_rt60(rt60, size);
            assert!(
                (got - rt60).abs() < rt60 * 0.3 + 0.1,
                "asked {rt60} s, got {got} s"
            );
        }
    }

    #[test]
    fn a_low_pass_takes_the_highs() {
        let mut f = LowPass::new(500.0);
        let mut n = Noise::new(9);
        let (mut e_in, mut e_out) = (0.0, 0.0);
        for _ in 0..48_000 {
            let x = n.sample();
            let y = f.run(x);
            e_in += x * x;
            e_out += y * y;
        }
        // White noise keeps about 2 * 500 / 48000 of its power below 500 Hz.
        assert!(e_out < e_in * 0.1, "{e_out} of {e_in}");
    }

    #[test]
    fn panning_keeps_the_power() {
        for p in [-1.0, -0.3, 0.0, 0.5, 1.0] {
            let (l, r) = pan(1.0, p);
            assert!((l * l + r * r - 1.0).abs() < 1e-5);
        }
        let (l, r) = pan(1.0, -1.0);
        assert!(l > 0.99 && r.abs() < 1e-5);
    }
}
