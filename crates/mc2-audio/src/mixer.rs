//! The mix: every voice dulled by what stands between it and the listener,
//! panned to where it is, arriving when sound would, then the place's
//! first echoes and its reverb over the lot.

use crate::acoustics::{Acoustics, Occlusion};
use crate::dsp::RATE;
use crate::dsp::{Delay, LowPass, Reverb, pan};
use crate::synth::{BedVoices, Beds, Sound, Voice};
use glam::Vec3;

/// Voices playing at once, most; the oldest makes way.
const MAX_VOICES: usize = 48;
/// The longest early echo, seconds.
const MAX_ECHO: f32 = 0.16;

/// A sound to play: what, how loud, from where (-1 left .. 1 right), how
/// much of the world is in the way, and how long before it arrives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emit {
    pub sound: Sound,
    pub gain: f32,
    pub pan: f32,
    pub occlusion: Occlusion,
    /// Seconds before it is heard: its distance over the speed of sound.
    pub delay: f32,
}

struct Playing {
    voice: Voice,
    gain: f32,
    pan: f32,
    filter: LowPass,
    wait: u32,
}

/// Mixes voices and beds through the place's acoustics.
pub struct Mixer {
    playing: Vec<Playing>,
    pub beds: BedVoices,
    reverb: Reverb,
    early: Delay,
    /// Early echoes: delay in samples, left and right gains.
    taps: Vec<(usize, f32, f32)>,
    wet: f32,
    wet_target: f32,
    pub volume: f32,
    seed: u32,
}

impl Default for Mixer {
    fn default() -> Self {
        Self::new()
    }
}

impl Mixer {
    pub fn new() -> Mixer {
        Mixer {
            playing: Vec::new(),
            beds: BedVoices::default(),
            reverb: Reverb::new(),
            early: Delay::new((MAX_ECHO * RATE) as usize + 1),
            taps: Vec::new(),
            wet: 0.0,
            wet_target: 0.0,
            volume: 0.8,
            seed: 1,
        }
    }

    pub fn play(&mut self, e: Emit) {
        if self.playing.len() >= MAX_VOICES {
            self.playing.remove(0);
        }
        self.seed = self.seed.wrapping_add(1);
        self.playing.push(Playing {
            voice: Voice::new(e.sound, self.seed),
            gain: e.gain * e.occlusion.gain,
            pan: e.pan,
            filter: LowPass::new(e.occlusion.cutoff),
            wait: (e.delay.max(0.0) * RATE) as u32,
        });
    }

    pub fn set_beds(&mut self, beds: Beds) {
        self.beds.target = beds;
    }

    /// Takes on the sound of a place, heard facing so that `right` is the
    /// listener's right.
    pub fn set_place(&mut self, a: &Acoustics, right: Vec3) {
        self.reverb.set(a.rt60, a.size, a.damping);
        self.wet_target = a.wet;
        self.taps = a
            .taps
            .iter()
            .filter(|t| t.delay < MAX_ECHO)
            .map(|t| {
                let (l, r) = pan(t.gain, t.dir.dot(right));
                (((t.delay * RATE) as usize).max(1), l, r)
            })
            .collect();
    }

    pub fn playing(&self) -> usize {
        self.playing.len()
    }

    /// Fills `out` with interleaved stereo samples.
    pub fn render(&mut self, out: &mut [f32]) {
        let (frames, _) = out.as_chunks_mut::<2>();
        for frame in frames {
            let (mut l, mut r) = (0.0f32, 0.0f32);
            for p in &mut self.playing {
                if p.wait > 0 {
                    p.wait -= 1;
                    continue;
                }
                let x = p.filter.run(p.voice.sample()) * p.gain;
                let (a, b) = pan(x, p.pan);
                l += a;
                r += b;
            }
            let bed = self.beds.sample();
            l += bed * 0.7;
            r += bed * 0.7;
            // The first echoes of the dry sound off the nearest surfaces.
            self.early.write((l + r) * 0.5);
            for &(back, gl, gr) in &self.taps {
                let e = self.early.read(back);
                l += e * gl;
                r += e * gr;
            }
            self.wet += (self.wet_target - self.wet) * (1.0 / (0.3 * RATE));
            let (wl, wr) = self.reverb.run(l * self.wet, r * self.wet);
            frame[0] = soft_clip((l + wl) * self.volume);
            frame[1] = soft_clip((r + wr) * self.volume);
        }
        self.playing.retain(|p| !p.voice.done());
    }
}

/// Keeps a loud mix inside -1..1 without a hard edge.
fn soft_clip(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::Tap;
    use crate::synth::Ground;

    fn blast() -> Emit {
        Emit {
            sound: Sound::Blast { radius: 3.0 },
            gain: 0.5,
            pan: 0.0,
            occlusion: Occlusion::CLEAR,
            delay: 0.0,
        }
    }

    /// Energy in `from..to` seconds of what the mixer plays.
    fn energy(m: &mut Mixer, from: f32, to: f32) -> f32 {
        let mut buf = vec![0.0; (to * RATE) as usize * 2];
        m.render(&mut buf);
        buf[(from * RATE) as usize * 2..]
            .iter()
            .map(|x| x * x)
            .sum()
    }

    fn cave() -> Acoustics {
        Acoustics {
            rt60: 4.0,
            wet: 0.6,
            size: 1.5,
            damping: 9_000.0,
            open: 0.0,
            mean_free_path: 6.0,
            absorption: 0.03,
            taps: vec![Tap {
                delay: 0.02,
                gain: 0.4,
                dir: Vec3::X,
            }],
        }
    }

    #[test]
    fn silence_is_silent() {
        let mut m = Mixer::new();
        assert_eq!(energy(&mut m, 0.0, 0.5), 0.0);
    }

    #[test]
    fn a_blast_in_a_cave_rings_on_after_the_same_blast_in_the_open() {
        let mut open = Mixer::new();
        open.set_place(&Acoustics::default(), Vec3::X);
        open.play(blast());
        let mut cave_m = Mixer::new();
        cave_m.set_place(&cave(), Vec3::X);
        // Let the wet level settle into the cave first.
        energy(&mut cave_m, 0.0, 1.0);
        cave_m.play(blast());
        let tail_open = energy(&mut open, 3.5, 5.0);
        let tail_cave = energy(&mut cave_m, 3.5, 5.0);
        assert!(
            tail_cave > tail_open * 20.0 + 1e-3,
            "cave {tail_cave} vs open {tail_open}"
        );
    }

    #[test]
    fn a_sound_behind_a_wall_is_duller_and_thunder_arrives_late() {
        // Highs: how much the signal changes sample to sample.
        let highs = |occ: Occlusion| {
            let mut m = Mixer::new();
            m.play(Emit {
                sound: Sound::Step(Ground::Gravel),
                gain: 1.0,
                pan: 0.0,
                occlusion: occ,
                delay: 0.0,
            });
            let mut buf = vec![0.0; 9_600];
            m.render(&mut buf);
            let diff: f32 = buf.windows(4).map(|w| (w[2] - w[0]).powi(2)).sum();
            let total: f32 = buf.iter().map(|x| x * x).sum::<f32>().max(1e-9);
            diff / total
        };
        let clear = highs(Occlusion::CLEAR);
        let walled = highs(Occlusion {
            gain: 0.5,
            cutoff: 300.0,
        });
        assert!(walled < clear * 0.3, "walled {walled} vs clear {clear}");
        // Thunder a kilometre off is heard about three seconds after.
        let mut m = Mixer::new();
        m.play(Emit {
            sound: Sound::Thunder { distance: 1_000.0 },
            gain: 1.0,
            pan: 0.0,
            occlusion: Occlusion::CLEAR,
            delay: 1_000.0 / crate::acoustics::SOUND,
        });
        assert_eq!(energy(&mut m, 0.0, 2.8), 0.0);
        assert!(energy(&mut m, 0.0, 2.0) > 0.1);
    }
}
