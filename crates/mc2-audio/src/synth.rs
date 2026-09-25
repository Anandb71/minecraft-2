//! Sounds made from noise and oscillators, no recordings.
//!
//! A `Voice` plays one sound once; the beds (rain, fire, wind, an engine)
//! play all the time at a level the game sets.

use crate::dsp::{LowPass, Noise, RATE};
use std::f32::consts::TAU;

/// What a foot lands on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ground {
    /// Grass, soil, sand, snow, leaves.
    Soft,
    /// Rock, stone, brick, concrete, metal, glass.
    Hard,
    /// Planks and logs.
    Wood,
    /// Gravel.
    Gravel,
    /// Shallow water.
    Water,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sound {
    Step(Ground),
    /// Something breaking, by how hard it is (0..10).
    Break {
        hard: f32,
    },
    /// A block set down.
    Place,
    /// A blast of this radius, metres.
    Blast {
        radius: f32,
    },
    /// Thunder from this far away, metres: near, it cracks; far, it rolls.
    Thunder {
        distance: f32,
    },
    Splash,
}

impl Sound {
    /// How long it sounds, seconds.
    pub fn length(self) -> f32 {
        match self {
            Sound::Step(Ground::Water) => 0.35,
            Sound::Step(_) => 0.12,
            Sound::Break { .. } => 0.25,
            Sound::Place => 0.12,
            Sound::Blast { radius } => 2.0 + radius * 0.4,
            Sound::Thunder { distance } => 2.5 + (distance / 400.0).min(5.0),
            Sound::Splash => 0.6,
        }
    }
}

/// One sound playing.
#[derive(Clone, Debug)]
pub struct Voice {
    pub sound: Sound,
    /// Samples played.
    t: u32,
    len: u32,
    noise: Noise,
    tone: LowPass,
    shade: LowPass,
    phase: f32,
    /// Crackle and grain: when the next pop comes, samples.
    pop_at: u32,
}

impl Voice {
    pub fn new(sound: Sound, seed: u32) -> Voice {
        let (tone, shade) = match sound {
            Sound::Step(Ground::Soft) => (700.0, 700.0),
            Sound::Step(Ground::Hard) => (5_000.0, 3_000.0),
            Sound::Step(Ground::Wood) => (1_200.0, 900.0),
            Sound::Step(Ground::Gravel) => (6_000.0, 4_000.0),
            Sound::Step(Ground::Water) => (2_500.0, 1_500.0),
            Sound::Break { hard } => (800.0 + hard * 700.0, 2_000.0),
            Sound::Place => (600.0, 400.0),
            Sound::Blast { .. } => (3_000.0, 200.0),
            Sound::Thunder { distance } => ((4_000.0 - distance * 3.0).max(300.0), 150.0),
            Sound::Splash => (3_000.0, 1_200.0),
        };
        Voice {
            sound,
            t: 0,
            len: (sound.length() * RATE) as u32,
            noise: Noise::new(seed.wrapping_mul(2_654_435_761).max(1)),
            tone: LowPass::new(tone),
            shade: LowPass::new(shade),
            phase: 0.0,
            pop_at: 0,
        }
    }

    pub fn done(&self) -> bool {
        self.t >= self.len
    }

    /// The next sample, mono.
    pub fn sample(&mut self) -> f32 {
        if self.done() {
            return 0.0;
        }
        let s = self.t as f32 / RATE;
        let life = self.t as f32 / self.len as f32;
        self.t += 1;
        let n = self.noise.sample();
        match self.sound {
            Sound::Step(ground) => {
                let env = (-s * 45.0).exp() * (s * 900.0).min(1.0);
                match ground {
                    Ground::Soft => self.tone.run(n) * env * 0.8,
                    Ground::Hard => (self.tone.run(n) - self.shade.run(n)) * env * 1.4,
                    Ground::Wood => {
                        self.phase += TAU * 170.0 / RATE;
                        (self.phase.sin() * 0.9 + self.tone.run(n) * 0.5) * env
                    }
                    Ground::Gravel => {
                        // Grains: short pops scattered through the step.
                        if self.t >= self.pop_at {
                            self.pop_at = self.t + 60 + (self.noise.sample().abs() * 500.0) as u32;
                        }
                        let grain = if self.pop_at - self.t > 40 { 0.3 } else { 1.2 };
                        (self.tone.run(n) - self.shade.run(n)) * env * grain
                    }
                    Ground::Water => {
                        self.phase += TAU * (300.0 + 500.0 * life) / RATE;
                        let slosh = (-s * 12.0).exp() * (s * 200.0).min(1.0);
                        (self.tone.run(n) * 0.6 + self.phase.sin() * 0.15) * slosh
                    }
                }
            }
            Sound::Break { .. } => {
                let env = (-s * 18.0).exp() * (s * 2_000.0).min(1.0);
                if self.t >= self.pop_at {
                    self.pop_at = self.t + 200 + (self.noise.sample().abs() * 1_500.0) as u32;
                }
                let crack = if self.pop_at - self.t > 120 { 0.6 } else { 1.4 };
                self.tone.run(n) * env * crack
            }
            Sound::Place => {
                self.phase += TAU * 110.0 / RATE;
                let env = (-s * 40.0).exp();
                (self.phase.sin() * 0.8 + self.tone.run(n) * 0.4) * env
            }
            Sound::Blast { radius } => {
                // A falling boom under a roar that darkens as it goes.
                let f = 55.0 * (1.0 - 0.5 * life);
                self.phase += TAU * f / RATE;
                let boom = self.phase.sin() * (-s * 2.2).exp();
                self.tone.set(3_000.0 * (1.0 - life).powi(2) + 150.0);
                let roar = self.tone.run(n) * (-s * 1.6).exp() * (s * 400.0).min(1.0);
                let size = (radius / 3.0).clamp(0.5, 3.0);
                (boom * 1.2 + roar * 1.6) * size
            }
            Sound::Thunder { distance } => {
                // A crack if it is near, then a long roll that swells and
                // fades unevenly.
                let near = (1.0 - distance / 600.0).clamp(0.0, 1.0);
                let crack = self.tone.run(n) * (-s * 25.0).exp() * near * 2.0;
                self.phase += TAU * 0.7 / RATE;
                let swell = 0.6 + 0.4 * (self.phase * 3.0).sin() * (self.phase * 1.3).sin();
                let roll = self.shade.run(n) * (1.0 - life).powi(2) * swell * (s * 3.0).min(1.0);
                crack + roll * 3.5
            }
            Sound::Splash => {
                let env = (-s * 6.0).exp() * (s * 300.0).min(1.0);
                self.phase += TAU * (600.0 - 400.0 * life) / RATE;
                (self.tone.run(n) * 0.8 + self.phase.sin() * 0.1) * env
            }
        }
    }
}

/// Sounds that never stop, each at a level from 0 (silent).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Beds {
    /// Rain, by how hard it falls, 0..1.
    pub rain: f32,
    /// Fire, by how much is burning near, 0..1.
    pub fire: f32,
    /// Wind, by its speed, 0..1.
    pub wind: f32,
    /// An engine, by its speed (m/s) and throttle, when one is running.
    pub engine: Option<(f32, f32)>,
}

/// The beds' own oscillators and filters.
#[derive(Clone, Debug)]
pub struct BedVoices {
    pub level: Beds,
    /// What the levels ease toward.
    pub target: Beds,
    noise: Noise,
    rain: LowPass,
    fire: LowPass,
    wind: LowPass,
    engine: LowPass,
    phase: f32,
    gust: f32,
    pop_at: u32,
    t: u32,
}

impl Default for BedVoices {
    fn default() -> Self {
        BedVoices {
            level: Beds::default(),
            target: Beds::default(),
            noise: Noise::new(0x51ed),
            rain: LowPass::new(5_000.0),
            fire: LowPass::new(700.0),
            wind: LowPass::new(400.0),
            engine: LowPass::new(900.0),
            phase: 0.0,
            gust: 0.0,
            pop_at: 0,
            t: 0,
        }
    }
}

impl BedVoices {
    /// The next sample, mono.
    pub fn sample(&mut self) -> f32 {
        // Levels ease over about a fifth of a second.
        let k = 1.0 / (0.2 * RATE);
        let ease = |a: &mut f32, b: f32| *a += (b - *a) * k;
        ease(&mut self.level.rain, self.target.rain);
        ease(&mut self.level.fire, self.target.fire);
        ease(&mut self.level.wind, self.target.wind);
        self.level.engine = self.target.engine;
        self.t = self.t.wrapping_add(1);
        let n = self.noise.sample();
        let mut out = 0.0;
        if self.level.rain > 0.001 {
            out += self.rain.run(n) * self.level.rain * 0.5;
        }
        if self.level.fire > 0.001 {
            if self.t >= self.pop_at {
                let gap = 400.0 + self.noise.sample().abs() * 6_000.0 / self.level.fire;
                self.pop_at = self.t.wrapping_add(gap as u32);
            }
            let pop = if self.pop_at.wrapping_sub(self.t) < 90 {
                2.5
            } else {
                0.25
            };
            out += self.fire.run(n) * self.level.fire * pop;
        }
        if self.level.wind > 0.001 {
            self.gust += TAU * 0.13 / RATE;
            let gust = 0.7 + 0.3 * self.gust.sin() * (self.gust * 0.37).sin();
            self.wind.set(250.0 + 500.0 * self.level.wind * gust);
            out += self.wind.run(n) * self.level.wind * gust * 1.2;
        }
        if let Some((speed, throttle)) = self.level.engine {
            // A four-cylinder's firing note climbing with road speed.
            let f = 32.0 + speed * 6.0;
            self.phase = (self.phase + f / RATE).fract();
            let saw = self.phase * 2.0 - 1.0;
            self.engine.set(500.0 + 1_500.0 * throttle.abs());
            out += self.engine.run(saw + n * 0.2) * (0.25 + 0.35 * throttle.abs());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn energy(v: &mut Voice, seconds: f32) -> f32 {
        (0..(seconds * RATE) as usize)
            .map(|_| v.sample().powi(2))
            .sum()
    }

    #[test]
    fn every_sound_makes_a_sound_and_then_stops() {
        let all = [
            Sound::Step(Ground::Soft),
            Sound::Step(Ground::Hard),
            Sound::Step(Ground::Wood),
            Sound::Step(Ground::Gravel),
            Sound::Step(Ground::Water),
            Sound::Break { hard: 3.0 },
            Sound::Place,
            Sound::Blast { radius: 3.0 },
            Sound::Thunder { distance: 200.0 },
            Sound::Splash,
        ];
        for (i, s) in all.into_iter().enumerate() {
            let mut v = Voice::new(s, i as u32 + 1);
            let e = energy(&mut v, s.length());
            assert!(e > 0.5, "{s:?} is silent: {e}");
            assert!(v.done(), "{s:?} goes on");
            assert_eq!(v.sample(), 0.0);
            // Nothing blows up.
            let mut v = Voice::new(s, 7);
            for _ in 0..(s.length() * RATE) as usize {
                let x = v.sample();
                assert!(x.is_finite() && x.abs() < 8.0, "{s:?}: {x}");
            }
        }
    }

    #[test]
    fn near_thunder_cracks_and_far_thunder_only_rolls() {
        let first = |d: f32| {
            let mut v = Voice::new(Sound::Thunder { distance: d }, 3);
            energy(&mut v, 0.1)
        };
        assert!(first(50.0) > first(2_000.0) * 4.0);
    }

    #[test]
    fn beds_follow_their_levels() {
        let mut b = BedVoices::default();
        let quiet: f32 = (0..4_800).map(|_| b.sample().powi(2)).sum();
        assert!(quiet < 1e-6);
        b.target.rain = 1.0;
        b.target.engine = Some((10.0, 1.0));
        for _ in 0..48_000 {
            b.sample();
        }
        let loud: f32 = (0..4_800).map(|_| b.sample().powi(2)).sum();
        assert!(loud > 1.0, "{loud}");
    }
}
