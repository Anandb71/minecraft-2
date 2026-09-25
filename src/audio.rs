//! Plays the game's sounds.
//!
//! The mixer runs on the audio device's own thread. Each frame the game
//! thread hands it every sound heard (panned, attenuated with distance,
//! dulled by what the traced acoustics say stands between, and delayed by
//! how long sound takes to arrive), the beds' levels, and every so often
//! the listener's place, measured afresh. With no device it runs offline,
//! for writing what a demo sounds like to a file.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use glam::{DVec3, Vec3};
use mc2_audio::acoustics::SOUND;
use mc2_audio::dsp::RATE;
use mc2_audio::{Emit, Mixer};
use mc2_game::Game;
use mc2_game::sounds::{Heard, Sounds};
use std::sync::{Arc, Mutex};

/// Seconds between measurements of the listener's place.
const PROBE_S: f64 = 0.15;

pub struct Audio {
    mixer: Arc<Mutex<Mixer>>,
    /// Plays for as long as it is kept.
    _stream: Option<cpal::Stream>,
    /// Sounds waiting for the mixer, when its lock was busy.
    pending: Vec<Emit>,
    next_probe: f64,
}

impl Audio {
    /// Opens the default output device; with none, stays silent.
    pub fn start() -> Audio {
        let mixer = Arc::new(Mutex::new(Mixer::new()));
        let stream = match open(mixer.clone()) {
            Ok(s) => Some(s),
            Err(e) => {
                log::warn!("no sound: {e}");
                None
            }
        };
        Audio {
            mixer,
            _stream: stream,
            pending: Vec::new(),
            next_probe: 0.0,
        }
    }

    /// A mixer with no device, to render by hand.
    pub fn offline() -> Audio {
        Audio {
            mixer: Arc::new(Mutex::new(Mixer::new())),
            _stream: None,
            pending: Vec::new(),
            next_probe: 0.0,
        }
    }

    /// Takes the sounds heard since last time from the game and hands them
    /// to the mixer, heard at `ear` facing `yaw` (radians about +y; 0 faces
    /// +z). Played back offline from `origin` (game seconds), each is held
    /// back until its time comes; live, they are all fresh.
    pub fn update(&mut self, game: &mut Game, ear: DVec3, yaw: f32, origin: Option<f64>) {
        let (heard, beds, now) = {
            let mut s = game.world.resource_mut::<Sounds>();
            (s.take(), s.beds, s.now)
        };
        // The listener's right, level: +z forward, so +x is the left.
        let right = Vec3::new(-yaw.cos(), 0.0, yaw.sin());
        let world = &game.world.resource::<mc2_game::Voxels>().0;
        for h in heard {
            let late = origin.map_or(0.0, |o| (h.time - o).max(0.0) as f32);
            self.pending.push(emit(world, &h, ear, right, late));
        }
        let place = (now >= self.next_probe).then(|| {
            self.next_probe = now + PROBE_S;
            mc2_audio::measure(world, ear)
        });
        if let Ok(mut m) = self.mixer.try_lock() {
            for e in self.pending.drain(..) {
                m.play(e);
            }
            // Rain and wind are heard less the less open sky there is.
            let mut beds = beds;
            if let Some(p) = &place {
                let open = 0.25 + 0.75 * p.open;
                beds.rain *= open;
                beds.wind *= open;
                m.set_place(p, right);
            }
            m.set_beds(beds);
        }
    }

    /// Renders `seconds` of sound by hand (offline), interleaved stereo.
    pub fn render(&self, seconds: f32) -> Vec<f32> {
        let mut out = vec![0.0; (seconds * RATE) as usize * 2];
        if let Ok(mut m) = self.mixer.lock() {
            m.render(&mut out);
        }
        out
    }
}

/// How a heard sound reaches the ear.
fn emit(
    world: &mc2_voxel::world::VoxelWorld,
    h: &Heard,
    ear: DVec3,
    right: Vec3,
    late: f32,
) -> Emit {
    let to = h.at - ear;
    let d = to.length() as f32;
    let dir = if d > 1e-3 {
        (to / f64::from(d)).as_vec3()
    } else {
        Vec3::ZERO
    };
    // Loudness falls with distance, but big sounds carry.
    let reach = match h.sound {
        mc2_audio::Sound::Blast { radius } => 60.0 * radius,
        mc2_audio::Sound::Thunder { .. } => 2_000.0,
        _ => 6.0,
    };
    let gain = h.gain * reach / (reach + d * d / reach.max(1.0));
    Emit {
        sound: h.sound,
        gain,
        pan: dir.dot(right),
        occlusion: mc2_audio::occlusion(world, h.at, ear),
        delay: d / SOUND + late,
    }
}

/// Opens the default device's stream, feeding it from `mixer`.
fn open(mixer: Arc<Mutex<Mixer>>) -> Result<cpal::Stream, String> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no output device")?;
    // 48 kHz float if the device offers it, else its own default.
    let wanted = device
        .supported_output_configs()
        .map_err(|e| e.to_string())?
        .filter(|c| c.sample_format() == cpal::SampleFormat::F32 && c.channels() >= 2)
        .find_map(|c| c.try_with_sample_rate(cpal::SampleRate(RATE as u32)));
    let supported = match wanted {
        Some(c) => c,
        None => device.default_output_config().map_err(|e| e.to_string())?,
    };
    if supported.sample_format() != cpal::SampleFormat::F32 {
        return Err(format!(
            "device wants {:?} samples",
            supported.sample_format()
        ));
    }
    let config: cpal::StreamConfig = supported.into();
    let channels = config.channels as usize;
    if config.sample_rate.0 != RATE as u32 {
        log::warn!(
            "sound device runs at {} Hz; pitch is off by {:.0}%",
            config.sample_rate.0,
            (config.sample_rate.0 as f32 / RATE - 1.0) * 100.0
        );
    }
    let mut stereo = Vec::new();
    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                let frames = data.len() / channels.max(1);
                stereo.resize(frames * 2, 0.0);
                match mixer.lock() {
                    Ok(mut m) => m.render(&mut stereo),
                    Err(_) => stereo.fill(0.0),
                }
                for (f, out) in data.chunks_mut(channels).enumerate() {
                    let (l, r) = (stereo[f * 2], stereo[f * 2 + 1]);
                    match out.len() {
                        1 => out[0] = (l + r) * 0.5,
                        _ => {
                            out[0] = l;
                            out[1] = r;
                            for x in &mut out[2..] {
                                *x = 0.0;
                            }
                        }
                    }
                }
            },
            |e| log::warn!("sound stream: {e}"),
            None,
        )
        .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    Ok(stream)
}

/// Writes interleaved stereo samples as a 16-bit WAV file.
pub fn write_wav(path: &std::path::Path, stereo: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let data = (stereo.len() * 2) as u32;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&(RATE as u32).to_le_bytes())?;
    f.write_all(&(RATE as u32 * 4).to_le_bytes())?;
    f.write_all(&4u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data.to_le_bytes())?;
    for &x in stereo {
        f.write_all(&((x.clamp(-1.0, 1.0) * 32_767.0) as i16).to_le_bytes())?;
    }
    f.flush()
}
