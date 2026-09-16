//! Command line parsing. A handful of flags does not justify a parser crate.

use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq)]
pub enum Mode {
    Window,
    /// Render headless and write the final frame to a PNG.
    Capture(PathBuf),
    /// Render headless and print the profiler table.
    Bench,
}

#[derive(Clone, Debug)]
pub struct Args {
    pub mode: Mode,
    pub frames: u32,
    /// Set when `--frames` is given explicitly; the window then closes itself.
    pub exit_after: Option<u32>,
    pub size: (u32, u32),
    pub software: bool,
    /// Draw the profiler overlay into headless captures.
    pub hud: bool,
    /// Renderer debug view index.
    pub debug_view: u32,
    pub no_beam: bool,
    pub seed: u64,
    pub world_dir: PathBuf,
    /// Headless camera override: position and look-at target, metres.
    pub camera: Option<([f64; 3], [f64; 3])>,
    /// Scripted play before a headless capture.
    pub demo: Option<crate::demo::Demo>,
    pub quality: mc2_render::quality::Preset,
    /// Time of day in hours; headless runs freeze the clock there.
    pub time: Option<f64>,
    /// Overrides the preset's indirect light method.
    pub gi: Option<mc2_render::indirect::GiMethod>,
}

/// Quality settings from the preset plus command line overrides.
pub fn quality(args: &Args) -> mc2_render::quality::Quality {
    let mut q = args.quality.settings();
    if let Some(gi) = args.gi {
        q.gi = gi;
    }
    q
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::Window,
            frames: 120,
            exit_after: None,
            size: (1280, 720),
            software: false,
            hud: false,
            debug_view: 0,
            no_beam: false,
            seed: 42,
            world_dir: PathBuf::from("worlds/default"),
            camera: None,
            demo: None,
            quality: mc2_render::quality::Preset::UltraRealistic,
            time: None,
            gi: None,
        }
    }
}

pub const USAGE: &str = "\
usage: minecraft-2 [options]
  --capture <file.png>   render headless, save the last frame
  --bench                render headless, print GPU and CPU timings
  --frames <n>           frames to render (headless default 120; window exits)
  --size <w>x<h>         headless resolution (default 1280x720)
  --software             use the software adapter (WARP / lavapipe)
  --hud                  draw the profiler overlay into captures
  --no-beam              disable the beam prepass (A/B measurement)
  --seed <n>             world seed (default 42)
  --world <dir>          world directory (default worlds/default)
  --camera x,y,z,lx,ly,lz  headless camera position and look-at target (m)
  --demo build|lights    scripted play before a headless capture
  --time <hours>         time of day, e.g. 6.5 or 22 (headless: frozen)
  --gi restir|cascades   indirect light method (default: preset)
  --quality <tier>       0 Realistic, 1 Hyper Realistic, 2 Ultra Realistic
                         (default), 3 Super Ultra Crazy Duper Realistic
  --view <n>             debug view: 0 shaded, 1 LOD hits, 2 march iterations";

pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut out = Args::default();
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--capture" => out.mode = Mode::Capture(PathBuf::from(value("--capture")?)),
            "--bench" => out.mode = Mode::Bench,
            "--frames" => {
                out.frames = value("--frames")?
                    .parse()
                    .map_err(|e| format!("--frames: {e}"))?;
                out.exit_after = Some(out.frames);
            }
            "--size" => {
                let v = value("--size")?;
                let (w, h) = v.split_once('x').ok_or("--size expects WxH")?;
                out.size = (
                    w.parse().map_err(|e| format!("--size width: {e}"))?,
                    h.parse().map_err(|e| format!("--size height: {e}"))?,
                );
            }
            "--software" => out.software = true,
            "--hud" => out.hud = true,
            "--no-beam" => out.no_beam = true,
            "--seed" => {
                out.seed = value("--seed")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?
            }
            "--world" => out.world_dir = PathBuf::from(value("--world")?),
            "--demo" => {
                let name = value("--demo")?;
                out.demo =
                    Some(crate::demo::Demo::parse(&name).ok_or(format!("unknown demo `{name}`"))?);
            }
            "--camera" => {
                let v: Vec<f64> = value("--camera")?
                    .split(',')
                    .map(|s| s.trim().parse::<f64>())
                    .collect::<Result<_, _>>()
                    .map_err(|e| format!("--camera: {e}"))?;
                if v.len() != 6 {
                    return Err("--camera expects x,y,z,look_x,look_y,look_z".into());
                }
                out.camera = Some(([v[0], v[1], v[2]], [v[3], v[4], v[5]]));
            }
            "--time" => {
                let v: f64 = value("--time")?
                    .parse()
                    .map_err(|e| format!("--time: {e}"))?;
                if !(0.0..=24.0).contains(&v) {
                    return Err("--time expects hours in 0..=24".into());
                }
                out.time = Some(v);
            }
            "--gi" => {
                let v = value("--gi")?;
                out.gi = Some(
                    mc2_render::indirect::GiMethod::parse(&v)
                        .ok_or(format!("unknown indirect method `{v}`"))?,
                );
            }
            "--quality" => {
                let v = value("--quality")?;
                out.quality = mc2_render::quality::Preset::parse(&v)
                    .ok_or(format!("unknown quality `{v}`"))?;
            }
            "--view" => {
                out.debug_view = value("--view")?
                    .parse()
                    .map_err(|e| format!("--view: {e}"))?
            }
            "-h" | "--help" => return Err(USAGE.to_owned()),
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Args, String> {
        parse(s.split_whitespace().map(str::to_owned))
    }

    #[test]
    fn parses_capture_flags() {
        let a = p("--capture out.png --frames 30 --size 640x360 --software").unwrap();
        assert_eq!(a.mode, Mode::Capture(PathBuf::from("out.png")));
        assert_eq!(a.frames, 30);
        assert_eq!(a.size, (640, 360));
        assert!(a.software);
    }

    #[test]
    fn parses_quality_tiers() {
        use mc2_render::quality::Preset;
        assert_eq!(p("").unwrap().quality, Preset::UltraRealistic);
        assert_eq!(p("--quality 0").unwrap().quality, Preset::Realistic);
        assert_eq!(
            p("--quality super-ultra-crazy-duper-realistic")
                .unwrap()
                .quality,
            Preset::SuperUltraCrazyDuperRealistic
        );
        assert!(p("--quality low").is_err());
        assert_eq!(p("--time 21.5").unwrap().time, Some(21.5));
        assert!(p("--time 25").is_err());
    }

    #[test]
    fn rejects_bad_input() {
        assert!(p("--size 640").is_err());
        assert!(p("--frames").is_err());
        assert!(p("--wat").is_err());
        assert_eq!(p("").unwrap().mode, Mode::Window);
    }
}
