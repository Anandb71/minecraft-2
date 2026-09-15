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
    fn rejects_bad_input() {
        assert!(p("--size 640").is_err());
        assert!(p("--frames").is_err());
        assert!(p("--wat").is_err());
        assert_eq!(p("").unwrap().mode, Mode::Window);
    }
}
