//! WGSL source library with `#import` resolution and debug hot reload.
//!
//! Release builds read sources embedded at compile time, so the binary runs
//! from any directory. Debug builds read the same files from disk and poll
//! their modification times; any change bumps [`ShaderLibrary::generation`],
//! which pipeline owners compare against to rebuild. A shader that fails to
//! compile after an edit keeps the previous pipeline alive.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

pub type EmbeddedShaders = &'static [(&'static str, &'static str)];

#[derive(Debug)]
pub enum ShaderError {
    Missing(String),
    ImportCycle(String),
    Compile { name: String, message: String },
}

impl std::fmt::Display for ShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShaderError::Missing(n) => write!(f, "shader `{n}` not found"),
            ShaderError::ImportCycle(n) => write!(f, "import cycle through `{n}`"),
            ShaderError::Compile { name, message } => write!(f, "{name}: {message}"),
        }
    }
}

impl std::error::Error for ShaderError {}

pub struct ShaderLibrary {
    embedded: EmbeddedShaders,
    disk_root: Option<PathBuf>,
    mtimes: HashMap<String, SystemTime>,
    generation: u64,
    last_poll: Instant,
}

impl ShaderLibrary {
    /// `disk_root` enables hot reload; pass `None` to use embedded sources only.
    pub fn new(embedded: EmbeddedShaders, disk_root: Option<PathBuf>) -> Self {
        let mut lib = Self {
            embedded,
            disk_root: disk_root.filter(|p| p.is_dir()),
            mtimes: HashMap::new(),
            generation: 0,
            last_poll: Instant::now(),
        };
        lib.scan_mtimes();
        lib
    }

    pub fn hot_reload(&self) -> bool {
        self.disk_root.is_some()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn raw(&self, name: &str) -> Result<String, ShaderError> {
        if let Some(root) = &self.disk_root
            && let Ok(text) = std::fs::read_to_string(root.join(name))
        {
            return Ok(text);
        }
        self.embedded
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, src)| (*src).to_owned())
            .ok_or_else(|| ShaderError::Missing(name.to_owned()))
    }

    /// Full source with every `#import "file"` expanded once, depth first.
    pub fn source(&self, name: &str) -> Result<String, ShaderError> {
        let mut out = String::new();
        let mut included = Vec::new();
        let mut stack = Vec::new();
        self.expand(name, &mut out, &mut included, &mut stack)?;
        Ok(out)
    }

    fn expand(
        &self,
        name: &str,
        out: &mut String,
        included: &mut Vec<String>,
        stack: &mut Vec<String>,
    ) -> Result<(), ShaderError> {
        if stack.iter().any(|s| s == name) {
            return Err(ShaderError::ImportCycle(name.to_owned()));
        }
        if included.iter().any(|s| s == name) {
            return Ok(());
        }
        included.push(name.to_owned());
        stack.push(name.to_owned());
        let text = self.raw(name)?;
        for line in text.lines() {
            match parse_import(line) {
                Some(dep) => self.expand(dep, out, included, stack)?,
                None => {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }
        stack.pop();
        Ok(())
    }

    fn scan_mtimes(&mut self) -> bool {
        let Some(root) = &self.disk_root else {
            return false;
        };
        let Ok(entries) = std::fs::read_dir(root) else {
            return false;
        };
        let mut changed = false;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "wgsl") {
                continue;
            }
            let Ok(mtime) = entry.metadata().and_then(|m| m.modified()) else {
                continue;
            };
            let key = entry.file_name().to_string_lossy().into_owned();
            if self.mtimes.insert(key, mtime) != Some(mtime) {
                changed = true;
            }
        }
        changed
    }

    /// Checks the disk at most four times a second. Returns true on change.
    pub fn poll(&mut self) -> bool {
        if self.disk_root.is_none() || self.last_poll.elapsed() < Duration::from_millis(250) {
            return false;
        }
        self.last_poll = Instant::now();
        if self.scan_mtimes() {
            self.generation += 1;
            return true;
        }
        false
    }

    /// Compiles a module inside a validation error scope.
    pub fn module(
        &self,
        device: &wgpu::Device,
        name: &str,
    ) -> Result<wgpu::ShaderModule, ShaderError> {
        let src = self.source(name)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        match pollster::block_on(scope.pop()) {
            None => Ok(module),
            Some(e) => Err(ShaderError::Compile {
                name: name.to_owned(),
                message: e.to_string(),
            }),
        }
    }
}

fn parse_import(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("#import")?;
    let rest = rest.trim();
    rest.strip_prefix('"')?.strip_suffix('"')
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIB: EmbeddedShaders = &[
        ("a.wgsl", "#import \"common.wgsl\"\nfn a() {}\n"),
        (
            "b.wgsl",
            "#import \"common.wgsl\"\n#import \"a.wgsl\"\nfn b() {}\n",
        ),
        ("common.wgsl", "const PI: f32 = 3.14159;\n"),
        ("loop.wgsl", "#import \"loop.wgsl\"\n"),
    ];

    #[test]
    fn imports_expand_once() {
        let lib = ShaderLibrary::new(LIB, None);
        let src = lib.source("b.wgsl").unwrap();
        assert_eq!(src.matches("const PI").count(), 1);
        assert!(src.find("fn a()").unwrap() < src.find("fn b()").unwrap());
    }

    #[test]
    fn cycles_and_missing_files_error() {
        let lib = ShaderLibrary::new(LIB, None);
        assert!(matches!(
            lib.source("loop.wgsl"),
            Err(ShaderError::ImportCycle(_))
        ));
        assert!(matches!(
            lib.source("nope.wgsl"),
            Err(ShaderError::Missing(_))
        ));
    }

    #[test]
    fn compile_errors_are_captured_not_panicked() {
        let Some(gpu) = crate::device::test_gpu() else {
            return;
        };
        const BAD: EmbeddedShaders = &[("bad.wgsl", "fn broken( {")];
        let lib = ShaderLibrary::new(BAD, None);
        assert!(matches!(
            lib.module(&gpu.device, "bad.wgsl"),
            Err(ShaderError::Compile { .. })
        ));
    }
}
