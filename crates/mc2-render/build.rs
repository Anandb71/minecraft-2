//! Embeds every `shaders/*.wgsl` file so release binaries run from any
//! directory. Debug builds read the same files from disk for hot reload.

use std::fmt::Write;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let dir = manifest.join("../../shaders");
    println!("cargo:rerun-if-changed={}", dir.display());

    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".wgsl"))
                .collect()
        })
        .unwrap_or_default();
    names.sort();

    let mut out = String::from("pub const EMBEDDED: mc2_gpu::EmbeddedShaders = &[\n");
    for name in &names {
        let path = dir.join(name).canonicalize().expect("shader path");
        println!("cargo:rerun-if-changed={}", path.display());
        writeln!(
            out,
            "    ({name:?}, include_str!({:?})),",
            path.display().to_string()
        )
        .expect("write to string");
    }
    out.push_str("];\n");

    let dest = PathBuf::from(std::env::var("OUT_DIR").expect("out dir")).join("shaders.rs");
    std::fs::write(dest, out).expect("write embedded shader table");
}
