//! The renderer: frame graph passes over the voxel acceleration structure.

pub mod calibration;
pub mod camera;
pub mod debug_shade;
pub mod frame;
pub mod gizmo;
pub mod hud;
pub mod lights;
pub mod overlay;
pub mod present;
pub mod renderer;
pub mod sky;
pub mod vis;
pub mod voxel_gpu;

pub use frame::FrameCtx;
pub use renderer::Renderer;

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

    /// Disk location for hot reload, only in debug builds.
    pub fn disk_root() -> Option<std::path::PathBuf> {
        if cfg!(debug_assertions) {
            Some(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../shaders"))
        } else {
            None
        }
    }

    pub fn library() -> mc2_gpu::ShaderLibrary {
        mc2_gpu::ShaderLibrary::new(EMBEDDED, disk_root())
    }
}

#[cfg(test)]
mod tests {
    /// Every shader with an entry point must pass naga validation, so a
    /// broken WGSL edit fails `cargo test` without needing a GPU.
    #[test]
    fn every_entry_shader_validates() {
        let lib = super::shaders::library();
        let mut failures = Vec::new();
        for (name, raw) in super::shaders::EMBEDDED {
            if !["@compute", "@vertex", "@fragment"]
                .iter()
                .any(|e| raw.contains(e))
            {
                continue;
            }
            let src = lib.source(name).expect("imports resolve");
            let module = match naga::front::wgsl::parse_str(&src) {
                Ok(m) => m,
                Err(e) => {
                    failures.push(format!("{name}: {}", e.emit_to_string(&src)));
                    continue;
                }
            };
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            if let Err(e) = validator.validate(&module) {
                failures.push(format!("{name}: {}", e.emit_to_string(&src)));
            }
        }
        assert!(
            failures.is_empty(),
            "{}",
            failures.join(
                "
"
            )
        );
    }

    #[test]
    fn every_embedded_shader_parses() {
        let lib = super::shaders::library();
        for (name, _) in super::shaders::EMBEDDED {
            // Library files are imported by others and may not stand alone,
            // but every file must at least resolve its imports.
            lib.source(name)
                .unwrap_or_else(|e| panic!("shader {name} failed to resolve: {e}"));
        }
    }
}
