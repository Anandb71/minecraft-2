//! The renderer: frame graph passes over the voxel acceleration structure.

pub mod frame;
pub mod hud;

pub use frame::FrameCtx;

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
