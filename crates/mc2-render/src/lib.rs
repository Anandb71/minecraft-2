//! The renderer: frame graph passes over the voxel acceleration structure.

pub mod bodies;
pub mod calibration;
pub mod camera;
pub mod clouds;
pub mod debug_shade;
pub mod direct;
pub mod fluid;
pub mod fog;
pub mod frame;
pub mod gizmo;
pub mod hud;
pub mod indirect;
pub mod lights;
pub mod overlay;
pub mod post;
pub mod present;
pub mod quality;
pub mod renderer;
pub mod sky;
pub mod skymap;
pub mod svgf;
pub mod upsample;
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

    /// Surface patterns pick materials by id; each id the shader names must
    /// be the Rust material of that name.
    #[test]
    fn surface_detail_material_ids_match() {
        use mc2_voxel::material::ids;
        let expected = [
            ("M_GRANITE", ids::GRANITE),
            ("M_BASALT", ids::BASALT),
            ("M_LIMESTONE", ids::LIMESTONE),
            ("M_SANDSTONE", ids::SANDSTONE),
            ("M_SHALE", ids::SHALE),
            ("M_MARBLE", ids::MARBLE),
            ("M_SLATE", ids::SLATE),
            ("M_DIRT", ids::DIRT),
            ("M_GRASS", ids::GRASS),
            ("M_SAND", ids::SAND),
            ("M_GRAVEL", ids::GRAVEL),
            ("M_CLAY", ids::CLAY),
            ("M_SNOW", ids::SNOW),
            ("M_ICE", ids::ICE),
            ("M_WATER", ids::WATER),
            ("M_OBSIDIAN", ids::OBSIDIAN),
            ("M_GLASS", ids::GLASS),
            ("M_OAK_LOG", ids::OAK_LOG),
            ("M_PINE_LOG", ids::PINE_LOG),
            ("M_PLANKS", ids::PLANKS),
            ("M_LEAVES", ids::LEAVES),
            ("M_PINE_NEEDLES", ids::PINE_NEEDLES),
            ("M_COBBLESTONE", ids::COBBLESTONE),
            ("M_STONE_BRICK", ids::STONE_BRICK),
            ("M_COAL_ORE", ids::COAL_ORE),
            ("M_IRON_ORE", ids::IRON_ORE),
            ("M_COPPER_ORE", ids::COPPER_ORE),
            ("M_GOLD_ORE", ids::GOLD_ORE),
            ("M_IRON", ids::IRON),
            ("M_TNT", ids::TNT),
            ("M_WET_DIRT", ids::WET_DIRT),
            ("M_ROOF_TILE", ids::ROOF_TILE),
            ("M_WHITEWASH", ids::WHITEWASH),
            ("M_WOOL", ids::WOOL),
            ("M_STEEL", ids::STEEL),
            ("M_BEDROCK", ids::BEDROCK),
            ("M_RHYOLITE", ids::RHYOLITE),
            ("M_CONGLOMERATE", ids::CONGLOMERATE),
            ("M_CHALK", ids::CHALK),
            ("M_GNEISS", ids::GNEISS),
            ("M_FARMLAND", ids::FARMLAND),
            ("M_WHEAT", ids::WHEAT),
        ];
        let src = super::shaders::EMBEDDED
            .iter()
            .find(|(n, _)| *n == "surface_detail.wgsl")
            .expect("surface_detail.wgsl embedded")
            .1;
        let mut seen = 0;
        for line in src.lines() {
            let Some(rest) = line.strip_prefix("const M_") else {
                continue;
            };
            let (name, value) = rest.split_once(": u32 = ").expect("const M_*: u32 = Nu;");
            let value: u16 = value.trim_end_matches("u;").parse().expect("id");
            let name = format!("M_{name}");
            let id = expected
                .iter()
                .find(|(n, _)| *n == name)
                .unwrap_or_else(|| panic!("{name} is not checked here"))
                .1;
            assert_eq!(id.0, value, "{name}");
            seen += 1;
        }
        assert_eq!(seen, expected.len());
    }
}
