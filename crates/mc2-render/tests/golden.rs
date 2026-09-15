//! Golden image tests: renderer output compared against stored references.
//!
//! References are rendered on the software adapter (WARP on Windows,
//! lavapipe on Linux) so CI and developer machines agree. Regenerate with
//! `MC2_BLESS=1 cargo test -p mc2-render --test golden`.

use mc2_gpu::capture::{GoldenTolerance, check_golden, read_rgba8};
use mc2_gpu::{Gpu, GpuOptions};
use mc2_render::Renderer;
use mc2_render::renderer::{RenderMode, RendererOptions};
use mc2_render::voxel_gpu::GpuWorldConfig;
use std::path::PathBuf;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden")
}

fn software_gpu() -> Option<Gpu> {
    if std::env::var("MC2_SKIP_GOLDEN").is_ok_and(|v| v == "1") {
        eprintln!("MC2_SKIP_GOLDEN=1: references are rendered on WARP, skipping here");
        return None;
    }
    match Gpu::headless(&GpuOptions {
        force_fallback: true,
    }) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            if std::env::var("MC2_REQUIRE_GPU").is_ok_and(|v| v == "1") {
                panic!("golden tests need a software adapter: {e}");
            }
            eprintln!("skipping golden test: {e}");
            None
        }
    }
}

fn render(gpu: &Gpu, renderer: &mut Renderer, size: (u32, u32), frames: u32) -> Vec<u8> {
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("golden target"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    for _ in 0..frames {
        renderer.render(gpu, target.clone());
    }
    read_rgba8(&gpu.device, &gpu.queue, &target)
}

#[test]
fn calibration_display_transform() {
    let Some(gpu) = software_gpu() else {
        return;
    };
    let size = (320, 180);
    let mut renderer = Renderer::new(
        &gpu,
        FORMAT,
        size,
        RendererOptions {
            mode: RenderMode::Calibration,
            world: GpuWorldConfig {
                tree_words: 1 << 16,
                voxel_words: 1 << 16,
                ..Default::default()
            },
            render_scale: 1.0,
        },
    );
    renderer.frame.time = 1.0;
    let rgba = render(&gpu, &mut renderer, size, 1);
    let result = check_golden(
        &golden_dir(),
        "calibration",
        size.0,
        size.1,
        &rgba,
        &GoldenTolerance::default(),
    );
    if let Err(e) = result {
        panic!("{e}");
    }
}
