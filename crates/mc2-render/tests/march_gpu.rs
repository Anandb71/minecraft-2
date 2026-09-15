//! The GPU marcher must agree with the CPU reference marcher, pixel by pixel.

use glam::{DVec3, IVec3, Mat4, Vec4};
use mc2_gpu::capture::read_texture;
use mc2_render::Renderer;
use mc2_render::camera::Camera;
use mc2_render::quality::Quality;
use mc2_render::renderer::RendererOptions;
use mc2_render::voxel_gpu::GpuWorldConfig;
use mc2_voxel::march::raycast;
use mc2_voxel::samples::{Rng, rolling_terrain};

#[test]
fn gpu_hits_match_cpu_reference() {
    let Some(gpu) = mc2_gpu::device::test_gpu() else {
        return;
    };
    let size = (160u32, 90u32);
    let mut renderer = Renderer::new(
        &gpu,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        size,
        RendererOptions {
            world: GpuWorldConfig {
                tree_words: 1 << 20,
                voxel_words: 4 << 20,
                upload_budget: usize::MAX,
                proximity_m: 1.0e4,
                structure_budget_ms: f32::INFINITY,
            },
            // Every brick is resident; disable sub-pixel LOD so both
            // marchers see voxels.
            quality: Quality {
                render_scale: 1.0,
                lod_pixels: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let mut world = rolling_terrain(&mut Rng(0x9e37_79b9_7f4a_7c15));
    let mut camera = Camera {
        position: DVec3::new(6.3, 19.1, 4.7),
        ..Default::default()
    };
    camera.look_at(DVec3::new(34.0, 12.0, 26.0));
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    for _ in 0..12 {
        renderer.prepare(&gpu, &mut world, &camera, 0.016);
        renderer.render(&gpu, target.clone());
    }
    let ids = read_texture(
        &gpu.device,
        &gpu.queue,
        renderer.graph_texture("vis id").unwrap(),
    );
    let depth = read_texture(
        &gpu.device,
        &gpu.queue,
        renderer.graph_texture("vis depth").unwrap(),
    );

    let inv = Mat4::from_cols_array_2d(&renderer.frame.uniforms.inv_view_proj);
    let (mut checked, mut voxel_mismatch, mut presence_mismatch) = (0, 0, 0);
    for y in (0..size.1).step_by(3) {
        for x in (0..size.0).step_by(3) {
            let i = (y * size.0 + x) as usize;
            let w: Vec<u32> = ids[i * 16..i * 16 + 16]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            let d = f32::from_le_bytes(depth[i * 4..i * 4 + 4].try_into().unwrap());
            let uv = (
                (x as f32 + 0.5) / size.0 as f32,
                (y as f32 + 0.5) / size.1 as f32,
            );
            let far = inv * Vec4::new(uv.0 * 2.0 - 1.0, 1.0 - uv.1 * 2.0, 0.5, 1.0);
            let dir = (far.truncate() / far.w).normalize().as_dvec3();
            let cpu = raycast(&world, camera.position, dir, 4096.0);
            let gpu_hit = w[3] >> 30 != 0;
            assert_ne!(w[3] >> 30, 3, "pixel {x},{y} hit non-resident LOD");
            checked += 1;
            match cpu {
                Some(h) if gpu_hit => {
                    let gv = IVec3::new(w[0] as i32, w[1] as i32, w[2] as i32);
                    if gv != h.voxel {
                        voxel_mismatch += 1;
                    }
                    assert!(
                        (f64::from(d) - h.t).abs() < 0.01 + h.t * 1e-4,
                        "pixel {x},{y}: depth {d} vs {}",
                        h.t
                    );
                    assert_eq!(w[3] & 0xffff, u32::from(h.material.0), "pixel {x},{y}");
                }
                None if !gpu_hit => {}
                _ => presence_mismatch += 1,
            }
        }
    }
    eprintln!(
        "{checked} pixels, {voxel_mismatch} voxel and {presence_mismatch} presence mismatches"
    );
    assert!(voxel_mismatch * 50 <= checked, "too many voxel mismatches");
    assert!(
        presence_mismatch * 200 <= checked,
        "too many hit/miss mismatches"
    );
}

#[test]
fn feedback_streams_visible_bricks_only() {
    let Some(gpu) = mc2_gpu::device::test_gpu() else {
        return;
    };
    let size = (96u32, 54u32);
    let mut renderer = Renderer::new(
        &gpu,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        size,
        RendererOptions {
            world: GpuWorldConfig {
                tree_words: 1 << 20,
                voxel_words: 4 << 20,
                upload_budget: 400 * 356,
                proximity_m: 0.0,
                structure_budget_ms: f32::INFINITY,
            },
            quality: Quality {
                render_scale: 1.0,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    renderer.beam = false;
    let mut world = rolling_terrain(&mut Rng(0x9e37_79b9_7f4a_7c15));
    let total_bricks: usize = world.chunks().map(|(_, t)| t.brick_count()).sum();
    let mut camera = Camera {
        position: DVec3::new(6.3, 19.1, 4.7),
        ..Default::default()
    };
    camera.look_at(DVec3::new(34.0, 12.0, 26.0));
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let mut resident = Vec::new();
    let mut requested = 0;
    for _ in 0..40 {
        renderer.prepare(&gpu, &mut world, &camera, 0.016);
        renderer.render(&gpu, target.clone());
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        resident.push(renderer.world.stats.bricks_resident);
        requested += renderer.world.stats.feedback_requests_last_frame;
    }
    eprintln!(
        "resident after 40 frames: {} of {total_bricks}, {requested} requests",
        resident.last().unwrap()
    );
    assert_eq!(resident[0], 0, "nothing uploads before a ray asks");
    assert!(requested > 0, "marcher never reported a non-resident brick");
    assert!(resident.windows(2).all(|w| w[1] >= w[0]));
    let last = *resident.last().unwrap();
    assert!(last > 100, "only {last} bricks streamed");
    assert!(
        last < total_bricks / 2,
        "{last} of {total_bricks} resident: streaming is not visibility driven"
    );
}
