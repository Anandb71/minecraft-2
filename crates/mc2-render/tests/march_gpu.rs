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
    // Unjittered rays, so pixel centres match the CPU reference.
    renderer.jitter = false;
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
    renderer.jitter = false;
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

#[test]
fn gpu_body_hits_match_cpu_reference() {
    use glam::Quat;
    use mc2_physics::{BodyShape, PhysicsWorld};
    use mc2_render::bodies::BodyInstance;
    use mc2_voxel::material::{MaterialId, ids};
    use std::sync::Arc;

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
            quality: Quality {
                render_scale: 1.0,
                lod_pixels: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    renderer.jitter = false;
    let mut world = rolling_terrain(&mut Rng(0x9e37_79b9_7f4a_7c15));
    let mut camera = Camera {
        position: DVec3::new(6.3, 19.1, 4.7),
        ..Default::default()
    };
    let target_point = DVec3::new(34.0, 12.0, 26.0);
    camera.look_at(target_point);
    let forward = (target_point - camera.position).normalize();

    // Bodies of several shapes and materials, turned every which way, in
    // the view: a hollow crate, a plank, and a ragged lump.
    let mut physics = PhysicsWorld::new();
    let crate_voxels: Vec<MaterialId> = (0..12 * 12 * 12)
        .map(|i| {
            let v = IVec3::new(i % 12, (i / 12) % 12, i / 144);
            let edge = v.cmpeq(IVec3::ZERO) | v.cmpeq(IVec3::splat(11));
            if edge.x as u8 + edge.y as u8 + edge.z as u8 >= 2 {
                ids::PLANKS
            } else {
                MaterialId(0)
            }
        })
        .collect();
    let lump: Vec<MaterialId> = (0..9 * 7 * 5)
        .map(|i| {
            let v = IVec3::new(i % 9, (i / 9) % 7, i / 63);
            if (v.x * 7 + v.y * 3 + v.z * 5) % 4 != 0 {
                ids::GRANITE
            } else {
                MaterialId(0)
            }
        })
        .collect();
    let shapes = [
        BodyShape::from_voxels(IVec3::splat(12), crate_voxels).unwrap(),
        BodyShape::from_voxels(IVec3::new(24, 2, 4), vec![ids::PLANKS; 192]).unwrap(),
        BodyShape::from_voxels(IVec3::new(9, 7, 5), lump).unwrap(),
    ];
    for (k, shape) in shapes.into_iter().enumerate() {
        let shape = Arc::new(shape);
        for j in 0..3 {
            let along = 4.0 + 3.0 * k as f64 + j as f64;
            let side = DVec3::new(-forward.z, 0.0, forward.x) * (j as f64 - 1.0) * 1.2;
            let pos = camera.position + forward * along + side + DVec3::Y * (0.4 * k as f64 - 0.4);
            let rot = Quat::from_euler(
                glam::EulerRot::XYZ,
                0.7 * (k + j) as f32,
                1.3 * j as f32 + 0.2,
                0.4 * k as f32,
            );
            physics.spawn(shape.clone(), pos, rot);
        }
    }
    renderer.bodies = physics
        .bodies
        .iter()
        .map(|b| BodyInstance {
            key: b.id.0,
            shape: b.shape.clone(),
            grid_origin: b.grid_origin(),
            grid_rotation: b.grid_rotation(),
        })
        .collect();
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
    assert_eq!(renderer.world.bodies.stats.drawn, 9);
    let ids_tex = read_texture(
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
    let (mut bodies_seen, mut mismatch) = (0, 0);
    for y in 0..size.1 {
        for x in 0..size.0 {
            let i = (y * size.0 + x) as usize;
            let w: Vec<u32> = ids_tex[i * 16..i * 16 + 16]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            let d = f64::from(f32::from_le_bytes(
                depth[i * 4..i * 4 + 4].try_into().unwrap(),
            ));
            let uv = (
                (x as f32 + 0.5) / size.0 as f32,
                (y as f32 + 0.5) / size.1 as f32,
            );
            let far = inv * Vec4::new(uv.0 * 2.0 - 1.0, 1.0 - uv.1 * 2.0, 0.5, 1.0);
            let dir = (far.truncate() / far.w).normalize().as_dvec3();
            let world_t = raycast(&world, camera.position, dir, 4096.0).map_or(f64::MAX, |h| h.t);
            let body = physics
                .raycast(camera.position, dir, 4096.0)
                .filter(|(_, h)| h.t < world_t);
            let gpu_body = w[3] >> 30 == 2;
            match body {
                Some((id, h)) => {
                    bodies_seen += 1;
                    if !gpu_body {
                        mismatch += 1;
                        continue;
                    }
                    assert_eq!(((w[3] >> 19) & 0x7ff) as u64, id.0 & 0x7ff, "pixel {x},{y}");
                    assert!(
                        (d - h.t).abs() < 0.01,
                        "pixel {x},{y}: depth {d} vs {}",
                        h.t
                    );
                    assert_eq!(w[3] & 0xffff, u32::from(h.material.0), "pixel {x},{y}");
                    let gv = IVec3::new(w[0] as i32, w[1] as i32, w[2] as i32);
                    let body = physics.body(id).unwrap();
                    let grid_dir = body.grid_rotation().inverse() * dir.as_vec3();
                    let face = h.axis as u32 * 2 + u32::from(grid_dir[h.axis] < 0.0);
                    if gv != h.voxel || (w[3] >> 16) & 7 != face {
                        mismatch += 1;
                    }
                }
                None if gpu_body => mismatch += 1,
                None => {}
            }
        }
    }
    eprintln!("{bodies_seen} body pixels, {mismatch} mismatches");
    assert!(
        bodies_seen > 250,
        "bodies barely visible: {bodies_seen} pixels"
    );
    assert!(mismatch * 50 <= bodies_seen, "too many body mismatches");
}
