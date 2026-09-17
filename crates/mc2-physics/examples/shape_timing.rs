fn main() {
    use glam::IVec3;
    use mc2_physics::BodyShape;
    use mc2_voxel::material::ids;
    for n in [8, 16, 32, 64] {
        let voxels = vec![ids::STONE_BRICK; (n * n * n) as usize];
        let t = std::time::Instant::now();
        let mut count = 0;
        for _ in 0..20 {
            count += BodyShape::from_voxels(IVec3::splat(n), voxels.clone())
                .unwrap()
                .solid_count;
        }
        println!(
            "{n}^3: {:.3} ms each ({count})",
            t.elapsed().as_secs_f64() * 1000.0 / 20.0
        );
    }
}
