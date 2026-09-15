fn main() {
    let instance = mc2_gpu::Gpu::instance();
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    for a in adapters {
        let i = a.get_info();
        println!(
            "{:?} {:?} {} | ts={} | max_buf={}MB",
            i.backend,
            i.device_type,
            i.name,
            a.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            a.limits().max_buffer_size / (1 << 20)
        );
    }
}
