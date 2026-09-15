//! Adapter selection and logical device creation.

use std::fmt;

/// Owned GPU context shared by every pass.
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    /// Timestamp queries on pass descriptors are available.
    pub timestamps: bool,
}

#[derive(Debug)]
pub enum GpuError {
    NoAdapter(String),
    Device(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoAdapter(e) => write!(f, "no suitable GPU adapter: {e}"),
            GpuError::Device(e) => write!(f, "device creation failed: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Knobs for adapter selection, mostly for tests and CI.
#[derive(Clone, Debug, Default)]
pub struct GpuOptions {
    /// Use a software rasteriser (WARP or lavapipe) if one is available.
    pub force_fallback: bool,
}

impl Gpu {
    pub fn instance() -> wgpu::Instance {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        // Vulkan and DX12 first, Metal second. GL cannot run the compute pipeline.
        desc.backends = wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL;
        if let Ok(b) = std::env::var("MC2_BACKEND") {
            desc.backends = match b.to_ascii_lowercase().as_str() {
                "vulkan" => wgpu::Backends::VULKAN,
                "dx12" => wgpu::Backends::DX12,
                "metal" => wgpu::Backends::METAL,
                _ => desc.backends,
            };
        }
        wgpu::Instance::new(desc)
    }

    /// Picks the high performance adapter and creates a device with every
    /// feature the renderer depends on, plus timestamp queries when offered.
    pub fn new(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
        opts: &GpuOptions,
    ) -> Result<Self, GpuError> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: opts.force_fallback,
            compatible_surface: surface,
            apply_limit_buckets: false,
        }))
        .map_err(|e| GpuError::NoAdapter(e.to_string()))?;
        let info = adapter.get_info();
        let offered = adapter.features();

        let wanted_optional = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::FLOAT32_FILTERABLE;
        let required = wgpu::Features::empty();
        let features = required | (offered & wanted_optional);
        let timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);

        let al = adapter.limits();
        let mut limits = wgpu::Limits::default();
        // The voxel pools are the largest buffers we bind; take what the adapter gives.
        limits.max_buffer_size = al.max_buffer_size;
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_storage_buffers_per_shader_stage =
            al.max_storage_buffers_per_shader_stage.min(16);
        limits.max_storage_textures_per_shader_stage =
            al.max_storage_textures_per_shader_stage.min(16);
        limits.max_sampled_textures_per_shader_stage =
            al.max_sampled_textures_per_shader_stage.min(24);
        limits.max_bind_groups = al.max_bind_groups.min(8);
        limits.max_texture_dimension_3d = al.max_texture_dimension_3d;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mc2 device"),
            required_features: features,
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| GpuError::Device(e.to_string()))?;

        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            panic!("wgpu uncaptured error: {e}");
        }));

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            info,
            timestamps,
        })
    }

    /// Headless device for captures, benchmarks and golden tests.
    pub fn headless(opts: &GpuOptions) -> Result<Self, GpuError> {
        Self::new(Self::instance(), None, opts)
    }

    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?}, driver {} {})",
            self.info.name,
            self.info.backend,
            self.info.device_type,
            self.info.driver,
            self.info.driver_info
        )
    }
}

/// Returns a headless GPU for tests, or `None` when no adapter exists and the
/// environment does not demand one. CI sets `MC2_REQUIRE_GPU=1` so a missing
/// software adapter fails loudly instead of skipping the golden tests.
pub fn test_gpu() -> Option<Gpu> {
    let force_fallback = std::env::var("MC2_FORCE_FALLBACK").is_ok_and(|v| v == "1");
    match Gpu::headless(&GpuOptions { force_fallback }) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            if std::env::var("MC2_REQUIRE_GPU").is_ok_and(|v| v == "1") {
                panic!("MC2_REQUIRE_GPU=1 but {e}");
            }
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn headless_device_initialises() {
        let Some(gpu) = super::test_gpu() else {
            return;
        };
        eprintln!("adapter: {}", gpu.describe());
        let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert_eq!(buf.size(), 64);
    }
}
