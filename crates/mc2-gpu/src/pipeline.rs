//! Hot-reloadable pipelines and bind group layout shorthand.

use crate::shader::ShaderLibrary;

/// Binding type shorthands; visibility is supplied by the layout builder.
pub mod bind {
    pub fn uniform() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        }
    }

    pub fn storage(read_only: bool) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        }
    }

    pub fn texture(dim: wgpu::TextureViewDimension, filterable: bool) -> wgpu::BindingType {
        wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: dim,
            multisampled: false,
        }
    }

    pub fn texture_2d() -> wgpu::BindingType {
        texture(wgpu::TextureViewDimension::D2, true)
    }

    pub fn utexture_2d() -> wgpu::BindingType {
        wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        }
    }

    pub fn storage_texture(
        format: wgpu::TextureFormat,
        dim: wgpu::TextureViewDimension,
        access: wgpu::StorageTextureAccess,
    ) -> wgpu::BindingType {
        wgpu::BindingType::StorageTexture {
            access,
            format,
            view_dimension: dim,
        }
    }

    pub fn write_2d(format: wgpu::TextureFormat) -> wgpu::BindingType {
        storage_texture(
            format,
            wgpu::TextureViewDimension::D2,
            wgpu::StorageTextureAccess::WriteOnly,
        )
    }

    pub fn sampler(filtering: bool) -> wgpu::BindingType {
        wgpu::BindingType::Sampler(if filtering {
            wgpu::SamplerBindingType::Filtering
        } else {
            wgpu::SamplerBindingType::NonFiltering
        })
    }
}

/// Builds a layout where binding `i` is `types[i]`.
pub fn layout(
    device: &wgpu::Device,
    label: &str,
    visibility: wgpu::ShaderStages,
    types: &[wgpu::BindingType],
) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = types
        .iter()
        .enumerate()
        .map(|(i, ty)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility,
            ty: *ty,
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

/// Bind group whose binding `i` is `resources[i]`.
pub fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    resources: &[wgpu::BindingResource<'_>],
) -> wgpu::BindGroup {
    let entries: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(i, r)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: r.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// Compute pipeline rebuilt whenever the shader library generation moves.
///
/// Uses an explicit layout, so bind groups survive a reload. A failed
/// rebuild logs the error and keeps the previous pipeline; a failure with
/// no previous pipeline (first build) panics, since the frame cannot run.
pub struct HotCompute {
    shader: &'static str,
    entry: &'static str,
    layout: wgpu::PipelineLayout,
    pipeline: Option<wgpu::ComputePipeline>,
    generation: u64,
}

impl HotCompute {
    pub fn new(
        device: &wgpu::Device,
        lib: &ShaderLibrary,
        shader: &'static str,
        entry: &'static str,
        groups: &[&wgpu::BindGroupLayout],
    ) -> Self {
        let groups: Vec<Option<&wgpu::BindGroupLayout>> = groups.iter().map(|g| Some(*g)).collect();
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(shader),
            bind_group_layouts: &groups,
            immediate_size: 0,
        });
        let mut hc = Self {
            shader,
            entry,
            layout,
            pipeline: None,
            generation: u64::MAX,
        };
        hc.refresh(device, lib);
        hc
    }

    fn refresh(&mut self, device: &wgpu::Device, lib: &ShaderLibrary) {
        if self.generation == lib.generation() && self.pipeline.is_some() {
            return;
        }
        self.generation = lib.generation();
        let built = lib.module(device, self.shader).and_then(|module| {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(self.shader),
                layout: Some(&self.layout),
                module: &module,
                entry_point: Some(self.entry),
                compilation_options: Default::default(),
                cache: None,
            });
            match pollster::block_on(scope.pop()) {
                None => Ok(p),
                Some(e) => Err(crate::shader::ShaderError::Compile {
                    name: format!("{}::{}", self.shader, self.entry),
                    message: e.to_string(),
                }),
            }
        });
        match built {
            Ok(p) => {
                if self.pipeline.is_some() {
                    log::info!("reloaded {}::{}", self.shader, self.entry);
                }
                self.pipeline = Some(p);
            }
            Err(e) if self.pipeline.is_some() => log::error!("shader reload failed: {e}"),
            Err(e) => panic!("shader build failed: {e}"),
        }
    }

    pub fn get(&mut self, device: &wgpu::Device, lib: &ShaderLibrary) -> &wgpu::ComputePipeline {
        self.refresh(device, lib);
        self.pipeline.as_ref().expect("pipeline built in refresh")
    }
}

/// Dispatch count covering `size` with workgroups of `group`.
pub fn groups(size: u32, group: u32) -> u32 {
    size.div_ceil(group)
}

type RenderBuilder =
    Box<dyn Fn(&wgpu::Device, &wgpu::ShaderModule, &wgpu::PipelineLayout) -> wgpu::RenderPipeline>;

/// Render pipeline counterpart of [`HotCompute`]; the builder closure fills in
/// vertex layout, targets and blend state against a freshly compiled module.
pub struct HotRender {
    shader: &'static str,
    layout: wgpu::PipelineLayout,
    builder: RenderBuilder,
    pipeline: Option<wgpu::RenderPipeline>,
    generation: u64,
}

impl HotRender {
    pub fn new(
        device: &wgpu::Device,
        lib: &ShaderLibrary,
        shader: &'static str,
        groups: &[&wgpu::BindGroupLayout],
        builder: RenderBuilder,
    ) -> Self {
        let groups: Vec<Option<&wgpu::BindGroupLayout>> = groups.iter().map(|g| Some(*g)).collect();
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(shader),
            bind_group_layouts: &groups,
            immediate_size: 0,
        });
        let mut hr = Self {
            shader,
            layout,
            builder,
            pipeline: None,
            generation: u64::MAX,
        };
        hr.refresh(device, lib);
        hr
    }

    fn refresh(&mut self, device: &wgpu::Device, lib: &ShaderLibrary) {
        if self.generation == lib.generation() && self.pipeline.is_some() {
            return;
        }
        self.generation = lib.generation();
        let built = lib.module(device, self.shader).and_then(|module| {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let p = (self.builder)(device, &module, &self.layout);
            match pollster::block_on(scope.pop()) {
                None => Ok(p),
                Some(e) => Err(crate::shader::ShaderError::Compile {
                    name: self.shader.to_owned(),
                    message: e.to_string(),
                }),
            }
        });
        match built {
            Ok(p) => self.pipeline = Some(p),
            Err(e) if self.pipeline.is_some() => log::error!("shader reload failed: {e}"),
            Err(e) => panic!("shader build failed: {e}"),
        }
    }

    pub fn get(&mut self, device: &wgpu::Device, lib: &ShaderLibrary) -> &wgpu::RenderPipeline {
        self.refresh(device, lib);
        self.pipeline.as_ref().expect("pipeline built in refresh")
    }
}
