//! Resolves the HDR scene target onto the display image.

use crate::frame::FrameCtx;
use bytemuck::{Pod, Zeroable};
use mc2_gpu::{HotRender, Pass, PassBuilder, PassContext, TexHandle, bind, bind_group, layout};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PresentUniforms {
    exposure: f32,
    tonemap: u32,
    _pad: [f32; 2],
}

pub struct PresentPass {
    scene: TexHandle,
    output: TexHandle,
    pipeline: HotRender,
    bgl: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    sampler: wgpu::Sampler,
    bind_group: Option<(u64, wgpu::BindGroup)>,
}

impl PresentPass {
    pub fn new(
        device: &wgpu::Device,
        lib: &mc2_gpu::ShaderLibrary,
        scene: TexHandle,
        output: TexHandle,
        format: wgpu::TextureFormat,
    ) -> Self {
        let bgl = layout(
            device,
            "present",
            wgpu::ShaderStages::FRAGMENT,
            &[bind::uniform(), bind::texture_2d(), bind::sampler(true)],
        );
        let pipeline = HotRender::new(
            device,
            lib,
            "present.wgsl",
            &[&bgl],
            Box::new(move |device, module, layout| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("present"),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(format.into())],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            }),
        );
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present uniforms"),
            size: std::mem::size_of::<PresentUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("present linear clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            scene,
            output,
            pipeline,
            bgl,
            uniforms,
            sampler,
            bind_group: None,
        }
    }
}

impl Pass<FrameCtx> for PresentPass {
    fn name(&self) -> &'static str {
        "present"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.scene);
        b.write(self.output);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        ctx.queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&PresentUniforms {
                exposure: ctx.frame.exposure,
                tonemap: u32::from(ctx.frame.tonemap),
                _pad: [0.0; 2],
            }),
        );
        let generation = ctx.graph.generation();
        if self
            .bind_group
            .as_ref()
            .is_none_or(|(g, _)| *g != generation)
        {
            let bg = bind_group(
                ctx.device,
                "present",
                &self.bgl,
                &[
                    self.uniforms.as_entire_binding(),
                    wgpu::BindingResource::TextureView(ctx.graph.view(self.scene)),
                    wgpu::BindingResource::Sampler(&self.sampler),
                ],
            );
            self.bind_group = Some((generation, bg));
        }
        let pipeline = self.pipeline.get(ctx.device, &ctx.frame.shaders);
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("present"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.graph.view(self.output),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: ctx.timestamps.render(),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group.as_ref().expect("bind group").1, &[]);
        pass.draw(0..3, 0..1);
    }
}
