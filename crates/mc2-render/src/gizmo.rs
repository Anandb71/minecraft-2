//! World-space line gizmos: build previews, selections, debug shapes.

use crate::frame::FrameCtx;
use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use mc2_gpu::{HotRender, Pass, PassBuilder, PassContext, TexHandle, bind, bind_group, layout};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GizmoVertex {
    /// Camera-relative position, metres.
    pub rel: [f32; 3],
    pub color: [f32; 4],
}

/// Line list for one frame, positions stored camera-relative.
#[derive(Default)]
pub struct Gizmos {
    pub camera: DVec3,
    pub vertices: Vec<GizmoVertex>,
}

impl Gizmos {
    pub fn begin(&mut self, camera: DVec3) {
        self.camera = camera;
        self.vertices.clear();
    }

    pub fn line(&mut self, a: DVec3, b: DVec3, color: [f32; 4]) {
        for p in [a, b] {
            self.vertices.push(GizmoVertex {
                rel: (p - self.camera).as_vec3().to_array(),
                color,
            });
        }
    }

    pub fn aabb(&mut self, min: DVec3, max: DVec3, color: [f32; 4]) {
        let c = |i: u32| {
            DVec3::new(
                if i & 1 == 0 { min.x } else { max.x },
                if i & 2 == 0 { min.y } else { max.y },
                if i & 4 == 0 { min.z } else { max.z },
            )
        };
        for i in 0..8u32 {
            for bit in [1u32, 2, 4] {
                if i & bit == 0 {
                    self.line(c(i), c(i | bit), color);
                }
            }
        }
    }

    pub fn sphere(&mut self, centre: DVec3, radius: f64, color: [f32; 4]) {
        const SEGMENTS: usize = 32;
        for axis in 0..3 {
            for k in 0..SEGMENTS {
                let point = |j: usize| {
                    let a = j as f64 / SEGMENTS as f64 * std::f64::consts::TAU;
                    let (s, c) = (a.sin() * radius, a.cos() * radius);
                    centre
                        + match axis {
                            0 => DVec3::new(0.0, s, c),
                            1 => DVec3::new(s, 0.0, c),
                            _ => DVec3::new(s, c, 0.0),
                        }
                };
                self.line(point(k), point(k + 1), color);
            }
        }
    }
}

pub struct GizmoPass {
    depth: TexHandle,
    output: TexHandle,
    pipeline: HotRender,
    bgl: wgpu::BindGroupLayout,
    bind_group: Option<(u64, wgpu::BindGroup)>,
    buffer: wgpu::Buffer,
    capacity: usize,
}

impl GizmoPass {
    pub fn new(
        device: &wgpu::Device,
        ctx: &FrameCtx,
        depth: TexHandle,
        output: TexHandle,
        format: wgpu::TextureFormat,
    ) -> Self {
        let bgl = layout(
            device,
            "gizmo",
            wgpu::ShaderStages::FRAGMENT,
            &[bind::texture(wgpu::TextureViewDimension::D2, false)],
        );
        let pipeline = HotRender::new(
            device,
            &ctx.shaders,
            "gizmo.wgsl",
            &[&ctx.frame_layout, &bgl],
            Box::new(move |device, module, layout| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("gizmo"),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<GizmoVertex>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                        })],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::LineList,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            }),
        );
        let capacity = 1024;
        Self {
            depth,
            output,
            pipeline,
            bgl,
            bind_group: None,
            buffer: Self::buffer(device, capacity),
            capacity,
        }
    }

    fn buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo vertices"),
            size: (capacity * std::mem::size_of::<GizmoVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

impl Pass<FrameCtx> for GizmoPass {
    fn name(&self) -> &'static str {
        "post.gizmo"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.depth);
        b.read(self.output);
        b.write(self.output);
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let verts = &ctx.frame.gizmos.vertices;
        if verts.is_empty() {
            return;
        }
        if verts.len() > self.capacity {
            self.capacity = verts.len().next_power_of_two();
            self.buffer = Self::buffer(ctx.device, self.capacity);
        }
        ctx.queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(verts));
        let generation = ctx.graph.generation();
        if self
            .bind_group
            .as_ref()
            .is_none_or(|(g, _)| *g != generation)
        {
            let bg = bind_group(
                ctx.device,
                "gizmo",
                &self.bgl,
                &[wgpu::BindingResource::TextureView(
                    ctx.graph.view(self.depth),
                )],
            );
            self.bind_group = Some((generation, bg));
        }
        let pipeline = self.pipeline.get(ctx.device, &ctx.frame.shaders);
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gizmo"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.graph.view(self.output),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: ctx.timestamps.render(),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &ctx.frame.frame_bind_group, &[]);
        pass.set_bind_group(1, &self.bind_group.as_ref().expect("bind group").1, &[]);
        pass.set_vertex_buffer(0, self.buffer.slice(..));
        pass.draw(0..verts.len() as u32, 0..1);
    }
}

/// Convenience colours.
pub const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 0.9];
pub const RED: [f32; 4] = [1.0, 0.25, 0.2, 0.9];
pub const AMBER: [f32; 4] = [1.0, 0.75, 0.2, 0.9];
