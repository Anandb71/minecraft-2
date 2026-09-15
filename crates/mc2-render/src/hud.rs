//! Immediate-mode HUD: text and rectangles drawn over the final image.

use crate::frame::FrameCtx;
use bytemuck::{Pod, Zeroable};
use mc2_gpu::{HotRender, Pass, PassBuilder, PassContext, TexHandle, bind, bind_group, layout};

pub const GLYPH_PX: f32 = 8.0;
const FILLED_CELL: u32 = 127;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GlyphInstance {
    pub rect: [f32; 4],
    pub glyph: u32,
    pub color: u32,
}

pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    u32::from_le_bytes([r, g, b, a])
}

/// Collects quads for one frame.
#[derive(Default)]
pub struct HudCanvas {
    instances: Vec<GlyphInstance>,
}

impl HudCanvas {
    pub fn clear(&mut self) {
        self.instances.clear();
    }

    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: u32) {
        self.instances.push(GlyphInstance {
            rect: [x, y, w, h],
            glyph: FILLED_CELL,
            color,
        });
    }

    /// Draws ASCII text; returns the advance width in pixels.
    pub fn text(&mut self, x: f32, y: f32, scale: f32, color: u32, s: &str) -> f32 {
        let size = GLYPH_PX * scale;
        let mut cx = x;
        for ch in s.chars() {
            let code = ch as u32;
            if code != u32::from(b' ') && code < FILLED_CELL {
                self.instances.push(GlyphInstance {
                    rect: [cx, y, size, size],
                    glyph: code,
                    color,
                });
            }
            cx += size;
        }
        cx - x
    }

    pub fn text_width(s: &str, scale: f32) -> f32 {
        s.chars().count() as f32 * GLYPH_PX * scale
    }

    pub fn instances(&self) -> &[GlyphInstance] {
        &self.instances
    }
}

/// 128x64 R8 atlas: 16 columns by 8 rows of 8x8 cells, one per ASCII code.
fn atlas_pixels() -> Vec<u8> {
    let mut px = vec![0u8; 128 * 64];
    for code in 0..128usize {
        let (cx, cy) = (code % 16, code / 16);
        let rows = font8x8::legacy::BASIC_LEGACY[code];
        for (y, row) in rows.iter().enumerate() {
            for x in 0..8 {
                let on = code == FILLED_CELL as usize || row & (1 << x) != 0;
                if on {
                    px[(cy * 8 + y) * 128 + cx * 8 + x] = 255;
                }
            }
        }
    }
    px
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HudUniforms {
    screen: [f32; 2],
    _pad: [f32; 2],
}

pub struct HudPass {
    target: TexHandle,
    pipeline: HotRender,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instances: wgpu::Buffer,
    capacity: usize,
}

impl HudPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lib: &mc2_gpu::ShaderLibrary,
        target: TexHandle,
        format: wgpu::TextureFormat,
    ) -> Self {
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hud atlas"),
            size: wgpu::Extent3d {
                width: 128,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            atlas.as_image_copy(),
            &atlas_pixels(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(128),
                rows_per_image: Some(64),
            },
            atlas.size(),
        );
        let atlas_view = atlas.create_view(&Default::default());
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud uniforms"),
            size: std::mem::size_of::<HudUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = layout(
            device,
            "hud",
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            &[
                bind::uniform(),
                bind::texture(wgpu::TextureViewDimension::D2, false),
            ],
        );
        let bind_group = bind_group(
            device,
            "hud",
            &bgl,
            &[
                uniforms.as_entire_binding(),
                wgpu::BindingResource::TextureView(&atlas_view),
            ],
        );
        let pipeline = HotRender::new(
            device,
            lib,
            "hud_text.wgsl",
            &[&bgl],
            Box::new(move |device, module, layout| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("hud"),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<GlyphInstance>() as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &wgpu::vertex_attr_array![
                                0 => Float32x4, 1 => Uint32, 2 => Uint32
                            ],
                        })],
                    },
                    primitive: wgpu::PrimitiveState::default(),
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
        let capacity = 4096;
        Self {
            target,
            pipeline,
            uniforms,
            bind_group,
            instances: Self::instance_buffer(device, capacity),
            capacity,
        }
    }

    fn instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud instances"),
            size: (capacity * std::mem::size_of::<GlyphInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

impl Pass<FrameCtx> for HudPass {
    fn name(&self) -> &'static str {
        "post.hud"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) {
        b.read(self.target);
        b.write(self.target);
        b.output();
    }

    fn execute(&mut self, ctx: &mut PassContext<'_, FrameCtx>) {
        let quads = ctx.frame.hud.instances();
        if quads.is_empty() {
            return;
        }
        if quads.len() > self.capacity {
            self.capacity = quads.len().next_power_of_two();
            self.instances = Self::instance_buffer(ctx.device, self.capacity);
        }
        let extent = ctx.graph.extent(self.target);
        ctx.queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&HudUniforms {
                screen: [extent.width as f32, extent.height as f32],
                _pad: [0.0; 2],
            }),
        );
        ctx.queue
            .write_buffer(&self.instances, 0, bytemuck::cast_slice(quads));
        let pipeline = self.pipeline.get(ctx.device, &ctx.frame.shaders);
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hud"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.graph.view(self.target),
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
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.draw(0..6, 0..quads.len() as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_has_glyph_pixels_where_expected() {
        let px = atlas_pixels();
        let lit = |code: usize| {
            let (cx, cy) = (code % 16, code / 16);
            (0..64)
                .filter(|i| px[(cy * 8 + i / 8) * 128 + cx * 8 + i % 8] > 0)
                .count()
        };
        assert_eq!(lit(b' ' as usize), 0);
        assert!(lit(b'A' as usize) > 10);
        assert_eq!(lit(FILLED_CELL as usize), 64);
    }

    #[test]
    fn text_advances_by_glyph_size() {
        let mut c = HudCanvas::default();
        let w = c.text(0.0, 0.0, 2.0, 0, "ab c");
        assert_eq!(w, 64.0);
        assert_eq!(c.instances().len(), 3);
    }
}
