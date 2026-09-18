//! Immediate-mode HUD: text, rectangles and icons drawn over the final image.
//!
//! Icons come from an RGBA atlas the application paints once and hands
//! over (`HudCanvas::set_icons`); an icon quad is glyph `ICON_BASE + index`.

use crate::frame::FrameCtx;
use bytemuck::{Pod, Zeroable};
use mc2_gpu::{HotRender, Pass, PassBuilder, PassContext, TexHandle, bind, bind_group, layout};

pub const GLYPH_PX: f32 = 8.0;
const FILLED_CELL: u32 = 127;
/// Glyph ids from here on are icons.
pub const ICON_BASE: u32 = 256;

/// Square icons in a square sRGB RGBA8 atlas, row-major cells.
#[derive(Clone, Debug)]
pub struct IconAtlas {
    /// Atlas width and height, pixels.
    pub size: u32,
    /// Icon width and height, pixels.
    pub cell: u32,
    pub pixels: Vec<u8>,
}

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
    icons: Option<std::sync::Arc<IconAtlas>>,
    /// Bumped whenever the atlas changes, so the pass re-uploads it.
    icons_version: u32,
}

impl HudCanvas {
    pub fn clear(&mut self) {
        self.instances.clear();
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Inserts a rectangle beneath everything drawn after index `at`, for
    /// backgrounds sized after their contents are laid out.
    pub fn rect_behind(&mut self, at: usize, x: f32, y: f32, w: f32, h: f32, color: u32) {
        self.instances.insert(
            at.min(self.instances.len()),
            GlyphInstance {
                rect: [x, y, w, h],
                glyph: FILLED_CELL,
                color,
            },
        );
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

    /// Draws icon `index` of the atlas into a square, tinted by `color`
    /// (white leaves it as painted).
    pub fn icon(&mut self, x: f32, y: f32, size: f32, index: u32, color: u32) {
        self.instances.push(GlyphInstance {
            rect: [x, y, size, size],
            glyph: ICON_BASE + index,
            color,
        });
    }

    pub fn set_icons(&mut self, atlas: std::sync::Arc<IconAtlas>) {
        self.icons = Some(atlas);
        self.icons_version += 1;
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
    /// Icon cells per atlas row, and the icon cell size in atlas UV.
    icon_grid: [f32; 2],
}

pub struct HudPass {
    target: TexHandle,
    pipeline: HotRender,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instances: wgpu::Buffer,
    capacity: usize,
    layout: wgpu::BindGroupLayout,
    font_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    icons: Option<wgpu::Texture>,
    icons_version: u32,
    icon_grid: [f32; 2],
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
                bind::texture(wgpu::TextureViewDimension::D2, true),
                bind::sampler(true),
            ],
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hud icons"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // Until the application paints icons, a single clear texel.
        let placeholder = Self::icon_texture(device, 1, 1);
        let bind_group = Self::bind_group(
            device,
            &bgl,
            &uniforms,
            &atlas_view,
            &placeholder.create_view(&Default::default()),
            &sampler,
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
            layout: bgl,
            font_view: atlas_view,
            sampler,
            icons: None,
            icons_version: 0,
            icon_grid: [1.0, 1.0],
        }
    }

    fn icon_texture(device: &wgpu::Device, size: u32, mips: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hud icons"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniforms: &wgpu::Buffer,
        font: &wgpu::TextureView,
        icons: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        bind_group(
            device,
            "hud",
            layout,
            &[
                uniforms.as_entire_binding(),
                wgpu::BindingResource::TextureView(font),
                wgpu::BindingResource::TextureView(icons),
                wgpu::BindingResource::Sampler(sampler),
            ],
        )
    }

    /// Uploads a new icon atlas with a mip chain, so icons drawn small
    /// stay smooth.
    fn upload_icons(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, atlas: &IconAtlas) {
        let mips = atlas.cell.max(1).ilog2().min(6);
        let tex = Self::icon_texture(device, atlas.size, mips);
        let mut level = atlas.pixels.clone();
        let mut size = atlas.size;
        for mip in 0..mips {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: mip,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &level,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 4),
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
            );
            level = downsample(&level, size);
            size /= 2;
        }
        self.bind_group = Self::bind_group(
            device,
            &self.layout,
            &self.uniforms,
            &self.font_view,
            &tex.create_view(&Default::default()),
            &self.sampler,
        );
        let cells = (atlas.size / atlas.cell.max(1)).max(1) as f32;
        self.icon_grid = [cells, 1.0 / cells];
        self.icons = Some(tex);
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
        if ctx.frame.hud.icons_version != self.icons_version
            && let Some(atlas) = ctx.frame.hud.icons.clone()
        {
            self.upload_icons(ctx.device, ctx.queue, &atlas);
            self.icons_version = ctx.frame.hud.icons_version;
        }
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
                icon_grid: self.icon_grid,
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

/// Halves an sRGB RGBA8 image, averaging in linear light weighted by
/// alpha so transparent texels do not darken the edges.
fn downsample(px: &[u8], size: u32) -> Vec<u8> {
    let half = (size / 2).max(1);
    let to_lin = |c: u8| (f32::from(c) / 255.0).powf(2.2);
    let mut out = vec![0u8; (half * half * 4) as usize];
    for y in 0..half {
        for x in 0..half {
            let mut rgb = [0.0f32; 3];
            let mut a = 0.0f32;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let sx = (x * 2 + dx).min(size - 1);
                let sy = (y * 2 + dy).min(size - 1);
                let i = ((sy * size + sx) * 4) as usize;
                let alpha = f32::from(px[i + 3]) / 255.0;
                for c in 0..3 {
                    rgb[c] += to_lin(px[i + c]) * alpha;
                }
                a += alpha;
            }
            let o = ((y * half + x) * 4) as usize;
            for c in 0..3 {
                let v = if a > 0.0 { rgb[c] / a } else { 0.0 };
                out[o + c] = (v.powf(1.0 / 2.2) * 255.0).round() as u8;
            }
            out[o + 3] = (a / 4.0 * 255.0).round() as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_keeps_colour_at_transparent_edges() {
        // One opaque red texel among clear black ones stays pure red.
        let mut px = vec![0u8; 2 * 2 * 4];
        px[0..4].copy_from_slice(&[255, 0, 0, 255]);
        let half = downsample(&px, 2);
        assert_eq!(&half[0..3], &[255, 0, 0]);
        assert_eq!(half[3], 64);
    }

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
