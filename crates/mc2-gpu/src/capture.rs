//! Texture readback, PNG encode/decode and golden image comparison.

use std::path::Path;

/// Blocking readback of an 8-bit RGBA or BGRA texture as tightly packed RGBA.
pub fn read_rgba8(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let size = texture.size();
    let (w, h) = (size.width, size.height);
    let bgra = matches!(
        texture.format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let bytes_per_row = (4 * w).div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capture staging"),
        size: u64::from(bytes_per_row) * u64::from(h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        size,
    );
    queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll during capture");
    rx.recv()
        .expect("map callback")
        .expect("capture buffer mapping failed");
    let view = staging.get_mapped_range(..).expect("mapped capture range");
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h as usize {
        let start = y * bytes_per_row as usize;
        let row = &view[start..start + 4 * w as usize];
        if bgra {
            for px in row.chunks_exact(4) {
                out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
            }
        } else {
            out.extend_from_slice(row);
        }
    }
    drop(view);
    staging.unmap();
    out
}

pub fn save_png(path: &Path, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(std::io::Error::other)?;
    writer
        .write_image_data(rgba)
        .map_err(std::io::Error::other)?;
    Ok(())
}

pub fn load_png(path: &Path) -> std::io::Result<(u32, u32, Vec<u8>)> {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    let mut reader = decoder.read_info().map_err(std::io::Error::other)?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| std::io::Error::other("png too large"))?;
    let mut buf = vec![0; size];
    let info = reader.next_frame(&mut buf).map_err(std::io::Error::other)?;
    buf.truncate(info.buffer_size());
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => buf
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        other => return Err(std::io::Error::other(format!("unsupported png {other:?}"))),
    };
    Ok((info.width, info.height, rgba))
}

#[derive(Debug, Clone, Copy)]
pub struct ImageDiff {
    /// Mean absolute channel error in 0..255 units.
    pub mean_abs: f32,
    /// Fraction of pixels where any channel differs by more than the threshold.
    pub bad_fraction: f32,
}

pub fn diff(a: &[u8], b: &[u8], threshold: u8) -> ImageDiff {
    assert_eq!(a.len(), b.len(), "image sizes differ");
    let mut sum = 0u64;
    let mut bad = 0usize;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let mut worst = 0u8;
        for c in 0..3 {
            let d = pa[c].abs_diff(pb[c]);
            sum += u64::from(d);
            worst = worst.max(d);
        }
        if worst > threshold {
            bad += 1;
        }
    }
    let pixels = (a.len() / 4).max(1);
    ImageDiff {
        mean_abs: sum as f32 / (pixels * 3) as f32,
        bad_fraction: bad as f32 / pixels as f32,
    }
}

/// Tolerances absorb driver-level float differences, not visual changes.
pub struct GoldenTolerance {
    pub channel_threshold: u8,
    pub max_mean_abs: f32,
    pub max_bad_fraction: f32,
}

impl Default for GoldenTolerance {
    fn default() -> Self {
        Self {
            channel_threshold: 12,
            max_mean_abs: 1.5,
            max_bad_fraction: 0.005,
        }
    }
}

/// Compares against `dir/name.png`. With `MC2_BLESS=1` the reference is
/// (re)written instead. On failure the actual image and a difference image
/// are written to `target/golden-out` for inspection.
pub fn check_golden(
    dir: &Path,
    name: &str,
    w: u32,
    h: u32,
    rgba: &[u8],
    tol: &GoldenTolerance,
) -> Result<ImageDiff, String> {
    let reference = dir.join(format!("{name}.png"));
    if std::env::var("MC2_BLESS").is_ok_and(|v| v == "1") {
        save_png(&reference, w, h, rgba).map_err(|e| e.to_string())?;
        return Ok(ImageDiff {
            mean_abs: 0.0,
            bad_fraction: 0.0,
        });
    }
    let (rw, rh, expected) = load_png(&reference).map_err(|e| {
        format!(
            "missing golden {} ({e}); run with MC2_BLESS=1",
            reference.display()
        )
    })?;
    if (rw, rh) != (w, h) {
        return Err(format!("golden {name} is {rw}x{rh}, got {w}x{h}"));
    }
    let d = diff(rgba, &expected, tol.channel_threshold);
    if d.mean_abs > tol.max_mean_abs || d.bad_fraction > tol.max_bad_fraction {
        let out = dir.join("../target/golden-out");
        let _ = save_png(&out.join(format!("{name}.actual.png")), w, h, rgba);
        let delta: Vec<u8> = rgba
            .chunks_exact(4)
            .zip(expected.chunks_exact(4))
            .flat_map(|(a, b)| {
                let m = (0..3).map(|c| a[c].abs_diff(b[c])).max().unwrap_or(0);
                [m.saturating_mul(4), 0, 0, 255]
            })
            .collect();
        let _ = save_png(&out.join(format!("{name}.diff.png")), w, h, &delta);
        return Err(format!(
            "golden {name} regressed: mean abs {:.3} (max {}), bad pixels {:.4}% (max {}%)",
            d.mean_abs,
            tol.max_mean_abs,
            d.bad_fraction * 100.0,
            tol.max_bad_fraction * 100.0
        ));
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_counts_bad_pixels() {
        let a = vec![10u8, 10, 10, 255, 200, 200, 200, 255];
        let b = vec![10u8, 12, 10, 255, 100, 200, 200, 255];
        let d = diff(&a, &b, 8);
        assert_eq!(d.bad_fraction, 0.5);
        assert!((d.mean_abs - 102.0 / 6.0).abs() < 1e-4);
    }

    #[test]
    fn png_roundtrip() {
        let dir = std::env::temp_dir().join("mc2_png_roundtrip");
        let path = dir.join("t.png");
        let px: Vec<u8> = (0..16u8).flat_map(|i| [i * 10, 255 - i, i, 255]).collect();
        save_png(&path, 4, 4, &px).unwrap();
        let (w, h, back) = load_png(&path).unwrap();
        assert_eq!((w, h), (4, 4));
        assert_eq!(back, px);
    }
}
