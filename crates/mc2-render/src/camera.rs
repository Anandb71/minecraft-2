//! Camera and the per-frame uniform block.

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, IVec3, Mat4, Vec2, Vec3};

pub const VOXELS_PER_METRE: f64 = 16.0;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// World position in metres.
    pub position: DVec3,
    /// Radians; yaw 0 looks along +z, positive yaw turns toward +x.
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: DVec3::new(8192.0, 200.0, 8192.0),
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 70f32.to_radians(),
            near: 0.05,
        }
    }
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
    }

    /// Screen-right in a right-handed, y-up world.
    pub fn right(&self) -> Vec3 {
        Vec3::new(-self.yaw.cos(), 0.0, self.yaw.sin())
    }

    /// Camera-relative view matrix (no translation).
    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_to_mat4(Vec3::ZERO, self.forward(), Vec3::Y)
    }

    /// Reverse-Z infinite projection, shared by rasterised passes.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        glam::camera::rh::proj::directx::perspective_infinite_reverse(self.fov_y, aspect, self.near)
    }

    pub fn look_at(&mut self, target: DVec3) {
        let d = (target - self.position).normalize().as_vec3();
        self.pitch = d.y.clamp(-1.0, 1.0).asin();
        self.yaw = d.x.atan2(d.z);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct FrameUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub camera_voxel: [i32; 3],
    pub frame_index: u32,
    pub camera_frac: [f32; 3],
    pub time: f32,
    pub prev_camera_delta: [f32; 3],
    pub dt: f32,
    pub render_size: [f32; 2],
    pub output_size: [f32; 2],
    pub jitter: [f32; 2],
    pub prev_jitter: [f32; 2],
    pub sun_dir: [f32; 3],
    pub exposure: f32,
    pub debug_mode: u32,
    pub restir_candidates: u32,
    pub pixel_angle: f32,
    pub lod_pixels: f32,
    pub beam: u32,
    pub trace_stride: u32,
    pub light_count: u32,
    pub _pad: u32,
    pub sun_illuminance: [f32; 3],
    pub sun_angular_radius: f32,
    pub moon_dir: [f32; 3],
    pub moon_illuminance: f32,
    pub moon_phase: f32,
    pub star_rotation: f32,
    pub latitude: f32,
    pub sky_flags: u32,
}

/// `sky_flags` bit: the moon-lit sky LUTs are rendered this frame.
pub const SKY_MOON: u32 = 1;

/// Sun, moon and stars as seen from the world, in world axes (+X east, +Y up,
/// -Z north). Illuminance is above the atmosphere, in lux.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Celestial {
    pub sun_dir: Vec3,
    pub sun_illuminance: Vec3,
    pub moon_dir: Vec3,
    pub moon_illuminance: f32,
    /// 0 new moon, 0.5 full.
    pub moon_phase: f32,
    /// Star field rotation about the celestial pole, radians.
    pub star_rotation: f32,
    /// Observer latitude, radians; tilts the celestial pole.
    pub latitude: f32,
}

impl Default for Celestial {
    fn default() -> Self {
        Self {
            sun_dir: Vec3::new(0.4, 0.8, 0.3).normalize(),
            sun_illuminance: Vec3::splat(127_500.0),
            moon_dir: Vec3::new(-0.4, -0.8, -0.3).normalize(),
            moon_illuminance: 0.32,
            moon_phase: 0.5,
            star_rotation: 0.0,
            latitude: 47f32.to_radians(),
        }
    }
}

impl Celestial {
    /// Whether moonlight is worth its own sky LUTs: the moon is up (or just
    /// set, still lighting the upper atmosphere) and the sun is not.
    pub fn moon_sky(&self) -> bool {
        self.moon_illuminance > 0.0 && self.moon_dir.y > -0.25 && self.sun_dir.y < 0.05
    }
}

/// Halton (2,3) sequence, centred, for sub-pixel jitter.
pub fn halton_jitter(index: u32) -> Vec2 {
    fn halton(mut i: u32, base: u32) -> f32 {
        let mut f = 1.0;
        let mut r = 0.0;
        while i > 0 {
            f /= base as f32;
            r += f * (i % base) as f32;
            i /= base;
        }
        r
    }
    let i = index % 16 + 1;
    Vec2::new(halton(i, 2) - 0.5, halton(i, 3) - 0.5)
}

pub struct FrameInputs {
    pub camera: Camera,
    pub prev_camera: Camera,
    pub render_size: (u32, u32),
    pub output_size: (u32, u32),
    pub frame_index: u32,
    pub time: f32,
    pub dt: f32,
    pub jitter: bool,
    pub celestial: Celestial,
    pub exposure: f32,
    pub debug_mode: u32,
    pub restir_candidates: u32,
    pub lod_pixels: f32,
    pub beam: bool,
    pub trace_stride: u32,
    pub light_count: u32,
}

impl FrameUniforms {
    pub fn build(i: &FrameInputs) -> Self {
        let aspect = i.render_size.0 as f32 / i.render_size.1.max(1) as f32;
        let vp = i.camera.projection(aspect) * i.camera.view();
        let prev_vp = i.prev_camera.projection(aspect) * i.prev_camera.view();
        let voxel_pos = i.camera.position * VOXELS_PER_METRE;
        let base = voxel_pos.floor();
        let (jitter, prev_jitter) = if i.jitter {
            (
                halton_jitter(i.frame_index),
                halton_jitter(i.frame_index.wrapping_sub(1)),
            )
        } else {
            (Vec2::ZERO, Vec2::ZERO)
        };
        Self {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            prev_view_proj: prev_vp.to_cols_array_2d(),
            camera_voxel: base.as_ivec3().to_array(),
            frame_index: i.frame_index,
            camera_frac: (voxel_pos - base).as_vec3().to_array(),
            time: i.time,
            prev_camera_delta: (i.prev_camera.position - i.camera.position)
                .as_vec3()
                .to_array(),
            dt: i.dt,
            render_size: [i.render_size.0 as f32, i.render_size.1 as f32],
            output_size: [i.output_size.0 as f32, i.output_size.1 as f32],
            jitter: jitter.to_array(),
            prev_jitter: prev_jitter.to_array(),
            sun_dir: i.celestial.sun_dir.normalize().to_array(),
            exposure: i.exposure,
            debug_mode: i.debug_mode,
            restir_candidates: i.restir_candidates,
            pixel_angle: 2.0 * (i.camera.fov_y * 0.5).tan() / i.render_size.1.max(1) as f32,
            lod_pixels: i.lod_pixels,
            beam: u32::from(i.beam),
            trace_stride: i.trace_stride,
            light_count: i.light_count,
            _pad: 0,
            sun_illuminance: i.celestial.sun_illuminance.to_array(),
            sun_angular_radius: 0.004_675,
            moon_dir: i.celestial.moon_dir.normalize_or(Vec3::NEG_Y).to_array(),
            moon_illuminance: i.celestial.moon_illuminance,
            moon_phase: i.celestial.moon_phase,
            star_rotation: i.celestial.star_rotation,
            latitude: i.celestial.latitude,
            sky_flags: if i.celestial.moon_sky() { SKY_MOON } else { 0 },
        }
    }

    pub fn camera_voxel(&self) -> IVec3 {
        IVec3::from_array(self.camera_voxel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_block_matches_wgsl_size() {
        // 3 mat4 (192) + 3 x 16 + 4 x vec2 (32) + 6 x 16, as laid out in frame.wgsl
        assert_eq!(std::mem::size_of::<FrameUniforms>(), 368);
    }

    #[test]
    fn forward_and_look_at_agree() {
        let mut c = Camera {
            position: DVec3::ZERO,
            ..Default::default()
        };
        c.look_at(DVec3::new(3.0, 4.0, -5.0));
        let want = Vec3::new(3.0, 4.0, -5.0).normalize();
        assert!((c.forward() - want).length() < 1e-5);
        assert!(c.right().dot(c.forward()).abs() < 1e-5);
        // Right-handed: forward x right points down.
        assert!(c.forward().cross(c.right()).y < 0.0);
    }

    #[test]
    fn centre_pixel_ray_is_forward() {
        let c = Camera {
            position: DVec3::new(100.3, 50.7, 20.1),
            yaw: 0.7,
            pitch: -0.2,
            ..Default::default()
        };
        let u = FrameUniforms::build(&FrameInputs {
            camera: c,
            prev_camera: c,
            render_size: (64, 64),
            output_size: (64, 64),
            frame_index: 0,
            time: 0.0,
            dt: 0.0,
            jitter: false,
            celestial: Celestial::default(),
            exposure: 1.0,
            debug_mode: 0,
            restir_candidates: 32,
            lod_pixels: 1.0,
            beam: false,
            trace_stride: 1,
            light_count: 0,
        });
        let inv = Mat4::from_cols_array_2d(&u.inv_view_proj);
        let far = inv * glam::Vec4::new(0.0, 0.0, 0.5, 1.0);
        let dir = (far.truncate() / far.w).normalize();
        assert!(
            (dir - c.forward()).length() < 1e-4,
            "{dir} vs {}",
            c.forward()
        );
        assert_eq!(u.camera_voxel, [1604, 811, 321]);
    }
}
