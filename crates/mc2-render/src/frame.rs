//! Per-frame context threaded through every pass.

use crate::hud::HudCanvas;

pub struct FrameCtx {
    pub shaders: mc2_gpu::ShaderLibrary,
    pub time: f32,
    pub frame_index: u32,
    pub hud: HudCanvas,
    pub exposure: f32,
    pub tonemap: bool,
}

impl FrameCtx {
    pub fn new(shaders: mc2_gpu::ShaderLibrary) -> Self {
        Self {
            shaders,
            time: 0.0,
            frame_index: 0,
            hud: HudCanvas::default(),
            exposure: 1.0,
            tonemap: true,
        }
    }
}
