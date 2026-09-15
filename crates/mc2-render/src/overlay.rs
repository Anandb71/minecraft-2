//! Profiler HUD: GPU budget lines, individual passes and CPU scopes, each
//! with last and rolling p99. A line whose p99 exceeds its budget turns red.

use crate::hud::{HudCanvas, rgba};
use mc2_core::RollingStats;
use mc2_core::profiler::ProfileRow;
use mc2_gpu::GpuProfiler;

pub struct BudgetLine {
    pub group: &'static str,
    pub label: &'static str,
    pub ms: f32,
}

/// The 16.6 ms frame budget at 1440p on a mid range 2024 discrete GPU.
pub const BUDGETS: &[BudgetLine] = &[
    BudgetLine {
        group: "vis",
        label: "Primary visibility march",
        ms: 2.5,
    },
    BudgetLine {
        group: "direct",
        label: "Direct light (ReSTIR)",
        ms: 2.0,
    },
    BudgetLine {
        group: "indirect",
        label: "Indirect light",
        ms: 2.5,
    },
    BudgetLine {
        group: "refl",
        label: "Reflections",
        ms: 1.0,
    },
    BudgetLine {
        group: "denoise",
        label: "Denoise and upsample",
        ms: 2.0,
    },
    BudgetLine {
        group: "vol",
        label: "Volumetrics and clouds",
        ms: 2.0,
    },
    BudgetLine {
        group: "sky",
        label: "Atmosphere and sky",
        ms: 0.5,
    },
    BudgetLine {
        group: "sim",
        label: "Fluid and fire (amortized)",
        ms: 1.5,
    },
    BudgetLine {
        group: "post",
        label: "Post and present",
        ms: 1.0,
    },
];

pub const FRAME_BUDGET_MS: f32 = 16.6;

const WHITE: u32 = 0xffff_ffff;
const DIM: u32 = 0xffb0_b0b0;
const RED: u32 = 0xff40_40ff;
const GREEN: u32 = 0xff60_ff60;
const YELLOW: u32 = 0xff40_e0ff;

pub struct OverlayInput<'a> {
    pub gpu: &'a GpuProfiler,
    pub cpu: &'a [ProfileRow],
    pub frame_ms: &'a RollingStats,
    pub adapter: &'a str,
    pub resolution: (u32, u32),
    pub render_resolution: (u32, u32),
    pub extra: &'a [String],
}

fn row(c: &mut HudCanvas, y: f32, scale: f32, label: &str, s: &RollingStats, budget: Option<f32>) {
    let line = GLYPH * scale;
    let over = budget.is_some_and(|b| s.p99() > b);
    let color = if over { RED } else { WHITE };
    c.text(
        8.0,
        y,
        scale,
        if budget.is_some() { color } else { DIM },
        label,
    );
    let x = 8.0 + 30.0 * line;
    c.text(x, y, scale, color, &format!("{:6.2}", s.last()));
    c.text(x + 7.0 * line, y, scale, color, &format!("{:6.2}", s.p99()));
    if let Some(b) = budget {
        c.text(x + 14.0 * line, y, scale, DIM, &format!("{b:5.1}"));
    }
}

const GLYPH: f32 = crate::hud::GLYPH_PX;

/// Draws the overlay in the top-left corner. Returns the bottom y.
pub fn draw_profiler(c: &mut HudCanvas, input: &OverlayInput<'_>, scale: f32) -> f32 {
    let line = GLYPH * scale + 2.0;
    let rows =
        6 + BUDGETS.len() + input.gpu.rows().len() + input.cpu.len().min(24) + input.extra.len();
    let width = 52.0 * GLYPH * scale;
    c.rect(
        0.0,
        0.0,
        width,
        rows as f32 * line + 12.0,
        rgba(0, 0, 0, 170),
    );

    let mut y = 6.0;
    let fps = 1000.0 / input.frame_ms.mean().max(0.001);
    let frame_color = if input.frame_ms.p99() > FRAME_BUDGET_MS {
        RED
    } else {
        GREEN
    };
    c.text(
        8.0,
        y,
        scale,
        frame_color,
        &format!(
            "{fps:5.1} fps  frame {:5.2} ms  p99 {:5.2} ms",
            input.frame_ms.last(),
            input.frame_ms.p99()
        ),
    );
    y += line;
    c.text(
        8.0,
        y,
        scale,
        DIM,
        &format!(
            "{}  {}x{} -> {}x{}",
            input.adapter,
            input.render_resolution.0,
            input.render_resolution.1,
            input.resolution.0,
            input.resolution.1
        ),
    );
    y += line * 1.5;

    let head_x = 8.0 + 30.0 * GLYPH * scale;
    c.text(8.0, y, scale, YELLOW, "GPU budget");
    c.text(head_x, y, scale, YELLOW, "  last    p99  budget");
    y += line;
    let empty = RollingStats::new(1);
    for b in BUDGETS {
        let stats = input
            .gpu
            .groups()
            .iter()
            .find(|g| g.name == b.group)
            .map_or(&empty, |g| &g.stats);
        row(c, y, scale, b.label, stats, Some(b.ms));
        y += line;
    }
    if !input.gpu.enabled() {
        c.text(
            8.0,
            y,
            scale,
            RED,
            "timestamp queries unavailable on this adapter",
        );
        y += line;
    }
    row(
        c,
        y,
        scale,
        "GPU total",
        input.gpu.frame_total(),
        Some(FRAME_BUDGET_MS - 1.6),
    );
    y += line * 1.5;

    c.text(8.0, y, scale, YELLOW, "GPU passes");
    y += line;
    for r in input.gpu.rows() {
        row(c, y, scale, &format!("  {}", r.name), &r.stats, None);
        y += line;
    }
    y += line * 0.5;
    c.text(8.0, y, scale, YELLOW, "CPU scopes");
    y += line;
    for r in input.cpu.iter().take(24) {
        row(c, y, scale, &format!("  {}", r.name), &r.stats, None);
        y += line;
    }
    for e in input.extra {
        c.text(8.0, y, scale, DIM, e);
        y += line;
    }
    y
}
