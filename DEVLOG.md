# Devlog

One entry per system: what was built, what it costs, what was rejected.

## Test hardware

Development machine: Intel Core i5-13450HX, 16 GB RAM, Intel UHD Graphics
(Raptor Lake-S mobile, 16 execution units) through Vulkan. The laptop's
discrete GPU was switched off for the duration of the build, so every GPU
number in this log comes from the integrated GPU. The frame budget in the
README targets a mid range 2024 discrete GPU, which is roughly an order of
magnitude faster in both compute throughput and memory bandwidth; figures
here are reported raw and scaled comparisons are called out explicitly.

## Step 1: Harness (`v0.1-harness`)

**Built.** A cargo workspace (`mc2-core`, `mc2-gpu`, `mc2-render`, and the
`minecraft-2` binary).

- `mc2-core::stats::RollingStats`: 256 sample ring with mean, max and nearest
  rank percentile. Every HUD row is one of these.
- `mc2-core::profiler`: cross-thread CPU scopes summed per frame. A scope that
  does not run in a frame records zero, so amortised systems report a true
  per-frame p99. Scopes are forwarded to puffin.
- `mc2-gpu::timing::GpuProfiler`: one shared timestamp query set, resolved
  into a ring of four staging buffers mapped asynchronously. When all four
  are in flight the frame is left untimed instead of stalling the queue.
  Pass names are `group.pass`; group sums feed the budget table.
- `mc2-gpu::graph::FrameGraph`: passes declare reads and writes. Compile
  validates that transient textures are written before being read, culls
  passes whose outputs are not consumed (history textures keep their writers
  alive), and reallocates on resize with a generation counter so passes
  rebuild bind groups lazily.
- `mc2-gpu::shader::ShaderLibrary`: `#import` expansion with cycle detection.
  Release embeds `shaders/*.wgsl` through a generated `include_str!` table;
  debug reads the files and polls modification times four times a second.
  `HotCompute`/`HotRender` rebuild pipelines inside validation error scopes
  and keep the old pipeline if an edit fails to compile.
- Profiler HUD from an 8x8 bitmap font atlas in one instanced draw.
- AgX display transform and a calibration compute pass (exposure ramp over
  16 stops, saturated bars over 14 stops, moving bar for pacing).
- Headless `--capture` and `--bench` modes, golden image rig on WARP, CI.

**Measured** (`--bench --frames 300 --size 2560x1440`, integrated GPU):

| Pass | mean ms | p99 ms |
|---|---|---|
| `debug.calibration` (compute, 3.7 M px) | 1.39 | 1.93 |
| `post.present` (AgX, fullscreen triangle) | 2.58 | 3.43 |
| `post.hud` | 0.00 | 0.00 |
| CPU `renderer.frame` | 0.31 | 0.85 |

Present at 2.6 ms is over the 1.0 ms post budget on this GPU; the same
shader is a handful of arithmetic ops per pixel and lands far inside budget on
the discrete target. It is the first number to revisit in the optimisation pass.

**Rejected.**
- `notify` for hot reload: polling a directory's mtimes every 250 ms costs
  microseconds and avoids a platform watcher dependency.
- Blocking timestamp readback: simple, but a `poll(Wait)` per frame hides
  exactly the GPU stalls the profiler exists to show.
- Hardware golden references: see D3.
