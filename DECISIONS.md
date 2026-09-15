# Decisions

Architectural forks, the options considered, what was measured, and the call.
Newest at the bottom. Dependencies each get one line stating why they beat
writing the thing ourselves.

## Dependencies

| Crate | Why it beats writing it |
|---|---|
| `wgpu` | Portable Vulkan/DX12/Metal with WGSL and timestamp queries; writing three backends is not the product. |
| `winit` | Window, input and event loop across three OS APIs; zero engine value in redoing it. |
| `glam` | SIMD vector and matrix math with `bytemuck` support; hand-rolled SIMD math is weeks of work for parity. |
| `bytemuck` | Sound zero-copy casts of `#[repr(C)]` uniforms to bytes without `unsafe` at every call site. |
| `pollster` | Four-line blocking executor for wgpu's async adapter/device requests. |
| `puffin` | Flame graph capture compatible with an existing viewer; our HUD profiler covers the rest. |
| `png` | Deflate compression for screenshots and golden references; stored-block PNGs would be 11 MB per 1440p shot. |
| `font8x8` | Public domain 8x8 glyph bitmaps. Data, not code: the HUD renderer that uses it is ours. |
| `log` | The logging facade wgpu already emits through; our logger is 20 lines. |

## D1. Who owns the frame graph

- **Bevy renderer.** Mesh and material centric extract/prepare/queue phases. Every pass here is a custom compute dispatch over a custom acceleration structure, so nearly all of it would be bypassed.
- **Ad hoc pass list.** Fastest to start, but texture lifetimes, resize handling and culling of disabled features end up duplicated in every pass.
- **Own frame graph (chosen).** Passes declare reads and writes; the graph validates order, culls passes whose outputs are unused, allocates and resizes textures, and wraps every pass in a CPU scope and a GPU timestamp scope. Measured overhead at step 1: `renderer.frame` CPU scope 0.31 ms mean for three passes including submission bookkeeping.

## D2. HUD text

- **egui.** Full widget toolkit, but its wgpu integration pins wgpu versions and it drags a large dependency tree into a renderer we otherwise own.
- **fontdue plus an embedded TTF.** Proper glyph shaping, but needs a redistributable font file and a glyph cache.
- **8x8 bitmap atlas, instanced quads (chosen).** One 128x64 R8 texture, one instanced draw. `post.hud` measured 0.06 ms at 1280x720 with the full profiler table on screen.

## D3. Golden image determinism

- **Hardware adapter.** Results differ across vendors and driver versions; references would only be valid on one machine.
- **lavapipe on Linux CI.** Deterministic, but differs from any adapter a Windows developer can run locally.
- **WARP on Windows CI and locally (chosen).** `MC2_FORCE_FALLBACK=1` selects the Microsoft Basic Render Driver on both sides. Tolerance is mean absolute channel error 1.5 and at most 0.5% of pixels off by more than 12. Validated by perturbing one AgX coefficient by 0.02: mean error 2.48, test failed.

## D4. Display transform

- **Reinhard.** Cheap, but desaturates and hue-shifts bright saturated emitters toward white along the wrong path, and has no toe.
- **ACES fitted (Narkowicz).** Popular, but its RRT skews saturated blues toward purple and oranges toward yellow at high exposure: exactly the lava and torch colours this world is full of.
- **AgX (chosen).** Log-encoded inset/outset with a polynomial contrast fit. Saturated emitters path to white without hue skew; the calibration bars in `golden/calibration.png` show it across 14 stops.
