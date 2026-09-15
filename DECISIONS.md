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

## D5. World acceleration structure

- **Flat brickmap.** A dense grid of brick pointers over the loaded area. VoxelRT measures XBrickMap at 165.6 Mrays/s against 182.6 for Tree64 on primary rays, with the simplest edits. But a dense pointer grid over a 16 km world at 0.5 m is 32000^2 x 1024 cells; even sparse sector grids of it cost a pointer per empty cell, and it offers no LOD.
- **Single giant 64-tree.** Best empty space skipping and natural LOD, but the dubiousconst282 guide itself recommends "many smaller trees at a top-level grid rather than a single giant tree" because streaming and memory management of one tree across a 16 km world is painful: every edit rewrites a path from the root.
- **Brickmap leaves under per-chunk 64-trees, under a sector grid (chosen).** Bricks (8^3 voxels, 4-bit palettes, a 64-bit occupancy mask per 4^3 subblock) remain the authoritative storage and upload unit. Each 32 m chunk is a three-level 64-tree whose leaves are brick cells: empty, uniform (one material reference, no brick allocated) or a palette brick. Chunks sit under 128 m and 512 m tree levels per sector. A chunk edit rewrites that chunk's small node block; brick edits upload single bricks. Upper levels carry LOD.

Measured with the CPU reference marcher on the sample terrain (64 m x 37.5 m of carved terrain, 17 MB): 569 ns/ray and 20.8 DDA iterations per hit, against 6349 ns/ray and 105.9 iterations for a voxel walk. 11x.

Also from the paper, adopted: robust stepping clamps the entry point into the next cell's bounds rather than biasing t (guide, "getting stuck in place"), and child index `x + z*4 + y*16` so the 2x2x2 coalescing mask `0x00330033` from the guide applies unchanged on the GPU.
