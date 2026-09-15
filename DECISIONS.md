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

`naga` (dev-dependency only): already compiled as part of wgpu; the test suite uses it directly to validate every entry-point shader without a GPU, so a broken WGSL edit fails CI on every platform.

## D6. GPU traversal state

- **Recursive-style re-initialisation** (the CPU marcher's approach, re-deriving each level's DDA from the entry point on every pop). Simple, but repeats floor and clamp work on every ascent.
- **Mantissa bit tricks in [1,2)** (the 64-tree guide). Fastest on its author's hardware, but `f32` leaves 23 mantissa bits for a tree spanning 13 bits of sector plus 3 of brick, and the robustness argument relies on exact float layouts across three shader compilers.
- **Integer cells with per-level stacks (chosen).** Each level is an Amanatides-Woo DDA; descending pushes (node, min, cell, tMax), leaving pops them so the parent resumes exactly. Cell bounds come from integer subtraction relative to the camera voxel, so precision is independent of world position.

Measured on the integrated GPU at 960x540 over the sample terrain: the first version kept the stack as an array of `Dda` structs written through element chains and took 132.4 ms; moving the active level into plain locals with per-field stacks took 46.5 ms for identical iterations. The struct-chain version also failed FXC compilation on DX12. Removing wgpu's injected loop bounding made the Intel Vulkan driver 3.4x slower (446 ms), so runtime checks stay on.

## D7. Primary ray start distance

- **March every ray from the camera.** 42.8 mean iterations per ray after coalescing, 34.8 ms.
- **Reproject last frame's depth as a start distance.** Cheap, but disocclusion makes it non-conservative exactly when the camera moves fast.
- **Beam prepass (chosen, after Laine and Karras).** Coarse rays through every 4x4 block corner stop at the first occupied brick cell or at a node twice the beam width; each primary ray starts at its block's minimum corner distance less one brick and the beam width. 17.6 mean iterations, vis.beam 2.2 ms + vis.march 16.3 ms = 18.5 ms. Verified against the CPU marcher (0 of 1620 pixels differ) and the golden image.

## D8. Brick residency

- **Upload everything in view distance.** Predictable, but pays for surfaces behind hills and inside caves that no ray will reach.
- **CPU visibility estimation** (frustum and occlusion culling of bricks). Duplicates the marcher's knowledge approximately, and still over-uploads occluded bricks.
- **Marcher feedback (chosen).** A non-resident brick is shaded from its dominant material and writes its leaf word offset into a 4096-slot hash table; the table is read back asynchronously and the CPU uploads the nearest requests within a byte budget. A 12 m proximity radius covers what the player can touch before a ray sees it. Measured: after 40 frames at 96x54 over the sample terrain, 2297 of 39982 bricks were resident.

## D9. Where erosion runs

- **GPU compute** (the Mei et al. paper's own setting). Fastest, and we already have a device. But float results differ across vendors and drivers, and the eroded terrain is the world: two players with the same seed on different GPUs would get different mountains, and a cached world regenerated on another machine would not match its saved edits.
- **Tile-local erosion at stream time.** Bounded cost, but erosion does not decompose: drainage at a tile edge depends on the whole upstream basin, and neighbouring tiles disagree at the seam.
- **Whole-world CPU erosion at creation, cached (chosen).** Jacobi passes over rows with rayon are deterministic for any thread count. 1024^2 cells at 16 m, 600 iterations: 46.3 s first version, 32.8 s after buffer reuse (bit-identical output). The result is cached with a checksum and parameter fingerprint; later launches load it in 0.07 s.

## D10. Detail with distance

- **Everything at full detail within view distance.** A full-detail surface chunk is 1.7 MB of CPU memory and 164 KB of tree words; the 1536 m radius holds about 21 000 surface chunks.
- **A separate far-terrain heightfield renderer.** Cheap, but a second representation that the marcher, shadows and GI would all need to understand, and it cannot show overhangs or excavations.
- **The same tree at four levels of detail (chosen).** Full voxels within 64 m, 0.5 m cells to 192 m, uniform 2 m nodes to 512 m, uniform 8 m nodes to 1536 m, with hysteresis. Chunks wholly below the lowest possible surface of their footprint are not generated at all (no ray reaches them) except a thin layer under the player. Per surface chunk, single threaded: Full 89 ms / 1.7 MB, Cell 32 ms / 323 KB, Node2 1.5 ms / 55 KB, Node8 0.05 ms / 1.5 KB.

## D11. Keeping streaming off the frame

- **Generate and upload on the main thread.** Simplest; a camera turn cost seconds.
- **Generate on workers, flatten and upload on the main thread.** Generation left the frame but flattening a 50 000-cell chunk took up to 74 ms: voxel_gpu.update averaged 12.3 ms while streaming.
- **Generate and flatten on workers; relocate and write within a budget (chosen).** Workers attach a flattening to each new tree (invalidated by any edit). The main thread inserts finished chunks within 3 ms and uploads structure nearest-first within 4 ms. voxel_gpu.update while streaming: 2.7 ms mean, 6.0 ms p99.
