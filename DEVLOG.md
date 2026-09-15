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

## Step 2: Brickmap, 64-tree, palette compression, CPU reference marcher (`v0.2-brickmap`)

**Read first.** Amanatides and Woo's DDA; Laine and Karras's ESVO for the
contour and beam ideas deferred to LOD; Museth's HDDA talk (level-adaptive
leapfrogging, re-initialised on descent); dubiousconst282's 64-tree guide and
the VoxelRT benchmark table, which shaped D5.

**Built.** New crate `mc2-voxel`.

- `coords`: every unit is a power of two above the 6.25 cm voxel: 25 cm
  subblock, 50 cm brick, 1 m block, 2 m and 8 m tree nodes, 32 m chunk, 128 m
  and 512 m upper nodes. The world is 32 x 1 x 32 sectors, 16 384 m square
  and 512 m tall, addressed in `i32` voxels.
- `material`: 51 materials with render (albedo, roughness, metallic,
  emission, IOR), structural (density, compressive and tensile strength,
  hardness), thermal (conductivity, specific heat, ignition and melt points,
  flammability) and acoustic (absorption) properties. Later systems read
  these instead of inventing their own tables.
- `brick::Brick`: palette slot 0 is air. Narrow bricks store 4-bit indices
  inline (256 bytes); a seventeenth material promotes the brick to 8-bit
  indices in a boxed wide variant, after first recycling slots whose voxel
  count dropped to zero. Occupancy is eight `u64`, one per subblock.
  `VoxelState` packs damage, wetness, temperature band and burn progress into
  one byte, and the 512 byte state array exists only while some voxel is
  non-default.
- `tree::ChunkTree`: `Sparse64<T>` compact child arrays indexed by popcount,
  three levels deep over brick cells. Setting a voxel in a uniform cell
  materialises a brick; a brick that returns to one material collapses back
  to a uniform cell, and emptied nodes are pruned upward. Dirty tracking
  separates structure changes from brick content changes for upload.
- `world::VoxelWorld`: FxHash map of chunk trees, voxel get/set across chunk
  boundaries, `fill_box` that writes whole uniform cells inside the box and
  edits only boundary bricks, `fill_sphere` returning changed voxels.
- `march`: hierarchical DDA. Chunk grid, then 128/32/8 voxel tree levels,
  then brick subblocks (skipped when their mask is zero), then voxels. Each
  child DDA starts from its parent's entry time and locates its first cell by
  clamping into bounds. Accepts a material predicate so audio and physics can
  ignore foliage or glass.

**Measured** (`cargo run --release -p mc2-voxel --example march_bench`,
200 000 random rays, max 40 m, single thread):

| Marcher | ns/ray | iterations/hit |
|---|---|---|
| hierarchical | 569 | 20.8 |
| voxel walk | 6349 | 105.9 |

Correctness: 3000 rays, a share of them axis-aligned or confined to a plane,
agree with the voxel walk on voxel, distance (1e-9 m) and material.

Memory: the sample terrain costs 17 MB for 2400 m^2 of fully detailed
surface bricks, about 400 bytes per brick including palette counts.

**Rejected.**
- Mantissa bit tricks on the CPU: `f64` headroom makes plain floor-and-clamp
  exact at world scale and keeps the reference marcher obviously correct.
  They return on the GPU where `f32` makes them worth it.
- Quantising bricks above 16 materials to the nearest material: lossy, and
  the wide fallback costs nothing for bricks that never need it.

## Step 3: GPU marcher, visibility buffer, brick streaming (`v0.3-marcher`)

**Built.**

- `mc2-voxel::gpu_layout`: the GPU word format (D5) and chunk flattening.
  LOD words take the dominant material by voxel coverage, never averaged
  colour, and a normal from the occupancy gradient with a spread term.
- `mc2-gpu::alloc::RangeAllocator`: segregated free lists over fixed
  buffers, with exact classes for 89-word narrow bricks, 273-word wide
  bricks and 128-word state blocks.
- `mc2-render::voxel_gpu::GpuWorld`: tree and brick pools, chunk structure
  upload on change, sector blocks holding copies of chunk roots under 128 m
  nodes, feedback readback, nearest-first brick uploads within a byte budget.
- `shaders/march.wgsl`: six tree levels from the 32 x 32 sector grid down to
  brick cells, then subblocks and voxels. Empty 2x2x2 child groups are left
  in one step. Non-resident data returns LOD hits and requests upload.
- `vis.beam` and `vis.march` write the visibility buffer: world voxel id,
  material, face and hit kind (`rgba32uint`), hit distance (`r32float`),
  motion vectors and octahedral normal (`rgba16float`). Nothing is shaded
  during traversal; `debug.shade` turns the buffer into a picture.
- A free-flying camera, F4 debug views (shaded, LOD hits, iteration
  heatmap), and a naga validation test so a broken WGSL edit fails CI
  without a GPU.

**Measured** (sample terrain, 960x540, integrated GPU, Vulkan):

| Change | vis cost | iterations/ray |
|---|---|---|
| first working marcher | 132.4 ms | 51.2 |
| state in locals, per-field stacks | 46.5 ms | 51.2 |
| 2x2x2 empty group coalescing | 35.6 ms | 42.8 |
| cached node mask | 34.6 ms | 42.8 |
| beam prepass | 18.5 ms (2.2 + 16.3) | 17.6 |

DX12 runs the same shaders (50.2 ms against 46.5 ms on Vulkan, measured
before coalescing) after replacing dynamic-index vector writes that FXC
cannot address.

Correctness: the GPU visibility buffer matches the CPU reference marcher on
every one of 1620 sampled pixels (voxel id, material, depth within 1 cm), and
a golden image of the terrain renders on WARP in CI.

Streaming: with proximity uploads off, 40 frames at 96x54 made 2297 of 39982
bricks resident. The rest never cost GPU memory.

At 1440p with 65% internal resolution the march covers about 3x this pixel
count, roughly 55 ms here; scaled to the discrete target that is still above
the 2.5 ms budget line. Remaining ideas, deferred to the optimisation pass:
octant mirroring, skipping the sector and 128 m levels when the beam distance
lands inside a known chunk, and half-resolution beams for secondary rays.

**Rejected.** Disabling wgpu's loop bounding (3.4x slower on this driver);
disabling bounds checks (no change, not worth `unsafe`); reprojected depth
as a start distance (D7).
