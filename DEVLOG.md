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

## Step 4: Worldgen v1, region streaming, LOD (`v0.4-worldgen`)

**Read first.** Mei, Decaudin and Hu's virtual-pipe erosion (flux update,
outflow scaling, velocity from flux, capacity from tilt and speed,
semi-Lagrangian sediment transport, evaporation).

**Built.** New crate `mc2-worldgen`, plus supporting work in the voxel crate.

- Uniform 2 m and 8 m nodes in chunk trees and in the GPU layout: a solid
  chunk is 64 material references instead of 262 144 cells. One-pass
  construction from a dense cell array.
- `noise`: seeded improved Perlin, fBm, ridged multifractal, Worley.
- `strata`: a gneiss basement with granite plutons, a repeating sedimentary
  cycle (conglomerate, sandstone, shale, limestone, chalk) with per-cycle
  thickness jitter, regional dip and folds, and volcanic provinces where
  basalt flows and rhyolite tuff replace limestone and chalk. The layer
  index is monotonic in elevation so generation can prove a node is one rock.
- `terrain` and `erosion`: a continent falling to sea at the world edge,
  ridged mountain belts, broad hills, then Mei erosion over the whole 16 km
  at 16 m with dissolving scaled by the hardness of whatever stratum is
  currently exposed, the sea as a fixed reservoir, and thermal slumping.
  Cached per seed (checksum, parameter fingerprint, atomic rename).
- `amplify`: surface detail whose amplitude follows the coarse field (crags
  on steep eroded slopes, swells on plains, almost none where discharge is
  high) and surface materials from slope, altitude, sediment and water.
- `chunkgen`: top-down generation. Uniform nodes wherever rock is one
  stratum; voxel bricks only where the surface or an exposed boundary
  crosses a cell; buried boundaries kept at cell resolution with
  `detail_brick` to restore voxels when something digs there. Ore bodies
  follow their host rock. Four levels of detail.
- `stream`: distance bands with hysteresis, buried-chunk skipping, pinning of
  edited chunks, rayon generation, budgeted inserts. GPU residency uploads
  structure nearest-first within a budget using worker-prepared flattenings.
- The app loads terrain in the background, spawns on open land facing the
  highest nearby ground, and streams around the camera.

**Measured** (integrated GPU, i5-13450HX):

| What | Cost |
|---|---|
| World terrain, 1024^2 x 600 erosion iterations | 32.8 s once, 0.07 s from cache |
| Full-detail surface chunk | 89 ms, 1.7 MB CPU, 164 KB tree words |
| Cell / Node2 / Node8 chunk | 32 / 1.5 / 0.05 ms |
| Streaming the spawn area (21 389 chunks) | 24 s on 16 threads |
| voxel_gpu.update while streaming | 2.7 ms mean, 6.0 ms p99 |
| vis.beam + vis.march, 1280x720, generated world | 5.1 + 29.8 ms |

Chunk generation went 190 -> 187 ms with dense construction, then 187 -> 89
ms by caching per-cycle layer bounds (every rock voxel had been hashing
eight thickness jitters).

Bugs found by measurement: the headless warm-up closed no profiler frames,
so hundreds of frames' scopes summed into one "9 second frame"; a proximity
scan that stopped at its limit was never repeated; and building trees from
dense arrays briefly dropped 97% of bricks because children are visited in
tree order, not index order (caught by the chunk benchmark's brick count).

**Rejected.** Noise-stack terrain (the brief), GPU erosion (D9), a separate
far-terrain renderer (D10), and noise-worm caves: caves wait for karst
dissolution in worldgen v2.

## Step 5: Player, collision, carve and place (`v0.5-player`)

**Built.** New crate `mc2-game` on standalone `bevy_ecs`: the voxel world,
streaming, input, the fixed clock, the block layer and the interaction state
are resources; the player is an entity with `Player` and `Body` components.
Three schedules run per frame: look and toggles, then the fixed 120 Hz
movement ticks that are due, then the view camera and interaction.

- `collide`: boxes swept one axis at a time in steps shorter than a voxel,
  snapping flush to the face they meet. Foliage and liquids do not block.
- `player`: walk, sprint, crouch (standing only when there is headroom),
  jump, fly. Ledges up to 0.55 m are climbed automatically with the camera
  eased over the step; on 6.25 cm terrain that is most of walking.
  Movement pauses over chunks that have not streamed in.
- `blocks`: the macro layer. A generated block is its dominant material,
  derived from voxels on demand, so natural terrain costs nothing; only
  placed blocks with identity (torch, lantern, window, slab) are recorded.
  Every block is a 16^3 voxel model.
- `interact`: block mode breaks the targeted 1 m cell (its voxels go to the
  inventory by material volume, so half a carved block is half a block of
  stone) and places models against the targeted face; carve mode cuts or
  deposits spheres of 1 to 16 voxels. Before an edit touches a brick cell
  stored coarsely (buried rock), the generator restores its voxel detail, so
  a tunnel wall shows real layer boundaries.
- `view`: first person, or third person pulled in front of walls.
- The renderer gained a gizmo pass for placement and carve previews that
  dims where the marched depth hides them, and the app a crosshair and
  hotbar HUD. `--demo build` drives the real systems with scripted input for
  headless captures.

**Measured.**

| What | Cost |
|---|---|
| Flattening a full-detail chunk after an edit | 2.26 ms (was 5.62 ms before stack histograms) |
| Scripted build demo, game.update | 1.00 ms mean, 16.0 ms worst frame |

The first demo run found a 213.8 ms frame: a 15-voxel carve touched about 64
coarse cells and resampled the chunk's surface and strata once per cell.
Batching restoration per chunk brought the worst frame to 16.0 ms.

Controller tests run against a voxel arena: lands flush at 10.000 m, climbs a
0.5 m ledge, stops flush at a 1.5 m wall, jumps 1.16 m, stays crouched under
a 1.5 m ceiling. An ECS integration test breaks a block through the full
schedule and finds 4096 voxels of sandstone in the inventory.

**Rejected.**
- A capsule collider: smoother on slopes, but capsule against 6.25 cm voxels
  needs per-voxel sphere tests where the box needs a range scan, and step-up
  already smooths what the capsule would.
- Explicit block ids for generated terrain: 32 768 ids per chunk that only
  restate the voxels.
- Physics-thread player movement: the player joins the 120 Hz physics thread
  with rigid bodies in step 9; until then the same fixed clock runs on the
  main thread.

## Step 6: Sun, sky LUTs, traced shadows, ReSTIR direct light (`v0.6-lighting`)

**Read first.** Hillaire 2020, "A Scalable and Production Ready Sky and
Atmosphere Rendering Technique" (LUT set, sky-view parameterisation, the
multiple scattering approximation); Bruneton and Neyret 2008 for the
transmittance parameterisation; Bitterli et al. 2020, "Spatiotemporal
reservoir resampling for real-time ray tracing with dynamic direct
lighting" (Algorithms 3-5, the M = 32 initial candidates, the 20x temporal M
clamp, k = 5 spatial neighbours within 30 px, the 10% depth and 25 degree
normal rejection, visibility before reuse); Lagarde and de Rousiers 2014 for
photometric units and EV100; Krawczyk et al. 2005 for a luminance-dependent
exposure key.

**Built.**

- Atmosphere (`sky.rs`, `atmosphere.wgsl`): Earth-like Rayleigh, Mie and
  ozone media in kilometres. Transmittance (256x64) and multiple scattering
  (32x32) LUTs are built once and rebuilt after a shader reload; the sky-view
  LUT (200x100, horizon-dense latitudes, longitude from the light) and a
  32^3 aerial perspective volume reaching 8 km along quadratic slices are
  rendered every frame. Everything is per unit illuminance, so the sun
  (127.5 klx above the atmosphere) and the moon share passes: the moon's
  copies render only while moonlight matters.
- Traced visibility (`visibility_trace.wgsl`, `visibility_resolve.wgsl`):
  per pixel, a shadow ray toward a random point on the sun's or moon's disc
  (the penumbra widened 4x) and a cosine-weighted sky visibility ray to 96 m,
  through the same marcher as primary rays with 8x coarser LOD. Rays are
  traced for one pixel per stride x stride block (a disocclusion first,
  otherwise the scheduled pixel) and resolved into per-pixel history, which
  is reused only where the reprojected pixel saw the exact same world voxel:
  voxel ids make that test exact instead of a depth heuristic.
- ReSTIR (`lights.rs`, `restir_*.wgsl`): every emissive brick cluster or
  uniform node is a spherical light in a stable slot; a Vose alias table
  over luminance x area / max(d^2, 16 m^2) within 192 m is the source
  distribution. Candidates, visibility of the survivor, temporal reuse at the
  reprojected pixel (exact voxel match, history M clamped to 20x),
  two spatial rounds, then shading with a final visibility ray. The pass is
  skipped when no light is in range.
- Composition (`shade_compose.wgsl`): sun and moon through transmittance and
  traced visibility with GGX specular, sky irradiance from both sky LUTs
  scaled by sky visibility, ReSTIR emitters, emission, then aerial
  perspective. Sky pixels add a limb-darkened sun, a Lambert-lit moon disc,
  about 8000 stars on a cube map rotated by sidereal time, and airglow.
- Exposure (`exposure.wgsl`): centre-weighted log average to EV100, adapted
  exponentially, with a key that falls with luminance so moonlight reads as
  night.
- World clock (`mc2-game::clock`): solar declination and hour angle, a moon
  that trails the sun by its synodic phase, illuminance by phase. `--time`,
  `T`, `[` and `]`.
- Quality presets (`quality.rs`): Realistic, Hyper Realistic, Ultra
  Realistic and Super Ultra Crazy Duper Realistic set render scale, LOD
  threshold, visibility trace stride and ReSTIR candidates. `--quality`, `F6`.
- `--demo lights` places lanterns and torches through the interaction code;
  `MC2_DUMP=1` writes the lighting intermediates next to a capture.
- Golden tests `lit_terrain_afternoon` and `lit_terrain_night` stream to
  quiescence, restart the frame counter and accumulate 64 frames, so every
  random sequence is the same on every run.

**Measured** (integrated GPU, 1280x720 output, 120 frames, static camera;
GPU timestamps per pass):

| Scene, preset (internal size) | Frame mean / p99 | vis.beam + vis.march | direct.sun | direct.restir mean / p99 | sky.frame | compose + exposure + present |
|---|---|---|---|---|---|---|
| Day terrain, Realistic (640x360, stride 2) | 30.4 / 38.7 ms | 1.4 + 10.9 ms | 5.4 ms | skipped | 0.74 ms | 2.7 ms |
| Day terrain, Hyper Realistic (854x480, stride 2) | 37.5 / 39.9 ms | 2.4 + 17.6 ms | 9.4 ms | skipped | 0.70 ms | 3.6 ms |
| Day terrain, Ultra Realistic (960x540) | 76.0 / 92.9 ms | 3.0 + 20.5 ms | 42.3 ms | skipped | 0.69 ms | 4.3 ms |
| Day terrain, Super Ultra Crazy Duper Realistic (1280x720) | 130.8 / 139.0 ms | 5.2 + 31.5 ms | 75.7 ms | skipped | 0.69 ms | 6.6 ms |
| Night, lights demo, Realistic | 95.3 / 209.3 ms | 1.4 + 11.2 ms | 10.2 ms | 22.2 / 52.4 ms | 1.39 ms | 3.0 ms |
| Night, lights demo, Ultra Realistic | 259.0 / 356.9 ms | 2.9 + 22.3 ms | 83.2 ms | 52.7 / 118.6 ms | 1.39 ms | 5.0 ms |

The LUTs cost 1.04 ms on the frame they rebuild. Two optimisation passes:

- ReSTIR with no emitter in range: 44.6 ms -> skipped.
- Hyper Realistic visibility: 21.3 ms skipping untraced pixels inside the
  full-resolution dispatch -> 9.0 ms dispatching over blocks. The first
  version saved a third instead of three quarters because each 8x8
  workgroup still held a tracing pixel, and a SIMD group pays for its
  slowest lane.

**Budget.** Direct light's line is 2.0 ms at 1440p on the target GPU. Ultra
at 1440p is 4x these pixels on a GPU roughly 10x faster, so day direct light
extrapolates to about 17 ms and night to about 54 ms: an 8-27x violation,
logged as the open bug for this line. The fixes are structural and belong to
the next step: lighting at half the render resolution under the SVGF
reconstruction and temporal upsampler, and one visibility ray shared by
ReSTIR's candidate test and final shading.

Bugs found by looking: the sky-view uv halves were swapped (a flat grey
sky); an LOD normal flipped the hit face so distant shadow rays started
inside nodes; the multiple scattering LUT ignored the planet's shadow and
kept the night sky orange at 127 klx; `arrayLength` on an over-allocated
light buffer sent most candidates to slot 0; Vose's loop popped an entry it
then lost (caught by the alias unit test); FXC rejected dynamic vector
writes and a `vec3<u32>` pad made a uniform 32 bytes.

**Rejected.**
- Preetham or Hosek-Wilkie analytic skies: no aerial perspective, no night,
  no altitude (D13).
- Shadow maps: a second representation of a world that changes per voxel,
  and wrong for 6.25 cm detail at kilometre range (D14).
- One shadow ray per light per pixel: a lava field is thousands of lights
  (D15).
