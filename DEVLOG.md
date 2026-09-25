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

## Step 7: Indirect light, denoising, temporal upsampling (`v0.7-indirect`)

**Read first.** Sannikov, "Radiance Cascades: A Novel Approach to
Calculating Global Illumination" (the penumbra condition, radiance
intervals and their merging, section 4.5's screen-space probes holding
world-space intervals); Ouyang et al. 2021, "ReSTIR GI: Path Resampling for
Real-Time Path Tracing" (Algorithms 2-4, the Eq. 9 and 10 targets, the Eq. 11
reconnection Jacobian, M clamps of 30 and 500, uniform hemisphere
sampling); Schied et al. 2017, "Spatiotemporal Variance-Guided Filtering"
(alpha 0.2, moments variance with a 7x7 spatial fallback under four frames,
five a-trous levels, sigma_z 1, sigma_n 128, sigma_l 4, the first level as
colour history, albedo demodulation); Karis 2014 on temporal antialiasing.

**Built.**

- Temporal upsampling (`upsample.rs`, `taa_upsample.wgsl`): the camera
  jitters on a Halton (2,3) sequence; each display pixel gathers 3x3 render
  samples under a Gaussian of their jittered centres, reprojects a
  display-resolution history through a Catmull-Rom filter, clips it to the
  YCoCg variance box, and blends by how close a sample landed, on
  tonemapped weights. Realistic renders 640x360 and resolves 1280x720.
- History reuse everywhere now asks `same_surface`: exact voxel ids until
  jitter or half resolution moves samples, then ids within the sample
  footprint plus one voxel on the same face.
- SVGF (`svgf.rs`, `svgf_*.wgsl`) for any demodulated signal at render or
  half resolution. Depth is compared as distance from the centre pixel's
  tangent plane in pixel footprints, which is exact for the planes voxels
  are made of. It filters ReSTIR's emitter irradiance (render resolution,
  only when emitters are sampled) and indirect irradiance (half resolution).
- Sky ambient cube (`sky_ambient.wgsl`): above-horizon sky irradiance for
  the six axis normals, 144 cosine directions per face, once per frame.
  Composition reads it instead of sampling the sky-view LUT per pixel.
- Sky map (`skymap.rs`, `skymap.wgsl`): the highest solid 0.5 m cell per
  column within 224 m, a toroidal 1024^2 texture updated per changed chunk
  (4 KB of tops per chunk). Answers "is the sky open above this point" and,
  by a short 2D march, "does the sun reach it".
- Indirect hits (`gi_common.wgsl`): a hit that last frame saw on the same
  surface returns last frame's surface radiance (direct light with traced
  shadows, sky light, emitters and indirect light), so bounces compound
  frame to frame for free; off screen, hits are lit through transmittance,
  sky-map visibility and the ambient cube. Composition now writes that
  surface radiance for every pixel. Indirect rays carry only light from
  surfaces; sky light stays with the traced sky visibility, so nothing is
  counted twice.
- ReSTIR GI (`restir_gi_*.wgsl`) and radiance cascades (`rc_*.wgsl`), both
  at half resolution writing the same noisy irradiance, selectable with
  `--gi restir|cascades`. Cascades are the default in every preset (D19).
- `MC2_GI_COMPARE=<frames>` measures the active method against an unbiased
  reference: ReSTIR GI's initial samples with reuse disabled (debug view 5),
  averaged over that many frames.

**Measured.** Indirect methods on the build demo at noon, Realistic
(640x360 render, 320x180 GI). Errors are relative RMSE on luminance
against the reference, whose own noise is 0.243 at 512 frames (the first
and third rows used 256 reference frames).

| Method | Cost | Relative RMSE | Bias | Frame-to-frame change |
|---|---|---|---|---|
| ReSTIR GI, first version | 9.4 ms | 2.059 | +25.3% | 29.5% |
| ReSTIR GI, spacing-aware history, own-reservoir fallback | 9.4 ms | 1.938 | +32.8% | 27.0% |
| Radiance cascades, doubling intervals (15.75 m reach) | 2.5 ms | 1.646 | -14.3% | 6.1% |
| Radiance cascades, fourfold intervals (341 m reach) | 3.0 ms | 1.541 | +3.5% | 5.3% |

Frame costs at 1280x720 output (GPU timestamps, 120 frames, static camera):

| Scene, preset | Frame mean / p99 | vis | direct.sun | direct.restir | denoise.emitters | indirect | denoise.gi | compose | denoise.upsample |
|---|---|---|---|---|---|---|---|---|---|
| Build demo, Realistic, cascades | 45.8 / 118.4 ms | 12.4 ms | 5.6 ms | skipped | skipped | 3.0 ms | 4.6 ms | 0.9 ms | 4.3 ms |
| Build demo, Realistic, ReSTIR GI | 56.1 / 143.8 ms | 12.5 ms | 5.7 ms | skipped | skipped | 9.4 ms | 4.6 ms | 0.9 ms | 4.4 ms |
| Build demo, Ultra, cascades | 120.1 / 217.3 ms | 23.7 ms | 46.0 ms | skipped | skipped | 6.6 ms | 10.8 ms | 2.1 ms | 4.5 ms |
| Build demo, Ultra, ReSTIR GI | 137.2 / 235.7 ms | 23.7 ms | 46.0 ms | skipped | skipped | 22.5 ms | 10.8 ms | 2.2 ms | 4.5 ms |
| Night lights, Realistic, cascades | 110.5 / 231.9 ms | 12.1 ms | 9.4 ms | 20.4 ms | 6.5 ms | 2.8 ms | 3.9 ms | 0.9 ms | 4.2 ms |
| Night lights, Ultra, cascades | 290.0 / 431.4 ms | 23.7 ms | 76.7 ms | 48.4 ms | 15.1 ms | 6.2 ms | 9.2 ms | 2.1 ms | 4.4 ms |

**Budget.** Scaled to 1440p on the target (4x pixels, roughly 10x faster),
Ultra's indirect line is about 2.6 ms against 2.5 and denoise plus upsample
about 6.1 ms against 2.0. SVGF is the second open budget bug: every tap
rebuilds its world position from a matrix product; a half-resolution
position and normal target would remove most of that. Frame means exceed
the sum of GPU passes by 8 to 90 ms in these scenes; that time is not yet
attributed and is logged for step 18.

Bugs found: parallel golden tests crashed WARP with an access violation
(each passes alone; they now run one at a time); sky light entered twice
once indirect rays that escaped returned sky radiance; a removal script
took composition's constants with it (the GPU march test caught the shader
error). Caught in review before it showed: a 17-chunk sky map window
would have aliased on its 16-chunk torus.

**Rejected.**
- ReSTIR GI as the default: three times the cost with more error and five
  times the flicker in these measurements (D19).
- A world-space irradiance cache: bounces already compound through the
  previous frame's surface radiance, and the sky map covers off-screen hits
  without a second structure to keep in step with edits (D20).
- Separate shadow and sky rays at every indirect hit: they would triple the
  ray count for light the sky map predicts well.

## Step 8: Reflections, volumetrics, clouds, post stack, photo mode (`v0.8-atmosphere`)

**Read first.** Schneider and Vos 2015, "The Real-time Volumetric
Cloudscapes of Horizon Zero Dawn" (Perlin-Worley and Worley noise volumes,
height gradients per cloud type, weather coverage, Beer and powder terms, the
six-sample light cone, cheap-then-full marching, 1-in-16 amortisation);
Hillaire 2015, "Physically Based and Unified Volumetric Rendering in
Frostbite" (froxels, exponential depth, jittered temporal integration,
energy-conserving scattering integration); Wrenninge's multiple scattering
octaves as used by Hillaire 2016; Heitz 2018, "Sampling the GGX
Distribution of Visible Normals" (Listing 1 and the F G2 / G1 estimator);
Jimenez 2014, "Next Generation Post Processing in Call of Duty: Advanced
Warfare" (13-tap downsample, Karis average, tent upsample).

**Built.**

- Clouds (`clouds.rs`, `cloud_noise.wgsl`, `clouds_*.wgsl`): tiling 128^3
  Perlin-Worley and 32^3 Worley volumes generated on the GPU; a 1.5-4 km
  spherical shell with stratus-to-cumulus gradients, coverage varying over
  24 km, detail erosion; lighting from the sun or moon through a six-step
  cone plus a far sample, three multiple-scattering octaves with a dual-lobe
  phase and a powder term, and sky ambient by height; one texel in four of a
  half-resolution buffer traced per frame and the rest reprojected by view
  direction; a 256^2 toroidal cloud shadow map on the base plane that dims
  the sun and moon on the ground, in fog and at indirect hits.
- Froxel fog (`fog.rs`, `fog_*.wgsl`): height fog in 64 exponential slices
  to 192 m at 1/8 resolution, lit by the sun or moon (transmittance, sky map
  visibility, cloud shadow, Henyey-Greenstein) and the sky, jittered and
  reprojected, integrated front to back and applied after aerial
  perspective. As in the paper, one sample per froxel is jittered along the
  view ray, material and light with the same offset, and integrated
  analytically per slice; the history keeps 90% rather than Frostbite's 95%
  so moving cloud shadows leave shorter trails. The frame graph gained a
  render-relative volume size.
- Traced reflections (`reflect_trace.wgsl`, `indirect.rs`): VNDF-sampled
  rays for surfaces up to roughness 0.5, shaded like indirect hits with the
  sky, denoised by SVGF and added in composition. SVGF now skips pixels a
  signal does not cover.
- Post (`post.rs`, `post_*.wgsl`, `present.wgsl`): six-level bloom, camera
  motion blur over half a frame, thin-lens depth of field, then in present
  the Purkinje shift at low adaptation, AgX, contrast-adaptive sharpening,
  cos^4 vignetting and film grain.
- Photo mode (`src/photo.rs`): `P` stops time and detaches a free camera;
  focal length on the wheel (12-400 mm), focus, aperture in full stops and
  exposure in thirds; `Enter` renders 96 frames at Super Ultra Crazy Duper
  Realistic without motion blur and saves `captures/photo_<time>.png`.
  `--dof focus,f` gives headless captures the same lens.
- `--demo mirror` lays steel, obsidian and ice slabs by a marble wall.

**Measured** (integrated GPU, 1280x720 output, 120 frames, static camera,
GPU time per budget group summed over its passes):

| Scene, preset | Frame mean / p99 | vis | direct | indirect | refl | denoise | clouds + fog | post |
|---|---|---|---|---|---|---|---|---|
| Terrain, Realistic | 47.9 / 52.4 ms | 11.0 | 6.3 | 2.7 | 3.9 | 8.1 | 1.2 + 2.0 | 6.4 |
| Terrain, Hyper Realistic | 71.2 / 74.0 ms | 18.6 | 11.2 | 3.8 | 7.2 | 11.4 | 2.4 + 3.4 | 6.3 |
| Terrain, Ultra Realistic | 116.5 / 126.0 ms | 22.0 | 44.5 | 5.9 | 9.0 | 13.3 | 3.6 + 4.2 | 6.2 |
| Terrain, Super Ultra Crazy Duper Realistic | 188.4 / 192.3 ms | 33.8 | 77.8 | 9.1 | 16.3 | 20.7 | 7.3 + 7.5 | 6.1 |
| Mirror demo, Realistic | 61.5 / 67.5 ms | 11.8 | 7.9 | 3.3 | 6.6 | 9.1 | 1.1 + 1.9 | 6.5 |
| Mirror demo, Ultra Realistic | 176.1 / 185.6 ms | 23.5 | 60.2 | 7.3 | 16.5 | 16.0 | 3.3 + 4.0 | 6.5 |

Skipping uncovered pixels in SVGF took reflections on plain terrain from
3.9 ms (trace plus denoise, Realistic) to 0.2 ms; the whole frame went from
47.9 to 45.6 ms.

**Budget.** Scaled to the target (Ultra at 1440p, 4x the pixels on a GPU
about 10x faster): clouds plus fog about 3.1 ms against 2.0, reflections
about 6.6 ms against 1.0 when glossy surfaces fill the view, post about
2.5 ms against 1.0. Three more open budget bugs; bloom's half-resolution first
level and the display-resolution combine are the obvious post costs.

Bugs found: the half-resolution cloud buffer was sampled with render
coordinates (a dark strip on the right edge); the first cloud model was a
uniform overcast because the dilated noise clustered around 0.75; an
extinction of 0.04/m left cloud interiors grey; nearest-texel cloud shadows
showed 50 m squares.

**Rejected.**
- Billboard clouds or sky domes (D21).
- Analytic height fog (D22): no shafts.
- Screen-space reflections (D23): the reflected world is mostly off screen.
- Per-effect post passes with separate histories (D24).

## Step 9: Rigid bodies, contacts, destruction (`v0.9-destruction`)

**Read first.** Müller, Macklin, Chentanez, Jeschke and Kim 2020, "Detailed
Rigid Body Simulation with Extended Position Based Dynamics" (Algorithm 2:
many substeps with one position iteration each; generalised inverse masses
and positional corrections, Eqs. 2-9; static friction as a positional
constraint; dynamic friction and restitution in the velocity pass, with no
bounce below 2|g|h). Amanatides and Woo 1987 again, for marching body grids.

**Built.**

- `mc2-physics`, a new crate:
  - Voxel body shapes: mass, centre of mass and principal inertia from
    per-material density (Jacobi eigen solver), and up to 64 collision
    samples, the outermost surface voxel in each 4^3 cell.
  - The XPBD step: 8 substeps at 120 Hz, gyroscopic torque, exact rotation
    per substep.
  - Contacts: sample spheres against the world (read through a per-body
    voxel window cached for the step) and against the other body's grid
    after a sort-and-sweep broad phase. Penetration is re-evaluated as
    earlier corrections move the bodies, and overlap a substep did not
    cause is resolved at no more than 1 m/s.
  - Kinematic obstacles (the player), with the surface velocity carried
    into the contact.
  - Sleeping judged by displacement over the step; only moving bodies wake
    sleepers they touch. Bodies with no neighbour run all their substeps
    as one parallel task.
  - Explosions carve per brick cell, planned in parallel. Material from the
    blast shell becomes 3-7 voxel fragments thrown up and out; bodies
    already nearby are pushed.
  - Settled bodies are baked back into terrain. Ray casts against bodies.
- Game (`mc2-game::physics`):
  - TNT: E lifts a TNT block out as a body with a 4 s fuse, and it
    detonates wherever it ends up. A blast lights every TNT block within
    reach with a 0.15-0.75 s fuse.
  - Blasts restore generated detail before carving. Edits wake resting
    debris.
  - Debris at rest for 15 s, or the oldest sleepers above 600 bodies,
    becomes terrain again.
  - The player stands on bodies of 40 kg and up and kicks lighter ones.
- Rendering (`bodies.rs`, `bodies.wgsl`):
  - Bodies are dense voxel grids in a 16 MB pool plus a per-frame table
    holding this frame's and last frame's pose.
  - A culling grid of at most 32^3 cells covers their combined bounds.
  - `trace_scene` marches the world, then the bodies in front of the world
    hit, in each body's grid frame over occupancy blocks and voxels.
  - Every ray type goes through it: primary, sun and moon visibility,
    ReSTIR shadow rays, indirect and reflection rays. Sky visibility rays
    skip bodies (below).
  - Visibility ids carry an 11-bit body tag, so history follows a body and
    never matches the world. Body faces use the stored normal, and motion
    vectors come from last frame's pose.
- `--demo blast` (two charges, one lit by hand, the other by the first
  blast), `--hide-bodies`, a physics line on the F3 HUD, F2 screenshots,
  and the `blast_profile` benchmark example.

**Measured.**

*Physics* (dev CPU, `cargo run --release -p mc2-game --example
blast_profile`: three overlapping 3 m blasts in granite under a metre of
dirt, then 10 s of physics):

| | Before | After |
|---|---|---|
| Blasts (406k / 299k / 299k voxels) | 287 / 143 / 184 ms | 8.2 / 3.8 / 3.5 ms (first includes thread pool start) |
| Step, 48 fragments, mean | 2.28 ms | 1.36 ms (windows, islands, fast pair transform) |
| Pair search, 48 fragments | 0.87 ms | 0.11 ms |
| Step, 144 fragments | | mean 2.0 ms, p99 4.3 ms, max 8.5 ms; 17 awake after 10 s |

*Correctness.* The GPU body tracer agrees with the CPU body ray cast on all
325 body pixels of nine rotated bodies (depth, material, tag, grid voxel
and face). A lit debris golden joins the suite.

*GPU cost* of the blast demo's 74 fragments, integrated GPU, 1280x720, 120
frames, bodies hidden vs drawn:

| Preset | Frame (hidden / drawn) | vis | direct |
|---|---|---|---|
| Realistic | 51.8 / 53.2 ms | 12.8 / 13.5 | 7.5 / 7.4 |
| Ultra Realistic | 130.9 / 137.2 ms | 25.2 / 27.0 | 51.4 / 51.7 |

Before sky rays skipped bodies, Ultra spent 56.0 ms in direct light
(+4.6 ms) and the whole frame was 146.7 ms.

**Budget.**

- Physics: the tick p99 with 144 active fragments is 4.3 ms against 4.0
  ms, a budget bug.
- Blasts run on the main thread and cost a 4-8 ms hitch each.
- Debris adds about 0.7 ms to primary visibility at the target (1.8 ms x
  0.4), inside the march line's share only when the rest of it shrinks
  (step 18).

**Bugs found.**

- Stacked cubes exploded when they first touched: pair penetration was
  not re-evaluated, so every sample applied full depth.
- The upper cube then slid off: the nearest-face normal of a sample
  already inside the other grid pointed sideways. Pairs now use the same
  sphere probe as the world.
- Free spin decayed under the paper's linearised quaternion update.
- Debris spawned by a blast was pushed by the same blast, often downward.
- Rest time stopped counting at sleep, so time-based baking never ran.
- Crater piles spun at 40 rad/s: overlap between neighbours was resolved
  in one substep, and awake neighbours re-woke each other forever.
- A headless edit landed in the streaming warm-up loop instead of the
  capture loop, so bodies simulated but were never drawn.
- The blast demo placed its second charge into the first one's cell.

**Rejected.** Sequential impulses (D25), convex hulls or SDFs for collision
(D26), rasterised or re-voxelised bodies (D27), particle-only debris (D28).

## Step 10: Structural integrity and collapse (`v0.10-structure`)

**Read first.** Macklin, Müller and Chentanez 2016, "XPBD: Position-Based
Simulation of Compliant Constrained Dynamics" (compliance as inverse
stiffness folded into the time step, the total Lagrange multiplier, and the
constraint force estimates it gives, which is how one would build breakable
joints); Erin Catto, "Solver2D" (2024), on soft constraints as a
spring-damper with a frequency and damping ratio, sub-stepping, the relax
pass, and the warning that deriving velocity from position differences
loses precision far from the origin — this codebase keeps body positions
in f64 world metres, so the error stays in the last bits; Dennis
Gustafsson's Teardown material on voxel volumes per object and deterministic
destruction commands. His structural integrity prototype is shown but not
described anywhere I could fetch, so the load graph here follows the brief.

**Built.**

- Physics on its own thread (`mc2-game::physics_host`), as the brief asks.
  The game thread owns the voxel world; the physics world sees only the
  collision cells it is sent. Each step reports the brick cells its bodies
  needed and lacked, and those bodies wait a tick; the next job carries
  them, plus fresh copies of mirrored cells that were edited. The game
  thread never waits on physics: it sends a job when the thread is idle and
  reads the newest snapshot of the bodies when one arrives. Spawns and
  removals are visible to gameplay at once. Inline mode runs the same jobs
  synchronously for tests and scripted captures.
- `mc2-physics::collision`: solid occupancy per brick cell, from the world
  (reusing brick occupancy masks when every material in a brick is solid)
  or from the mirror the physics thread keeps.
- `mc2-structure`, the load graph:
  - Each 1 m block is a node: mass, dominant material, and the solid voxels
    it offers each neighbour.
  - Dijkstra from the anchors gives every node a path to support; resting
    on the node below costs 1, leaning sideways 1.5, hanging 4. Nodes the
    search never reaches are islands.
  - Load and bending moment are gathered from the farthest node inward,
    split between the neighbours nearer to support in proportion to shared
    area. Moments are horizontal vectors, so the opposite arms of a
    symmetric roof cancel.
  - Every hand-off is checked in compression, bending and shear against the
    weaker material: a quarter strength between placed blocks (mortar and
    nails), 30% of intact strength for natural rock mass, and natural
    ground is never crushed, only pulled apart.
  - Islands are cut out of the world as bodies, in pieces of a chosen size.
- Structural integrity in the game (`mc2-game::structure`): edits mark
  blocks dirty; a region grows from them over untouched terrain within 8 m
  and every placed block connected to it, however far that reaches.
  Terrain outside the box anchors it, except above, where it is open;
  unloaded chunks hold. Gathering costs at most a millisecond a frame and
  reads cached block summaries, the solve runs on a worker thread, and the
  verdict is applied when it arrives: failed blocks crumble into half-metre
  pieces, unsupported islands fall as 2 m pieces lowest first, and islands
  open to the top of their region are re-examined in a taller one.
- `--demo collapse` raises a stone tower, blows its base out and stops
  while it comes down; `--physics-thread` runs headless captures the way
  the window does; the F3 HUD reports both systems.

**Measured** (dev CPU).

*Debris piles*, `cargo run --release -p mc2-game --example pile_profile`,
2 m pieces dropped in a heap, 600 steps, before and after finding contacts
once per step instead of once per substep:

| Bodies | Step mean before | after | p99 after |
|---|---|---|---|
| 50 | 0.97 ms | 0.64 ms | 4.6 ms |
| 150 | 3.27 ms | 1.46 ms | 13.0 ms |
| 300 | 75.2 ms | 1.98 ms | 18.6 ms |
| 600 | 161 ms | 11.5 ms | 103 ms |

Before the change the 300-body pile never went to sleep; now every pile
settles. Body shapes also got cheaper to build: a 2 m piece 1.00 -> 0.46 ms,
a 4 m piece 7.4 -> 3.4 ms.

*The collapse demo* (a 3 m stone tower 14 m tall under a 5 m cap, its base
blown out), integrated GPU, 1280x720:

| | Inline physics | On its own thread |
|---|---|---|
| game.update mean | 32.4 ms | 2.2 ms |
| worst frame | 138 ms | 28.9 ms |

A typical collapse: 8 regions solved, 10402 nodes in the last of them,
20.7 ms of gathering spread over frames at a millisecond each, 21.5 ms to
solve on the worker thread, 32 blocks failed, one island of the tower came
down as 182 pieces, 230 bodies at the end.

**Budget.**

- Physics is off the frame: `game.update` falls from 32 ms to 2 ms with the
  thread, and a busy step (15 ms with 159 awake pieces) costs physics
  latency, not frames. It is still four times the 4 ms tick budget while a
  tower is in the air, so a collapse runs in slow motion for a moment.
- A detonation still costs 25 ms on the game thread: 17 ms carving and
  7.7 ms restoring generated detail. Carving is the next thing to move off
  the frame.
- Uploading the changed chunks after a collapse costs up to 24 ms in
  `voxel_gpu.update`, over its share of the frame.

**Bugs found.**

- Moments were scalars, so the four arms of a symmetric roof added instead
  of cancelling and an intact tower "failed" at a stress ratio of 2.3.
- Finding contacts once per step with a margin wider than a voxel made the
  sample spheres probe past their neighbouring cell, so two cubes sank 3 cm
  into each other before anything pushed back and then jumped apart at
  4 m/s. The margin is now half a voxel.
- The depenetration speed cap turned resolved overlap into velocity that
  nothing removed once the contact separated; with contacts that live for a
  whole step the cap is unnecessary and it is gone.
- Verdicts were validated by re-reading every block of an island, which
  cost more than the solve; they now compare against the summaries the
  cache still holds, which edits invalidate anyway.

**Rejected.** A lock around the voxel world (D29), structural integrity as
breakable XPBD joints or as finite elements (D30), whole islands as single
bodies (D31).
## Interlude: surfaces, living worlds, settlements and survival

Feedback after step 10 was blunt: the world looked blank, glass looked
wrong, there were no trees, no crafting, no towns. Step 11's solver was done
but its game integration waited while this was fixed; the work below pulls
parts of steps 13 (worldgen v2) and 17 (inventory, crafting, tools) forward.

**Surfaces.** Every material now has a voxel-scale pattern, computed in the
compose pass from the visibility buffer's voxel coordinates and face
(`surface_detail.wgsl`): boards and grain on planks, a running bond on brick
and stone brick, cobbles, bark and rings on logs, seams in ore, blades on
grass, the letters on TNT. Normals tilt per voxel so the patterns catch
light. A test checks that the shader's material ids match the Rust table.

**Glass and water.** Rays can now pass through chosen kinds of voxel
(`Ray.pass_kinds`), so the sun, sky and lamps shine through glass and into
water. A refraction pass at render resolution traces what lies behind each
clear surface (Snell's law for water, straight through thin glass), with
Beer-Lambert absorption and scattering in water and a traced sun shadow on
what it finds; compose blends it with the reflection by Fresnel.

**Living worlds.** A climate (temperature and wetness, cooler with height)
picks one of ten biomes, and each biome grows its plants: broad oaks with
low spreading limbs, pines, birches, bushes, cacti, boulders and fallen
logs, and ground cover of grass, flowers and moss. Plants are built from a
handful of primitives (capsules for wood, noisy ellipsoids for foliage) and
stamped at every level of detail, so forests read from the horizon. Stamping
them voxel by voxel cost 3386 ms a chunk; classifying each 8^3 cell against
only the parts that reach it, with the edge noise interpolated from the
cell's corners, brought that to 466 ms.

**Settlements.** Villages grow round a cobbled square and well on a stone
plinth: streets graded in straight four-metre runs, houses in four styles
(timber frame, stone, brick, thatched cottage), lanterns, lamp posts and
fields. About a third of sites that are flat enough become towns: a street
grid with dashed centre lines, zebra crossings, raised pavements and street
lights; glass towers on podiums of shops, brick apartment blocks over
shopfronts, concrete offices with ribbon windows, and parks with fountains.
Light panels under some floors make windows glow at night. Settlements are
ordered lists of shapes where the last one wins; chunks ask only the shapes
whose 8 m tiles they touch. A chunk in the middle of a town generates in 279
ms at full detail and 9 ms at the coarsest level. The player spawns at the
corner of the nearest village square.

**Survival.** Blocks have become items. Breaking a block takes time set by
its hardness and the tool in hand, and drops what it yields: rock needs a
pickaxe, iron ore a stone one; grass gives dirt, gravel sometimes flint,
coal ore coal. Tools come in wood, stone, iron and steel and wear out. A
36-slot inventory holds stacks, and carving collects loose voxel volume that
becomes a block every 4096 voxels. Forty-odd recipes run from logs to planks
to sticks and torches, through crafting tables, furnaces and tools, to
glass, ingots, steel, bricks, concrete, gunpowder and TNT. The furnace burns
fuel into heat for several smelts at a time.

The inventory screen (I, or use a crafting table or furnace) has the pack,
the hotbar and a recipe book that lists what can be made now first. Its
icons are painted at start-up by ray casting each item's voxel model into an
atlas: blocks in three-quarter view with their textures, tools and sticks
face-on, lumps and ingots like blocks, each with an outline and a soft
shadow. `--creative` keeps the old infinite hotbar and adds a catalogue of
every item.

**Goldens.** The golden images now render terrain without plants or
settlements: on the software adapter a village multiplies the time several
fold, and a bless already takes 50 minutes. The test world's centre is snowy
taiga, so its terrain is white.

## Step 11: Water (`v0.11-fluids`)

**What.** Free-surface lattice Boltzmann water (D3Q19, BGK with
Smagorinsky subgrid viscosity, Guo forcing for gravity) in 0.5 m cells,
eight voxels a side, stepped at 240 Hz. Cells are liquid, interface, gas or
solid; interface cells carry a mass and fill and empty as it flows, and gas
populations streaming into the surface are rebuilt from the atmosphere's
equilibrium ("only missing" reconstruction). The lattice lives in sparse
8^3 tiles allocated where water goes and freed when they have been dry for
a while; still tiles sleep and motion wakes their neighbours.

The CPU version in `mc2-fluid` is the reference. The GPU version in
`mc2-render` runs the same passes (stream and collide, flag, apply, share,
gather, activity) over a pool of tiles, dispatched indirectly over the
awake list the GPU builds for itself; the CPU owns only the tile table and
sends edits. A test steps both side by side and holds the mean difference
in water column height under a thousandth of a cell, and mass to 1e-4.

**In the world.** A levels pass packs each cell's fill in eighths, four
cells to a word, and copies them out a frame or two later; the game writes
them into the voxel world, filling a cell's open voxels to that height or
draining them. The water drawn is the water that flows, and it is lit,
refracted and reflected like the rest of the world. The simulation sees a
cell as solid where the voxel at its centre is, where the world has not
loaded, and where still water stands that it does not own. Sea and lake
water stays still until something disturbs it: an edit beside it takes the
water around (8 m, 8000 cells at most) into the simulation, and cells
along the edge of what was taken, below the surface, are refilled once
they drain, as the rest of the sea would refill them. Buckets carry water:
three iron ingots make one, it fills from any water and pours a cubic
metre where it is aimed. `--demo flood` opens a 7 m tank three metres deep
toward the camera; `--demo shore` digs a trench from a beach and lets the
sea in.

**Measurements.** The dam break benchmark (`fluid_bench`: water 16 by 32 m
and 12 m deep released into a 64 by 32 m basin, four steps a frame):

| | Mean GPU time a frame |
|---|---|
| Intel RaptorLake iGPU, populations in a private array | 20.8 ms |
| iGPU, populations as separate values | 7.1 ms |
| iGPU, converting cells mark their neighbours | 5.8 ms |
| NVIDIA RTX 3050 Laptop, over 12 s (650 tiles awake) | about 2 ms, p99 under 4.5 ms once running |

Mass drifts 0.05% over 12 s of the dam break. After digging a trench to the
sea, 260 tiles simulate; the flood demo settles into 142.

**Bugs found.**

- A 12 m dam break blew up after 9 s: splash cells passed 0.3 lattice units
  (Mach 0.5). Speeds are held to 0.18, about 22 m/s, far above anything
  falling or flowing at this scale.
- A droplet smaller than a cell never became gas and hung in the air while
  gravity sped it up. Interface cells with no water around, or under half
  full with neither liquid nor ground beside them, now empty (the mass is
  counted as lost).
- Excess mass with only liquid around it was dropped and overdrained cells
  went negative: the dam break gained 1.3% in 12 s. Liquid neighbours now
  take shares into their populations; the same run holds to 0.05%.
- Water stuck to walls and still water was slow to settle: walls are now
  specular with a little friction (0.995 kept), and new water starts at
  hydrostatic density.
- On the iGPU the 19 populations of a cell, held in a private array
  indexed by loop counter, lived in scratch memory: half the frame went to
  that array.
- FXC (DX12) rejects writes to a vector component through a dynamic index,
  which would fail the fluid pipelines there.
- Refilling the sea's edge every frame, whenever a cell was short of full,
  reset the surface at every ripple and kept the whole promoted sea awake.
- A levels readback recorded before new water reached the GPU would have
  drained the sea it was about to hold; tiles that have just taken still
  water cannot be lowered for eight frames.

**Known.** Shallow sheets of water spreading over flat ground stay awake
long after they look still (the flood's 142 tiles, the benchmark's 650 of
684 after 12 s): at a millisecond or two on the RTX 3050 it is affordable,
but tuning sleep for thin films is still to do. Cells are half a metre, so
a flowing surface steps in 0.5 m columns with 6.25 cm levels.

**Rejected.** Heightfield water, SPH and FLIP (D37), no-slip walls (D38),
hanging droplets (D39), drawing the lattice in the marcher (D40), and
simulating the whole sea (D41).

## Step 12: Fire, heat, wind and weather (`v0.12-weather`)

**Fire.** Fire burns 1 m blocks. A block holds so many seconds of fuel by
what is in it (a log 40 s, planks 25, leaves 5, grass 1.5), judged by 27
voxels through it so a trunk thinner than its block or a block of scattered
leaves still counts. Ten times a second its flames are drawn again: tongues
of fire voxels two across at the root, up to a metre tall, in the air
beside its flammable voxels, lighting everything around through the same
emitter path as torches. Its flammable voxels char from the outside in
(wood to charcoal, leaves and grass to nothing, wool and thatch to ash),
and it tries its neighbours: more readily above it (fire climbs) and
downwind. Water beside a burning block puts it out and rain does now and
then where it reaches; its heat melts snow and ice. Charring is an edit
like any other, so the structure solver judges a burning building as it
weakens. Flint and steel lights things; blasts set what is flammable
around them alight; TNT that catches goes off.

**Weather.** The sky drifts between clear, cloudy, rain and storm, each
lasting minutes; cloud cover, rain and wind ease toward what it calls for.
The renderer takes cover, precipitation and wind for the cloud layer,
thickens the fog in rain, and draws rain or snow by the climate where the
camera stands (nothing falls under a roof). Open ground soaks while it
rains (45 s) and dries slowly after (4 min): surfaces the traced sky
visibility reaches darken and turn glossy, and water stands in the hollows
of level ground, flat as mirrors; compose and the reflection trace agree
on the wet roughness, so puddles really reflect. Rain and snow are drawn at
display resolution in four layers of air 1.5 to 12 m out, gridded over the
view direction's angles so they hold still as the camera turns, sheared
along the wind and hidden behind anything nearer.

Storms strike every few seconds within 90 m: a jagged bolt of lightning
voxels, 60 m tall with a fork or two, for a quarter second (lighting the
scene and showing in the puddles), a flash over the whole image, and
whatever it hits may catch fire. `--weather clear|cloudy|rain|storm`
starts in a sky and holds it in captures; `--demo wildfire` and
`--demo lightning` show both.

**Measurements** (RTX 3050 Laptop, 1280x720):

| | Before | After |
|---|---|---|
| fire.tick, a canopy catching (350 blocks, mean / p99) | 1.76 / 28.4 ms | 1.11 / 10.4 ms |
| lights.update while water flows (mean) | 8.75 ms | 0.84 ms |

The fire runs ten times a second, so its cost lands on one frame in six.
Lights and sky heights were rebuilt for every chunk an edit touched,
every frame; a chunk now skips the emitter scan unless it had lights or a
changed brick holds an emissive voxel, and a chunk scanned in the last 12
frames waits.

**Bugs found.**

- A trunk thinner than its block looked like air to the eight probes that
  name a block's material, so fire never climbed a tree.
- Grass was floored at 0.3 flammability for spreading and set whole
  meadows alight.
- Charring stopped at the outer shell of a log: voxels were exposed only
  next to air, fire or ash, not to the charcoal the shell had become.
- A test of spreading fire failed until its ground had bedrock under it:
  the structure solver rightly dropped an unanchored slab and the logs on
  it as debris.
- The weather hooks landed in the commit before the function they call; the
  five commits were rebuilt from the same final tree in order before they
  were pushed.

**Known.** Fire is block-grained: heat does not flow through voxels, and
smoke is not drawn yet. The GPU frame at 720p is around 40 ms on the RTX
3050, most of it ReSTIR direct light: step 18's business.

**Rejected.** A voxel heat field (D42), particle rain (D43), lightning as
a screen effect (D44).

## Worldgen v2, in progress

Noise-worm caves were rejected in step 4. Passages follow two joint sets and
a bedding plane in limestone, chalk and marble. Steep ground thins the roof
so a hillside cuts into a joint; flat ground keeps three metres of cover.
Sinkholes drop where a river sits over a shaft. The dry part of a passage
grows stalactites and stalagmites. Node8 stays solid.

Uplift comes from kinematic plates (D46): colliding edges raise ranges,
parting edges open rifts, sliding edges leave a scarp. The coarse-terrain
cache is version 2, so an old eroded world is rebuilt rather than mixed
with the new field. North faces run colder, and carbonate ground keeps a
thin soil (D47).

The GPU tree pool no longer leaks when chunks stream out. Structure upload
is nearest-first and budgeted, so distant unloads never reached `free_chunk`;
`reclaim_unloaded` sweeps every frame, and a full pool evicts the farthest
resident chunk (preferring a range big enough to reuse) instead of dropping
the near upload on the floor.

A connected pad walks and looks with the sticks, opens the pack on Back,
and drives the inventory cursor. Photo mode stays on P.

## Step 14: Animation, IK, ragdolls (`v0.14-animation`)

**Built.** `mc2-anim`, a humanoid of eleven bones whose parts are voxel
grids at 6.25 cm, 29 voxels (1.81 m) tall.

- Forward kinematics from a root (feet, yaw) and per-bone local rotations,
  with a hip drop for crouching and a stride's bob.
- Looks from a seed: four skins, four hair colours, six shirts, four
  trousers, sleeves, long hair, beards and belts, painted voxel by voxel
  into the parts. Parts are `BodyShape`s, so a part can become a rigid
  body as it is.
- A procedural gait driven by distance, not time: the stride lengthens
  from 1.3 m walking to 2.4 m running, arms counter-swing, the torso leans
  and breathes, and jumping and crouching blend in.
- Feet planted by an analytic two-bone solve on the voxel top nearest
  each foot, the hips dropping as far as the lower foot needs; the head
  turns toward what the eyes point at within the neck's reach.
- `mc2-game::character`: every frame each character is posed from its
  body's interpolated feet and velocity and its parts are handed to the
  body renderer under keys above 2^40, clear of physics ids. The player
  has one, drawn in third person.
- Ball joints with cone limits in `mc2-physics`, and hinges for knees and
  elbows (axes held together, the bend kept between two angles): all hard
  XPBD position constraints every substep, with the relative spin damped
  at the velocity stage. Jointed bodies advance in
  the coupled group, bodies sharing a group never collide (a ragdoll's
  parts overlap at every joint), and a still pair falls asleep together.
- Characters caught in a blast become ragdolls of their own parts, hung
  at their joints with a swing per joint (a little at the neck and waist,
  far at the shoulder; knees fold back and elbows forward only), each part
  thrown away from the blast by how near it stood, so limbs fly first.

**Measured** (i5-13450HX, while a capture held the GPU and a core):

| | |
|---|---|
| posing a character (gait, IK, look, placement) | 1.56 us a frame |
| 10 ragdolls, 110 bodies, one 120 Hz step | 0.85 ms mean, 2.0 ms worst |

**Rejected.**
- Skinned meshes: the renderer draws voxels, and a skinned character would
  be the one thing in the world that is not made of them.
- Keyframed clips: a stride keyed to time slides on slopes and at any
  speed but the authored one; a distance-driven cycle cannot.
- Per-pair collision filters for ragdolls: a group id per body is one
  compare in the broad phase and covers the arm touching the torso as
  well as the joint itself.

**Known.** Villagers stand where a demo puts them; step 15 gives them
somewhere to go. A ragdoll never gets up again.

## Interlude: the site

The landing page was a CPU raymarcher drawing a toy scene into a canvas. It
could never look like the engine, so it is now a trailer told in the
engine's own stills.

- Every still is a headless capture at 2560x1440 on the top quality tier
  (`--clean` keeps the HUD out), saved as WebP at 1280 and 2560 wide.
- Chapters pin a full-screen stage and crossfade between their stills as
  you scroll; a trailer button plays all of them full screen with the
  chapters' titles. Reduced motion gets the same page, still.
- Every claim on the page was checked against the code; three captions
  were cut back to what their pictures show.

**Found while shooting.**
- Worldgen v2 moved the settlements: the spawn village is at (9855, 6871)
  and the largest town at (6411, 10673), a five by five street grid.
- Distant glass towers turn into floating floor slabs at coarse LOD: the
  thin glass drops out of the downsample and only the slabs survive.
  Visible on the horizon at dusk; a step 18 item.
- The shore demo's trench did not fill within 240 frames at the new beach
  (10296, 7392); the flood demo carries the water chapter.
- A 2560x1440 capture on the top tier spends about three minutes
  streaming before its frames on the RTX 3050 Laptop GPU.

## Step 15: People, trade, vehicles, in progress

**Built.**

- Homes: a village records each house's door (on the ground a stride past
  its step) and a spot on its floor.
- Walking: A* over half-metre columns of the voxels themselves. A column
  is standable with a solid top and 1.9 m of head room; a neighbour is
  reachable within 0.55 m up or 1.1 m down and with nothing between at
  knee, waist or head height, because a house wall (a quarter metre) is
  thinner than a column. Searches run 300 columns a tick across everyone,
  so a long one spreads over several ticks.
- Villagers: up to eight a village, peopled from its homes as the player
  comes near and emptied when it leaves. By day they go to the square, to
  one another's doors and back; at dusk they walk home and go in. They
  keep to the ground as it is now and think again when the way has gone,
  and turn their heads to anyone close.
- Trade: each villager is a woodcutter, mason, smith or miner. Trades are
  recipes made at a villager of the right trade, gold ingots the money.
  E on a villager in reach opens its trades; it stands and faces you until
  you are done.
- Cars: a body on raycast wheels (`mc2-physics::vehicle`). Each wheel's
  ray finds the ground each substep; a spring and damper hold the body up
  there and the tyre stops the sideways slide, drives and brakes within
  its grip, all as velocity impulses before integration. The car is one
  4 m voxel body (1215 kg): painted shell, glass cabin, seats, chrome,
  headlights and tail lights that light the road, and wheels held 3 cm
  clear of the ground by the springs. Every village keeps one by its
  square. E gets in and out; a chase camera follows.

**Measured** (i5-13450HX):

| | |
|---|---|
| worst `game.update`, trade demo, before searches were sliced | 99.5 ms |
| the same after | 6.7 ms |
| village demo, twenty seconds of villagers, worst `game.update` | 0.9 ms |
| drive demo, `game.update` mean / worst | 1.25 / 5.2 ms |

**Rejected.** See D50 and D51.

**Known.** Towns have no homes, so nobody lives in them yet, and nothing
parks on their streets. A car's wheels do not turn. Villagers walk
through one another.
