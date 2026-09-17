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
