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

## D12. The macro layer

- **Dense block ids per chunk, authoritative.** Familiar, but 32 768 ids per chunk that restate the voxels for natural terrain, and every micro edit has to decide what the block has become.
- **Voxels only, no blocks.** Simplest, but inventory, crafting and building want identity: a torch is not "some planks and some flame".
- **Implicit blocks plus explicit placements (chosen).** A block with no explicit entry is its dominant material, sampled from eight interior voxels on demand; placed blocks with identity are recorded in a sparse map. Breaking returns voxel volumes by material, so macro and micro interaction share one economy (4096 voxels to a block).

`bevy_ecs` (game crate): archetype storage, change detection and system scheduling for the game state; writing an ECS is not the product, and taking only the ECS (not Bevy's renderer) is what the brief asks.

## D13. Sky and atmosphere

- **Analytic sky (Preetham, Hosek-Wilkie).** Microseconds per frame, but only a sky dome seen from the ground in daylight: no aerial perspective over a 16 km world, no view from a mountain top, nothing after sunset.
- **Bruneton and Neyret precomputed scattering.** Physically complete, but 4D scattering tables that rebuild slowly when media change, and weather will change them.
- **Hillaire 2020 LUTs (chosen).** A fixed transmittance and multiple scattering pair built once (1.04 ms), then a 200x100 sky-view LUT and a 32^3 aerial perspective volume per frame (0.71 ms by day, 1.37 ms at night when the moon gets its own pair). Per unit illuminance, so the moon is the same pipeline with a different light.

## D14. Sun and moon shadows

- **Cascaded shadow maps.** The standard, but it needs a rasterised depth representation of a world that is not meshed, and at 6.25 cm detail and kilometre view distances cascades either alias the voxels or miss the mountains.
- **Screen-space shadows plus a coarse heightfield.** Cheap, but wrong for anything off screen or overhanging, which is most of a voxel world once people dig.
- **Traced shadow rays through the marcher (chosen).** One ray per pixel toward a random point on a widened disc, temporally accumulated with exact voxel id reuse. The same structure that renders the world shadows it, edits included, with no second representation. It is the most expensive line of the frame (see D16).

## D15. Emissive voxel lighting

- **A shadow ray per light per pixel, or clustered forward lights.** Correct for a handful of torches; a lava lake is thousands of emissive bricks.
- **Light BVH with one importance-sampled light per pixel.** Bounded cost, but noisy for many similar lights and needs a BVH refit whenever voxels change.
- **ReSTIR (chosen).** 32 candidates from an alias table over distance-weighted power, then reservoir reuse across frames and neighbours: effectively hundreds of candidates for two visibility rays per pixel. Lights are emissive clusters per brick in stable slots, rebuilt only for chunks the GPU world reports changed (lights.update 0.08 ms mean while streaming).

## D16. Scheduling visibility rays

- **Trace every pixel every frame.** 41.2 ms at 960x540 on the integrated GPU.
- **Skip pixels with settled history inside the full-resolution dispatch.** 21.3 ms at 854x480 with a 2x2 stride where a quarter of the rays should have cost about 10 ms: every 8x8 workgroup still contains tracing pixels, and SIMD lanes run until the slowest ray finishes.
- **Dispatch over blocks, resolve per pixel (chosen).** One invocation per 2x2 block traces its most starved pixel; a full-resolution resolve accumulates, carries or borrows. 9.0 ms. Stride is a quality knob: 1 at Ultra and above, 2 below.

## D17. Quality presets

- **A few global levels read by each pass.** Compact, but every pass interprets "high" separately and the budget table cannot say what a level costs.
- **Hundreds of independent settings.** Maximal control and an untestable matrix.
- **Four named presets that are plain settings structs (chosen).** Realistic, Hyper Realistic, Ultra Realistic and Super Ultra Crazy Duper Realistic each fill a `Quality` struct; knobs join the struct as the systems they scale land, and a test keeps each knob monotonic across tiers. The budget table is measured at Ultra Realistic.

## D18. Reconstruction: denoising and upsampling

- **Spatial blur only (edge-aware bilateral per frame).** Cheap and immediate, but one sample per pixel at half resolution needs a footprint that erases contact detail, and it flickers.
- **A single temporal accumulator for the final image.** TAA alone smears lighting noise into ghosts and cannot tell texture from noise.
- **SVGF per demodulated lighting signal, then temporal upsampling of the final image (chosen).** Lighting is denoised without albedo, at its own resolution, with variance-guided edge stopping; the composed image then accumulates jittered render samples into display resolution. Measured on the integrated GPU: SVGF 4.6 ms at 320x180, 10.8 ms at 480x270; upsampling 4.4 ms at 1280x720 output.

## D19. Indirect light: radiance cascades against ReSTIR GI

Both were built behind one output with the same hit shading and denoiser, and measured on the same scene against an unbiased reference (DEVLOG, step 7).

- **ReSTIR GI.** 9.4 ms at Realistic, relative RMSE 1.94, bias +33%, 27% frame-to-frame change after SVGF. Reservoirs were healthy (temporal M at the 30 clamp for most pixels) but the weights' tail (99th percentile W 16x the median) survives denoising as flicker, and biased spatial reuse brightens.
- **Radiance cascades with the paper's doubling intervals.** 2.5 ms, RMSE 1.65, bias -14% (six cascades reach only 15.75 m), 6% change.
- **Radiance cascades with fourfold intervals (chosen).** 3.0 ms, RMSE 1.54, bias +3.5%, 5% change. Deterministic per frame, so the denoiser only has to smooth probe interpolation. ReSTIR GI stays available with `--gi restir`.

## D20. Lighting indirect hits off screen

- **Trace a shadow ray and a sky ray at every hit.** Correct, and triples the ray count of either method.
- **A world-space radiance cache (hash grid of voxel faces).** Handles off-screen multi-bounce well, but it is a second structure to update on every edit and to budget.
- **Last frame's surface radiance on screen, the sky map off screen (chosen).** Bounces compound through the previous frame for free; off-screen hits get sun and sky visibility from column heights (texture reads and an 18-step 2D march). Overhangs deeper than the map's 0.5 m column resolution can leak a little sky light.

## D21. Clouds

- **Cloud billboards or a painted sky dome.** Cheap and art-directable, but static: no time of day, no weather, nothing to fly under or cast shadows.
- **Clouds as voxels in the world structure.** Consistent with everything else, but a cloud layer is kilometres of mostly empty, constantly changing volume; streaming and editing it would cost more than the terrain.
- **Procedural ray-marched layer (chosen, after Schneider and Vos 2015).** Two tiling noise volumes, height gradients per cloud type, weather-driven coverage, a cone light march with Wrenninge's multiple scattering octaves, one quarter of the half-resolution texels traced per frame and the rest reprojected. 1.3 ms at Realistic, 3.2-3.8 ms at Ultra on the integrated GPU. A 256^2 toroidal shadow map feeds the sun term, fog and indirect hits.

## D22. Local fog and light shafts

- **Analytic height fog in composition.** Free, but it cannot be shadowed: no shafts through clouds or past cliffs.
- **Ray-marched fog per pixel.** Correct and shadowed, but tens of samples per pixel.
- **Froxel volume with temporal reprojection (chosen, Hillaire 2015).** 64 exponential slices at 1/8 resolution, lit once per froxel with sky-map sun visibility and cloud shadows, reprojected 90%, integrated once per column. 2.0 ms at Realistic, 4.2 ms at Ultra.

## D23. Glossy reflections

- **Screen-space reflections.** Cheap, but a voxel world is full of off-screen and occluded geometry, and the sky dominates what water and ice reflect.
- **Pre-filtered environment probes.** Stable, but they misplace everything nearby and go stale with every edit.
- **Traced reflections through the marcher, denoised (chosen).** One VNDF-sampled ray per half-resolution pixel for surfaces up to roughness 0.5, hits shaded like indirect light, SVGF on the result. Rough surfaces keep the analytic sun highlight only. refl.trace 1.9 ms and refl.denoise 5.0 ms at Realistic when glossy surfaces fill the view.

## D24. Post and photo mode

- **No post beyond tonemapping.** Honest, but images read as renders: no glare around the sun, no film response, no night vision.
- **Screen-space effects stacked ad hoc.** Every effect a pass with its own resolution and history.
- **One display-resolution post pass plus the present shader (chosen).** Bloom, motion blur and depth of field on the upsampled HDR image; Purkinje shift, AgX, contrast-adaptive sharpening, vignetting and grain in present, all from one settings struct. Photo mode reuses the same passes, raising quality to the top tier for 96 frames before saving.

## D25. Rigid body solver

- **Sequential impulses with warm starting.** The standard in game engines, but stable stacks need many iterations and persistent contact manifolds, which voxel fragments with changing sample contacts do not have.
- **Shape matching over voxel particles.** Every voxel a particle: handles any shape and breaks naturally, but costs per voxel and bodies look soft.
- **Extended position based dynamics with substeps (chosen, Müller et al. 2020).** One position iteration per substep, contacts as positional constraints, friction and restitution in a velocity pass. Two cubes stack on the first try once penetration is re-evaluated; 144 fragments step in 2.0 ms.

## D26. Collision representation

- **Convex hulls or boxes per fragment.** A cheap narrow phase, but it fills in the concave shapes that blasts produce, and it needs a second representation beside the voxels.
- **Signed distance fields per body.** Smooth normals, but building and storing one for every fragment costs more than the fragment.
- **Surface sample spheres against voxel grids (chosen).** Up to 64 samples per body; one sphere probe serves world and body contacts, with the world read through a window cached per step. Resting contact is blocky at the scale of one voxel, which is the scale of everything else.

## D27. Drawing bodies

- **Rasterise body meshes into the visibility buffer.** Fast primary visibility, but a second geometry path that shadow, indirect and reflection rays would not see.
- **Voxelise moving bodies into the world every frame.** One structure, but rotation aliases, and every frame would edit and re-upload bricks.
- **Trace posed voxel grids beside the world (chosen).** The same ids, lighting, history and motion vectors as terrain. A culling grid keeps a ray that misses the debris at one slab test. 74 fragments cost 1.4 ms (Realistic) and 6.3 ms (Ultra) on the integrated GPU, once sky rays skip them.

## D28. Destruction

- **Remove voxels only.** Craters without anything thrown: cheap and lifeless.
- **Particles for debris.** Many cheap sprites or points, but they cannot be stood on, kicked, stacked or baked back into the world.
- **Rigid fragments from the blast shell, baked back when settled (chosen).** Up to 48 fragments per blast, sleeping when still and becoming terrain after 15 s, so a battlefield does not accumulate bodies.

## D29. Physics on its own thread

- **A lock around the voxel world.** Simplest to write, but the renderer takes the world mutably every frame to stream bricks, so a frame would wait on a physics step: exactly what the brief forbids.
- **Copy the world, or the chunks near the bodies, each tick.** No contention, but a chunk tree is megabytes and debris sits in several of them; copy-on-write would move that cost to the next edit instead of removing it.
- **A mirror of collision cells, fed by the game thread (chosen).** The physics thread owns solid occupancy for the brick cells around its bodies, a few thousand at a time, and asks for what it lacks; bodies whose surroundings have not arrived wait a tick, which is invisible. The game thread never blocks, and the mirror is about 60 bytes a cell.

## D30. What structural integrity is made of

- **Breakable XPBD joints between blocks.** The 2016 paper's total Lagrange multiplier gives a constraint force, so joints could break on force, and the solver already exists. But every block of every structure would become a simulated body with constraints, at 120 Hz, whether or not anything is happening to it.
- **Finite elements over the voxel grid.** The honest way to get stress, and far too expensive for a 16 km world edited continuously.
- **A load graph over the 1 m block layer (chosen).** Only blocks near an edit are examined, a few thousand at a time, on a worker thread. Compression, bending and shear per connection reproduce what the brief asks for: a stone arm breaks past about 3 m, a plank one past about 23 m, cutting a column drops the tower, and soil cannot overhang at all. It ignores arching and load history, which a game does not miss.

## D31. What falls, and in what pieces

- **The whole island as one body.** True to the structure, but a collapsing wall would need a dense voxel grid over its bounding box: a hollow tower would cost a hundred megabytes and one body that cannot break further.
- **Voxel-level fragmentation.** Every voxel its own body is the most detailed and the least affordable.
- **Fixed 2 m pieces, half-metre crumbs for failed blocks (chosen).** A collapse becomes a few hundred bodies, each a 32^3 grid built in under half a millisecond, cut a few per frame lowest first so the pile builds from the bottom. Rubble that settles for 15 s becomes terrain again, so a battlefield does not accumulate bodies.
