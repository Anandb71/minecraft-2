# MINECRAFT 2

A voxel sandbox that meshes nothing. The world is a sparse voxel structure ray
marched directly in compute shaders, built in Rust on wgpu. Resolution, traced
lighting, arbitrary destruction and traced audio all come from the same
acceleration structure.

> Status: under construction. Systems land in the order listed in
> [Build order](#build-order); each is tagged when it runs.

## Build and run

Requires Rust stable (edition 2024) and a GPU with Vulkan, DX12 or Metal.

```bash
cargo run --release
```

Headless modes:

```bash
cargo run --release -- --capture shot.png --frames 120 --size 2560x1440
cargo run --release -- --bench --frames 600 --size 2560x1440
```

The first launch erodes the whole world (about 30 s on a laptop) and caches it
under `worlds/default`; later launches load it instantly. `--seed <n>` picks
another world, `--camera x,y,z,lx,ly,lz` places a headless camera, `--time 21.5`
sets the hour, `--demo build|lights|mirror|blast|collapse|glass|craft|workshop`
plays a scripted scene before capturing, `--creative` starts in creative play, `--dof 12,2.8` renders with a thin lens focused at 12 m
at f/2.8, `--hide-bodies` simulates debris without drawing it and
`--physics-thread` steps physics off the main thread as the window does.

### Graphics quality

`--quality <tier>` or `F6` in game:

| Tier | Render scale | Visibility rays | ReSTIR candidates | Cloud steps |
|---|---|---|---|---|
| 0 Realistic | 50% | 1 per 2x2 block | 8 | 32 |
| 1 Hyper Realistic | 67% | 1 per 2x2 block | 16 | 48 |
| 2 Ultra Realistic (default) | 75% | every pixel | 32 | 64 |
| 3 Super Ultra Crazy Duper Realistic | 100% | every pixel | 64 | 96 |

Indirect light runs at half the render resolution in every tier; `--gi restir`
swaps radiance cascades for ReSTIR GI.

`--software` selects the software adapter (WARP on Windows, lavapipe on Linux).
`MC2_BACKEND=vulkan|dx12|metal` forces a backend.

## Controls

| Key | Action |
|---|---|
| Left click | Capture mouse; then break block (hold) / carve |
| Right click | Place the held block / deposit; use a crafting table or furnace |
| I | Inventory and recipes |
| Middle click (held) | Preview the targeted block instead of the placement |
| W A S D | Move |
| Space | Jump (fly: up) |
| Ctrl or C | Crouch (fly: down) |
| Shift | Sprint |
| F | Toggle flying |
| F5 | First / third person |
| Tab | Block mode / carve mode |
| E | Light the targeted TNT block (4 s fuse) |
| 1-9, mouse wheel | Hotbar slot (carve mode: wheel changes radius) |
| F3 | Toggle profiler HUD |
| F4 | Debug view: shaded, LOD hits, march iteration heatmap, unlit materials |
| F6 | Cycle graphics quality |
| T | Pause or resume time |
| [ ] | One hour back or forward |
| V | Toggle vsync |
| P | Photo mode |
| F2 | Screenshot |
| Esc | Release mouse, then quit |

### Photo mode

Time stops and the camera flies free.

| Key | Action |
|---|---|
| W A S D, Space, Ctrl | Fly |
| Mouse wheel | Focal length, 12-400 mm |
| Q / E | Focus nearer / farther |
| Z / X | Aperture one stop wider / narrower |
| , / . | Exposure down / up a third of a stop |
| G | Depth of field on or off |
| Enter or F2 | Render 96 frames at Super Ultra Crazy Duper Realistic and save `captures/photo_<time>.png` |
| P | Leave |

### Survival

The game starts in survival with empty hands. Hold left click to break a
block: logs and dirt give way to bare hands, rock needs a pickaxe, iron ore a
stone one, and better tools break things faster until they wear out. Press I
for the inventory: drag stacks between the pack and the hotbar (right click
splits, shift click moves) and click a recipe to make it. Logs become planks,
planks sticks and a crafting table; at the table come tools, a furnace, stone
bricks, windows, lanterns, concrete and TNT (gunpowder is coal and flint from
gravel); the furnace, fed coal or wood, makes glass, iron and steel ingots,
charcoal and bricks. `--creative` gives an endless hotbar and a catalogue of
every item instead.

## Features


- A 16 km world grown from geology: continents and ridged mountain belts eroded once by a virtual-pipe hydraulic model, stratified rock (basement, cyclic sediments, folds, volcanic provinces) exposed by that erosion, ores that follow their host rock
- Four levels of detail streamed around the camera on worker threads; buried rock is never generated until someone digs
- Ten biomes from a climate that cools with height: oak and birch forests, pine taiga, snowy taiga, plains of grass and flowers, deserts with cacti, bare alpine slopes and snowfields, beaches and seas; every tree, bush, boulder and fallen log built from voxels and visible from the horizon
- Villages round a cobbled square and well (timber, stone, brick and thatched houses, graded streets, lamps, fields) and towns on a street grid with road markings, glass towers, brick apartments over shops, offices and parks, lit at night
- A pattern on every material at voxel scale: boards and grain, brick bond, cobbles, bark and growth rings, ore seams, grass blades, the letters on TNT
- Glass, ice and water you can see through: refraction by Snell's law, absorption and scattering with depth, sunlight and lamplight traced through them, Fresnel reflection on top
- Survival: hardness and tools decide how long a block takes and what it drops; tools in wood, stone, iron and steel wear out; 36 slots of stacks; 44 recipes by hand, at a crafting table and in a furnace that burns fuel
- An inventory screen and hotbar whose icons are ray cast at start-up from each item's voxel model
- The world meshes nothing: 6.25 cm voxels in 4-bit palette bricks under per-chunk sparse 64-trees, ray marched in a compute shader
- Physically based atmosphere (Hillaire 2020): sky-view and aerial perspective LUTs in photometric units, sunsets, moonlit nights with a phased moon, stars rotating with sidereal time
- Sun and moon positions from latitude, season and time of day
- Traced soft shadows and sky visibility through the same marcher that draws the world, accumulated with exact voxel-id reprojection
- ReSTIR direct light from every emissive voxel cluster: torches, lanterns, lava
- Indirect light from radiance cascades (ReSTIR GI selectable), bouncing frame over frame through last frame's lit surfaces
- SVGF denoising of emitter, indirect and reflected light; temporal upsampling from the render resolution to the display
- Volumetric clouds (Schneider and Vos 2015) in a 1.5-4 km shell, lit by sun or moon with multiple scattering octaves, casting shadows on the ground, the fog and bounced light
- Froxel height fog with light shafts, integrated energy-conservingly (Hillaire 2015)
- Traced glossy reflections with visible-normal sampling for metals, ice and polished stone
- Bloom, motion blur, thin-lens depth of field, Purkinje night vision, sharpening, vignetting and film grain
- Photo mode with a free camera, focal length, focus, aperture and exposure
- Rigid body debris (extended position based dynamics): blasts carve craters and throw fragments that tumble, stack, can be stood on or kicked, and become terrain again when they settle
- TNT with fuses and chain reactions; lit charges fly as bodies and go off where they land
- Bodies are traced through the same lighting as the world: shadows, bounce light, reflections and exact reprojection
- Structural integrity over the 1 m block layer: load and bending moment flow to bedrock through real materials, so cutting a load bearing column drops the tower, an unsupported stone arm snaps at about 3 m and a plank one at 23 m, and soil cannot overhang at all
- Collapses fall as rigid pieces that pile up, can be walked on, and become terrain again once they settle
- Auto exposure in EV100 with a night-aware key
- Four quality tiers up to Super Ultra Crazy Duper Realistic
- Visibility buffer with exact world voxel ids, material, face, depth and motion vectors
- First and third person player with voxel collision, automatic step-up, crouch and fly
- Build on the 1 m block grid or carve and deposit 6.25 cm voxels; digging restores generated detail so tunnels show real strata
- Beam prepass and empty-space coalescing; the GPU marcher is verified pixel for pixel against a CPU reference marcher
- Bricks stream to the GPU on demand from marcher feedback, so unseen surfaces never cost VRAM
- Own frame graph: declared reads/writes, validation, culling, resize-aware allocation
- GPU timestamp query on every pass, CPU scopes everywhere, rolling p99 on screen
- WGSL hot reload in debug builds with `#import`
- AgX display transform
- Golden image tests on WARP in CI

## Frame budget

1440p, 60 fps, mid range 2024 discrete GPU. 16.6 ms. The profiler HUD shows
each line's last and p99 cost and turns red when p99 exceeds its budget.

| Line | Budget |
|---|---|
| Primary visibility march | 2.5 ms |
| Direct light (ReSTIR) | 2.0 ms |
| Indirect light | 2.5 ms |
| Reflections | 1.0 ms |
| Denoise and upsample | 2.0 ms |
| Volumetrics and clouds | 2.0 ms |
| Atmosphere and sky | 0.5 ms |
| Fluid and fire (amortized) | 1.5 ms |
| Post and present | 1.0 ms |
| Headroom | 1.6 ms |

Physics runs on its own thread at a fixed 120 Hz with a 4 ms tick budget,
never blocking a frame; structural integrity is gathered a millisecond a
frame and solved on another thread.

## Build order

1. Harness: window, device, frame graph, WGSL hot reload, timestamp queries, profiler HUD, golden image rig, CI. **(v0.1-harness)**
2. Brickmap, 64-tree, palette compression, CPU reference marcher. **(v0.2-brickmap)**
3. GPU marcher, visibility buffer, brick streaming with feedback. **(v0.3-marcher)**
4. Worldgen v1, region streaming, LOD. **(v0.4-worldgen)**
5. Player controller, collision, carve and place. **(v0.5-player)**
6. Sun, sky LUTs, traced shadows, ReSTIR direct light. **(v0.6-lighting)**
7. Indirect light, denoise, temporal upsample. **(v0.7-indirect)**
8. Reflections, volumetrics, clouds, post stack, photo mode. **(v0.8-atmosphere)**
9. Rigid bodies, contacts, destruction. **(v0.9-destruction)**
10. Structural integrity and collapse. **(v0.10-structure)**
11. Fluids.
12. Fire, heat, materials, wind, weather.
13. Worldgen v2: tectonics, erosion, climate, hydrology, caves, ecology.
14. Animation, motion matching, IK, ragdolls.
15. NPCs, settlements, economy, vehicles.
16. Traced audio.
17. Inventory, crafting, tools, UI, settings, persistence.
18. Optimisation and content pass.

## Repository

- `crates/mc2-core`: profiler, statistics
- `crates/mc2-gpu`: device, frame graph, shader library, timestamp profiler, capture
- `crates/mc2-voxel`: bricks, 64-trees, materials, GPU layout, CPU reference marcher
- `crates/mc2-game`: ECS game state, player, collision, blocks, interaction
- `crates/mc2-worldgen`: noise, strata, erosion, amplification, chunk generation, streaming
- `crates/mc2-physics`: rigid bodies (XPBD), contacts, explosions, baking
- `crates/mc2-structure`: the load graph, stress and collapse
- `crates/mc2-render`: passes, GPU voxel residency, rigid bodies and the renderer
- `shaders/`: WGSL
- `golden/`: reference images for renderer tests
- [`DEVLOG.md`](DEVLOG.md): how each system works and what it costs
- [`DECISIONS.md`](DECISIONS.md): architectural forks and the measurements behind them
