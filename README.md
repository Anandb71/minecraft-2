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
another world, `--camera x,y,z,lx,ly,lz` places a headless camera.

`--software` selects the software adapter (WARP on Windows, lavapipe on Linux).
`MC2_BACKEND=vulkan|dx12|metal` forces a backend.

## Controls

| Key | Action |
|---|---|
| Left click | Capture mouse; then break block / carve |
| Right click | Place block / deposit material |
| Middle click (held) | Preview the targeted block instead of the placement |
| W A S D | Move |
| Space | Jump (fly: up) |
| Ctrl or C | Crouch (fly: down) |
| Shift | Sprint |
| F | Toggle flying |
| F5 | First / third person |
| Tab | Block mode / carve mode |
| 1-9, mouse wheel | Hotbar slot (carve mode: wheel changes radius) |
| F3 | Toggle profiler HUD |
| F4 | Debug view: shaded, LOD hits, march iteration heatmap |
| V | Toggle vsync |
| Esc | Release mouse, then quit |

## Features

- A 16 km world grown from geology: continents and ridged mountain belts eroded once by a virtual-pipe hydraulic model, stratified rock (basement, cyclic sediments, folds, volcanic provinces) exposed by that erosion, ores that follow their host rock
- Four levels of detail streamed around the camera on worker threads; buried rock is never generated until someone digs
- The world meshes nothing: 6.25 cm voxels in 4-bit palette bricks under per-chunk sparse 64-trees, ray marched in a compute shader
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

Physics runs on its own thread at a fixed 120 Hz with a 4 ms tick budget.

## Build order

1. Harness: window, device, frame graph, WGSL hot reload, timestamp queries, profiler HUD, golden image rig, CI. **(v0.1-harness)**
2. Brickmap, 64-tree, palette compression, CPU reference marcher. **(v0.2-brickmap)**
3. GPU marcher, visibility buffer, brick streaming with feedback. **(v0.3-marcher)**
4. Worldgen v1, region streaming, LOD. **(v0.4-worldgen)**
5. Player controller, collision, carve and place. **(v0.5-player)**
6. Sun, sky LUTs, traced shadows, ReSTIR direct light.
7. Indirect light, denoise, temporal upsample.
8. Reflections, volumetrics, clouds, post stack, photo mode.
9. Rigid bodies, contacts, destruction.
10. Structural integrity and collapse.
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
- `crates/mc2-render`: passes, GPU voxel residency and the renderer
- `shaders/`: WGSL
- `golden/`: reference images for renderer tests
- [`DEVLOG.md`](DEVLOG.md): how each system works and what it costs
- [`DECISIONS.md`](DECISIONS.md): architectural forks and the measurements behind them
