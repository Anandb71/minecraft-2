# Getting started

Rust stable (edition 2024) and a GPU with Vulkan, DX12 or Metal.

```bash
git clone https://github.com/Anandb71/minecraft-2.git
cd minecraft-2
cargo run --release
```

The first launch erodes a 16 km world (about 30 s on a laptop) and caches it
under `worlds/default`. Later launches load that cache. `--seed <n>` picks
another world.

Left click captures the mouse. An Xbox-layout pad starts play on the first
stick or face-button press; you do not need the mouse.

## Survival and creative

Default is survival: empty hands, hardness, tools that wear out, an inventory
on `I`. `--creative` fills the hotbar and unlocks the full catalogue.

## Headless

```bash
cargo run --release -- --capture shot.png --frames 120 --size 2560x1440
cargo run --release -- --bench --frames 600 --size 2560x1440
```

Useful flags:

| Flag | What it does |
|---|---|
| `--seed <n>` | World seed (default 42) |
| `--world <dir>` | World directory (default `worlds/default`) |
| `--time <hours>` | Time of day, e.g. `6.5` or `22`. Headless freezes there. |
| `--camera x,y,z,lx,ly,lz` | Headless camera position and look-at, metres |
| `--quality <0-3>` | Realistic … Super Ultra Crazy Duper Realistic |
| `--gi restir\|cascades` | Indirect light method |
| `--dof <metres>,<f>` | Thin lens for captures, e.g. `--dof 12,2.8` |
| `--demo <name>` | Scripted scene: `build`, `lights`, `mirror`, `blast`, `collapse`, `glass`, `craft`, `workshop`, `flood`, `shore`, `wildfire`, `lightning`, `people`, `ragdoll` |
| `--weather <sky>` | `clear`, `cloudy`, `rain` or `storm` (held in captures) |
| `--creative` | Nothing runs out |
| `--software` | WARP (Windows) or lavapipe (Linux) |
| `--hud` | Draw the profiler into a capture |
| `--hide-bodies` | Simulate debris without drawing it |
| `--physics-thread` | Headless physics on its own thread, paced in real time |

`MC2_BACKEND=vulkan|dx12|metal` forces a backend.

## Quality tiers

`--quality <tier>` or `F6` in game. Indirect light is half the render
resolution in every tier.

| Tier | Name | Render scale | Visibility rays | ReSTIR candidates | Cloud steps |
|---|---|---|---|---|---|
| 0 | Realistic | 50% | 1 per 2×2 | 8 | 32 |
| 1 | Hyper Realistic | 67% | 1 per 2×2 | 16 | 48 |
| 2 | Ultra Realistic (default) | 75% | every pixel | 32 | 64 |
| 3 | Super Ultra Crazy Duper Realistic | 100% | every pixel | 64 | 96 |

## Tests

```bash
cargo test --workspace
```

Golden images run on WARP in CI. On Linux, or without a GPU:

```bash
MC2_SKIP_GOLDEN=1 cargo test --workspace
```

More: [controls](controls.md), [architecture](architecture.md),
[CONTRIBUTING.md](../CONTRIBUTING.md).
