# Contributing

Thanks for wanting to touch this. The game is a cargo workspace of small
crates and a pile of WGSL. Read this, then [docs/architecture.md](docs/architecture.md)
and [DECISIONS.md](DECISIONS.md) before rewriting something that was already
measured.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be
decent. Report problems to [anandbiju71@gmail.com](mailto:anandbiju71@gmail.com).

## What to work on

Useful:

- bugs you can reproduce (`--seed`, `--camera`, a screenshot)
- frame-budget holes the profiler HUD already names
- missing tests around a change you are making
- docs that were wrong after a change you shipped

Talk first:

- new dependencies (each one needs a line in `DECISIONS.md`)
- a different acceleration structure, lighting method, or physics solver
- anything that makes worlds non-deterministic across GPUs
- large visual restyles of the HUD or inventory

The build order in the README is the product plan. A surprise rewrite of a
finished step is rarely the right move.

## Setup

Rust stable, edition 2024. A GPU with Vulkan, DX12 or Metal. On Windows the
golden image tests run on WARP; on Linux CI they are skipped and lavapipe
covers the rest.

```bash
git clone https://github.com/Anandb71/minecraft-2.git
cd minecraft-2
cargo test --workspace
cargo run --release
```

The first launch erodes the world (about 30 s) and caches it under
`worlds/default`. Later launches skip that.

## Day to day

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Linux, or a machine without a GPU:

```bash
MC2_SKIP_GOLDEN=1 cargo test --workspace
```

Force a backend: `MC2_BACKEND=vulkan|dx12|metal`. Software adapter:
`--software` (WARP on Windows, lavapipe on Linux).

`cargo fmt` and clippy-as-errors are what CI runs. Match them locally.

## Style

- Small crates, small modules, comments that say why.
- No new parser crates for a handful of flags. No egui. See `DECISIONS.md`.
- Worldgen and erosion must stay CPU-side and seed-deterministic. Two
  players with the same seed on different GPUs have to get the same
  mountains.
- Prefer a test over a screenshot in the PR description. Prefer a
  measurement over "it felt faster".

## Pull requests

1. One idea per PR.
2. `cargo fmt`, clippy, tests.
3. If you changed how a system works, add a paragraph to `DEVLOG.md`.
4. If you chose between two architectures, add a `D*` entry to `DECISIONS.md`.
5. Update the README or `docs/` when a player-facing flag, key, or feature
   moved.

Use the pull request template. I will review from Tamil Nadu, usually on
evenings.

## Issues

Bug reports need: OS, GPU, backend (`MC2_BACKEND` or whatever the HUD
printed), seed, and what you did. Feature ideas are welcome if they name the
player-facing outcome, not only the algorithm.

Security: [SECURITY.md](SECURITY.md), not a public issue.
