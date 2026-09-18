# Security policy

This is a local voxel game. It does not talk to a network, store passwords, or
run untrusted shaders from the internet. The realistic risks are still worth
reporting: panics on bad CLI input, unbounded memory from a crafted world
directory, path traversal in `--capture` / `--world`, or shader compiler
crashes that take the process down.

## Reporting

Email **[anandbiju71@gmail.com](mailto:anandbiju71@gmail.com)** with:

- a short description of the issue
- steps to reproduce (command line, seed, platform)
- whether you believe it is exploitable beyond crashing the game

Please do not open a public issue for something that could be used to harm
players' machines. I will reply when I have looked at it.

## Supported versions

Only `main` is supported. There are no numbered releases yet.

## Scope

In scope: this repository's Rust and WGSL, its CLI, world cache files, and
the way it writes screenshots.

Out of scope: bugs in wgpu, your GPU driver, or "the game is slow on my
integrated chip" (that belongs in a normal issue).
