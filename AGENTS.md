# Contributor Guide

## Build Output

Cargo never removes stale artifacts from a `target` directory. Every change to
a crate's dependencies, features, or toolchain leaves the old build behind
under a new hash, and each provider crate links its own test binaries, so a
long-lived `target` grows without bound.

- A worktree builds into its own `target`. Remove the worktree, and with it
  that `target`, as soon as its branch lands or is abandoned, and delete the
  branch.
- Never point `CARGO_TARGET_DIR` at a temporary or scratch directory that
  outlives the task. Write logs there, not builds.
- Tests that need files make them under a `tempfile` directory that removes
  itself, never a fixed path under the system temp directory.
- When a `target` grows large, `cargo clean` it. A compiler cache such as
  sccache keeps the next build reasonable.
