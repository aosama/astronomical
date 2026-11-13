# Building and developing Astronomical

For people who build Astronomical from source or contribute to it. Read [CONTRIBUTING.md](../CONTRIBUTING.md) first. To use the released app, see the [README](../README.md).

## Two copies that never mix: Stable and Development

The app you use every day and the app you are changing are kept apart, so a broken build cannot disturb your daily one.

| | Stable | Development |
| --- | --- | --- |
| What it is | The released app you rely on | The app built from your checkout |
| State folder | `~/.astronomical` | `~/.astronomical-dev` |
| Address | `127.0.0.1:6732` | `127.0.0.1:6733` |
| App | `~/Applications/Astronomical.app` | `target/astronomical-macos-development.noindex/Astronomical Development.app` |

- **Kept apart:** settings, logs, prompt caches, background processes, process locks and addresses. Each copy refuses a configured address that belongs to the other.
- **Shared:** both may read the same model folders, which they never write to. Both also share the Mac's graphics chip, memory and SSD, so running a real model in Development slows Stable down.
- **Tests use Development only.** Tests that run a model read model locations and policy from Development. Tests that write settings, caches or logs use a temporary `.astronomical-dev` folder and never `~/.astronomical`. The only exception is settings tests that build a temporary Stable fixture to prove the two stay apart.

The settings file each copy creates is described in [Configuration](configuration.md).

## What you need

- Xcode with the macOS 26 software development kit.
- CMake.
- Rust nightly-2026-11-07 (rustc 1.101.0-nightly). The checked-in toolchain file selects it for you.
- Swift 6.2.
- Homebrew coreutils, jq, curl and sccache, used by the repository's checks.

## Build and run the Development app

1. Fetch the native dependencies. Each one is pinned to a version and checked against a checksum:

       cargo xtask bootstrap-native-dependencies

2. Build the Development app and open it. This never touches Stable:

       cargo xtask app dev

   Add `--background` to open it behind your other windows, so the app you are working in stays in front.

The first build compiles everything. Later builds are incremental and replace the running Development app in seconds. A failed build leaves the running app as it was.

To control the Development copy without its window, for example to start it with exported `ASTRONOMICAL_*` experiment flags:

    cargo xtask app instance status|start|stop|restart

**About the built app:**

- It is signed ad hoc, for local use only.
- The `.noindex` folder name keeps built apps out of Spotlight.
- Development's icon carries a small badge so you can tell it from Stable.

To rehearse launching any built app in isolation:

    cargo xtask app smoke-test --app-bundle PATH

Add `--real-model MODEL_ID` to send one prompt through a model. This uses the Development copy and the graphics chip, so do it only when nothing else needs them.

To build only the optimized background service and worker:

    CARGO_BUILD_JOBS="$(sysctl -n hw.logicalcpu)" cargo build --release \
      -p astronomical-inference-worker --bin astronomical-inference-worker \
      -p astronomical-daemon --bin astronomicald

## Check your change

Check Rust formatting, then run the Rust integration and REST suites:

    cargo fmt --all -- --check
    cargo test-integration-and-rest

Run the macOS menu contracts:

    cargo xtask menu-contracts

Tests that drive MLX directly or need a real model are run on request, because they need Apple graphics hardware or models on your Mac.

## Build and install Stable (release only)

Installing Stable and publishing a release use release-only commands, `cargo xtask release ...` and the scripts under `scripts/release/`. They are never part of checking an ordinary commit or push.

To build a clean Stable app and install it outside the repository's build folder:

    cargo xtask release build-and-install-stable

- **Preview first** with `--dry-run`: it builds and validates Stable and shows what installing would do, without doing it.
- **Run the two stages separately** with `cargo xtask app package --channel stable` and `cargo xtask release install-stable`.
- **The previous app goes to the Trash.** Installing never deletes it.
- **A running Stable is left alone.** The build checks the signature, resources, metadata and the bundled background service without launching over a running Stable, and installing does not restart it.
- **Commit first.** Stable builds require a Git worktree with no uncommitted changes.

### What checks the packaging

On every pull request:

- the release steps' decisions, through `cargo test -p xtask`,
- the typed packaging contracts, as `cargo test-app-builder`,
- the icon renderer contract, as `swift scripts/internal/render-macos-app-icon-contract.swift`.

A real Stable app is built with `cargo xtask app package --channel stable` and launched with `cargo xtask app smoke-test` by the macOS packaging workflow. That runs weekly on main, on demand, and on pull requests that change the packaging pipeline.

## Version numbers

Astronomical follows pre-1.0 Semantic Versioning, beginning at `0.2.0`. The app's screens and `/v1/status` show the version, Stable or Development, the short Git commit, and a marker when a Development build has uncommitted changes.
