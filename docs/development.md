# Building and developing Astronomical

For people who build Astronomical from source or contribute to it. Read [CONTRIBUTING.md](../CONTRIBUTING.md) first. To use the released app, see the [README](../README.md).

## Stable and Development instances

Astronomical keeps the trusted daily driver separate from repository development:

| Channel | State | REST API | Application |
| --- | --- | --- | --- |
| Stable | `~/.astronomical` | `127.0.0.1:6732` | `~/Applications/Astronomical.app` |
| Development | `~/.astronomical-dev` | `127.0.0.1:6733` | `target/astronomical-macos-development.noindex/Astronomical Development.app` |

Config, logs, prompt caches, daemon ownership, process locks, and loopback endpoints are isolated. A standard instance rejects a configured endpoint belonging to the other channel. Both configs may reference the same read-only model directories. Real-model development still shares the Mac's GPU, wired memory, and storage bandwidth with Stable.

Serving and acceptance tests read user-selected model locations and policy from Development only. Their mutable config, cache, and logging fixtures use temporary `.astronomical-dev` state and never `~/.astronomical`. Config boundary tests may construct temporary Stable fixtures solely to prove channel separation. Explicit app validation with `--real-model` uses the Development instance.

The configuration file each instance creates is described in [Configuration](configuration.md).

## Build the app

Required tools:

- Xcode with the macOS 26 software development kit.
- CMake.
- Rust nightly-2026-11-07 (rustc 1.101.0-nightly) through the checked-in toolchain file.
- Swift 6.2.
- Homebrew coreutils, jq, curl, and sccache for repository verification and application validation.

Provision the pinned, checksum-verified native dependencies:

    cargo xtask bootstrap-native-dependencies

Build the Development app and open it, without touching Stable:

    cargo xtask app dev

The first run compiles everything; later runs rebuild incrementally and replace the running Development app in seconds. A failed build leaves the running app untouched. Control the Development instance headless, for example to start its daemon with exported `ASTRONOMICAL_*` experiment flags:

    cargo xtask app instance status|start|stop|restart

The ad-hoc signed local bundle is written to `target/astronomical-macos-development.noindex/Astronomical Development.app`. The `.noindex` build directory keeps generated app bundles out of Spotlight while preserving them for validation and direct launch. Stable uses the clean product icon, while Development uses a restrained channel badge. To rehearse a launch of any built bundle in isolation, run `cargo xtask app smoke-test --app-bundle PATH`; add `--real-model MODEL_ID` to send one prompt through an advertised model, only when shared GPU pressure is acceptable.

Stable installation and release publication are release-only commands (`cargo xtask release ...` and `scripts/release/`). They are never part of ordinary commit or push verification. Build a clean Stable candidate and explicitly promote it outside the repository build tree with one release-only command:

    cargo xtask release build-and-install-stable

Use `--dry-run` to build and validate Stable while previewing, rather than performing, the installation. `cargo xtask app package --channel stable` and `cargo xtask release install-stable` remain available when the two stages need to run independently. Installing moves the previously installed app to the Trash rather than deleting it. The packaging contracts run on every pull request: the release steps' decisions through `cargo test -p xtask`, the typed packaging contracts as `cargo test-app-builder`, and the icon renderer contract as `swift scripts/internal/render-macos-app-icon-contract.swift`. A real Stable bundle is built with `cargo xtask app package --channel stable` and launched with `cargo xtask app smoke-test` by the macOS packaging workflow, weekly on main, on demand, and on pull requests that change the packaging pipeline.

Stable candidate builds perform signature, resource, metadata, and bundled-daemon validation without launching over a running Stable instance. Promotion does not restart the running app. Stable builds require a clean Git worktree.

Astronomical follows pre-1.0 Semantic Versioning beginning at `0.2.0`. User interfaces and `/v1/status` show the semantic version, Stable or Development channel, short Git commit, and dirty-development marker.

To build only the optimized daemon and worker:

    CARGO_BUILD_JOBS="$(sysctl -n hw.logicalcpu)" cargo build --release \
      -p astronomical-inference-worker --bin astronomical-inference-worker \
      -p astronomical-daemon --bin astronomicald

## Verification

Check Rust formatting, then run the Rust integration and REST suites:

    cargo fmt --all -- --check
    cargo test-integration-and-rest

Run the macOS menu contracts:

    cargo xtask menu-contracts

Direct MLX and real-model acceptance lanes remain explicit because they require Apple graphics hardware or local model artifacts.
