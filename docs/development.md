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

- **Kept apart:** settings, logs, astro SSD prompt caches, background processes, process locks and addresses. Each copy refuses a configured address that belongs to the other.
- **Shared:** both may read the same model folders, which they never write to. Both also share the Mac's graphics chip, memory and SSD, so running a real model in Development slows Stable down.
- **Tests use Development only.** Tests that run a model read model locations and policy from Development. Tests that write settings, astro SSD prompt caches or logs use a temporary `.astronomical-dev` folder and never `~/.astronomical`. The only exception is settings tests that build a temporary Stable fixture to prove the two stay apart.

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

## Building and testing on Linux

Linux support is in progress ([#2424](https://github.com/aosama/astronomical/issues/2424)). The whole workspace builds and its tests run on Linux with MLX's CPU backend. The macOS app and its packaging do not.

You need:

- CMake and a C and C++ compiler, which CMake finds as `cc` and `c++`. The native build identity records their versions.
- A BLAS and a LAPACK with the `cblas.h` and `lapacke.h` headers, for example OpenBLAS.
- libclang, which `mlx-bridge` uses to generate its bindings.

Then fetch the native dependencies, build and test:

    cargo xtask bootstrap-native-dependencies
    cargo xtask provision-bindgen-headers
    cargo check --workspace --all-targets
    cargo test --workspace

`provision-bindgen-headers` gives the MLX-C coverage contract the headers it checks the bindings against.

On Linux the build differs from macOS in these ways:

- **No Metal and no Metallib.** MLX runs on the CPU, and the runtime has no kernel archive to load or verify.
- **The BLAS that MLX's configure finds is the one that gets linked.** The build installs MLX's own CMake package and publishes its `share/cmake/MLX/MLXTargets.cmake` in the store entry; the Rust link line links the exact library files that file records, by name.
- **Pointing CMake at another BLAS builds a new native runtime.** If your BLAS is outside the system library folders, point CMake at it with its standard `CMAKE_PREFIX_PATH` and `CMAKE_INCLUDE_PATH`. Those settings, and the other BLAS search settings MLX's configure reads, are part of the native build identity. If a recorded library file disappears later, the link stops with a message naming it.
- **Run paths cover only `runtime-integration`'s own test binaries.** Other programs need a BLAS outside the system folders on the loader's path.
- **Only the host's own triple is built.** Cross-compiling to another architecture or C library is refused.
- **The memory-contract probe and the GPU working-set probe are macOS-only.** Both measure Metal.
- **Entries that cannot be linked faithfully are reported as build warnings, not linked.** These are linker flags, conditional generator expressions and CMake targets.
- **A static BLAS is linked, with a build warning.** Its own dependencies, such as a Fortran runtime, are not recorded by MLX and may need linking by hand.
- **The native build and its dependencies use the same folders under `~/Library/Caches/Astronomical` as on macOS.**

What runs differently on Linux:

- **MLX runs on its CPU stream.** The runtime uses the GPU stream wherever MLX has a GPU backend and the CPU stream when it has none.
- **Astronomical's custom Metal kernels are demoted.** Their capability probes fail on Linux, and the engine takes the public-MLX route each kernel already falls back to on a GPU that cannot run it. Tests of the kernels themselves print `skipped … needs a custom Metal kernel` and pass. Training decision-model adapters needs the gated-delta kernel, so it does not run on Linux yet.
- **Core ML and the Neural Engine report themselves unavailable.** Loading or compiling a program fails with a message saying so, and the device report finds no Neural Engine. Their tests print `skipped … needs Core ML`.
- **The expert record reader works as on macOS.** Only the direct reads that bypass the `macos_file_cache` are macOS-only.
- **The daemon's telemetry reads memory pressure from `/proc/pressure/memory`.** GPU utilization reads as unavailable.
- **The worker reads physical memory from `/proc/meminfo`.** It claims no wired residency, because that is a Metal claim, and serves from ordinary pageable memory.
- **Run tests with `--show-output`** to see which contracts were skipped and why.

## Running a model on Linux

The daemon and worker serve chat models on Linux with MLX's CPU backend. Build them optimized, because the CPU spends its time in MLX and in the worker's own loops, and debug builds of the latter are many times slower:

    cargo build --release -p astronomical-daemon -p astronomical-inference-worker

Put a model in its own folder under a models directory, named as the model is (for example `Qwen3.5-0.8B-MLX-4bit`, the smallest chat model in `registry/download_catalog.json`), and point an instance at that directory. A state directory holds the instance's `config.json`:

    {"schema_version":1,"runtime":{"model_directories":["/path/to/models"],"maximum_mlx_memory_gb":6}}

Start the daemon, and send a chat request to the address it prints:

    target/release/astronomicald --instance development --state-directory /path/to/state
    curl http://127.0.0.1:PORT/v1/chat/completions -H 'content-type: application/json' \
      -d '{"model":"Qwen3.5-0.8B-MLX-4bit","messages":[{"role":"user","content":"Who wrote the play Romeo and Juliet?"}],"max_tokens":60,"reasoning_effort":"none"}'

What to expect:

- **It is slow.** On MLX's CPU backend the 0.8B model reads and writes on the order of one token a second, so even a one-sentence answer takes minutes, most of them spent reading the chat template. `"reasoning_effort":"none"` keeps the model from spending its token budget thinking.
- **Prompts are read 64 tokens at a time by default** on builds without Metal, against 2,048 (4,096 for SSD-paged models) with Metal. The engine reads one chunk at load to measure its working buffers, and a larger chunk would keep a CPU loading for many minutes. `chunking.fixed_prompt_processing_chunk_size_tokens` in `config.json` sets another size, and a saved 2,048 written by an earlier Linux build is moved to the new default.
- **The daemon gives a model ten minutes to load**, against one minute with Metal, for the same reason.
- **Other models still compute in half precision on the CPU,** including their quantized projections, which MLX's CPU backend runs about ten times slower in BF16 and float16 than in float32 on CPUs without native half-precision arithmetic.
- **Qwen3.5 computes in float32 on the CPU.** MLX's CPU backend runs it in BF16 inaccurately enough to change its answers, so the resident engine binds every half-precision language tensor in float32 ([#2441](https://github.com/aosama/astronomical/issues/2441)). Its decoder state is float32 too, about twice the BF16 bytes, and memory admission counts it. The widened weights take more memory than the files on disk: every quantization scale and bias doubles, so a 4-bit weight carries an eighth of a byte of overhead instead of a sixteenth, and any unquantized BF16 tensor doubles. The ledger counts the bound bytes after load, but estimates made from the files before a load (such as the catalog's memory figure) are lower than what the model then uses. The vision tower stays BF16 and its rows are widened as they join the prompt; layers the streaming engine reads from storage bind as stored. The smallest model's answers then match MLX's own float32 reference; what remains wrong is the 0.8B model's own knowledge.
- **SSD-paged sparse models are not a Linux target yet.**
- **Nothing in MLX enforces the active memory ceiling on Linux.** The memory ledger still plans against it, and `/v1/status` reports `"active_memory_ceiling_enforcement": "unavailable"` in the ledger snapshot (it reads `"enforced_by_mlx"` on macOS). See [#2425](https://github.com/aosama/astronomical/issues/2425).
- **The bound is the machine's memory, lowered by any cgroup limit** (cgroup v2's `memory.max`, or the v1 memory controller's limit on older hosts) on the daemon, and `maximum_mlx_memory_gb` lowers it further.

## How the code is laid out

Every Rust crate sits in a folder named for its role. `cargo xtask check-architecture` holds the dependency edges that matter most:
- no crate under `crates/families/`, nor `engine-core`, `model-primitives`, `artifact-validation` or the `jangtq2-qwen35` weight format, depends on `model-serving`, the worker's serving loop;
- no family crate depends on another family of its kind. The family rules are read from the folders, not from a list of names, so a new family crate is held to them without editing the checker;
- `engine-core` never depends on the `qwen35` family;
- `memory-policy`, `model-discovery`, `model-family-registry` and every family's recognition crate stay free of the MLX bindings, because the daemon discovers models through them;
- `qwen35` never depends on `quantization`.

| Folder | What lives there |
|---|---|
| `crates/contracts/` | The messages programs exchange: the REST API types, the daemon-to-worker protocol, and the speech and decision request shapes. No model code and no MLX. |
| `crates/foundation/` | What everything builds on: settings (`config`), model discovery (`model-discovery`: the rules every family shares, naming no family, never linking MLX), memory policy, the one place that talks to MLX (`runtime-integration`), the MLX bindings, artifact checks, the native build tooling, the English text tools speech reads, and shared test fixtures (`test-harness`). |
| `crates/engine/` | The machinery every model family shares: the engine contract and owner thread, the `astro_ssd_prompt_cache`, speculative decoding and expert paging (`engine-core`), the shared model building blocks (`model-primitives`), expert placement records, and the worker's serving loop (`model-serving`). |
| `crates/hardware/` | The Neural Engine crate. Its hardware journeys run only by hand and in their own CI lane, so it is left out of the default build. |
| `crates/families/<kind>/` | The model families, one folder per kind: `text`, `image`, `speech` and `decision`. Each kind has a shell crate holding what its families share (`speech/speech`, `image/image-generation`, `decision/decision-models`), and each family gets a crate of its own beside it, as `speech/kokoro`, `image/flux2-klein` and `decision/laya` do. A building block two families share, such as the ModernBERT trunk under Laya and the ModernBERT embedding family, lives in the shell. A family crate depends on its shell, and a shell depends on no crate of its kind folder; that is how the checker tells them apart. A family crate holds only what makes its family different. Inside each family's folder, `recognition/` holds how discovery recognizes that family's folders: its recognizers, its folder checks and its description. Beside the kind folders, `registry/model-family-registry` lists every family discovery recognizes, in classification order, and offers discovery with that list passed in; `config` and the daemon discover through it. |
| `crates/weight-formats/` | Weight formats more than one crate reads. A format only one family uses lives inside that family. |
| `crates/tooling/` | Tools around the models rather than inside them: quantization and its probe, the vision tower, the download catalog, and decision-head training. |
| `apps/` | The programs: the daemon, the inference worker, the command line, and the two macOS apps (the menu-bar agent and the chat app). |
| `verification/` | Journeys that run real models or a whole program: acceptance, end-to-end and Neural Engine research. |
| `xtask/` | Repository tooling, run as `cargo xtask <command>`. |

The root `Cargo.toml` lists these folders by glob, so a new crate in any of them is a workspace member without an edit there. A crate declares the test lanes it runs in under `[package.metadata.astronomical] test-lanes`, and `cargo xtask lane-cargo` builds each lane's command from those declarations.

**Adding a model family.** A family's code lives in its own folder, `crates/families/<kind>/<family>/` (the text kind still holds its families in one crate):
1. Its engine, in the folder's crate beside its kind's shell.
2. Its recognition crate in the folder's `recognition/`: the family's constant, recognizers and description.
3. One line in the family list, `crates/families/registry/model-family-registry/src/model_families.list`. The line names the family's description, which discovery reads, and its builder, which the worker reads. Every kind's builder is the kind's own, built from the family's type (a chat family implements `ChatFamilyConstruction`, an image family `ImageFamilyConstruction`, and so on), so a family needs no worker code.
4. Cargo needs every crate named once as a dependency, so the new crates also get a line each in the registry's and `model-serving`'s manifests.

Its catalog entries go in `registry/download_catalog.json`. `check-architecture` holds the new crates to the family and recognition rules from their folders, with no edit. Every chat family today prepares the Qwen 3.5 request through the Qwen 3.5 processor; a chat family of another shape first needs the text shell crate issue #2533 plans.

## Check your change

Check Rust formatting, then run the Rust integration and REST suites:

    cargo fmt --all -- --check
    cargo test-integration-and-rest

Run the macOS menu contracts:

    cargo xtask menu-contracts

Tests that drive MLX directly or need a real model are run on request, because they need Apple graphics hardware or models on your Mac.

## The OpenCode plugin

`integrations/opencode` holds the OpenCode plugin that lists the app's chat models in OpenCode's model picker. It is a standalone npm package with its own lockfile and its own version number, because it follows OpenCode's plugin interface rather than the app's releases. It needs Node, at the version in `.node-version`:

    cd integrations/opencode
    npm ci
    npm run verify

`verify` checks formatting and types, and runs the tests; there is no build, because the package ships its TypeScript sources and OpenCode runs them directly. The tests stand in for OpenCode and for the app, so they need neither running. CI runs the same command on Linux whenever the plugin, `.node-version` or the shared model list changes; a change to the plugin alone skips the macOS lanes.

The tests read the app's `/v1/models` listing from `crates/contracts/rest-contract/fixtures/openai_model_list.json`. A test in `rest-contract` holds that file to what the contract types serialize. When you change a field the listing carries, that test fails and prints the new listing to paste into the file; the plugin's tests then fail until the plugin reads the new field. 

To try the plugin in a real OpenCode against the Development app, add this entry to `plugins` in your OpenCode configuration, with the folder's full path: `{ "package": "/path/to/integrations/opencode", "options": { "channels": ["development"] } }`. Development's models are listed as **Astronomical Development**. The plugin's [README](../integrations/opencode/README.md) describes what users see.

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
