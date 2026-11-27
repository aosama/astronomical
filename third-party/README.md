# Native Third-Party Dependency Manifest

This directory owns Astronomical's native (C/C++) dependency tree: the
immutable version pins in `pins/`, the local patch series in `patches/`, and
the manifest consumed by the build in `native-dependency-manifest.cmake`. It
is the single place a maintainer can read what is pinned, why each patch
exists, how the digests are verified, and how to bump or reproduce a build.
The files under `pins/` are canonical; this document explains them.

## Inventory

Every native dependency enters the build only through a pin file that names
its version, upstream Git commit, source archive URL, and SHA-256. The build
never resolves an upstream repository live.

| Dependency | Pinned version | Upstream ref | Pin file | Class |
| --- | --- | --- | --- | --- |
| MLX | 0.32.5 | release tag `v0.32.5` (commit `cfc675e9e`) | `pins/mlx-v0.32.5.cmake` | core runtime |
| MLX-C | 0.8.0 | release tag `v0.8.0` (commit `a26f2d584`) | `pins/mlx-c-v0.8.0.cmake` | C-ABI bridge |
| metal-cpp | 26 | Apple release download `metal-cpp_26.zip` | declared inside `pins/mlx-v0.32.5.cmake` | MLX transitive |
| nlohmann/json | 3.11.3 | release artifact `v3.11.3/json.tar.xz` | declared inside `pins/mlx-v0.32.5.cmake` | MLX transitive |
| fmt | 12.1.0 | release tag archive `12.1.0.tar.gz` | declared inside `pins/mlx-v0.32.5.cmake` | MLX transitive |
| MLX performance profiler | 0.7.0 | release tag `v0.7.0` (commit `0425dc05e`) | `pins/mlx-performance-profiler-v0.7.0.cmake` | measurement tooling, not shipped |
| CCCL | 3.1.3 | release asset `v3.1.3/cccl-v3.1.3.zip` | `pins/mlx-cuda-sources-v0.32.5.cmake` | MLX CUDA transitive |
| NVTX | 3.1.1 | tag `v3.1.1` (commit `6230bdf71`) | `pins/mlx-cuda-sources-v0.32.5.cmake` | MLX CUDA transitive |
| cuDNN frontend | 1.16.0 | tag `v1.16.0` (commit `be6c079be`) | `pins/mlx-cuda-sources-v0.32.5.cmake` | MLX CUDA transitive |
| CUTLASS | 4.4.2 | tag `v4.4.2` (commit `da5e086da`) | `pins/mlx-cuda-sources-v0.32.5.cmake` | MLX CUDA transitive |

metal-cpp, nlohmann/json, and fmt are source dependencies that MLX's own
CMake declares through FetchContent, and so are CCCL, NVTX, the cuDNN
frontend and CUTLASS for its CUDA backend, which only the Linux CUDA recipe
extracts (issue #2699; every platform provisions them so one manifest serves
every machine). Astronomical pre-declares them with
verified local archives; FetchContent's first-declaration-wins rule means
MLX's nested requests are always satisfied from the pinned digests instead of
the network.

The pin files are canonical for digests and URLs; the current values are
recorded here for review:

- MLX `v0.32.5`: `https://github.com/ml-explore/mlx/archive/cfc675e9e59cb6ee29a647cf8fe87ef844eff879.tar.gz` — SHA-256 `932473c50f96f776b60cb75914840c3e472c412558c3c9cb95c8c6fe87a93c44`
- MLX-C `v0.8.0`: `https://github.com/ml-explore/mlx-c/archive/a26f2d584a14ec619ae8db9a09be03becd47598a.tar.gz` — SHA-256 `b2cd55408fb15e5cf44c622355ff3445b58fb835f691db07d620047dff24548c`
- metal-cpp `26`: `https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip` — SHA-256 `4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4`
- nlohmann/json `3.11.3`: `https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz` — SHA-256 `d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d`
- fmt `12.1.0`: `https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz` — SHA-256 `ea7de4299689e12b6dddd392f9896f08fb0777ac7168897a244a6d6085043fea`
- MLX performance profiler `0.7.0`: `https://github.com/aosama/mlx-performance-profiler/releases/download/v0.7.0/mlx-performance-profiler-0.7.0.tar.gz` — SHA-256 `3466770490611f4131cda49e1568c58ed73fffd2162548c251adffdaf7040233`
- CCCL `3.1.3`: `https://github.com/NVIDIA/cccl/releases/download/v3.1.3/cccl-v3.1.3.zip` — SHA-256 `30f388ef784eb691d7de9d2cf918d53ab33464672500a17945b99d16af136cd6`
- NVTX `3.1.1`: `https://github.com/NVIDIA/NVTX/archive/6230bdf710bc94f44d433acceba735aaa9090ba5.tar.gz` — SHA-256 `1c515cf0e6c05597b62f3b3a190f7a759446597aa88a27afdc08d1730a905e29`
- cuDNN frontend `1.16.0`: `https://github.com/NVIDIA/cudnn-frontend/archive/be6c079be8aaffa0fc079fcf039887e637c289c7.tar.gz` — SHA-256 `bf3eef8b07394a40f73a510f6ce1d1deb11de253c5cff46a71adc9ab6fee3b00`
- CUTLASS `4.4.2`: `https://github.com/NVIDIA/cutlass/archive/da5e086dab31d63815acafdac9a9c5893b1c69e2.tar.gz` — SHA-256 `7f735e3e8cf03ef3d898edf7a7329f92b5cf7bc5bb9049a0299f524d43d37919`

Tooling pins (not part of the native runtime image): `tool-pins/Cargo.toml`
pins `sccache 0.19.0` and `cargo-about 0.9.3`, the latter for the
license-notice pipeline driven by `about.toml`. The pins are exact `=X.Y.Z`
dependencies of a crate that is never built, so Dependabot opens weekly bump
pull requests for them. `RUST_DEPENDENCY_NOTICES` and
`THIRD_PARTY_NOTICES` are its generated outputs.

## Upstream tracking and offsets

Every pin sits on an official upstream release artifact. There are no
development-commit pins, and no undocumented versions exist anywhere in the
build path: the pin files are the only place a version, URL, or digest
appears, and every other build file consumes them through CMake variables.

No MLX-C release pairs with the pinned MLX v0.32.5 yet, so MLX-C stays at
v0.8.0, the release authored against MLX v0.32.4, and carries two flagged
compatibility bridge patches (`attention-window-size-for-mlx-0.32.5` and
`include-format-in-io-header`, in the registry below). Each is deleted at the
MLX-C release that makes it unnecessary.

## Patch registry

Patches apply in the order declared by the `PATCH_COMMAND` list in
`crates/foundation/runtime-integration/native/CMakeLists.txt`, each through
`apply_patch_if_needed.cmake`, which fails the configure loudly when a patch
no longer applies to the pinned archive. Every patch below is reviewed at
each pin bump and retired the moment the pinned upstream tree contains the
equivalent behavior; each patch's rationale and evidence are
recorded inline in this registry below.

MLX patches (`mlx-0.32.5-*.patch`):

- `astronomical-active-memory-ceiling` — Astronomical feature. Gives the
  Metal allocator an enforced active-memory ceiling with a distinct error
  marker (`ASTRONOMICAL_MLX_ACTIVE_MEMORY_LIMIT_EXCEEDED`) so the Rust memory
  manager can treat limit exhaustion separately from ordinary allocation
  failure. It also moves the allocator's cache-collection line from a
  fixed 95% of Metal's recommended working set to the configured limit
  (never above the recommended working set), because a limit above 95%
  otherwise drains the buffer cache on every allocation and a large model
  replies at about a third of its speed. Retires never; it is serving
  behavior upstream does not offer.
- `streaming-safetensors-writer` — Astronomical feature. Streams safetensors
  output through the bounded descriptor writer and orders payloads so the
  allocator can reuse one contiguous materialization. Retires never.
- `fast-synch-setting` — upstream defect fix. MLX orders a CPU step among
  GPU steps with a fence, and has two kinds: Metal shared events, or a
  spin-wait on shared memory it calls fast synch. A host step inside a graph
  (`crates/engine/graph-host-step`) costs about 9 microseconds a layer with
  fast synch and about 170 without (`cargo measure-graph-host-step`). The
  patch ends a fast-synch wait when the command buffer that should have
  published the awaited value fails: upstream's CPU stream would spin on it
  forever, where the patched one lets the evaluation report the failure.
  Through v0.32.4 the patch also made the choice a setting code could change;
  that half retired at the v0.32.5 bump, where MLX's `config` module offers
  it (`mlx::core::config::update("MLX_METAL_FAST_SYNCH", 1)`, read each time
  an evaluation makes its fences), and `graph-host-step` calls that. Retires
  when upstream ends that wait itself.
- `evaluation-counts` — Astronomical feature. Counts, since the process
  started, the evaluations that walked a graph and waited, the ones that
  returned without waiting, the waits for arrays already handed to an
  earlier evaluation, and the times one stream waited inside an evaluation
  for another's output, and the primitives the evaluator ran on the GPU and
  on the CPU stream (`mlx::core::evaluation_counts`). Each is a hand-off
  between the caller, the GPU and the CPU stream, and a reply token's speed
  depends on how many it makes; the engine reports them a token
  (`crates/engine/graph-host-step`, `evaluation_counts`). Counted inside MLX
  because a host readback evaluates there too, where no caller can see it.
  It also counts the command buffers the GPU finished and the time it spent
  executing them, from Metal's own timestamps (recorded in the Metal
  backend's completion handler), so a span's wall time can be split into
  the GPU working and the GPU waiting.
  Retires if upstream reports the same counts.
- `cuda-jit-header-directories` — Astronomical feature (issue #2699). MLX
  compiles some CUDA kernels at run time with NVRTC (its quantized matrix
  multiplies among them) and looks for their CUTLASS, CuTe and CCCL headers
  in the `include/` folder beside the binary MLX lives in, and for the CUDA
  runtime headers through `CUDA_HOME`, `CUDA_PATH` or `/usr/local/cuda`
  (`jit_module.cpp`, `include_path_args`). Astronomical links MLX
  statically, so that binary is the worker and nothing is beside it; every
  such kernel then fails with `cannot open source file
  "cute/numeric/numeric_types.hpp"`. The patch adds
  `mlx::core::cu::set_jit_header_directories`, which names both folders
  before the first compile (a no-op in a build without CUDA, through
  `no_cuda.cpp`); the native build publishes the headers into its store
  entry and the runtime passes the folders at start-up. Retires when
  upstream lets a program name those folders.
- `stable-prefill-split-k` — serving performance. Keeps the quantized matmul
  split-K topology stable once row tiles already expose parallelism, so
  prefill numerics do not change with batch shape. Retires if upstream
  adopts an equivalent stability rule for the split-K heuristic.
- `nax-qmm-tile-variants` — retired at the v0.32.5 bump. It selected the
  transposed-prefill tile from the token count, a `(64, 64, 64, 4, 1)` tile
  in `(bm, bn, bk, wm, wn)` order above 32 rows. Upstream's staged
  `qmm_t_nax` kernel (`use_qmm_t_nax_staged` in `quantized.cpp`) now selects
  a 64-row, four-simdgroup tile itself, and its own split-K rule for that
  kernel replaced the code the patch edited. The retirement was verified by
  re-diffing the patch against the v0.32.5 archive, where its dispatch hunk
  no longer has a target, and by the prompt-reading figures recorded on the
  bump's pull request.
- `jit-qmv-template-arity` — retired at the v0.32.4 bump. Upstream's
  partial-row qmv fast-kernel restructure (the `partial_rows` template
  parameter and the unconditional `results_per_simdgroup` arity) declares
  the full template arity on every `qmv`/`qmv_fast` template, which is the
  defect the patch carried; the retirement was verified by inspecting the
  template declarations in the v0.32.4 archive and compiling without the
  patch.
- `jit-sdpa-dsplit-template-arity` — retired at the v0.32.5 bump.
  `get_steel_attention_nax_kernel` forwarded the `bv` template argument to
  `attention_nax_dsplit`, whose template has no `BV` parameter, so every
  head-dimension 256 or 512 attention failed Metal JIT compilation. Upstream
  fixed the generator the same way (it now builds the `attention_nax_dsplit`
  definition without `bv`); the retirement was verified by reading the
  generator in the v0.32.5 archive and by the wide-head-dimension attention
  test in
  `verification/acceptance/tests/gpu_contracts/direct_mlx/attention/masked_attention.rs`
  passing without the patch.

MLX-C patches (`mlx-c-0.8.0-*.patch`):

- `expose-array-detach` — Astronomical feature. Adds `mlx_array_detach`, which
  drops an array's graph and siblings while keeping its realized buffer, so
  long-lived weights do not hold a whole graph alive. The Rust declaration
  lives in the `mlx-c-fork` crate. Retires when MLX-C exposes an
  equivalent.

- `attention-window-size-for-mlx-0.32.5` — compatibility bridge. MLX v0.32.5
  inserted `int window_size = -1` between `scale` and `mask_mode` in
  `fast::scaled_dot_product_attention`, for sliding window attention, so
  MLX-C v0.8.0's positional call hands a string to an integer parameter and
  stops compiling. The patch passes MLX's own default, which keeps
  `mlx_fast_scaled_dot_product_attention` at its v0.8.0 signature and its
  unwindowed behavior. The published `mlx-bridge` crate carries the same
  edit for its own native build. Retires at the MLX-C release that forwards
  `window_size` itself.

- `include-format-in-io-header` — compatibility bridge. MLX-C v0.8.0's
  `mlx/c/private/io.h` calls `std::format` but includes only `<iostream>`,
  `<streambuf>` and `mlx/mlx.h`. Against MLX v0.32.4 that compiled because
  MLX's `mlx/random.h` included `<chrono>`, which in libstdc++ brings in
  `<format>`; MLX v0.32.5's `mlx/random.h` no longer does, so GCC on Linux
  stops with `'format' is not a member of 'std'` (issue #2682). The patch adds
  `#include <format>`. The macOS build compiles against MLX v0.32.5 without
  it, so with libc++ the header was already reachable and the patch changes
  nothing there. Retires at the MLX-C release whose `io.h` includes
  `<format>` itself.

- `cuda-runtime-settings` — Astronomical feature (issue #2699). Adds
  `mlx_cuda_set_jit_header_directories`, the C side of the MLX patch above,
  and `mlx_set_tf32_enabled`, which sets MLX's process-wide
  `MLX_ENABLE_TF32` through `mlx::core::config::update` (Metal's NAX
  kernels read it too, but only CUDA builds set it). MLX's CUDA backend runs float32
  matrix multiplies on TF32 tensor cores by default; on Laya's projections
  that made the largest difference from a float32 product 200 to 1,000
  times larger, and Astronomical never lowers a model's precision, so the
  runtime turns it off at start-up instead of through an environment
  variable. The Rust declarations live in the `mlx-c-fork` crate. Retires
  when MLX-C exposes both settings.

- `gather-qmm-global-scale-compatibility` — retired at the v0.8.0 bump. MLX
  carries, since v0.32.3, a `global_scale` parameter in `gather_qmm` before
  `sorted_indices`; MLX-C v0.7.0 did not expose it, so the patch forwarded the
  upstream default. The retirement was verified by reading `mlx_gather_qmm` in
  the v0.8.0 archive, which declares `global_scale` as an optional argument,
  and by the quantized gather contracts in the direct-MLX lane passing without
  the patch.

## Pinning and digest verification, end to end

1. A pin file in `pins/` records the version, upstream repository and commit,
   archive URL, archive file name, and SHA-256. Pins are immutable: a bump
   replaces the file, it does not edit around it.
2. `cargo xtask bootstrap-native-dependencies` downloads each archive by URL,
   verifies it against the pinned SHA-256, and provisions the platform cache
   directory. The Astronomical build itself never fetches source.
3. At every CMake configure,
   `crates/foundation/runtime-integration/native/CMakeLists.txt` re-verifies each
   archive digest through `require_pinned_archive()` before extraction, so a
   tampered or truncated cache fails the build, not the model.
4. `FetchContent_Declare` consumes only the verified local archive with
   `URL_HASH`, and the pre-declared transitive archives (metal-cpp, json,
   fmt) satisfy MLX's nested FetchContent requests without network access.
5. `third-party/native-dependency-manifest.cmake` writes the resolved
   inventory (file name, URL, digest, description) to a manifest file for
   build-time auditability.

Archive paths are always the pinned filenames under the provisioned dependency cache, selected with `cargo xtask bootstrap-native-dependencies --cache-dir ABSOLUTE_PATH` and `ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR` when a nondefault cache is needed. Per-archive environment overrides are not read. The `ASTRONOMICAL_*_SOURCE_ARCHIVE` CMake variables are internal resolved paths, not configuration inputs; inherited environment entries and stale CMake cache definitions cannot redirect them. Both the Cargo build script and standalone native-build tool use this same resolution and native build identity.

## Bump-and-release process

The reconciled-version policy: both MLX and MLX-C track the latest upstream
release tags. Development-commit pins are not used. When MLX-C lags the pinned
MLX, a compatibility bridge patch is the sanctioned, flagged way to span the
gap; it is deleted at the next MLX-C bump.

To bump a pinned dependency:

1. Download the candidate release archive and compute its SHA-256.
2. Replace the pin file under `pins/` with the new commit, URL, and digest.
3. Run `cargo xtask bootstrap-native-dependencies`, then
   `cargo xtask prewarm-native-build --profile core`. Every patch must apply
   cleanly; a failure means the patch set must be reconciled with the new
   tree. Re-diff each patch against the new archive before deciding it is
   superseded — never judge by release notes alone. For the two JIT
   template-arity patches, also run the direct-MLX attention lane: the
   wide-head-dimension test there exercises the JIT generator paths in
   seconds.
4. Reconcile the Rust FFI surface: the bridge is the published
   `mlx-bridge` crate (crates.io), which bindgen-generates from its own
   vendored stock MLX-C headers at build time. Bump the workspace
   `mlx-bridge` dependency to the release matching the new pin, reconcile
   call sites against its surface, and check the fork delta: the pinned
   patch set adds three C symbols (`mlx_array_detach`,
   `mlx_cuda_set_jit_header_directories` and `mlx_set_tf32_enabled`, owned
   by `crates/foundation/mlx-c-fork`); a new patch that adds more symbols must
   extend that crate and its `FORK_BRIDGED_FUNCTIONS`. The coverage contract
   (run by `cargo xtask mlx-c-coverage-inventory`) fails on any
   unbridged, stale, or unjustified symbol across both crates. Before the
   first native build of the new pin, diff the C header surface itself with
   the bindgen header provisioning (see the next section); header-level
   differences found there explain almost every binding-level difference
   before any compile runs. Since the crate's 0.32.4 line, its build script
   can also emit its own link flags through a `link` feature's discovery
   ladder, whose default rung resolves the crate's own stock shared pair —
   the wrong libraries for a patched fork. The workspace therefore depends
   bindings-only (`default-features = false`): the crate emits zero link
   directives and runtime-integration owns the linker end to end. That mode
   arrived in the crate's 0.32.4-alpha.2 release (upstream
   aosama/mlx-bridge#19); keep the dependency bindings-only.
   One lockfile property rides along with the crate: its build dependency
   bindgen declares a wildcard `syn = ">=2, <4"` range but only compiles
   against the syn 3 line, so the lockfile pairs bindgen with `syn 3.0.3`.
   A broad `cargo update` can silently re-resolve that edge onto the syn 2
   entry already in the graph and fail deep inside bindgen; a crate bump
   that turns the edge must set it back.
 5. Run the integration lanes, then invoke each real-model acceptance Cargo
   command directly with `--test-threads=1` (GPU journeys are never parallel).
6. Prove performance parity for the `resident_sparse_moe` registry model
   with `cargo xtask measure-speed --reference-set` and compare against the
   release's ledger under `registry/release_speed`.
7. Update this registry and the repo discovery guide, then commit
   after `cargo fmt --all -- --check` and
   `cargo test-integration-and-rest` pass.

## Bindgen headers and the C surface diff

The MLX-C binding surface spans two crates. The published `mlx-bridge`
crate (crates.io, versioned against the MLX release line its bindings
target — the 0.32.5 line matches the pinned MLX v0.32.5; the earlier
0.7.0-alpha spelling tracked the MLX-C release and was yanked upstream)
owns the
raw bindgen declarations over its vendored stock headers, the
captured-error translation, the owned handle types, the per-worker bindings
context, and every operation wrapper family; it never invokes CMake.
`crates/foundation/mlx-c-fork` owns the Astronomical patch delta over that stock
surface — today `mlx_array_detach` and the two CUDA run-time settings.
`crates/foundation/runtime-integration` owns the Astronomical policy over that
boundary — memory-limit enforcement, metallib resolution and verification,
the safetensors vtable policies, and native build orchestration — and links
both bridge crates against the pinned patched native image.

`cargo xtask provision-bindgen-headers` extracts the pinned MLX and MLX-C
archives from the verified native dependency cache and applies the same patch
pipeline as the native build (the patch list is parsed from the native
CMakeLists, so the two can never drift). The extraction is keyed by the
source-only native build identity, so a pin or patch edit invalidates it
automatically, and a second run re-verifies the published tree against its
recorded hash manifests without re-extracting. The script never downloads;
run `cargo xtask bootstrap-native-dependencies` first.

```bash
cargo xtask provision-bindgen-headers
cargo xtask provision-bindgen-headers --verify-headers
```

`--verify-headers` resolves the completed native build for the current
identity in the native build store and proves the extracted MLX-C headers are
byte-identical to the staged headers of that build (`include/mlx` in the store
entry); any mismatch is reported per file. This is the fast surface-diff step
of a dependency bump: within seconds of changing a pin — and before any
CMake build — you have the header tree the new pin will produce, so the
previous pin's extraction directory can be diffed directory against directory
to enumerate the exact C surface change. After the new pin's first native
build, `--verify-headers` closes the loop by proving the extraction matches
what the build actually staged.

The `bindgen_headers` tests in the xtask crate (`cargo test -p xtask --bin xtask bindgen_headers`)
cover extraction, patch application, idempotent reuse, self-healing,
tampered-archive refusal, no-network behavior, and verification against local
store fixtures.

## Native code written against MLX's C++

Astronomical's own C++ that has to run inside MLX, such as a primitive the
evaluator calls while it walks a graph, lives in a crate of its own and
compiles against the provisioned extraction above, never against a copy of
MLX's sources and never as a target of the native MLX build. The extraction is
the pinned archives with this repository's patches applied, so the headers are
the ones the linked MLX image was built from; compiling in the crate, with
`cc`, means an edit rebuilds one file and leaves the native build identity
alone.

`crates/engine/graph-host-step` is the first such crate: a step of a graph
that runs on MLX's CPU stream between two GPU steps. Its build script finds
the extraction through `native_build_orchestration::provisioned_mlx_sources`
and fails with the provisioning command when it is missing. A hook such code
needs inside MLX itself is a patch in `patches/`, recorded in the registry
above like any other.

## The MLX performance profiler

`crates/tooling/mlx-performance-profiler` compiles the pinned MLX performance
profiler (Apache-2.0) the same way: with `cc`, from its digest-verified
release archive, against the provisioned extraction, so it never links a
second MLX. It is measurement tooling: only the acceptance crate's
measurement tests depend on it, and nothing of it enters the worker or the
app bundle, which is why it has no entry in `THIRD_PARTY_NOTICES`.

It measures one MLX job node by node, and compares two implementations of
one job over twelve interleaved rounds with a verdict
(`cargo measure-register-order-verdict` is the first journey built on it).
Every page it writes opens the job's execution graph as an interactive view:
each operation coloured by where its time went, a hot path, and a large job
shown as its stages. Since 0.7.0 each report also says, for every operation
(or stage), whether moving its data or doing its arithmetic held it back,
against reading and arithmetic ceilings it measures on the machine with
MLX's own kernels. The viewer is embedded in the page, so a report is one
HTML file that opens offline. Its release archive vendors Cytoscape.js and
cytoscape-dagre (MIT) for that view; the crate's build embeds them with the
profiler's own CMake script, run in script mode, and they travel only inside
the pages a measurement writes.
Beyond MLX's public API it relies on the behaviours its README lists
(the Metal command encoder's command buffer, `array::is_available`,
`Primitive::name` and `stream`, MLX-C's array handle layout,
`CustomKernel`'s state holding the kernel's name, and, since 0.7.0, the
input order of the matrix-multiply primitives and
`ScaledDotProductAttention`'s state saying whether it is causal); none of the
patches above touches them, and a patch that would has to be checked against
that list.

To bump it: replace its pin file with the new release's archive and digest
(the build finds the pin by the variables it sets, not by its file name),
bring the source list and the embedded asset list in the crate's `build.rs`
in line with the release's `CMakeLists.txt`, and run the crate's tests.

## Reproducing a native build from scratch

Prerequisites: macOS on Apple Silicon, Xcode command line tools, Rust
toolchain, CMake 3.24 or newer, and `patch`.

```bash
cargo xtask bootstrap-native-dependencies
cargo xtask prewarm-native-build --profile core
cargo build -p astronomical-inference-worker
```

The first command provisions and verifies every pinned archive into the
platform cache. The second configures and compiles the native runtime image
(static MLX, MLX-C, and the Astronomical bridges) and publishes it to the
content-addressed native build store. The third links the worker against the
published store; if the store is empty, the cargo artifact lifecycle builds
it first. Verification afterwards needs no network and no model
weights:

```bash
cargo test -p runtime-integration --test integration_tests
```
