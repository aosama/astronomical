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
| MLX | 0.32.3 | release tag `v0.32.3` (commit `64ea011cb`) | `pins/mlx-v0.32.3.cmake` | core runtime |
| MLX-C | 0.7.0 | release tag `v0.7.0` (commit `a341b4925`) | `pins/mlx-c-v0.7.0.cmake` | C-ABI bridge |
| metal-cpp | 26 | Apple release download `metal-cpp_26.zip` | declared inside `pins/mlx-v0.32.3.cmake` | MLX transitive |
| nlohmann/json | 3.11.3 | release artifact `v3.11.3/json.tar.xz` | declared inside `pins/mlx-v0.32.3.cmake` | MLX transitive |
| fmt | 12.1.0 | release tag archive `12.1.0.tar.gz` | declared inside `pins/mlx-v0.32.3.cmake` | MLX transitive |

metal-cpp, nlohmann/json, and fmt are source dependencies that MLX's own
CMake declares through FetchContent. Astronomical pre-declares them with
verified local archives; FetchContent's first-declaration-wins rule means
MLX's nested requests are always satisfied from the pinned digests instead of
the network.

The pin files are canonical for digests and URLs; the current values are
recorded here for review:

- MLX `v0.32.3`: `https://github.com/ml-explore/mlx/archive/64ea011cb65f14d9ce2737e60db9a4ae91ed7441.tar.gz` — SHA-256 `428070f9b74ab39b5f65ae90a0a409c14b1a7a75911d041864788bfbdb9f7ef0`
- MLX-C `v0.7.0`: `https://github.com/ml-explore/mlx-c/archive/a341b4925024b88b2c593468f16e12f5e17315da.tar.gz` — SHA-256 `4424dd3f6225708d111b691be4041bba9bc6da09719ae8d32e6900744852c7f6`
- metal-cpp `26`: `https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip` — SHA-256 `4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4`
- nlohmann/json `3.11.3`: `https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz` — SHA-256 `d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d`
- fmt `12.1.0`: `https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz` — SHA-256 `ea7de4299689e12b6dddd392f9896f08fb0777ac7168897a244a6d6085043fea`

Tooling pins (not part of the native runtime image): `sccache-version` pins
`sccache 0.17.0` and `cargo-about-version` pins `cargo-about 0.9.2` for the
license-notice pipeline driven by `about.toml`. `RUST_DEPENDENCY_NOTICES` and
`THIRD_PARTY_NOTICES` are its generated outputs.

## Upstream tracking and offsets

Every pin sits on an official upstream release artifact. There are no
development-commit pins, and no undocumented versions exist anywhere in the
build path: the pin files are the only place a version, URL, or digest
appears, and every other build file consumes them through CMake variables.

The one intentional offset in the tree is between the two Astronomical-facing
pins themselves: MLX-C v0.7.0 was authored against MLX v0.32.2, one minor
release behind the pinned MLX v0.32.3. That gap is bridged by the single
compatibility patch listed below and is re-evaluated at every MLX-C bump.

## Patch registry

Patches apply in the order declared by the `PATCH_COMMAND` list in
`crates/runtime-integration/native/CMakeLists.txt`, each through
`apply_patch_if_needed.cmake`, which fails the configure loudly when a patch
no longer applies to the pinned archive. Every patch below is reviewed at
each pin bump and retired the moment the pinned upstream tree contains the
equivalent behavior; decisions and evidence are recorded in
`docs/performance-optimizations-lessons.md`.

MLX patches (`mlx-0.32.3-*.patch`):

- `astronomical-active-memory-ceiling` — Astronomical feature. Gives the
  Metal allocator an enforced active-memory ceiling with a distinct error
  marker (`ASTRONOMICAL_MLX_ACTIVE_MEMORY_LIMIT_EXCEEDED`) so the Rust memory
  manager can treat limit exhaustion separately from ordinary allocation
  failure. Retires never; it is serving behavior upstream does not offer.
- `streaming-safetensors-writer` — Astronomical feature. Streams safetensors
  output through the bounded descriptor writer and orders payloads so the
  allocator can reuse one contiguous materialization. Retires never.
- `stable-prefill-split-k` — serving performance. Keeps the quantized matmul
  split-K topology stable once row tiles already expose parallelism, so
  prefill numerics do not change with batch shape. Retires if upstream
  adopts an equivalent stability rule for the split-K heuristic.
- `nax-qmm-tile-variants` — serving performance. Adds transposed-prefill
  tile candidates for the NAX quantized matmul, ported from the OMLX
  project's Qwen3.5 prefill kernels. Retires if upstream grows an equivalent
  shape-driven tile set.
- `jit-qmv-template-arity` — upstream defect fix (JIT builds only). The
  quantized `qmv` JIT generator forwards one template argument more than the
  kernel templates accept, which aborts at runtime. Retires when an MLX
  release passes the correct arity; re-check at every bump by removing the
  patch and compiling.
- `jit-sdpa-dsplit-template-arity` — upstream defect fix (JIT builds only,
  new at the v0.32.3 tag). `get_steel_attention_nax_kernel` forwards the
  `bv` template argument to `attention_nax_dsplit`, whose template has no
  `BV` parameter (the head dimension is split across `WN` groups). The
  shifted arguments make every head-dimension 256 or 512 SDPA fail Metal JIT
  compilation with a misleading overload-resolution error. The prebuilt
  metallib path upstream ships by default never executes this generator,
  which is why the defect reached a release tag. Retires when an MLX release
  fixes the generator; the wide-head-dimension attention test in
  `crates/model-serving/tests/direct_mlx/attention/masked_attention.rs`
  fails within seconds if the patch is dropped while the defect persists.

MLX-C patches (`mlx-c-0.7.0-*.patch`):

- `mlx-0.32.3-gather-qmm-global-scale-compatibility` — the only compatibility
  bridge in the tree. MLX v0.32.3 inserts a `global_scale` parameter into
  `gather_qmm` before `sorted_indices`; the MLX-C v0.7.0 C API does not
  expose it, so the patch forwards the upstream default. Retires at the
  first MLX-C release authored against MLX 0.32.3 or newer.

## Pinning and digest verification, end to end

1. A pin file in `pins/` records the version, upstream repository and commit,
   archive URL, archive file name, and SHA-256. Pins are immutable: a bump
   replaces the file, it does not edit around it.
2. `scripts/bootstrap-native-dependencies.sh` downloads each archive by URL,
   verifies it against the pinned SHA-256, and provisions the platform cache
   directory. The Astronomical build itself never fetches source.
3. At every CMake configure,
   `crates/runtime-integration/native/CMakeLists.txt` re-verifies each
   archive digest through `require_pinned_archive()` before extraction, so a
   tampered or truncated cache fails the build, not the model.
4. `FetchContent_Declare` consumes only the verified local archive with
   `URL_HASH`, and the pre-declared transitive archives (metal-cpp, json,
   fmt) satisfy MLX's nested FetchContent requests without network access.
5. `third-party/native-dependency-manifest.cmake` writes the resolved
   inventory (file name, URL, digest, description) to a manifest file for
   build-time auditability.

## Bump-and-release process

The reconciled-version policy: both MLX and MLX-C track the latest upstream
release tags. Development-commit pins are not used. When MLX-C lags MLX by
one release, the gather-qmm bridge patch is the sanctioned, flagged bridge;
it is deleted at the next MLX-C bump.

To bump a pinned dependency:

1. Download the candidate release archive and compute its SHA-256.
2. Replace the pin file under `pins/` with the new commit, URL, and digest.
3. Run `scripts/bootstrap-native-dependencies.sh`, then
   `scripts/prewarm-native-build.sh --profile core`. Every patch must apply
   cleanly; a failure means the patch set must be reconciled with the new
   tree. Re-diff each patch against the new archive before deciding it is
   superseded — never judge by release notes alone. For the two JIT
   template-arity patches, also run the direct-MLX attention lane: the
   wide-head-dimension test there exercises the JIT generator paths in
   seconds.
4. Reconcile the Rust FFI surface: bindings are bindgen-generated at build
   time from the pinned headers via the allowlist in
   `crates/runtime-integration/build_bindings.rs`. Diff the generated
   surface, update the allowlist and call sites, and leave zero compiler
   warnings. Before the first native build of the new pin, diff the C header
   surface itself with the bindgen header provisioning (see the next
   section); header-level differences found there explain almost every
   binding-level difference before any compile runs.
5. Run the hermetic lanes, then the real-model acceptance journeys serially
   through `scripts/run-bounded-cargo-test.sh` (GPU journeys are never
   parallel).
6. Prove performance parity for the `resident_sparse_moe` registry model
   with the performance-throughput journeys and compare against the recorded
   band in `apps/inference-worker/tests/performance_throughput/throughput-history.jsonl`.
7. Update this registry, the repo discovery guide, and
   `docs/performance-optimizations-lessons.md`, then commit through
   `scripts/verify-before-commit.sh`.

## Bindgen headers and the C surface diff

`scripts/provision-bindgen-headers.sh` extracts the pinned MLX and MLX-C
archives from the verified native dependency cache and applies the same patch
pipeline as the native build (the patch list is parsed from the native
CMakeLists, so the two can never drift). The extraction is keyed by the
source-only native build identity, so a pin or patch edit invalidates it
automatically, and a second run re-verifies the published tree against its
recorded hash manifests without re-extracting. The script never downloads;
run `scripts/bootstrap-native-dependencies.sh` first.

```bash
scripts/provision-bindgen-headers.sh
scripts/provision-bindgen-headers.sh --verify-headers
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

The contract suite in `scripts/test-provision-bindgen-headers-contract.sh`
covers extraction, patch application, idempotent reuse, self-healing,
tampered-archive refusal, offline behavior, and verification against hermetic
store fixtures.

## Reproducing a native build from scratch

Prerequisites: macOS on Apple Silicon, Xcode command line tools, Rust
toolchain, CMake 3.24 or newer, and `patch`.

```bash
scripts/bootstrap-native-dependencies.sh
scripts/prewarm-native-build.sh --profile core
cargo build -p astronomical-inference-worker
```

The first script provisions and verifies every pinned archive into the
platform cache. The second configures and compiles the native runtime image
(static MLX, MLX-C, and the Astronomical bridges) and publishes it to the
content-addressed native build store. The third links the worker against the
published store; if the store is empty, the cargo artifact lifecycle builds
it first. Hermetic verification afterwards needs no network and no model
weights:

```bash
TEST_TIMEOUT_SECONDS=120 scripts/run-bounded-cargo-test.sh \
  cargo test -p astronomical-runtime-integration --test hermetic_tests
```
