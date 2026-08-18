# Astronomical

**Run bigger local language and vision models on Apple Silicon without requiring every sparse expert to live in RAM.**

Astronomical is a performance-first local model runner for Mac users who want serious models, private inference, and direct control over memory. Set the maximum model RAM your laptop can spare and Astronomical automatically balances hot expert weights, live context, runtime work, and solid-state-drive streaming under that ceiling.

Read the product story and public engineering reports at [aosama.github.io/astronomical](https://aosama.github.io/astronomical/).

![Astronomical running a Qwen3.6 35B mixture-of-experts model with an 11 GB model RAM ceiling and RAM plus SSD streaming](https://aosama.github.io/astronomical/assets/astronomical-ram-ssd-streaming.jpeg)

*A captured development run of Qwen3.6-35B-A3B-oQ4e-mtp: 21.61 GB on disk, an 11 GB model-memory ceiling, automatic RAM plus SSD expert streaming, and live prompt-processing telemetry. This demonstrates the operating mode, not a universal throughput guarantee; results vary by model, context, storage, and Mac.*

## Built for the RAM you have

A mixture-of-experts model may contain many expert weights while activating only a small selection for each token. Astronomical takes advantage of that structure instead of treating the complete model as one permanent RAM allocation.

- **Choose one model-memory ceiling.** On a 32 GB Mac, for example, a lower ceiling such as 16 GB can leave room for macOS and everyday applications.
- **Stream selected experts from SSD.** When the complete sparse payload does not fit, router-selected expert pages move into MLX memory on demand.
- **Keep the complete model hot when it fits.** All sparse experts use contiguous resident arrays when safe; context pressure switches the whole model to paging and idle recovery can restore residency.
- **Adapt automatically.** Resident and RAM plus SSD streaming are observed outcomes, not modes the user has to tune.
- **See the truth.** The menu reports the effective ceiling, expert residency, model core, runtime work, live context, headroom, GPU utilization, prompt progress, and prompt reuse.

The model core, active context, selected expert pages, and temporary execution work must still fit within the configured ceiling. Model compatibility and practical speed therefore depend on the artifact, context length, SSD, and available memory. Astronomical is designed to adapt rather than promise that every model fits every Mac.

Expert mode always describes the complete sparse payload, never a mixture of resident and paged layers. A model can be resident while idle, switch to paging before or during a large request, and return to resident after synchronized request cleanup when the complete payload fits again.

## Performance from the whole stack

Astronomical is not a generic server wrapped around an off-the-shelf model loop. The serving path is built in Rust and C++, executes through Apple MLX and MLX-C, and contains model-specific optimizations for Apple graphics processors.

| Optimization | What it does for the user |
| --- | --- |
| Automatic sparse-expert residency and paging | Keeps the complete sparse model resident when safe and uses SSD capacity when memory is needed elsewhere. |
| Adaptive memory admission | Projects context growth and observed transient work before allocation, then reclaims only the expert memory required for the next operation. |
| Deterministic prompt chunking | Uses a qualified fixed size, exact terminal remainders, and bounded memory-capacity reduction without spending user latency on online exploration. |
| Persistent prompt reuse | Restores validated prompt state, recurrent snapshots, and projected image embeddings from a bounded SSD store so repeated prefixes need less work. |
| GPU-native sampling and decode | Keeps sampling on the graphics processor and submits one token ahead to reduce avoidable host synchronization. |
| Sorted mixture-of-experts execution | Groups larger routing batches by expert and reuses one ordering across projections to reduce scattered quantized matrix work. |
| Compiled and fused model operations | Reuses compiled MLX graphs and dedicated Metal kernels for recurrent state, activation, gating, and expert combination where measured paths benefit. |
| Chunk-level allocator cleanup | Materializes the required decoder state, releases completed intermediates, and prevents long prompts from accumulating stale allocator memory. |
| Lazy loading and model hot-swap | Starts with no model resident and loads only the model requested through the local API. |

## Precision is not the sacrifice

Low-memory operation should not silently make a model less accurate.

- Astronomical does not add hidden quantization when memory pressure increases.
- Expert paging preserves the artifact's declared data types and quantization parameters.
- Persistent prompt state preserves the model execution state required for reuse.
- Unsupported architecture, precision, or artifact combinations fail validation instead of being guessed from a filename.

## Local, visible, and private

The embedded Observatory and native menu make model serving observable without a separate monitoring stack.

- Live prompt-processing and generation progress.
- Current and peak MLX memory attribution.
- Resident or RAM plus SSD streaming state.
- GPU utilization and session throughput.
- Prompt reuse rate and cumulative reused tokens.
- Effective model-memory ceiling with live adjustment.

Prompts, responses, model files, and persistent prompt state stay on the local Mac. Operational diagnostics retain bounded metadata and payload digests, not request headers, prompt text, tool arguments, or generated output text.

## Current scope

Astronomical is experimental and deliberately focused:

- Apple Silicon M5 and later.
- macOS 26 and later.
- Validated Qwen3.5/Qwen3.6 and Laguna artifacts supported by the repository.
- One local user and one active generation at a time.
- OpenAI-compatible chat completions, responses, model discovery, and server-sent event streaming over loopback only.

Astronomical does not bundle, download, or redistribute model weights. Model licenses remain separate and must permit the intended use.

## Stable and Development instances

Astronomical keeps the trusted daily driver separate from repository development:

| Channel | State | REST API | Application |
| --- | --- | --- | --- |
| Stable | `~/.astronomical` | `127.0.0.1:6732` | `~/Applications/Astronomical.app` |
| Development | `~/.astronomical-dev` | `127.0.0.1:6733` | `target/astronomical-macos-development.noindex/Astronomical Development.app` |

Config, logs, prompt caches, daemon ownership, process locks, and loopback endpoints are isolated. A standard instance rejects a configured endpoint belonging to the other channel. Both configs may reference the same read-only model directories. Real-model development still shares the Mac's GPU, wired memory, and storage bandwidth with Stable.

Serving and qualification tests read user-selected model locations and policy from Development only. Their mutable config, cache, and logging fixtures use temporary `.astronomical-dev` state and never `~/.astronomical`. Config boundary tests may construct temporary Stable fixtures solely to prove channel separation. Explicit app validation with `--real-model` uses the Development instance.

On first launch each instance creates its own `config.json`. Add one or more absolute directories to scan recursively:

    {
      "model_directories": ["/path/to/models"],
      "maximum_mlx_memory_gb": 16,
      "persistent_prompt_cache_enabled": true,
      "chunking": {
        "fixed_prompt_processing_chunk_size_tokens": 2048
      },
      "prompt_cache_max_size_gb": 50
    }

The memory value uses decimal gigabytes. Remove maximum_mlx_memory_gb to use the Mac-reported MLX ceiling. Set persistent_prompt_cache_enabled to false to disable SSD-backed prompt reuse. Both channels default to qualified fixed prompt-processing chunks of 2,048 tokens. Override the fixed chunk size with fixed_prompt_processing_chunk_size_tokens; a smaller fixed_ssd_streaming_prompt_processing_chunk_size_tokens can accelerate paged-expert prefill.

## Build the app

Required tools:

- Xcode with the macOS 26 software development kit.
- CMake.
- Rust 1.97.1 through the checked-in toolchain file.
- Swift 6.2.
- Homebrew coreutils, jq, curl, and sccache for repository verification and application validation.

Provision the pinned, checksum-verified native dependencies:

    scripts/bootstrap-native-dependencies.sh

Build and validate the Development app without replacing or stopping Stable:

    scripts/build-development-app.sh

The signed local bundle is written to `target/astronomical-macos-development.noindex/Astronomical Development.app`. The `.noindex` build directory keeps generated app bundles out of Spotlight while preserving them for validation and direct launch. Stable uses the clean product icon, while Development uses a restrained channel badge. Default validation does not load a second model. Use the explicit `--real-model` validation option only when shared GPU pressure is acceptable.

Stable installation and release publication are isolated under `scripts/release/`. They are never part of ordinary commit or push verification. Build a clean Stable candidate and explicitly promote it outside the repository build tree with one release-only command:

    scripts/release/build-and-install-stable-app.sh

Use `--dry-run` to build and validate Stable while previewing, rather than performing, the installation. `scripts/release/build-stable-app.sh` and `scripts/release/install-stable-app.sh` remain available when the two stages need to run independently. Run `scripts/release/tests/test-release-contracts.sh` explicitly before release work.

Stable candidate builds perform signature, resource, metadata, and bundled-daemon validation without launching over a running Stable instance. Promotion does not restart the running app. Stable builds require a clean Git worktree.

Astronomical follows pre-1.0 Semantic Versioning beginning at `0.2.0`. User interfaces and `/v1/status` show the semantic version, Stable or Development channel, short Git commit, and dirty-development marker.

To build only the optimized daemon and worker:

    CARGO_BUILD_JOBS="$(sysctl -n hw.logicalcpu)" cargo build --release \
      -p astronomical-inference-worker --bin astronomical-inference-worker \
      -p astronomical-supervisor --bin astronomicald

## Connect locally

The daemon exposes its OpenAI-compatible API and embedded Observatory at the same loopback address. Stable uses [http://127.0.0.1:6732/](http://127.0.0.1:6732/) and Development uses [http://127.0.0.1:6733/](http://127.0.0.1:6733/). Astronomical refuses non-loopback bind addresses.

Point an OpenAI-compatible local client at that address, select a discovered model, and Astronomical loads or swaps the worker on demand.

## Verification

Run the bounded commit gate:

    scripts/verify-before-commit.sh

Run the macOS menu contracts:

    scripts/test-macos-menu-contracts.sh

Direct MLX and real-model qualification lanes remain explicit because they require Apple graphics hardware or local model artifacts. See the [repository discovery guide](repo-discovery-guide-for-agents.md) for the maintained command map.

## Contributing

Issues and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes and [SECURITY.md](SECURITY.md) before reporting a vulnerability.

## License

Astronomical is licensed under [Apache License 2.0](LICENSE). See [THIRD_PARTY_NOTICES](third-party/THIRD_PARTY_NOTICES) and [RUST_DEPENDENCY_NOTICES](third-party/RUST_DEPENDENCY_NOTICES) for complete distribution terms.
