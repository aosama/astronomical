# Astronomical

<a href="https://discord.gg/dc4E6r4WD"><img src="site/assets/discord.svg" alt="Discord" width="162" height="34" style="vertical-align:middle"></a>

**Run bigger local language and vision models on Apple Silicon without requiring every sparse expert to live in RAM.**

You set the model RAM your laptop can spare. Astronomical is a local model runner for Mac users who want serious models, private inference, and direct control over memory: it automatically balances hot expert weights, live context, runtime work, and solid-state-drive streaming under that ceiling.

Read the product story and public engineering reports at [aosama.github.io/astronomical](https://aosama.github.io/astronomical/).

[![Astronomical running a Qwen3.6 35B mixture-of-experts model with an 11 GB model RAM ceiling and RAM plus SSD streaming](https://aosama.github.io/astronomical/assets/astronomical-ram-ssd-streaming.jpeg)](https://aosama.github.io/astronomical/assets/astronomical-ram-ssd-streaming.mp4)

*A captured development run of Qwen3.6-35B-A3B-oQ4e-mtp: 21.61 GB on disk, an 11 GB model-memory ceiling, automatic RAM plus SSD expert streaming, and live prompt-processing telemetry. This demonstrates the operating mode, not a universal throughput guarantee; results vary by model, context, storage, and Mac. Click the image for the [25-second silent demo](https://aosama.github.io/astronomical/assets/astronomical-ram-ssd-streaming.mp4): a 23.55 GB mixture-of-experts model generating fully resident under a 30 GB ceiling, then switching to RAM plus SSD streaming after the ceiling is lowered. Captured run, not a throughput claim.*

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
| Deterministic prompt chunking | Uses a chosen fixed size, exact terminal remainders, and bounded memory-capacity reduction without spending user latency on online exploration. |
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

- Any Apple silicon M-series Mac.
- macOS 14 (Sonoma) and later on any Apple silicon M-series Mac, declared best-effort. Capability probes and public MLX fallbacks protect correctness across the range; combinations outside the tested set are supported best-effort.
- Validated Qwen3.5/Qwen3.6, Laguna, FLUX.2 Klein, and ModernBERT embedding artifacts supported by the repository.
- One local user and one active generation at a time.
- OpenAI-compatible chat completions, responses, embeddings, image generation, model discovery, and server-sent event streaming over loopback only.

A supported model runs on every Apple silicon M-series Mac that meets the operating-system floor. Measured custom Metal kernels keep capable GPUs on their fastest path; where a GPU cannot run one, the same request completes through public MLX APIs with the output precision the model artifact specifies. Throughput still depends on the GPU generation, the memory ceiling, the model, and storage speed.

Astronomical does not bundle or redistribute model weights. Observatory can download curated public Hugging Face catalog entries directly onto this Mac; each payload is verified before it becomes discoverable. Model licenses remain separate and must permit the intended use.

## Stable and Development instances

Astronomical keeps the trusted daily driver separate from repository development:

| Channel | State | REST API | Application |
| --- | --- | --- | --- |
| Stable | `~/.astronomical` | `127.0.0.1:6732` | `~/Applications/Astronomical.app` |
| Development | `~/.astronomical-dev` | `127.0.0.1:6733` | `target/astronomical-macos-development.noindex/Astronomical Development.app` |

Config, logs, prompt caches, daemon ownership, process locks, and loopback endpoints are isolated. A standard instance rejects a configured endpoint belonging to the other channel. Both configs may reference the same read-only model directories. Real-model development still shares the Mac's GPU, wired memory, and storage bandwidth with Stable.

Serving and acceptance tests read user-selected model locations and policy from Development only. Their mutable config, cache, and logging fixtures use temporary `.astronomical-dev` state and never `~/.astronomical`. Config boundary tests may construct temporary Stable fixtures solely to prove channel separation. Explicit app validation with `--real-model` uses the Development instance.

On first launch each instance creates its own versioned `config.json` and adjacent local JSON Schema. Open Library in Observatory to install a curated model under that instance's automatic models directory. When an existing unversioned configuration can be migrated without changing its behavior, Astronomical preserves its exact bytes once as `config.legacy-v0.json` before atomically replacing `config.json`. An unsafe migration leaves the original configuration unchanged and creates no backup. Additional existing model locations remain available as recursive scan roots:

    {
      "$schema": "./astronomical-config.schema.json",
      "schema_version": 1,
      "runtime": {
        "model_directories": ["/path/to/models"],
        "maximum_mlx_memory_gb": 16
      },
      "prompt_cache": {
        "enabled": true,
        "maximum_size_gb": 50
      }
    }

The memory value uses the gigabyte Apple uses to market the Mac: 1 GB = 1,073,741,824 bytes (2³⁰). Remove `maximum_mlx_memory_gb` to use the Mac-reported MLX ceiling. Per-model entries can set a lower context ceiling, generation defaults, chunking overrides, and multi-token-prediction depth. Explicit request values override generation defaults. Overrides for temporarily absent models remain dormant and are reported as unmatched rather than preventing other models from serving. Global chunking controls remain available under `chunking`, and each model may override individual values. See the [version 1 configuration schema](https://aosama.github.io/astronomical/schemas/config/v1/astronomical-config.schema.json) for the complete contract.

## Edit the configuration

Observatory's **Configuration** page edits every setting in that instance's `config.json`: set or change a value, remove it to return to its default, add settings for one model, or reset everything. The page is generated from the version 1 schema. Saving validates the whole document, refuses to overwrite a file that changed since the page loaded it, and reports whether the change applied, is waiting for a running request, or needs a restart. Reset keeps the old file beside the new one as `config.previous.json`.

## Metal residency refresh

macOS can release the Metal residency of wired memory while the GPU is idle, so model weights may have to be made resident again before the next prompt. Set `metal_residency_refresh_enabled` under `runtime`, or use the switch on Observatory's **Configuration** page, to have the inference worker re-request residency once per second. It is off by default, changing it restarts the worker, and the renewal can keep that memory wired through long idle periods.

## Neural Engine prefill (experimental)

Dense models can run part of each full prompt chunk on the Apple Neural Engine next to the GPU: 47 percent of each chunk's MLP work, and the DeltaNet layers' output gates. Set `experimental_neural_engine_prefill_enabled` under `chunking`, globally or for one model; it is off by default and applies at the next model load.

Measured on Ornith-1.5-9B on one M5-class Mac, with 10,000-token Shakespeare prompts through the daemon and the prompt cache on: prefill rose from about 1,207 tokens per second to between 1,362 and 1,444, or 1.13× to 1.2×, and direct model runs without the cache's work measured about 1.27×. Decode did not change. The first long prompt after a load sees less of that gain, and only full 2,048-token chunks use the Neural Engine, so shorter prompts see little or none. The costs are real:

- **Memory:** the programs' cost is measured for each model on your Mac the first time they run, and depends on the model: 1.8 GB on a 0.8B model, 3.6 GB on a 2B model and 15 GB on a 9B model. Each prefill chunk uses the Neural Engine only when the memory MLX is projected to reach plus that cost fits under the MLX memory ceiling, so a long prompt can start on both engines and finish on the GPU. A model is refused only when its programs could not fit even for one chunk, which on the 9B model means a ceiling below about 20 GB. A ceiling just above that leaves `/v1/status` saying `active` while chunks run on the GPU, so allow a few more gigabytes; the worker log records each change of engine. The driver releases the programs' wired memory 10 to 20 seconds after work stops.
- **Which models gain:** this is found out for each model on your Mac. During its first long prompts, full chunks take turns between the GPU alone and both engines, which costs those prompts a few percent. That takes at least three prompts and 15 comparisons, which is eight 10,000-token prompts or three 25,000-token ones. A model must gain at least 2 percent to keep its programs. Otherwise they are released, the status says why, and later loads skip them until a macOS update. To ask again sooner, remove `program-speed.txt` from that model's program set under `neural-engine-programs`. Through the app, a 9B model read 1.28× and a 2B model read 1.03× to 1.11×, while a 0.8B model read 0.90× to 0.98× and was turned off. `/v1/status` reports the figure as `measured_speed_percent`, and the menu, the Configuration page and the Overview's Neural Engine panel show it. That panel also shows the share of prompt tokens the Neural Engine really ran and an estimate of the time saved since launch. The estimate counts only prompts that began with the programs already awake, because a prompt that wakes them can run slower overall, so it can undercount a large model and won't claim time a small one lost. A 2B model whose first prompts all arrive after idle spells can read too low and be turned off, which costs only the few percent it would have gained on prompts in quick succession.
- **After an idle spell:** the driver releases the programs 10 to 20 seconds after their last use, and waking them costs time. A prompt after 10 seconds of idle therefore starts with one chunk on the GPU, because a first chunk on both engines took 380 to 600 ms longer than a warm one, and the same programs waking on the second chunk took 140 to 180 ms longer. Over a whole 20,000-token prompt sent after 25 seconds of idle, a 9B model still ran 1.11× the GPU alone, but a 2B model ran 0.94×, against 1.11× for prompts in quick succession. So a small model gains only when its long prompts arrive close together.
- **A request while programs are measured:** the first two loads wire the whole set to measure it. A request arriving then cancels the measurement and starts at once, and the set can stay wired beside it for up to 20 seconds, past the MLX memory ceiling if the ceiling is tight.
- **Preparation time and disk:** a model serves on the GPU as soon as it loads, while its programs are prepared in the background, and the menu app raises banners while they prepare and when they engage. On a 9B model with its compiled programs already on disk, preparation took 94 seconds on the first load after the app itself changed, then 43, 13 and about 3 seconds on the next loads, as macOS rebuilt its own compiled copies. A model's very first preparation also writes and compiles its programs, then measures them on its own thread once the GPU has been idle for 30 seconds, which takes about 3 seconds and delays no request. A request restarts the wait, so a steady stream of requests holds the programs back until there is a pause, with their 1.2 GB of surfaces held meanwhile. Measuring happens on the first two loads, and once the two readings agree the result is kept until the macOS build or the Mac changes. A model keeps about 10 GB of compiled programs under the instance's `neural-engine-programs` folder, and macOS keeps its copies in the app's cache folder besides. Performance attribution, when enabled, removes the speed gain by adding synchronization.
- **Output:** part of each prompt computes in fp16, so output can differ slightly from the GPU alone. In six exact-answer tasks over 13,000-token prompts every answer stayed correct, though on one coding task the answer ignored "output only the code".

Mixture-of-experts models do not benefit, because their routed experts stay on the GPU. When the feature cannot run, serving stays on the GPU and `/v1/status` and the Configuration page say why.

## Build the app

Required tools:

- Xcode with the macOS 26 software development kit.
- CMake.
- Rust nightly-2026-10-18 (rustc 1.101.0-nightly) through the checked-in toolchain file.
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

## Connect locally

The daemon exposes its OpenAI-compatible API and embedded Observatory at the same loopback address. Stable uses [http://127.0.0.1:6732/](http://127.0.0.1:6732/) and Development uses [http://127.0.0.1:6733/](http://127.0.0.1:6733/). Astronomical refuses non-loopback bind addresses.

Point an OpenAI-compatible local client at that address, select a discovered model, and Astronomical loads or swaps the worker on demand.

## Verification

Check Rust formatting, then run the Rust integration and REST suites:

    cargo fmt --all -- --check
    cargo test-integration-and-rest

Run the macOS menu contracts:

    cargo xtask menu-contracts

Direct MLX and real-model acceptance lanes remain explicit because they require Apple graphics hardware or local model artifacts.

## Contributing

Issues and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes and [SECURITY.md](SECURITY.md) before reporting a vulnerability.

## License

Astronomical is licensed under [Apache License 2.0](LICENSE). See [THIRD_PARTY_NOTICES](third-party/THIRD_PARTY_NOTICES) and [RUST_DEPENDENCY_NOTICES](third-party/RUST_DEPENDENCY_NOTICES) for complete distribution terms.
