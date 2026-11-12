# How Astronomical works

What Astronomical does with memory, speed and precision, and what it supports today. To install and start using it, see the [README](../README.md).

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
- OpenAI-compatible chat completions, responses, embeddings, image generation, and model discovery, Anthropic-compatible Messages, and server-sent event streaming over loopback only.

A supported model runs on every Apple silicon M-series Mac that meets the operating-system floor. Measured custom Metal kernels keep capable GPUs on their fastest path; where a GPU cannot run one, the same request completes through public MLX APIs with the output precision the model artifact specifies. Throughput still depends on the GPU generation, the memory ceiling, the model, and storage speed.

Astronomical does not bundle or redistribute model weights. Observatory can download curated public Hugging Face catalog entries directly onto this Mac; each payload is verified before it becomes discoverable. Model licenses remain separate and must permit the intended use.

## Connect locally

The daemon exposes its OpenAI-compatible and Anthropic-compatible APIs and embedded Observatory at the same loopback address. Stable uses [http://127.0.0.1:6732/](http://127.0.0.1:6732/) and Development uses [http://127.0.0.1:6733/](http://127.0.0.1:6733/). Astronomical refuses non-loopback bind addresses.

Point an OpenAI-compatible or Anthropic-compatible local client at that address, select a discovered model, and Astronomical loads or swaps the worker on demand. `astronomical launch claude`, `astronomical launch codex`, or `astronomical launch opencode` starts that agent pointed at the running app for one session. The manual settings for other clients are on the [local API page](https://aosama.github.io/astronomical/local-api.html).
