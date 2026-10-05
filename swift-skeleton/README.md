# Swift Migration Skeleton

> **Status: INERT.** Every file in this tree is comments only. Nothing here
> compiles, nothing here is referenced by Cargo, and commits touching only this
> tree cannot trigger a GitHub Action. This tree exists so the Rust-to-Swift
> migration layout can be reviewed on its own, before any real Swift lands.

## Why this exists

Astronomical is migrating from Rust to Swift. Before wave 1 starts, this
skeleton fixes the target package layout so module boundaries — what goes
where — can be agreed on with zero build or CI (Continuous Integration)
surface.

## What keeps this inert (structural guarantees)

1. **`Package.swift` is comments only.** It is not a valid SwiftPM (Swift
   Package Manager) manifest. If anything ever tries to build this tree,
   packaging fails loudly instead of silently producing artifacts.
2. **Not a Cargo workspace member.** The root `Cargo.toml` lists members
   explicitly; `swift-skeleton/` is not among them.
3. **CI builds explicit package paths only.** The workflows invoke Swift
   builds for `apps/astronomical-menu` and `apps/thin-talk` via
   `--package-path`; a package directory on disk is never picked up
   implicitly.
4. **The change-scope classifier exempts `swift-skeleton/*`.** Commits
   touching only this tree classify as static-only and cannot trigger the
   `rust-core` or `swift-node` verification jobs. This is enforced by a
   contract case in `scripts/test-ci-native-cache-coordination.sh`.

## Layout

```
swift-skeleton/
  Package.swift                  comment-only sketch of the future manifest
  README.md                      this file
  Sources/
    AstronomicalConfig/          wave 1 — crates/config
      ModelDiscovery/            model_discovery/ (artifact scanning, bounded
                                  reads, classification, effective models,
                                  family shapes)
    IpcProtocol/                 wave 1 — crates/ipc-protocol
      ImageGeneration/           image_generation/
    RestContract/                wave 2 — crates/rest-contract (flat in Rust)
    Supervisor/                  wave 2 — apps/supervisor
      Library/                   library/
        DownloadJob/             library/download_job/
      WorkerHealth/              worker_health/
    AstronomicalCli/             wave 2 — apps/astronomical (flat in Rust)
    RuntimeIntegration/          wave 3 — crates/runtime-integration
      Experimental/              experimental/
      MlxRuntime/                mlx_runtime/
    ModelServing/                wave 3 — crates/model-serving
      ArtifactValidation/        artifact_validation/
      Attention/                 attention/
      DecoderCache/              decoder_cache/
      EngineBackedWorker/        engine_backed_worker/
      ExpertPaging/              expert_paging/
      InferenceEngine/           inference_engine/
      KernelCapability/          kernel_capability/
      Memory/                    memory/ — all memory-management code
      ModelFamilyRuntime/        model_family_runtime/
      PerformanceAttribution/    performance_attribution/
      PersistentCache/           persistent_cache/
      Safetensors/               safetensors/
      SparseExperts/             sparse_experts/
      StructuredGeneration/      structured_generation/
      DeepseekV4/                deepseek_v4/ (flat in Rust)
      Flux2Klein/                flux2_klein/ — Engine/, ImageEncoding/,
                                  TextConditioning/, Transformer/, Vae/
      K2HorizonMova/             k2_horizon_mova/ — Engine/, Artifacts/,
                                  Configuration/, Model/, Startup/, Text/
      Laguna/                    laguna/ — Engine/, Artifacts/, Model/, Moe/,
                                  Normalization/, Paging/,
                                  PromptProcessingChunkSizer/, Startup/, Text/
      ModernBERT/                modernbert/ (flat in Rust)
      Qwen3_5/                   qwen3_5/ — Artifacts/, Configuration/,
                                  Decoder/, Dense/, InferenceExecution/,
                                  Model/, MtpVerify/, MultiTokenPrediction/,
                                  Quantizations/, Text/, Vision/
      Qwen3_5MoE/                qwen3_5_moe/ — Artifacts/, ExpertPaging/,
                                  ExpertResidency/, Model/
      Qwen4Exp/                  qwen4_exp/ — Configuration/, Decoder/,
                                  HyperConnection/, Ple/, Qsa/
      QwenImage21/               qwen_image_21/ — Artifact/, Configuration/,
                                  Engine/, TensorProfiles/, TextEncoder/,
                                  Transformer/, Vae/
    InferenceWorker/             wave 3 — apps/inference-worker (flat in Rust)
  Tests/
    <Module>Tests/               one test target per module, mirroring Sources
```

Subfolder rule: a Swift subfolder exists exactly where the Rust crate has a
real subdirectory, one marker file per Rust sub-concern. Modules that are
flat files in Rust (RestContract, AstronomicalCli, InferenceWorker,
DeepseekV4, ModernBERT, and the root of most crates) stay flat here too —
the skeleton mirrors the Rust trees, it does not invent granularity.
Qwen3_5 and Qwen3_5MoE are separate module folders because Rust keeps
qwen3_5 and qwen3_5_moe as separate trees. Test trees deliberately stay one
marker per module at the target root.

## Rust-to-Swift mapping

| Swift module | Rust home today | Wave |
| --- | --- | --- |
| `AstronomicalConfig` | `crates/config` | 1 |
| `IpcProtocol` | `crates/ipc-protocol` | 1 |
| `RestContract` | `crates/rest-contract` | 2 |
| `Supervisor` | `apps/supervisor` | 2 |
| `AstronomicalCli` | `apps/astronomical` | 2 |
| `RuntimeIntegration` | `crates/runtime-integration` | 3 |
| `ModelServing` | `crates/model-serving` | 3 |
| `InferenceWorker` | `apps/inference-worker` | 3 |
| — retired | `crates/mlx-c-rust` (replaced by MLX-Swift) | 4 |
| — retired | `crates/native-build-tool`, third-party pins/patches, prewarm scripts | 4 |
| — retired | bounded-test shell scripts (replaced by native `swift test`) | 4 |
| unchanged | `apps/astronomical-menu`, `apps/thin-talk` (already Swift) | — |

Wave 1 modules are pure data contracts with no MLX dependency, so they can
land and be reviewed while the Rust side keeps running. Waves 2 and 3 then
follow the strangler seam: the supervisor and CLI (Command Line Interface)
move first while the worker keeps speaking the unchanged IPC (Inter-Process
Communication) protocol; the model-serving core moves last, over MLX-Swift.

## Carried contracts

These rules move with the code, regardless of language:

- All memory-management code lives in one place: `ModelServing/Memory`.
- Real-model GPU (Graphics Processing Unit) journeys never run in parallel;
  hermetic CPU tests may parallelize.
- Every test carries a built-in timeout capped at 120 seconds (endurance and
  out-of-memory reproduction tests excepted).
- The Romeo and Juliet fixture is the text input for LLM (Large Language
  Model) tests; no random text or tokens.
- Performance throughput journeys: at least 10000 input tokens and 1000
  output tokens (plus or minus 10 percent), a 1000-in/100-out warmup, SSD
  (Solid State Drive) cache disabled, and an optimized build.
- Performance logging is switchable through configuration and captures start
  and end time per operation, prioritizing model loading, prompt processing,
  tokenization, disk cache, expert paging, and token generation.
- End-user-facing sizes use decimal SI (Système International) gigabytes:
  1 GB = 1,000,000,000 bytes.
- Code files stay around the 500-line mark; split types rather than growing
  them.
- No developer-home paths, local model paths, or machine-specific hardware
  assumptions; resolve locations through configuration, environment, or
  platform-standard directories.

## Activation checklist (when wave 1 begins)

Activation must be a deliberate, visible sequence — never a side effect:

1. Replace the comment-only `Package.swift` with a real manifest listing only
   the wave-1 targets.
2. Remove `swift-skeleton/*` from the exempt list in
   `scripts/classify-ci-change-scope.sh` so the tree starts classifying as
   code.
3. Extend the contract test in
   `scripts/test-ci-native-cache-coordination.sh` to assert the new scope.
4. Wire the package paths into the `swift-node` job in the workflow.
5. Update the repo discovery guide: remove the inert-skeleton note and record
   the real build and test commands.
