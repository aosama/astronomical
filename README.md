# Astronomical

Astronomical is an experimental local large language model and vision-language model runner for Apple Silicon. It uses Rust for the local HTTP server and inference worker, Swift for the macOS menu application, and Apple MLX through MLX-C for model execution.

## Status

Astronomical currently targets:

- Apple Silicon M5 and later.
- macOS 26 and later.
- Qwen3.5-family dense and mixture-of-experts text or vision artifacts supported by the repository validators.
- One local user and one active generation at a time.

The project is under active development. Model compatibility is deliberately narrow and validated rather than inferred from filenames.

## Models

Astronomical does not bundle, download, or redistribute model weights. Supply an already-downloaded compatible model directory through the local configuration file.

On first launch Astronomical creates ~/.astronomical/config.json. Add one or more absolute directories to scan recursively:

    {
      "model_directories": ["/path/to/models"],
      "prefill_chunck_size_optimizer_enabled": true,
      "prompt_cache_max_size_gb": 50
    }

Model licenses are separate from Astronomical. Confirm that you may use each model artifact before loading it.

## Build

Required tools:

- Xcode with the macOS 26 SDK.
- CMake.
- Rust 1.97.1 through the checked-in toolchain file.
- Swift 6.2.
- Homebrew coreutils, jq, curl, and sccache for repository verification and app validation.

Build the optimized daemon, worker, and model preparer:

    CARGO_BUILD_JOBS="$(sysctl -n hw.logicalcpu)" cargo build --release \
      -p astronomical-inference-worker --bin astronomical-inference-worker \
      -p astronomical-supervisor --bin astronomicald \
      -p astronomical-model-serving --features direct-mlx --bin astronomical-model-preparer

Build and validate the complete menu-bar application after configuring at least one compatible local model:

    scripts/make-astronomical-app.sh

The signed local bundle is written to target/astronomical-macos-release/Astronomical.app.

## API

The daemon exposes an OpenAI-compatible local surface including chat completions, responses, and model discovery. The configured address must be loopback-only; Astronomical refuses non-loopback bind addresses.

The embedded Observatory console is served at the daemon root URL. The default address is http://127.0.0.1:6732/.

## Privacy

Prompts, responses, model files, and persistent prompt state stay on the local Mac. Operational logs contain bounded metadata and payload digests, not request headers, prompt text, tool arguments, or generated output text.

Persistent prompt state is a disposable local optimization under ~/.astronomical. It is not a portable model artifact or a compatibility promise.

## Verification

Run the bounded commit gate:

    scripts/verify-before-commit.sh

Run the macOS menu contracts:

    scripts/test-macos-menu-contracts.sh

Direct MLX and real-model qualification lanes are explicit because they require Apple graphics hardware or local model artifacts. See repo-discovery-guide-for-agents.md for the maintained command map.

## Contributing

Issues and pull requests are welcome. Read CONTRIBUTING.md before submitting changes and SECURITY.md before reporting a vulnerability.

## License

Astronomical is licensed under Apache License 2.0. See LICENSE, third-party/THIRD_PARTY_NOTICES, and the generated Rust dependency notices for complete distribution terms.
