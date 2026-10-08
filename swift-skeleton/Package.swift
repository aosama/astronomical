// swift-tools-version:6.1
import PackageDescription;

// Real manifest for the Rust-to-Swift migration tree. Every module below
// still holds comment-only markers except where a wave has landed, so the
// not-yet-ported modules build as empty modules until their units arrive.

let package: Package = Package(
    name: "Astronomical",
    platforms: [
        .macOS(.v15)
    ],
    dependencies: [
        // Wave 3 — the Metal acceleration substrate under crates/mlx-c-rust.
        // Pinned to the upstream release; never a developer-local checkout.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.32.3"),
        // Wave 3 engine epic #983 — upstream model definitions, KV caches,
        // tokenizer, and eval loops the Swift engine adapts instead of
        // porting. Exactly pinned so the pairing with mlx-swift above is
        // deliberate: 3.32.3 requires mlx-swift ~> 0.32.3.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", exact: "3.32.3"),
        // The tokenizer engine the mlx-swift-lm tokenizer bridge adapts;
        // 3.32.3 defines the bridge but ships no tokenizer implementation,
        // so the consuming package owns this dependency (upstream recipe).
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.0"),
        // The Library download engine (#1059): transport, tree listing,
        // gated-metadata preflight, ETag blob cache with HTTP Range
        // resume, progress, and Xet support. Already resolved transitively
        // via mlx-swift-lm; pinned exactly because the supervisor calls
        // its public download surface directly.
        .package(url: "https://github.com/huggingface/swift-huggingface.git", exact: "0.13.0")
    ],
    targets: [
        // Wave 1 — crates/config
        .target(
            name: "AstronomicalConfig",
            resources: [
                .copy("Resources/astronomical-config.schema.json")
            ]
        ),
        // Wave 1 — crates/ipc-protocol
        .target(name: "IpcProtocol"),
        // Wave 2 — crates/rest-contract (decodes over the shared JSON wire
        // library the way Rust serde derives decode over serde_json).
        .target(name: "RestContract", dependencies: ["IpcProtocol"]),
        // Wave 2 — apps/supervisor
        .target(
            name: "Supervisor",
            dependencies: ["AstronomicalConfig", "IpcProtocol", "RestContract",
                .product(name: "HuggingFace", package: "swift-huggingface")],
            resources: [
                // The Observatory console, symlinked to its one canonical
                // home under apps/supervisor/console so no copy drifts while
                // the Rust tree retires.
                .copy("Resources/console"),
                // The release download catalog, snapshotted from the
                // repository registry the way Rust's include_str! embeds it;
                // refresh when the registry document changes.
                .copy("Resources/download_catalog.json")
            ]),
        // Wave 2 — the astronomicald daemon binary itself.
        .executableTarget(
            name: "AstronomicalDaemon",
            dependencies: ["Supervisor", "AstronomicalConfig", "IpcProtocol"]),
        // Wave 2 — apps/astronomical
        .executableTarget(name: "AstronomicalCli", dependencies: ["AstronomicalConfig", "IpcProtocol", "Supervisor"]),
        // Wave 3 — crates/runtime-integration
        .target(
            name: "RuntimeIntegration",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift")
            ]),
        // Wave 3 — crates/model-serving
        .target(
            name: "ModelServing",
            dependencies: [
                "IpcProtocol",
                "RuntimeIntegration",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXGuidedGeneration", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]),
        // Shared hermetic fixtures (tiny tokenizer, tiny dense artifact)
        // consumed by every test target that synthesizes model directories.
        .target(
            name: "ModelServingTestSupport",
            dependencies: ["ModelServing"]),
        // The journey-category vocabulary: test-tier tags and the real-model
        // gate. Dependency-free so every test target adopts it without
        // pulling serving code into its dependency graph.
        .target(name: "JourneyCategories"),
        // Wave 3 — apps/inference-worker
        .executableTarget(
            name: "InferenceWorker",
            dependencies: ["IpcProtocol", "RuntimeIntegration", "ModelServing", "AstronomicalConfig"]),
        // The deterministic supervisor test worker, migrating the Rust
        // astronomical-supervisor-idle-worker fixture bin: a real framed
        // protocol speaker over stdin/stdout whose scripted behaviors the
        // worker journey suites steer by model identity. Built by swift test
        // like every executable; journeys locate it from the package root.
        .executableTarget(
            name: "SupervisorIdleWorker",
            dependencies: ["IpcProtocol"]),
        // The stderr-diagnostic probe fixture, migrating the Rust
        // astronomical-supervisor-stderr-probe-worker bin: one event, visible
        // stderr, clean exit — the raw material for the stream-closure
        // diagnostics journeys.
        .executableTarget(
            name: "SupervisorStderrProbeWorker",
            dependencies: ["IpcProtocol"]),
        // One test target per module, mirroring Sources/.
        .testTarget(name: "AstronomicalConfigTests", dependencies: ["AstronomicalConfig", "JourneyCategories"]),
        .testTarget(name: "IpcProtocolTests", dependencies: ["IpcProtocol", "JourneyCategories"]),
        .testTarget(name: "RestContractTests", dependencies: ["RestContract", "JourneyCategories"]),
        .testTarget(
            name: "SupervisorTests",
            dependencies: [
                "Supervisor",
                "JourneyCategories",
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ]),
        // The daemon-process journeys spawn the real astronomicald binary;
        // they live in their own target — a separate process under
        // `swift test`, like the Rust tree's separate integration-test
        // binary — so they can never starve the parallel hermetic suites.
        .testTarget(        name: "DaemonProcessJourneys",
        dependencies: ["Supervisor", "JourneyCategories"]),
        .testTarget(name: "AstronomicalCliTests", dependencies: ["AstronomicalCli", "AstronomicalConfig", "IpcProtocol", "Supervisor", "JourneyCategories"]),
        .testTarget(
            name: "RuntimeIntegrationTests",
            dependencies: [
                "RuntimeIntegration",
                "JourneyCategories",
                "ModelServingTestSupport",
                .product(name: "MLX", package: "mlx-swift")
            ]),
        .testTarget(
            name: "ModelServingTests",
            dependencies: [
                "ModelServing",
                "ModelServingTestSupport",
                "RuntimeIntegration",
                "JourneyCategories",
                .product(name: "Tokenizers", package: "swift-transformers")
            ]),
        .testTarget(
            name: "InferenceWorkerTests",
            dependencies: ["InferenceWorker", "ModelServingTestSupport", "JourneyCategories"])
    ]
);
