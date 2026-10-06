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
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.0")
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
        .target(name: "Supervisor", dependencies: ["AstronomicalConfig", "IpcProtocol", "RestContract"]),
        // Wave 2 — the astronomicald daemon binary itself.
        .executableTarget(
            name: "AstronomicalDaemon",
            dependencies: ["Supervisor", "AstronomicalConfig", "IpcProtocol"]),
        // Wave 2 — apps/astronomical
        .executableTarget(name: "AstronomicalCli", dependencies: ["AstronomicalConfig", "IpcProtocol", "Supervisor"]),
        // Wave 3 — crates/runtime-integration
        .target(name: "RuntimeIntegration"),
        // Wave 3 — crates/model-serving
        .target(
            name: "ModelServing",
            dependencies: [
                "IpcProtocol",
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
        // One test target per module, mirroring Sources/.
        .testTarget(name: "AstronomicalConfigTests", dependencies: ["AstronomicalConfig", "JourneyCategories"]),
        .testTarget(name: "IpcProtocolTests", dependencies: ["IpcProtocol", "JourneyCategories"]),
        .testTarget(name: "RestContractTests", dependencies: ["RestContract", "JourneyCategories"]),
        .testTarget(name: "SupervisorTests", dependencies: ["Supervisor", "JourneyCategories"]),
        .testTarget(name: "AstronomicalCliTests", dependencies: ["AstronomicalCli", "AstronomicalConfig", "IpcProtocol", "Supervisor", "JourneyCategories"]),
        .testTarget(name: "RuntimeIntegrationTests", dependencies: ["RuntimeIntegration"]),
        .testTarget(
            name: "ModelServingTests",
            dependencies: [
                "ModelServing",
                "ModelServingTestSupport",
                "JourneyCategories",
                .product(name: "Tokenizers", package: "swift-transformers")
            ]),
        .testTarget(
            name: "InferenceWorkerTests",
            dependencies: ["InferenceWorker", "ModelServingTestSupport", "JourneyCategories"])
    ]
);
