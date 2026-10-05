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
        // Wave 2 — crates/rest-contract
        .target(name: "RestContract"),
        // Wave 2 — apps/supervisor
        .target(name: "Supervisor"),
        // Wave 2 — apps/astronomical
        .executableTarget(name: "AstronomicalCli"),
        // Wave 3 — crates/runtime-integration
        .target(name: "RuntimeIntegration"),
        // Wave 3 — crates/model-serving
        .target(name: "ModelServing"),
        // Wave 3 — apps/inference-worker
        .executableTarget(name: "InferenceWorker"),
        // One test target per module, mirroring Sources/.
        .testTarget(name: "AstronomicalConfigTests", dependencies: ["AstronomicalConfig"]),
        .testTarget(name: "IpcProtocolTests", dependencies: ["IpcProtocol"]),
        .testTarget(name: "RestContractTests", dependencies: ["RestContract"]),
        .testTarget(name: "SupervisorTests", dependencies: ["Supervisor"]),
        .testTarget(name: "AstronomicalCliTests", dependencies: ["AstronomicalCli"]),
        .testTarget(name: "RuntimeIntegrationTests", dependencies: ["RuntimeIntegration"]),
        .testTarget(name: "ModelServingTests", dependencies: ["ModelServing"]),
        .testTarget(name: "InferenceWorkerTests", dependencies: ["InferenceWorker"])
    ]
);
