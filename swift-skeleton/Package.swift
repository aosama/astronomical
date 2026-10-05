// swift-tools-version: 6.1
//
// INERT MIGRATION SKELETON — COMMENTS ONLY.
//
// This file is deliberately NOT a valid manifest. It carries no
// `import PackageDescription` and no `Package(...)` declaration, so any
// attempt to build this tree fails loudly at manifest loading instead of
// silently producing artifacts. The sketch below records the intended shape
// for wave 1; it stays commented out until activation (see README.md).
//
// Intended platform and layout:
//
//   // swift-tools-version: 6.1
//   import PackageDescription
//
//   let package = Package(
//       name: "Astronomical",
//       platforms: [.macOS(.v15)],
//       targets: [
//           // Wave 1 — pure contracts, no MLX dependency.
//           .target(name: "AstronomicalConfig"),
//           .target(name: "IpcProtocol"),
//           // Wave 2 — strangler seam over unchanged IPC frames.
//           .target(name: "RestContract"),
//           .target(name: "Supervisor"),
//           .target(name: "AstronomicalCli"),
//           // Wave 3 — model serving over MLX-Swift.
//           .target(name: "RuntimeIntegration"),
//           .target(name: "ModelServing"),
//           .target(name: "InferenceWorker"),
//       ],
//       testTargets: [
//           .testTarget(name: "AstronomicalConfigTests", dependencies: ["AstronomicalConfig"]),
//           .testTarget(name: "IpcProtocolTests", dependencies: ["IpcProtocol"]),
//           .testTarget(name: "RestContractTests", dependencies: ["RestContract"]),
//           .testTarget(name: "SupervisorTests", dependencies: ["Supervisor"]),
//           .testTarget(name: "AstronomicalCliTests", dependencies: ["AstronomicalCli"]),
//           .testTarget(name: "RuntimeIntegrationTests", dependencies: ["RuntimeIntegration"]),
//           .testTarget(name: "ModelServingTests", dependencies: ["ModelServing"]),
//           .testTarget(name: "InferenceWorkerTests", dependencies: ["InferenceWorker"]),
//       ]
//   )
//
// The family engines under Sources/ModelServing/<Family>/ stay subfolders of
// the single ModelServing target, mirroring the Rust module layout rather
// than fragmenting into per-family packages.
