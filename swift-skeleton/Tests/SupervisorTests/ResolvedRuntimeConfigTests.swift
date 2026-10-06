import Foundation;
import XCTest;

import AstronomicalConfig;
import IpcProtocol;

@testable import Supervisor;

/**
 * Acceptance journeys for the resolved serving snapshot's pure surface: the
 * daemon can name one fully resolved configuration path-free (the resolved
 * generation), project the worker bootstrap DTO from it, and classify a
 * candidate config change into the three reload outcomes. Snapshots are
 * constructed directly: these types are pure data plus derived values.
 */
final class ResolvedRuntimeConfigTests: XCTestCase {

    private static func embeddingsSnapshot(
        configurationGeneration: String = "doc-gen-1"
    ) -> ResolvedRuntimeConfig {
        let discoveredModel: DiscoveryDiscoveredModel = DiscoveryDiscoveredModel(
            modelId: "modernbert-model",
            providerModelId: "org/modernbert",
            modelFamily: .modernbert,
            revision: "main",
            modelDirectory: FilePath(string: "/models/modernbert"),
            capabilities: .embeddings(DiscoveryEmbeddingModelCapabilities(
                vectorWidth: 768,
                maximumInputTokens: 8192)),
            license: nil,
            modelSizeBytes: 400_000_000);
        var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary<String, RuntimeModelPolicy>();
        modelPolicyCatalog["modernbert-model"] = try! ResolvedModelPolicyCatalog.resolve(
            userConfig: ResolvedRuntimeConfigTests.emptyConfig(),
            discoveredModels: [discoveredModel],
            artifactContextWindows: Dictionary<String, UInt32>())["modernbert-model"]!;
        return ResolvedRuntimeConfig(
            configurationGeneration: configurationGeneration,
            workerExecutablePath: FilePath(string: "/opt/astronomical/bin/inference-worker"),
            discoveredModels: [discoveredModel],
            modelDiscoveryDiagnostics: Array<DiscoveryModelDiscoveryDiagnostic>(),
            configuredModelDirectories: [FilePath(string: "/models")],
            modelPolicyCatalog: modelPolicyCatalog,
            unmatchedModelConfigIds: Array<String>(),
            maximumMlxMemoryBytes: nil,
            performanceAttributionEnabled: false,
            completionAttributionEnabled: false,
            experimentalQwenThinkingChannelSeedEnabled: false,
            persistentPromptCacheEnabled: true,
            configuredPersistentPromptCacheEnabled: nil,
            configuredPromptCacheMaximumSizeBytes: nil,
            promptCacheConfig: PromptCacheConfig(
                rootDirectory: FilePath(string: "/state/prompt-cache"),
                maximumSizeBytes: 50_000_000_000),
            bindAddress: "127.0.0.1:6733",
            loggingConfig: LoggingConfig(
                directory: FilePath(string: "/state/logs"),
                level: LogLevel.warn,
                retainedFiles: 7));
    }

    private static func emptyConfig() throws -> AstronomicalConfig {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "ares-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        try FileManager.default.createDirectory(
            atPath: instancePaths.stateDirectory.string,
            withIntermediateDirectories: true);
        try "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}".write(
            to: URL(fileURLWithPath: instancePaths.configFilePath.string),
            atomically: true,
            encoding: String.Encoding.utf8);
        return try AstronomicalConfig.loadFromInstancePaths(instancePaths);
    }

    func testResolvedGenerationIsStableAcrossEqualSnapshotsAndChangesWithContent() throws {
        let firstGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "doc-gen-1",
            discoveredModels: ResolvedRuntimeConfigTests.embeddingsSnapshot().discoveredModels,
            modelPolicyCatalog: ResolvedRuntimeConfigTests.embeddingsSnapshot().modelPolicyCatalog,
            unmatchedModelConfigIds: Array<String>());
        let secondGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "doc-gen-1",
            discoveredModels: ResolvedRuntimeConfigTests.embeddingsSnapshot().discoveredModels,
            modelPolicyCatalog: ResolvedRuntimeConfigTests.embeddingsSnapshot().modelPolicyCatalog,
            unmatchedModelConfigIds: Array<String>());
        XCTAssertEqual(firstGeneration, secondGeneration);
        XCTAssertEqual(firstGeneration.count, 64);

        let changedDocumentGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "doc-gen-2",
            discoveredModels: ResolvedRuntimeConfigTests.embeddingsSnapshot().discoveredModels,
            modelPolicyCatalog: ResolvedRuntimeConfigTests.embeddingsSnapshot().modelPolicyCatalog,
            unmatchedModelConfigIds: Array<String>());
        XCTAssertNotEqual(firstGeneration, changedDocumentGeneration);

        var unmatchedSnapshot: ResolvedRuntimeConfig = ResolvedRuntimeConfigTests.embeddingsSnapshot();
        unmatchedSnapshot.unmatchedModelConfigIds = ["ghost-model"];
        let changedUnmatchedGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "doc-gen-1",
            discoveredModels: unmatchedSnapshot.discoveredModels,
            modelPolicyCatalog: unmatchedSnapshot.modelPolicyCatalog,
            unmatchedModelConfigIds: unmatchedSnapshot.unmatchedModelConfigIds);
        XCTAssertNotEqual(firstGeneration, changedUnmatchedGeneration);
    }

    func testWorkerStartupConfigurationProjectsTheIpcBootstrap() throws {
        var snapshot: ResolvedRuntimeConfig = ResolvedRuntimeConfigTests.embeddingsSnapshot();
        snapshot.maximumMlxMemoryBytes = 32_000_000_000;
        snapshot.performanceAttributionEnabled = true;
        snapshot.persistentPromptCacheEnabled = false;
        snapshot.loggingConfig = LoggingConfig(
            directory: FilePath(string: "/state/logs"),
            level: LogLevel.debug,
            retainedFiles: 3);

        let workerStartupConfiguration: WorkerStartupConfiguration = snapshot.workerStartupConfiguration();

        XCTAssertEqual(workerStartupConfiguration.configurationGeneration, snapshot.configurationGeneration);
        XCTAssertEqual(
            workerStartupConfiguration.globalPromptCacheRootDirectory,
            "/state/prompt-cache");
        XCTAssertEqual(workerStartupConfiguration.globalPromptCacheMaximumSizeBytes, 50_000_000_000);
        XCTAssertEqual(workerStartupConfiguration.persistentPromptCacheEnabled, false);
        XCTAssertEqual(workerStartupConfiguration.configuredMaximumMlxMemoryBytes, 32_000_000_000);
        XCTAssertEqual(workerStartupConfiguration.performanceAttributionEnabled, true);
        XCTAssertEqual(workerStartupConfiguration.loggingDirectory, "/state/logs");
        XCTAssertEqual(workerStartupConfiguration.loggingLevel, WorkerLogLevel.debug);
        XCTAssertEqual(workerStartupConfiguration.retainedLogFileCount, 3);
    }

    func testReloadDiffClassifiesTheThreeOutcomesWithTheirFieldNames() throws {
        let current: ResolvedRuntimeConfig = ResolvedRuntimeConfigTests.embeddingsSnapshot();

        // A memory-ceiling edit applies in place.
        var memoryCandidate: ResolvedRuntimeConfig = current;
        memoryCandidate.maximumMlxMemoryBytes = 48_000_000_000;
        guard case let .noWorkerRestart(reloadedFields, _) = ConfigReloadDiff.compare(current: current, candidate: memoryCandidate) else {
            return XCTFail("a memory edit is an in-place reload");
        }
        XCTAssertEqual(reloadedFields, ["maximum_mlx_memory_gb"]);

        // A prompt-cache edit replaces the worker.
        var cacheCandidate: ResolvedRuntimeConfig = current;
        cacheCandidate.persistentPromptCacheEnabled = false;
        guard case let .restartWorker(reloadedWorkerFields, _) = ConfigReloadDiff.compare(current: current, candidate: cacheCandidate) else {
            return XCTFail("a prompt-cache edit requires a worker restart");
        }
        XCTAssertEqual(reloadedWorkerFields, ["persistent_prompt_cache_enabled"]);

        // A bind-address edit requires a full REST restart.
        var bindCandidate: ResolvedRuntimeConfig = current;
        bindCandidate.bindAddress = "127.0.0.1:6799";
        guard case let .restApiRestartRequired(reloadedInPlaceFields, restartRequiredFields, _) = ConfigReloadDiff.compare(current: current, candidate: bindCandidate) else {
            return XCTFail("a bind-address edit requires a REST restart");
        }
        XCTAssertEqual(reloadedInPlaceFields, Array<String>());
        XCTAssertEqual(restartRequiredFields, ["supervisor.bind_address"]);

        // Identical snapshots need nothing.
        guard case .noWorkerRestart = ConfigReloadDiff.compare(current: current, candidate: current) else {
            return XCTFail("identical snapshots need no restart");
        }
    }
}
