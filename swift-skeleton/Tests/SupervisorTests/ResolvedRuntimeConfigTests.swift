import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Acceptance journeys for the resolved serving snapshot's pure surface: the
 * daemon can name one fully resolved configuration path-free (the resolved
 * generation), project the worker bootstrap DTO from it, and classify a
 * candidate config change into the three reload outcomes. Snapshots are
 * constructed directly: these types are pure data plus derived values.
 */
@Suite(.tags(.hermeticJourney))
final class ResolvedRuntimeConfigTests {

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
            persistentPromptCacheEnabled: true,
            configuredPersistentPromptCacheEnabled: nil,
            configuredPromptCacheMaximumSizeBytes: nil,
            promptCacheConfig: PromptCacheConfig(
                rootDirectory: FilePath(string: "/state/prompt-cache"),
                maximumSizeBytes: 50_000_000_000),
            bindAddress: "127.0.0.1:6733",
            bindEndpoint: SocketEndpoint(host: "127.0.0.1", port: 6733),
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

    @Test
    func should_derive_a_stable_generation_for_equal_snapshots_and_a_changed_one_for_differing_content() throws {
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
        #expect(firstGeneration == secondGeneration);
        #expect(firstGeneration.count == 64);

        let changedDocumentGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "doc-gen-2",
            discoveredModels: ResolvedRuntimeConfigTests.embeddingsSnapshot().discoveredModels,
            modelPolicyCatalog: ResolvedRuntimeConfigTests.embeddingsSnapshot().modelPolicyCatalog,
            unmatchedModelConfigIds: Array<String>());
        #expect(firstGeneration != changedDocumentGeneration);

        var unmatchedSnapshot: ResolvedRuntimeConfig = ResolvedRuntimeConfigTests.embeddingsSnapshot();
        unmatchedSnapshot.unmatchedModelConfigIds = ["ghost-model"];
        let changedUnmatchedGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "doc-gen-1",
            discoveredModels: unmatchedSnapshot.discoveredModels,
            modelPolicyCatalog: unmatchedSnapshot.modelPolicyCatalog,
            unmatchedModelConfigIds: unmatchedSnapshot.unmatchedModelConfigIds);
        #expect(firstGeneration != changedUnmatchedGeneration);
    }

    @Test
    func should_project_the_worker_startup_configuration_from_the_resolved_snapshot() throws {
        var snapshot: ResolvedRuntimeConfig = ResolvedRuntimeConfigTests.embeddingsSnapshot();
        snapshot.maximumMlxMemoryBytes = 32_000_000_000;
        snapshot.performanceAttributionEnabled = true;
        snapshot.persistentPromptCacheEnabled = false;
        snapshot.loggingConfig = LoggingConfig(
            directory: FilePath(string: "/state/logs"),
            level: LogLevel.debug,
            retainedFiles: 3);

        let workerStartupConfiguration: WorkerStartupConfiguration = snapshot.workerStartupConfiguration();

        #expect(workerStartupConfiguration.configurationGeneration == snapshot.configurationGeneration);
        #expect(
            workerStartupConfiguration.globalPromptCacheRootDirectory
                == "/state/prompt-cache");
        #expect(workerStartupConfiguration.globalPromptCacheMaximumSizeBytes == 50_000_000_000);
        #expect(workerStartupConfiguration.persistentPromptCacheEnabled == false);
        #expect(workerStartupConfiguration.configuredMaximumMlxMemoryBytes == 32_000_000_000);
        #expect(workerStartupConfiguration.performanceAttributionEnabled == true);
        #expect(workerStartupConfiguration.loggingDirectory == "/state/logs");
        #expect(workerStartupConfiguration.loggingLevel == WorkerLogLevel.debug);
        #expect(workerStartupConfiguration.retainedLogFileCount == 3);
    }

    @Test
    func should_classify_a_config_change_into_the_three_reload_outcomes_with_their_field_names() throws {
        let current: ResolvedRuntimeConfig = ResolvedRuntimeConfigTests.embeddingsSnapshot();

        // A memory-ceiling edit applies in place.
        var memoryCandidate: ResolvedRuntimeConfig = current;
        memoryCandidate.maximumMlxMemoryBytes = 48_000_000_000;
        guard case let .noWorkerRestart(reloadedFields, _) = ConfigReloadDiff.compare(current: current, candidate: memoryCandidate) else {
            Issue.record("a memory edit is an in-place reload");
            return;
        }
        #expect(reloadedFields == ["maximum_mlx_memory_gb"]);

        // A prompt-cache edit replaces the worker.
        var cacheCandidate: ResolvedRuntimeConfig = current;
        cacheCandidate.persistentPromptCacheEnabled = false;
        guard case let .restartWorker(reloadedWorkerFields, _) = ConfigReloadDiff.compare(current: current, candidate: cacheCandidate) else {
            Issue.record("a prompt-cache edit requires a worker restart");
            return;
        }
        #expect(reloadedWorkerFields == ["persistent_prompt_cache_enabled"]);

        // A bind-address edit requires a full REST restart.
        var bindCandidate: ResolvedRuntimeConfig = current;
        bindCandidate.bindAddress = "127.0.0.1:6799";
        guard case let .restApiRestartRequired(reloadedInPlaceFields, restartRequiredFields, _) = ConfigReloadDiff.compare(current: current, candidate: bindCandidate) else {
            Issue.record("a bind-address edit requires a REST restart");
            return;
        }
        #expect(reloadedInPlaceFields == Array<String>());
        #expect(restartRequiredFields == ["supervisor.bind_address"]);

        // Identical snapshots need nothing.
        guard case .noWorkerRestart = ConfigReloadDiff.compare(current: current, candidate: current) else {
            Issue.record("identical snapshots need no restart");
            return;
        }
    }
}
