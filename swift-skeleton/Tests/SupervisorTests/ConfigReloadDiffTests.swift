import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Acceptance journeys for the reload classification, migrating the
 * classification family of apps/supervisor/tests/hermetic/config_reload.rs:
 * every field family that replaces the worker, the two fields that require a
 * full application restart, and the resolved-generation fallback that still
 * restarts the worker when only the generation digest moved.
 */
@Suite(.tags(.hermeticJourney))
final class ConfigReloadDiffTests {

    @Test
    func should_restart_worker_when_a_discovered_artifact_revision_changes() throws -> Void {
        var currentSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        currentSnapshot.discoveredModels = [ConfigReloadDiffTests.chatDiscoveredModel(revision: "revision-a")];
        var candidateSnapshot: ResolvedRuntimeConfig = currentSnapshot;
        candidateSnapshot.configurationGeneration = ConfigReloadDiffTests.CHANGED_GENERATION;
        candidateSnapshot.discoveredModels = [ConfigReloadDiffTests.chatDiscoveredModel(revision: "revision-b")];

        guard case let .restartWorker(reloadedFields, _) = ConfigReloadDiff.compare(
            current: currentSnapshot,
            candidate: candidateSnapshot) else {
            Issue.record("an artifact revision change must replace the worker");
            return;
        }
        #expect(reloadedFields == ["discovered_model_artifacts"]);
    }

    @Test
    func should_restart_worker_to_acknowledge_changed_dormant_model_policies() throws -> Void {
        let currentSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        var candidateSnapshot: ResolvedRuntimeConfig = currentSnapshot;
        candidateSnapshot.configurationGeneration = ConfigReloadDiffTests.CHANGED_GENERATION;
        candidateSnapshot.unmatchedModelConfigIds = ["temporarily-absent-model"];

        guard case let .restartWorker(reloadedFields, _) = ConfigReloadDiff.compare(
            current: currentSnapshot,
            candidate: candidateSnapshot) else {
            Issue.record("a new dormant policy identity must replace the worker");
            return;
        }
        #expect(reloadedFields == ["dormant_model_policies"]);
    }

    @Test
    func should_restart_worker_when_dormant_policy_content_changes_without_changing_its_id() throws -> Void {
        var currentSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        currentSnapshot.unmatchedModelConfigIds = ["temporarily-absent-model"];
        var candidateSnapshot: ResolvedRuntimeConfig = currentSnapshot;
        candidateSnapshot.configurationGeneration = ConfigReloadDiffTests.CHANGED_GENERATION;

        guard case let .restartWorker(reloadedFields, _) = ConfigReloadDiff.compare(
            current: currentSnapshot,
            candidate: candidateSnapshot) else {
            Issue.record("a content-only generation change must still replace the worker");
            return;
        }
        #expect(reloadedFields == ["resolved_configuration"]);
    }

    @Test
    func should_restart_worker_when_any_per_model_execution_policy_changes() throws -> Void {
        let currentSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        var candidateSnapshot: ResolvedRuntimeConfig = currentSnapshot;
        candidateSnapshot.modelPolicyCatalog["default"] = ConfigReloadDiffTests.chatPolicy(
            modelDirectory: "/fictional/models/default",
            fixedPromptProcessingChunkSizeTokens: 4_096);

        guard case let .restartWorker(reloadedFields, _) = ConfigReloadDiff.compare(
            current: currentSnapshot,
            candidate: candidateSnapshot) else {
            Issue.record("an execution policy change must replace the worker");
            return;
        }
        #expect(reloadedFields == ["model_policies"]);
    }

    @Test
    func should_require_an_application_restart_when_performance_attribution_changes() throws -> Void {
        var currentSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        currentSnapshot.performanceAttributionEnabled = false;
        var candidateSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        candidateSnapshot.performanceAttributionEnabled = true;

        guard case let .restApiRestartRequired(_, restartRequiredFields, _) = ConfigReloadDiff.compare(
            current: currentSnapshot,
            candidate: candidateSnapshot) else {
            Issue.record("the attribution toggle must require an application restart");
            return;
        }
        #expect(restartRequiredFields == ["diagnostics.performance_attribution_enabled"]);
    }

    @Test
    func should_classify_model_directories_change_as_worker_restart() throws -> Void {
        var currentSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        currentSnapshot.modelPolicyCatalog["default"] = ConfigReloadDiffTests.chatPolicy(
            modelDirectory: "/tmp/models-a",
            fixedPromptProcessingChunkSizeTokens: 2_048);
        var candidateSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        candidateSnapshot.modelPolicyCatalog["default"] = ConfigReloadDiffTests.chatPolicy(
            modelDirectory: "/tmp/models-b",
            fixedPromptProcessingChunkSizeTokens: 2_048);

        guard case let .restartWorker(reloadedFields, _) = ConfigReloadDiff.compare(
            current: currentSnapshot,
            candidate: candidateSnapshot) else {
            Issue.record("a model directories change must replace the worker");
            return;
        }
        #expect(reloadedFields.contains("model_policies") == true);
    }

    @Test
    func should_restart_worker_when_empty_configured_model_root_changes() throws -> Void {
        var currentSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        currentSnapshot.configuredModelDirectories = [FilePath(string: "/tmp/empty-model-root-a")];
        var candidateSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        candidateSnapshot.configuredModelDirectories = [FilePath(string: "/tmp/empty-model-root-b")];

        guard case let .restartWorker(reloadedFields, _) = ConfigReloadDiff.compare(
            current: currentSnapshot,
            candidate: candidateSnapshot) else {
            Issue.record("a configured model root change must replace the worker");
            return;
        }
        #expect(reloadedFields.contains("model_directories") == true);
    }

    @Test
    func should_classify_prompt_cache_capacity_change_as_worker_restart() throws -> Void {
        var currentSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        currentSnapshot.promptCacheConfig = PromptCacheConfig(
            rootDirectory: FilePath(string: "/tmp/prompt-cache"),
            maximumSizeBytes: 1_000_000_000);
        var candidateSnapshot: ResolvedRuntimeConfig = ConfigReloadDiffTests.classificationSnapshot();
        candidateSnapshot.promptCacheConfig = PromptCacheConfig(
            rootDirectory: FilePath(string: "/tmp/prompt-cache"),
            maximumSizeBytes: 2_000_000_000);

        guard case let .restartWorker(reloadedFields, _) = ConfigReloadDiff.compare(
            current: currentSnapshot,
            candidate: candidateSnapshot) else {
            Issue.record("a prompt-cache capacity change must replace the worker");
            return;
        }
        #expect(reloadedFields.contains("prompt_cache") == true);
    }

    // MARK: Journey fixtures

    private static let CHANGED_GENERATION: String = String(repeating: "2", count: 64);

    private static func classificationSnapshot() -> ResolvedRuntimeConfig {
        var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary<String, RuntimeModelPolicy>();
        modelPolicyCatalog["default"] = ConfigReloadDiffTests.chatPolicy(
            modelDirectory: "/tmp/models/default",
            fixedPromptProcessingChunkSizeTokens: 2_048);
        return ResolvedRuntimeConfig(
            configurationGeneration: String(repeating: "1", count: 64),
            workerExecutablePath: FilePath(string: "/tmp/astronomical-inference-worker"),
            discoveredModels: Array<DiscoveryDiscoveredModel>(),
            modelDiscoveryDiagnostics: Array<DiscoveryModelDiscoveryDiagnostic>(),
            configuredModelDirectories: Array<FilePath>(),
            modelPolicyCatalog: modelPolicyCatalog,
            unmatchedModelConfigIds: Array<String>(),
            maximumMlxMemoryBytes: nil,
            performanceAttributionEnabled: false,
            completionAttributionEnabled: false,
            persistentPromptCacheEnabled: true,
            configuredPersistentPromptCacheEnabled: nil,
            configuredPromptCacheMaximumSizeBytes: nil,
            promptCacheConfig: PromptCacheConfig(
                rootDirectory: FilePath(string: "/tmp/prompt-cache"),
                maximumSizeBytes: 50_000_000_000),
            bindAddress: "127.0.0.1:6733",
            bindEndpoint: SocketEndpoint(host: "127.0.0.1", port: 6733),
            loggingConfig: LoggingConfig(
                directory: FilePath(string: "/tmp/astronomical-logs"),
                level: LogLevel.warn,
                retainedFiles: 7));
    }

    private static func chatDiscoveredModel(revision: String) -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: "default",
            providerModelId: nil,
            modelFamily: .qwen35,
            revision: revision,
            modelDirectory: FilePath(string: "/fictional/models/default"),
            capabilities: .chat(DiscoveryChatModelCapabilities(
                contextWindowTokens: 65_536,
                maximumInputTokens: 65_535,
                maximumOutputTokens: 20_480,
                supportsVision: false,
                supportsReasoning: true,
                supportsToolCalls: true)),
            license: nil,
            modelSizeBytes: 1_000);
    }

    private static func chatPolicy(
        modelDirectory: String,
        fixedPromptProcessingChunkSizeTokens: UInt32
    ) -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: FilePath(string: modelDirectory),
            generationDefaults: RuntimeModelGenerationDefaults(
                maximumOutputTokens: 20_480,
                configuredMaximumOutputTokens: nil,
                temperatureThousandths: nil,
                topPThousandths: nil),
            configuredMaximumContextTokens: nil,
            defaultMaximumContextTokens: 65_536,
            configuredChunkingFields: ConfiguredChunkingFields(
                fixedPromptProcessingChunkSizeTokens: false,
                fixedSsdStreamingPromptProcessingChunkSizeTokens: false,
                fullAttentionKeyValueGrowthTokens: false,
                prefillGraphSubmissionLayerInterval: false,
                experimentalSsdPagingPrefillGraphSubmissionLayerInterval: false,
                experimentalSsdPagingGenerationGraphSubmissionLayerInterval: false,
                promptCacheBlockTokens: false,
                promptCacheCommonPrefixStrideBlocks: false,
                experimentalDecodeStageAttributionEnabled: false,
                experimentalQuantizedKvCacheEnabled: false,
                experimentalFusedMoeDecodeEnabled: false),
            workerModelConfiguration: .autoregressive(WorkerAutoregressiveModelConfiguration(
                modelId: "default",
                maximumContextTokens: 65_536,
                maximumOutputTokens: 20_480,
                chunking: WorkerChunkingConfiguration(
                    fixedPromptProcessingChunkSizeTokens: fixedPromptProcessingChunkSizeTokens,
                    fixedSsdStreamingPromptProcessingChunkSizeTokens: 2_048,
                    fullAttentionKeyValueGrowthTokens: 256,
                    prefillGraphSubmissionLayerInterval: 0,
                    experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                    experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 3,
                    promptCacheBlockTokens: nil,
                    promptCacheCommonPrefixStrideBlocks: 4,
                    experimentalDecodeStageAttributionEnabled: false,
                    experimentalQuantizedKvCacheEnabled: false,
                    experimentalFusedMoeDecodeEnabled: false))));
    }
}
