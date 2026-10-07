import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Acceptance journeys for the resolved generation digest, migrating the
 * generation family of apps/supervisor/tests/hermetic/config_reload.rs:
 * the digest moves when a discovered artifact revision, an image capability
 * bit, or an image policy's artifact revision changes — the same content
 * change the worker would have to acknowledge with a restart.
 */
@Suite(.tags(.hermeticJourney))
final class ResolvedConfigurationGenerationTests {

    @Test
    func should_change_resolved_generation_when_artifact_revision_changes() throws -> Void {
        var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary<String, RuntimeModelPolicy>();
        modelPolicyCatalog["default"] = ResolvedConfigurationGenerationTests.chatPolicy();
        let firstGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "document-generation",
            discoveredModels: [ResolvedConfigurationGenerationTests.chatDiscoveredModel(revision: "revision-a")],
            modelPolicyCatalog: modelPolicyCatalog,
            unmatchedModelConfigIds: Array<String>());
        let secondGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "document-generation",
            discoveredModels: [ResolvedConfigurationGenerationTests.chatDiscoveredModel(revision: "revision-b")],
            modelPolicyCatalog: modelPolicyCatalog,
            unmatchedModelConfigIds: Array<String>());

        #expect(firstGeneration != secondGeneration);
    }

    @Test
    func should_change_resolved_generation_when_image_capability_changes() throws -> Void {
        var imagePolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary<String, RuntimeModelPolicy>();
        imagePolicyCatalog["FLUX.2-klein-4B"] = ResolvedConfigurationGenerationTests.imagePolicy(revision: "revision-a");
        let firstGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "document-generation",
            discoveredModels: [ResolvedConfigurationGenerationTests.imageDiscoveredModel(
                revision: "revision-a",
                supportsImageEditing: false)],
            modelPolicyCatalog: imagePolicyCatalog,
            unmatchedModelConfigIds: Array<String>());
        let secondGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "document-generation",
            discoveredModels: [ResolvedConfigurationGenerationTests.imageDiscoveredModel(
                revision: "revision-a",
                supportsImageEditing: true)],
            modelPolicyCatalog: imagePolicyCatalog,
            unmatchedModelConfigIds: Array<String>());

        #expect(firstGeneration != secondGeneration);
    }

    @Test
    func should_change_resolved_generation_when_image_artifact_revision_changes() throws -> Void {
        var firstPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary<String, RuntimeModelPolicy>();
        firstPolicyCatalog["FLUX.2-klein-4B"] = ResolvedConfigurationGenerationTests.imagePolicy(revision: "revision-a");
        var secondPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary<String, RuntimeModelPolicy>();
        secondPolicyCatalog["FLUX.2-klein-4B"] = ResolvedConfigurationGenerationTests.imagePolicy(revision: "revision-b");
        let firstGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "document-generation",
            discoveredModels: [ResolvedConfigurationGenerationTests.imageDiscoveredModel(
                revision: "revision-a",
                supportsImageEditing: false)],
            modelPolicyCatalog: firstPolicyCatalog,
            unmatchedModelConfigIds: Array<String>());
        let secondGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: "document-generation",
            discoveredModels: [ResolvedConfigurationGenerationTests.imageDiscoveredModel(
                revision: "revision-b",
                supportsImageEditing: false)],
            modelPolicyCatalog: secondPolicyCatalog,
            unmatchedModelConfigIds: Array<String>());

        #expect(firstGeneration != secondGeneration);
    }

    // MARK: Journey fixtures

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

    private static func imageDiscoveredModel(
        revision: String,
        supportsImageEditing: Bool
    ) -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: "FLUX.2-klein-4B",
            providerModelId: "black-forest-labs/FLUX.2-klein-4B",
            modelFamily: .flux2Klein,
            revision: revision,
            modelDirectory: FilePath(string: "/fictional/models/FLUX.2-klein-4B"),
            capabilities: .imageGeneration(DiscoveryImageGenerationCapabilities(
                supportsTextToImage: true,
                supportsImageEditing: supportsImageEditing,
                supportsMultipleReferenceImages: false,
                defaultSteps: 4,
                minimumDimensionPixels: 64,
                maximumDimensionPixels: 1_024,
                dimensionMultiplePixels: 16)),
            license: nil,
            modelSizeBytes: 4_000);
    }

    private static func imagePolicy(revision: String) -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: FilePath(string: "/fictional/models/FLUX.2-klein-4B"),
            generationDefaults: RuntimeModelGenerationDefaults.inert(),
            configuredMaximumContextTokens: nil,
            defaultMaximumContextTokens: 0,
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
            workerModelConfiguration: .flux2Klein(WorkerFlux2KleinModelConfiguration(
                modelId: "FLUX.2-klein-4B",
                modelFamily: .flux2Klein,
                artifactRevision: revision)));
    }

    private static func chatPolicy() -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: FilePath(string: "/fictional/models/default"),
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
                    fixedPromptProcessingChunkSizeTokens: 2_048,
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
