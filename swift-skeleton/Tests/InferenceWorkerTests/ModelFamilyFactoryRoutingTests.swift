import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import InferenceWorker;

/**
 * Hermetic routing journeys for the model family factory: the synthesized
 * tiny MoE artifact classifies into the qwen3.5 family and routes onto the
 * paged MoE engine, while the dense twin keeps routing onto the dense
 * engine — the architecture branch, proven in both directions.
 */
extension InferenceWorkerMlxJourneyContainer {

    @Suite(.serialized, .tags(.hermeticMlxJourney))
    final class ModelFamilyFactoryRoutingTests {

        init() {
            signal(SIGPIPE, SIG_IGN);
            MLXMetallibLocator.overrideMetallibPathIfNecessary();
        }

        private static let MAX_OUTPUT_TOKENS: UInt32 = 256;

        /**
         * The MoE artifact directory routes through the factory onto the MoE
         * engine; the same factory call on the dense twin stays on the dense
         * engine.
         */
        @Test(.timeLimit(.minutes(2)))
        func should_route_each_architecture_to_its_matched_engine() throws {
            let (moeDirectoryUrl, _): (URL, TinyMoeArtifactFixture.SynthesizedLayout) =
                try TinyMoeArtifactFixture.writeModelDirectory(includeTokenizerFiles: true);
            defer { try? FileManager.default.removeItem(at: moeDirectoryUrl); }
            let (denseDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
                try TinyDenseArtifactFixture.writeModelDirectory(includeTokenizerFiles: true);
            defer { try? FileManager.default.removeItem(at: denseDirectoryUrl); }

            let factory: ModelFamilyFactory = ModelFamilyFactory();
            let moeRuntime: LoadedChatRuntime = try factory.createChatRuntime(
                modelDirectory: moeDirectoryUrl.path,
                modelConfiguration: WorkerModelConfiguration.autoregressive(
                    InferenceWorkerMlxJourneyContainer.ModelFamilyFactoryRoutingTests
                        .autoregressiveConfiguration(modelId: "qwen3.5-moe")));
            #expect(moeRuntime.engine is Qwen35MoeEngine,
                "the classified qwen3.5 MoE directory must route onto the MoE engine");

            let denseRuntime: LoadedChatRuntime = try factory.createChatRuntime(
                modelDirectory: denseDirectoryUrl.path,
                modelConfiguration: WorkerModelConfiguration.autoregressive(
                    InferenceWorkerMlxJourneyContainer.ModelFamilyFactoryRoutingTests
                        .autoregressiveConfiguration(modelId: "qwen3.5")));
            #expect(denseRuntime.engine is Qwen35DenseEngine,
                "the classified qwen3.5 dense directory must stay on the dense engine");
        }

        private static func autoregressiveConfiguration(
            modelId: String
        ) -> WorkerAutoregressiveModelConfiguration {
            return WorkerAutoregressiveModelConfiguration(
                modelId: modelId,
                maximumContextTokens: 4096,
                maximumOutputTokens: 1024,
                chunking: WorkerChunkingConfiguration(
                    fixedPromptProcessingChunkSizeTokens: 512,
                    fixedSsdStreamingPromptProcessingChunkSizeTokens: 512,
                    fullAttentionKeyValueGrowthTokens: 512,
                    prefillGraphSubmissionLayerInterval: 1,
                    experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                    experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 1,
                    promptCacheBlockTokens: nil,
                    promptCacheCommonPrefixStrideBlocks: 1,
                    experimentalDecodeStageAttributionEnabled: false,
                    experimentalQuantizedKvCacheEnabled: false,
                    experimentalFusedMoeDecodeEnabled: false));
        }
    }
}
