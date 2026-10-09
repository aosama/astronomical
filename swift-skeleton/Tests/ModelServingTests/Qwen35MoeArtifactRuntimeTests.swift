import Foundation;

import Testing;

import MLX;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing;

/**
 * Hermetic journeys for the production MoE artifact path: the validated
 * tiny MoE artifact streams through the same production runtime builder
 * the dense runtime uses, the architecture branch selects the MoE engine,
 * paged expert execution installs with an empty retained plan, and one
 * routed generation proves the page seam stays live.
 *
 * Assertions are structural and config-derived (expert count, layer count,
 * shard payload sums, paged residency mode) — never golden-master bytes.
 * The suite is serialized so the MLX journeys never overlap (the
 * repository's one-model-at-a-time rule).
 */
extension HermeticMlxJourneyContainer {

    @Suite(.tags(.hermeticMlxJourney))
    final class Qwen35MoeArtifactRuntimeTests {

    private static let ROMEO_AND_JULIET_PROMPT: String = "What is the play about?";

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    /**
     * The production builder routes the MoE artifact onto the MoE engine,
     * the generation start reports the paged residency mode, and the routed
     * decode observes experts through the page seam.
     */
    @Test(.timeLimit(.minutes(2)))
    func should_serve_a_paged_generation_from_the_moe_artifact_runtime() throws {
        let (modelDirectoryUrl, _): (URL, TinyMoeArtifactFixture.SynthesizedLayout) =
            try TinyMoeArtifactFixture.writeModelDirectory(includeTokenizerFiles: true);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }
        let runtime: LoadedChatRuntime = try Qwen35ChatRuntime.buildArtifactRuntime(
            modelDirectory: modelDirectoryUrl.path,
            modelConfiguration: WorkerModelConfiguration.autoregressive(
                Qwen35MoeArtifactRuntimeTests.autoregressiveConfiguration()));

        let moeEngine: Qwen35MoeEngine = try #require(
            runtime.engine as? Qwen35MoeEngine,
            "the MoE artifact must route onto the MoE engine");

        let activeGeneration: any ActiveChatGeneration = try runtime.processor.prepareChatGeneration(
            Qwen35MoeArtifactRuntimeTests.chatCommand(requestId: 251));
        #expect(activeGeneration.promptTokenCount > 0);

        let requestId: RequestId = RequestId(rawRequestId: 251);
        let generationStart: EngineGenerationStart = try runtime.engine
            .startGeneration(activeGeneration.inferenceRequest);
        #expect(generationStart.expertMemoryMode == .paged,
            "the empty retained plan must classify as the paged mode");

        #expect(moeEngine.pagedRouteObservationCount() == 0);
        var generatedTokenCount: Int = 0;
        var sawEndOfSequence: Bool = false;
        for _ in 0..<48 {
            let boundary: GeneratedToken = try runtime.engine.decodeNextToken(requestId: requestId);
            guard case let .tokenId(generatedTokenId, _, _, _, _, _) = boundary else {
                continue;
            };
            #expect(generatedTokenId < 256);
            generatedTokenCount += 1;
            if activeGeneration.isEndOfSequenceToken(generatedTokenId) {
                sawEndOfSequence = true;
                break;
            }
            if generatedTokenCount >= 8 {
                break;
            }
        }
        #expect(generatedTokenCount > 0);
        #expect(moeEngine.pagedRouteObservationCount() > 0,
            "the routed decode must observe experts through the page seam");

        let flushedOutputs: Array<ChatGenerationOutput> = try activeGeneration.finishOutputs();
        _ = try runtime.engine.cancelGeneration(requestId: requestId);
        #expect(flushedOutputs.count <= generatedTokenCount);
        #expect(sawEndOfSequence || generatedTokenCount >= 8);
    }

    /**
     * The validated MoE artifact's structural facts match the synthesized
     * config: MoE architecture, the fixture layer and expert counts, one
     * shard, and a payload equal to the synthesized tensor bytes.
     */
    @Test(.timeLimit(.minutes(1)))
    func should_validate_the_moe_artifact_with_config_derived_structure() throws {
        let (modelDirectoryUrl, layout): (URL, TinyMoeArtifactFixture.SynthesizedLayout) =
            try TinyMoeArtifactFixture.writeModelDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }

        let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
            .validate(
                modelDirectory: modelDirectoryUrl.path,
                maxOutputTokens: TinyMoeArtifactFixture.MAX_OUTPUT_TOKENS);
        let repositoryConfiguration: Qwen3_5Config = validatedArtifact.config();
        #expect(repositoryConfiguration.feedForwardArchitecture() == .mixtureOfExperts);
        #expect(Int(repositoryConfiguration.layerCount()) == layout.layerCount);
        #expect(Int(repositoryConfiguration.expertCount()) == layout.expertCount);
        #expect(validatedArtifact.shardCount() > 0);
        #expect(validatedArtifact.totalPayloadBytes() == layout.totalPayloadBytes,
            "total payload bytes must equal the synthesized tensor bytes");
    }

    /**
     * A dense artifact routed through the production builder never reaches
     * the MoE engine: the architecture branch sends it to the dense engine.
     */
    @Test(.timeLimit(.minutes(2)))
    func should_route_the_dense_artifact_to_the_dense_engine() throws {
        let (modelDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory(includeTokenizerFiles: true);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }
        let runtime: LoadedChatRuntime = try Qwen35ChatRuntime.buildArtifactRuntime(
            modelDirectory: modelDirectoryUrl.path,
            modelConfiguration: WorkerModelConfiguration.autoregressive(
                Qwen35MoeArtifactRuntimeTests.autoregressiveConfiguration()));
        #expect(runtime.engine is Qwen35DenseEngine,
            "the dense artifact must route onto the dense engine");
    }

    // MARK: - Fixtures

    private static func autoregressiveConfiguration() -> WorkerAutoregressiveModelConfiguration {
        return WorkerAutoregressiveModelConfiguration(
            modelId: "qwen3.5-moe",
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

    /// The Romeo and Juliet fixture prompt: the model-visible text the
    /// journey renders through the real chat template.
    private static func chatCommand(requestId: UInt64) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: "qwen3.5-moe",
            messages: [
                .system(content: "answer plainly"),
                .user(content: ROMEO_AND_JULIET_PROMPT, images: []),
            ],
            tools: [],
            toolChoice: .auto,
            settings: ChatGenerationSettings(
                maxOutputTokens: 16,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: 11,
                thinkingBudget: nil),
            structuredGeneration: nil);
    }
}

}
