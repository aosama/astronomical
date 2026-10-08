import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing;

/**
 * The gated real-model MoE journey: when this machine names an installed
 * Qwen3.5 MoE artifact directory through
 * `ASTRONOMICAL_QWEN35_MOE_ARTIFACT_DIRECTORY`, the production runtime
 * builder streams the real shard weights into the paged MoE engine and
 * serves the Romeo and Juliet fixture with every routed expert read from
 * disk through the page seam.
 *
 * Assertions stay structural and config-derived: matched engine type, the
 * paged residency mode, live route observations, positive token counts,
 * and non-empty assistant-visible text. The journey prints the measured
 * prompt-processing and token-generation throughput so the acceptance
 * record carries the numbers; no golden-master constant couples it to one
 * packaging variant. Without the environment variable the suite is
 * skipped and the default run stays hermetic by construction.
 */
@Suite(.serialized, .tags(.realModelJourney),
    .enabled(if: RealModelJourneyGate.qwen35MoeArtifactDirectory() != nil))
final class Qwen35MoeRealModelJourneyTests {

    private static let romeoAndJulietPrompt: String =
        "You are a concise literature assistant. In one sentence, name the play "
        + "these lines come from: \"O Romeo, Romeo, wherefore art thou Romeo?\"";

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    // The 35B artifact's shard load alone consumes about a minute, so the
    // ceiling exceeds the standard two minutes for this load-bound real
    // model journey.
    @Test(.timeLimit(.minutes(4)))
    func should_serve_the_romeo_and_juliet_fixture_through_the_paged_moe_runtime() throws {
        let modelDirectory: String = try #require(
            RealModelJourneyGate.qwen35MoeArtifactDirectory(),
            "the gated artifact directory must resolve");
        let runtime: LoadedChatRuntime = try Qwen35ChatRuntime.buildArtifactRuntime(
            modelDirectory: modelDirectory,
            modelConfiguration: WorkerModelConfiguration.autoregressive(
                Qwen35MoeRealModelJourneyTests.autoregressiveConfiguration()));
        guard let moeEngine: Qwen35MoeEngine = runtime.engine as? Qwen35MoeEngine else {
            Issue.record("the real MoE artifact must route onto the MoE engine");
            return;
        }

        let activeGeneration: any ActiveChatGeneration = try runtime.processor
            .prepareChatGeneration(Qwen35MoeRealModelJourneyTests.chatCommand());
        #expect(activeGeneration.promptTokenCount > 0);

        let requestId: RequestId = RequestId(rawRequestId: 401);
        let generationStart: EngineGenerationStart = try runtime.engine
            .startGeneration(activeGeneration.inferenceRequest);
        #expect(generationStart.expertMemoryMode == .paged,
            "the production MoE load must install the paged residency");

        let promptTokenCount: Int = activeGeneration.promptTokenCount;
        let generationStartClock: ContinuousClock.Instant = ContinuousClock.now;
        var firstTokenElapsedSeconds: Double = 0;
        var generatedTokenCount: Int = 0;
        var visibleText: String = "";
        var reasoningText: String = "";
        for _ in 0..<64 {
            let boundary: GeneratedToken = try runtime.engine.decodeNextToken(requestId: requestId);
            if case let .tokenId(generatedTokenId, _, _, _, _, _) = boundary {
                if generatedTokenCount == 0 {
                    firstTokenElapsedSeconds = Qwen35MoeRealModelJourneyTests
                        .elapsedSeconds(since: generationStartClock);
                }
                generatedTokenCount += 1;
                let translation: ModelGeneratedTokenTranslation = try activeGeneration
                    .translateGeneratedToken(generatedTokenId);
                for output: ChatGenerationOutput in translation.publicOutputs {
                    switch output {
                    case let .text(textChunk): visibleText += textChunk;
                    case let .reasoning(reasoningChunk): reasoningText += reasoningChunk;
                    case .toolCall: break;
                    }
                }
                if activeGeneration.isEndOfSequenceToken(generatedTokenId) {
                    break;
                }
            }
        }
        let totalElapsedSeconds: Double = Qwen35MoeRealModelJourneyTests
            .elapsedSeconds(since: generationStartClock);
        let decodeElapsedSeconds: Double = totalElapsedSeconds - firstTokenElapsedSeconds;

        let flushedOutputs: Array<ChatGenerationOutput> = try activeGeneration.finishOutputs();
        _ = try runtime.engine.cancelGeneration(requestId: requestId);
        // A thinking artifact spends its early budget in the reasoning
        // channel, so the structural answer proof is non-empty combined
        // assistant output across both channels.
        var flushedText: String = "";
        for output: ChatGenerationOutput in flushedOutputs {
            switch output {
            case let .text(textChunk): flushedText += textChunk;
            case let .reasoning(reasoningChunk): reasoningText += reasoningChunk;
            case .toolCall: break;
            }
        }
        visibleText += flushedText;

        #expect(moeEngine.pagedRouteObservationCount() > 0,
            "every routed decode step must observe experts through the page seam");
        #expect(generatedTokenCount > 0);
        #expect((visibleText + reasoningText).isEmpty == false,
            "the real model must produce assistant output for the fixture prompt");
        print("[qwen35-moe-real-journey] prompt_tokens=\(promptTokenCount) "
            + "prefill_tok_per_s=\(String(format: "%.1f", Double(promptTokenCount) / max(firstTokenElapsedSeconds, 0.000001))) "
            + "generated_tokens=\(generatedTokenCount) "
            + "decode_tok_per_s=\(String(format: "%.1f", Double(generatedTokenCount) / max(decodeElapsedSeconds, 0.000001))) "
            + "text_characters=\(visibleText.count) reasoning_characters=\(reasoningText.count)");
    }

    /// The elapsed seconds from one clock instant to now, as a fractional
    /// Double from the duration's second and attosecond components.
    private static func elapsedSeconds(since start: ContinuousClock.Instant) -> Double {
        let duration: Duration = start.duration(to: ContinuousClock.now);
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) * 1e-18;
    }

    private static func autoregressiveConfiguration() -> WorkerAutoregressiveModelConfiguration {
        return WorkerAutoregressiveModelConfiguration(
            modelId: "qwen3.5-moe",
            maximumContextTokens: 8192,
            maximumOutputTokens: 256,
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

    private static func chatCommand() -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: 401),
            model: "qwen3.5-moe",
            messages: [
                .system(content: "answer plainly"),
                .user(content: Qwen35MoeRealModelJourneyTests.romeoAndJulietPrompt, images: []),
            ],
            tools: [],
            toolChoice: .auto,
            settings: ChatGenerationSettings(
                maxOutputTokens: 48,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: 7,
                thinkingBudget: nil),
            structuredGeneration: nil);
    }
}
