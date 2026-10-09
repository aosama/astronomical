import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing;

extension MlxGpuJourneyContainer {

    @Suite(.serialized, .tags(.realModelJourney),
        .enabled(if: RealModelJourneyGate.installedArtifactDirectory(
            environmentVariableName: "ASTRONOMICAL_ORNITH35_MOE_ARTIFACT_DIRECTORY") != nil))
    final class Ornith35MoeThroughputJourneyTests {

    private static let MODEL_ID: String = "Ornith-1.5-35B-A3B-OptiQ-4bit";
    private static let MAXIMUM_CONTEXT_TOKEN_COUNT: UInt32 = 24_576;
    private static let PREFILL_CHUNK_TOKEN_COUNT: UInt32 = 2_048;
        private static let MAXIMUM_MEASURED_OUTPUT_TOKEN_COUNT: UInt16 = 128;
        private static let WARMUP_OUTPUT_TOKEN_COUNT: UInt16 = 8;
        private static let DECODE_PROGRESS_TOKEN_INTERVAL: Int = 32;

        init() {
            signal(SIGPIPE, SIG_IGN);
            MLXMetallibLocator.overrideMetallibPathIfNecessary();
        }

        @Test(.timeLimit(.minutes(2)))
        func should_measure_ornith_optiq_prefill_and_decode_throughput() throws {
            let modelDirectory: String = try #require(
                RealModelJourneyGate.installedArtifactDirectory(
                    environmentVariableName: "ASTRONOMICAL_ORNITH35_MOE_ARTIFACT_DIRECTORY"),
                "the Ornith OptiQ artifact directory must resolve");
            Self.emitProgress("loading the Ornith OptiQ 4-bit artifact");
            let runtime: LoadedChatRuntime = try Qwen35ChatRuntime.buildArtifactRuntime(
                modelDirectory: modelDirectory,
                modelConfiguration: WorkerModelConfiguration.autoregressive(
                    WorkerAutoregressiveModelConfiguration(
                        modelId: Self.MODEL_ID,
                        maximumContextTokens: Self.MAXIMUM_CONTEXT_TOKEN_COUNT,
                        maximumOutputTokens: UInt32(Self.MAXIMUM_MEASURED_OUTPUT_TOKEN_COUNT),
                        chunking: WorkerChunkingConfiguration(
                             fixedPromptProcessingChunkSizeTokens: Self.PREFILL_CHUNK_TOKEN_COUNT,
                             fixedSsdStreamingPromptProcessingChunkSizeTokens:
                                Self.PREFILL_CHUNK_TOKEN_COUNT,
                             fullAttentionKeyValueGrowthTokens: Self.PREFILL_CHUNK_TOKEN_COUNT,
                            prefillGraphSubmissionLayerInterval: 1,
                            experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                            experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 1,
                            promptCacheBlockTokens: nil,
                             promptCacheCommonPrefixStrideBlocks: 4,
                             experimentalDecodeStageAttributionEnabled: false,
                             experimentalQuantizedKvCacheEnabled: false,
                             experimentalFusedMoeDecodeEnabled: false))),
                performanceAttributionEnabled: true);
            Self.emitProgress("artifact loaded; preparing the Romeo and Juliet request");

            let promptContent: String = try Self.romeoAndJulietPrompt();
            let warmupOutcome: (promptTokenCount: Int, generatedTokenCount: Int,
                prefillTimeSeconds: Double, decodeTimeSeconds: Double,
                prefillTokensPerSecond: Double, decodeTokensPerSecond: Double) =
                try Self.runGeneration(
                    runtime: runtime,
                    promptContent: promptContent,
                    maximumOutputTokenCount: Self.WARMUP_OUTPUT_TOKEN_COUNT,
                    requestSequence: 0);
            Self.emitProgress("warmup complete: \(warmupOutcome.generatedTokenCount) output tokens");

            let measuredOutcome: (promptTokenCount: Int, generatedTokenCount: Int,
                prefillTimeSeconds: Double, decodeTimeSeconds: Double,
                prefillTokensPerSecond: Double, decodeTokensPerSecond: Double) =
                try Self.runGeneration(
                    runtime: runtime,
                    promptContent: promptContent,
                    maximumOutputTokenCount: Self.MAXIMUM_MEASURED_OUTPUT_TOKEN_COUNT,
                    requestSequence: 1);

            print("[ornith-throughput] model=\(Self.MODEL_ID) "
                + "prompt_tokens=\(measuredOutcome.promptTokenCount) "
                + "generated_tokens=\(measuredOutcome.generatedTokenCount) "
                + "prefill_seconds=\(String(format: "%.3f", measuredOutcome.prefillTimeSeconds)) "
                + "decode_seconds=\(String(format: "%.3f", measuredOutcome.decodeTimeSeconds)) "
                + "prefill_tokens_per_second=\(String(format: "%.2f", measuredOutcome.prefillTokensPerSecond)) "
                + "decode_tokens_per_second=\(String(format: "%.2f", measuredOutcome.decodeTokensPerSecond))");
            #expect(measuredOutcome.promptTokenCount > 0,
                "the Romeo and Juliet prompt must tokenize to input tokens");
            #expect(measuredOutcome.generatedTokenCount > 0,
                "the measured request must generate output tokens");
            #expect(measuredOutcome.prefillTokensPerSecond.isFinite
                && measuredOutcome.prefillTokensPerSecond > 0);
            #expect(measuredOutcome.decodeTokensPerSecond.isFinite
                && measuredOutcome.decodeTokensPerSecond > 0);
        }

        private static func runGeneration(
            runtime: LoadedChatRuntime,
            promptContent: String,
            maximumOutputTokenCount: UInt16,
            requestSequence: UInt64
        ) throws -> (promptTokenCount: Int, generatedTokenCount: Int,
            prefillTimeSeconds: Double, decodeTimeSeconds: Double,
            prefillTokensPerSecond: Double, decodeTokensPerSecond: Double) {
            let requestId: RequestId = RequestId(rawRequestId: 30_000 + requestSequence);
            let chatCommand: ChatGenerationCommand = ChatGenerationCommand(
                requestId: requestId,
                model: Self.MODEL_ID,
                messages: [.user(content: promptContent, images: [])],
                tools: [],
                toolChoice: .none,
                settings: ChatGenerationSettings(
                    maxOutputTokens: maximumOutputTokenCount,
                    temperatureThousandths: 1_000,
                    topPThousandths: nil,
                    seed: nil,
                    thinkingBudget: nil),
                structuredGeneration: nil);
            let activeGeneration: any ActiveChatGeneration = try runtime.processor
                .prepareChatGeneration(chatCommand);
            guard let preparedRequest: Qwen35PreparedInferenceRequest = activeGeneration
                .inferenceRequest as? Qwen35PreparedInferenceRequest
            else {
                throw Qwen35MoePromptCacheError.promptCacheNotAttached;
            }
            let promptTokenCount: Int = preparedRequest.promptTokenIds.count;
            let generationStartResult: EngineGenerationStart = try runtime.engine
                .startGeneration(preparedRequest);
            #expect(generationStartResult.expertMemoryMode == .resident,
                "the throughput comparison must use resident expert execution");
            Self.emitProgress(
                "execution mode: \(generationStartResult.expertMemoryMode?.wireName ?? "unknown")");
            Self.emitProgress("generation started: \(promptTokenCount) input tokens, "
                + "maximum \(maximumOutputTokenCount) output tokens");

            let generationStart: ContinuousClock.Instant = ContinuousClock.now;
            var firstTokenElapsedSeconds: Double = 0;
            var generatedTokenCount: Int = 0;
            while generatedTokenCount < Int(maximumOutputTokenCount) {
                let generatedBoundary: GeneratedToken = try runtime.engine
                    .decodeNextToken(requestId: requestId);
                guard case let .tokenId(generatedTokenId, _, _, _, _, generationFinalization) =
                    generatedBoundary
                else {
                    continue;
                }
                if generatedTokenCount == 0 {
                    firstTokenElapsedSeconds = Self.elapsedSeconds(since: generationStart);
                }
                generatedTokenCount += 1;
                if generatedTokenCount % Self.DECODE_PROGRESS_TOKEN_INTERVAL == 0 {
                    let elapsedSeconds: Double = Self.elapsedSeconds(since: generationStart);
                    Self.emitProgress("decoded \(generatedTokenCount) tokens in "
                        + "\(String(format: "%.1f", elapsedSeconds)) seconds");
                }
                if generationFinalization != nil
                    || activeGeneration.isEndOfSequenceToken(generatedTokenId) {
                    break;
                }
            }
            let totalGenerationElapsedSeconds: Double =
                Self.elapsedSeconds(since: generationStart);
            _ = try runtime.engine.cancelGeneration(requestId: requestId);
            _ = try activeGeneration.finishOutputs();

            let decodeTimeSeconds: Double =
                max(totalGenerationElapsedSeconds - firstTokenElapsedSeconds, 0.000_001);
            let prefillTimeSeconds: Double = max(firstTokenElapsedSeconds, 0.000_001);
            return (
                promptTokenCount: promptTokenCount,
                generatedTokenCount: generatedTokenCount,
                prefillTimeSeconds: prefillTimeSeconds,
                decodeTimeSeconds: decodeTimeSeconds,
                prefillTokensPerSecond: Double(promptTokenCount) / prefillTimeSeconds,
                decodeTokensPerSecond: Double(max(generatedTokenCount - 1, 1)) / decodeTimeSeconds);
        }

        private static func romeoAndJulietPrompt() throws -> String {
            let repositoryRoot: URL = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent();
            let fixtureUrl: URL = repositoryRoot
                .appendingPathComponent("apps/inference-worker/tests/fixtures")
                .appendingPathComponent("model_metrics_5000_romeo_and_juliet_words.txt");
            let fixtureText: String = try String(contentsOf: fixtureUrl, encoding: .utf8);
            return "Use the supplied Romeo and Juliet source. "
                + "Name the two households in one short sentence.\n\n"
                + fixtureText;
        }

        private static func elapsedSeconds(since start: ContinuousClock.Instant) -> Double {
            let duration: Duration = start.duration(to: ContinuousClock.now);
            return Double(duration.components.seconds)
                + Double(duration.components.attoseconds) * 1e-18;
        }

        private static func emitProgress(_ message: String) -> Void {
            print("[ornith-throughput-progress] \(message)");
            fflush(stdout);
        }
    }
}
