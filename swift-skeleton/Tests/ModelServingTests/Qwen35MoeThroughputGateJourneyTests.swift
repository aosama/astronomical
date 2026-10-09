import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing;

/**
 * The gated E6 throughput-parity gate (#983): with a real installed
 * Qwen3.5 MoE artifact named through
 * `ASTRONOMICAL_QWEN35_MOE_ARTIFACT_DIRECTORY`, a short warmup request
 * (about 1,000 input tokens, 100 output tokens) is followed by the
 * measured request (the Romeo and Juliet source plus instruction, about
 * 10,500 input tokens, 1,000 output tokens) with the persistent prompt
 * cache disabled, and the measured completion's prompt-processing and
 * decode rates must meet the recorded Rust baselines (2,508 prefill
 * tokens/second, 66 decode tokens/second at 10,545 in / 1,000 out). The
 * rates print with the full phase evidence for the durable history
 * record. Runs in the optimized build; without the environment variable
 * the suite skips and the default run stays hermetic.
 */
extension MlxGpuJourneyContainer {

    @Suite(.serialized, .tags(.realModelJourney),
        .enabled(if: RealModelJourneyGate.qwen35MoeArtifactDirectory() != nil))
    final class Qwen35MoeThroughputGateJourneyTests {

        private static let ROMEO_AND_JULIET_INSTRUCTION: String =
            "Write a detailed study guide that preserves the characters, "
            + "relationships, major events, and tragic ending.";
        private static let WARMUP_TARGET_INPUT_TOKENS: Int = 1_000;
        private static let WARMUP_OUTPUT_TOKEN_COUNT: UInt16 = 100;
        private static let MEASURED_TARGET_INPUT_TOKENS: Int = 10_000;
        private static let MEASURED_OUTPUT_TOKEN_COUNT: UInt16 = 1_000;
        private static let BASELINE_PREFILL_TOKENS_PER_SECOND: Double = 2_508;
        private static let BASELINE_DECODE_TOKENS_PER_SECOND: Double = 66;
        private static let DECODE_PROGRESS_TOKEN_INTERVAL: Int = 100;
        // One full Romeo and Juliet repetition tokenizes to about 7,700 tokens,
        // so whole repetitions cannot land the mandated prompt sizes (about
        // 1,000 warmup input tokens, 10,000-11,000 measured input tokens).
        // These character-prefix fractions of a second repetition close the
        // gap; the in-test token-count ranges verify them against the real
        // tokenizer and fail loudly if the fixture or tokenizer shifts.
        private static let WARMUP_SOURCE_PREFIX_FRACTION: Double = 0.13;
        private static let MEASURED_EXTRA_SOURCE_PREFIX_FRACTION: Double = 0.36;
        private static let WARMUP_EXPECTED_PROMPT_TOKEN_RANGE: ClosedRange<Int> = 900...1_100;
        private static let MEASURED_EXPECTED_PROMPT_TOKEN_RANGE: ClosedRange<Int> =
            10_000...11_000;

        init() {
            signal(SIGPIPE, SIG_IGN);
            MLXMetallibLocator.overrideMetallibPathIfNecessary();
        }

        // Endurance gate: artifact load plus a 1,000-token warmup and a
        // 10,500-token measured generation at 1,000 output tokens.
        @Test(.timeLimit(.minutes(12)))
        func should_meet_the_recorded_prompt_processing_and_decode_throughput_baselines() throws {
            let modelDirectory: String = try #require(
                RealModelJourneyGate.qwen35MoeArtifactDirectory(),
                "the gated artifact directory must resolve");
            Self.emitProgress("loading the 35B artifact from the gated directory");
            let runtime: LoadedChatRuntime = try Qwen35ChatRuntime.buildArtifactRuntime(
                modelDirectory: modelDirectory,
                modelConfiguration: WorkerModelConfiguration.autoregressive(
                    WorkerAutoregressiveModelConfiguration(
                        modelId: "qwen3.5-moe",
                        maximumContextTokens: 24_576,
                        maximumOutputTokens: UInt32(Self.MEASURED_OUTPUT_TOKEN_COUNT),
                        chunking: WorkerChunkingConfiguration(
                            fixedPromptProcessingChunkSizeTokens: 2_048,
                            fixedSsdStreamingPromptProcessingChunkSizeTokens: 2_048,
                            fullAttentionKeyValueGrowthTokens: 2_048,
                            prefillGraphSubmissionLayerInterval: 1,
                            experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                            experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 1,
                            promptCacheBlockTokens: nil,
                            promptCacheCommonPrefixStrideBlocks: 4,
                            experimentalDecodeStageAttributionEnabled: false,
                            experimentalQuantizedKvCacheEnabled: false,
                            experimentalFusedMoeDecodeEnabled: false))));

            Self.emitProgress("artifact loaded; starting the warmup generation");

            let romeoAndJulietSource: String = try Self.romeoAndJulietSourceText();
            let warmupPromptContent: String = Self.assemblePrompt(
                romeoAndJulietSource: romeoAndJulietSource,
                fullRepetitionCount: 0,
                extraPrefixFraction: Self.WARMUP_SOURCE_PREFIX_FRACTION);
            let measuredPromptContent: String = Self.assemblePrompt(
                romeoAndJulietSource: romeoAndJulietSource,
                fullRepetitionCount: 1,
                extraPrefixFraction: Self.MEASURED_EXTRA_SOURCE_PREFIX_FRACTION);

            // Warmup: about 1,000 input tokens, 100 output tokens; the result
            // primes allocators and compiled paths and is never recorded.
            let warmupOutcome: ThroughputOutcome = try Self.runMeasuredGeneration(
                runtime: runtime,
                promptContent: warmupPromptContent,
                maximumOutputTokens: Self.WARMUP_OUTPUT_TOKEN_COUNT,
                generationSequence: 0,
                expectedPromptTokenRange: Self.WARMUP_EXPECTED_PROMPT_TOKEN_RANGE);
            Self.emitProgress("warmup complete (not recorded): prompt-processing "
                + "\(String(format: "%.1f", warmupOutcome.prefillTokensPerSecond)) tokens/second, "
                + "decode \(String(format: "%.1f", warmupOutcome.decodeTokensPerSecond)) tokens/second");

            // Measured request: the full Romeo and Juliet source, 1,000 output.
            Self.emitProgress("starting the measured generation: the full Romeo and "
                + "Juliet source, 1,000 output tokens");
            let measuredOutcome: ThroughputOutcome = try Self.runMeasuredGeneration(
                runtime: runtime,
                promptContent: measuredPromptContent,
                maximumOutputTokens: Self.MEASURED_OUTPUT_TOKEN_COUNT,
                generationSequence: 1,
                expectedPromptTokenRange: Self.MEASURED_EXPECTED_PROMPT_TOKEN_RANGE);

            print("[throughput-gate] prompt_tokens=\(measuredOutcome.promptTokenCount) "
                + "generated_tokens=\(measuredOutcome.generatedTokenCount) "
                + "prefill_time_seconds=\(String(format: "%.3f", measuredOutcome.prefillTimeSeconds)) "
                + "decode_time_seconds=\(String(format: "%.3f", measuredOutcome.decodeTimeSeconds)) "
                + "prefill_tokens_per_second=\(String(format: "%.1f", measuredOutcome.prefillTokensPerSecond)) "
                + "decode_tokens_per_second=\(String(format: "%.1f", measuredOutcome.decodeTokensPerSecond))");

            #expect(measuredOutcome.generatedTokenCount >= 850,
                "the gate must generate close to the full 1,000 output tokens");
            #expect(measuredOutcome.prefillTokensPerSecond
                >= Self.BASELINE_PREFILL_TOKENS_PER_SECOND,
                "prompt-processing throughput must meet the recorded Rust baseline of 2,508 tokens per second");
            #expect(measuredOutcome.decodeTokensPerSecond
                >= Self.BASELINE_DECODE_TOKENS_PER_SECOND,
                "decode throughput must meet the recorded Rust baseline of 66 tokens per second");
        }

        // MARK: - Fixtures

        /// Builds the gate prompt: the instruction followed by whole Romeo and
        /// Juliet repetitions plus a character prefix of one more repetition,
        /// sized to land the prompt in its mandated token range.
        private static func assemblePrompt(
            romeoAndJulietSource: String, fullRepetitionCount: Int, extraPrefixFraction: Double
        ) -> String {
            let extraPrefixCharacterCount: Int =
                Int(Double(romeoAndJulietSource.count) * extraPrefixFraction);
            let extraPrefix: String = String(
                romeoAndJulietSource.prefix(extraPrefixCharacterCount));
            return ROMEO_AND_JULIET_INSTRUCTION + "\n\n"
                + String(repeating: romeoAndJulietSource, count: fullRepetitionCount)
                + extraPrefix;
        }

        private static func runMeasuredGeneration(
            runtime: LoadedChatRuntime, promptContent: String, maximumOutputTokens: UInt16,
            generationSequence: Int, expectedPromptTokenRange: ClosedRange<Int>
        ) throws -> ThroughputOutcome {
            // The engine records cancelled request ids for the runtime's lifetime
            // and never clears them, so every invocation must use fresh ids or
            // the previous cancel poisons the next generation.
            let commandRequestId: RequestId =
                RequestId(rawRequestId: UInt64(8_000 + generationSequence * 2));
            let engineRequestId: RequestId =
                RequestId(rawRequestId: UInt64(8_001 + generationSequence * 2));
            let chatCommand: ChatGenerationCommand = ChatGenerationCommand(
                requestId: commandRequestId,
                model: "qwen3.5-moe",
                messages: [.user(content: promptContent, images: [])],
                tools: [],
                toolChoice: .none,
                settings: ChatGenerationSettings(
                    maxOutputTokens: maximumOutputTokens,
                    temperatureThousandths: 1_000,
                    topPThousandths: nil,
                    seed: nil,
                    thinkingBudget: nil),
                structuredGeneration: nil);
            let activeGeneration: any ActiveChatGeneration = try runtime.processor
                .prepareChatGeneration(chatCommand);
            defer { _ = try? activeGeneration.finishOutputs(); }
            guard let preparedRequest: Qwen35PreparedInferenceRequest = activeGeneration
                .inferenceRequest as? Qwen35PreparedInferenceRequest
            else {
                throw Qwen35MoePromptCacheError.promptCacheNotAttached;
            }
            let requestId: RequestId = engineRequestId;
            let promptTokenCount: Int = preparedRequest.promptTokenIds.count;
            #expect(expectedPromptTokenRange.contains(promptTokenCount),
                "the prompt token count \(promptTokenCount) must land in the mandated range \(expectedPromptTokenRange.lowerBound)...\(expectedPromptTokenRange.upperBound) input tokens");
            Self.emitProgress("prompt prepared: \(promptTokenCount) "
                + "input tokens; prompt processing begins");
            _ = try runtime.engine.startGeneration(preparedRequest);

            let generationStartClock: ContinuousClock.Instant = ContinuousClock.now;
            var firstTokenElapsedSeconds: Double = 0;
            var generatedTokenCount: Int = 0;
            var sawTerminalFinalization: Bool = false;
            while generatedTokenCount < Int(maximumOutputTokens) {
                let boundary: GeneratedToken = try runtime.engine
                    .decodeNextToken(requestId: requestId);
                guard case let .tokenId(generatedTokenId, _, _, _, _, generationFinalization) =
                    boundary
                else {
                    continue;
                }
                if generatedTokenCount == 0 {
                    firstTokenElapsedSeconds = Self.elapsedSeconds(since: generationStartClock);
                }
                generatedTokenCount += 1;
                if generatedTokenCount % Self.DECODE_PROGRESS_TOKEN_INTERVAL == 0 {
                    let elapsedSoFarSeconds: Double =
                        Self.elapsedSeconds(since: generationStartClock);
                    let tokensPerSecondSoFar: Double = Double(generatedTokenCount)
                        / max(elapsedSoFarSeconds, 0.000_001);
                    Self.emitProgress("decoded \(generatedTokenCount) of "
                        + "\(Int(maximumOutputTokens)) tokens in "
                        + "\(String(format: "%.1f", elapsedSoFarSeconds)) seconds "
                        + "(\(String(format: "%.1f", tokensPerSecondSoFar)) "
                        + "tokens/second so far)");
                }
                if generationFinalization != nil
                    || activeGeneration.isEndOfSequenceToken(generatedTokenId) {
                    sawTerminalFinalization = true;
                    break;
                }
            }
            let totalElapsedSeconds: Double = Self.elapsedSeconds(since: generationStartClock);
            _ = try runtime.engine.cancelGeneration(requestId: requestId);
            #expect(generatedTokenCount > 0, "the gate generation must produce tokens");
            if sawTerminalFinalization == false && generatedTokenCount
                == Int(maximumOutputTokens) {
                // The budget exhausted exactly at the limit; the stream is complete.
            }
            let decodeTimeSeconds: Double = totalElapsedSeconds - firstTokenElapsedSeconds;
            let prefillTokensPerSecond: Double =
                Double(preparedRequest.promptTokenIds.count)
                / max(firstTokenElapsedSeconds, 0.000_001);
            let decodeTokensPerSecond: Double =
                Double(max(generatedTokenCount - 1, 1))
                / max(decodeTimeSeconds, 0.000_001);
            return ThroughputOutcome(
                promptTokenCount: preparedRequest.promptTokenIds.count,
                generatedTokenCount: generatedTokenCount,
                prefillTimeSeconds: firstTokenElapsedSeconds,
                decodeTimeSeconds: decodeTimeSeconds,
                prefillTokensPerSecond: prefillTokensPerSecond,
                decodeTokensPerSecond: decodeTokensPerSecond);
        }

        /// The repository's Romeo and Juliet fixture text, resolved relative to
        /// this file so no developer path is ever embedded.
        private static func romeoAndJulietSourceText() throws -> String {
            let repositoryRoot: URL = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()   // Tests/ModelServingTests
                .deletingLastPathComponent()   // Tests
                .deletingLastPathComponent()   // swift-skeleton
                .deletingLastPathComponent();  // repository root
            let fixtureUrl: URL = repositoryRoot
                .appendingPathComponent("apps/inference-worker/tests/fixtures")
                .appendingPathComponent("model_metrics_5000_romeo_and_juliet_words.txt");
            return try String(contentsOf: fixtureUrl, encoding: .utf8);
        }

        private static func elapsedSeconds(since start: ContinuousClock.Instant) -> Double {
            let duration: Duration = start.duration(to: ContinuousClock.now);
            return Double(duration.components.seconds)
                + Double(duration.components.attoseconds) * 1e-18;
        }

        /// The gate is dark for minutes during artifact load, prompt processing
        /// and decode; these prints stream phase progress so a watched run never
        /// looks hung.
        private static func emitProgress(_ message: String) {
            print("[throughput-gate-progress] \(message)");
            fflush(stdout);
        }
    }
}

/// One measured gate generation's phase evidence.
private struct ThroughputOutcome {

    let promptTokenCount: Int;

    let generatedTokenCount: Int;

    let prefillTimeSeconds: Double;

    let decodeTimeSeconds: Double;

    let prefillTokensPerSecond: Double;

    let decodeTokensPerSecond: Double;
}
