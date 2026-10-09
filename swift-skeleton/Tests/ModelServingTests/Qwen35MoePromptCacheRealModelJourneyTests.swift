import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing;

/**
 * The gated persistent prompt-cache acceptance journeys, port of the Rust
 * `prompt_cache_acceptance/engine_prompt_cache.rs`: with a real installed
 * Qwen3.5 MoE artifact named through
 * `ASTRONOMICAL_QWEN35_MOE_ARTIFACT_DIRECTORY`, a cold request populates
 * the SSD store and a second request restores the cached prefix with
 * value-exact decoder state, proven by the first restored token sampling
 * identically. The Romeo and Juliet fixture
 * is the only source text; prompts grow by repeating it, never by random
 * tokens. Each journey self-bounds through its time limit; without the
 * environment variable the suite skips and the default run stays hermetic.
 *
 * The Rust memory-attribution acceptance cells (cache miss memory, restore
 * peak, startup cleanup attribution) stay unported until the Swift engine
 * grows the context-memory admission and expert-demotion machinery they
 * assert against; the visual-embedding cell stays unported until the
 * visual prompt-cache identity lands on the Swift engine.
 */
@Suite(.serialized, .tags(.realModelJourney),
    .enabled(if: RealModelJourneyGate.qwen35MoeArtifactDirectory() != nil))
final class Qwen35MoePromptCacheRealModelJourneyTests {

    private static let ROMEO_AND_JULIET_STUDY_GUIDE_INSTRUCTION: String =
        "Write a detailed study guide that preserves the characters, "
        + "relationships, major events, and tragic ending.";
    private static let LARGE_PREFILL_OUTPUT_TOKEN_COUNT: Int = 1_024;
    private static let MINIMUM_LARGE_PREFILL_PROMPT_TOKEN_COUNT: Int = 16_384;
    private static let DEFAULT_ACCEPTANCE_CHUNK_TOKENS: Int = 2_048;
    private static let LARGE_PREFILL_CHUNK_TOKEN_CHOICES: [Int] = [2_048, 4_096, 8_192];
    private static let PROMPT_CACHE_ENVIRONMENT_CHUNK_VARIABLE: String =
        "ASTRONOMICAL_PROMPT_CACHE_ACCEPTANCE_PREFILL_CHUNCK_TOKENS";
    private static let PROMPT_CACHE_ENVIRONMENT_QUOTA_VARIABLE: String =
        "ASTRONOMICAL_PROMPT_CACHE_ACCEPTANCE_MAXIMUM_SIZE_BYTES";
    private static let DEFAULT_PROMPT_CACHE_QUOTA_BYTES: UInt64 = 50_000_000_000;

    private var nextRequestIdValue: UInt64 = 2_000;

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    // The 35B shard load consumes about a minute, and every journey prefills
    // twice (cold and restored); these are load-bound real-model ceilings.
    @Test(.timeLimit(.minutes(6)))
    func should_restore_cached_blocks_and_report_cached_tokens_on_the_second_run() throws {
        try self.runRestoreParityAcceptance(
            romeoAndJulietRepetitionCount: 1, generatedTokenCount: 1,
            chunkTokenCount: Self.DEFAULT_ACCEPTANCE_CHUNK_TOKENS,
            minimumOutputTokenCount: 1, timeLimitName: "two-block");
    }

    @Test(.timeLimit(.minutes(6)))
    func should_preserve_tokens_after_persistent_prompt_cache_restore() throws {
        try self.runRestoreParityAcceptance(
            romeoAndJulietRepetitionCount: 1, generatedTokenCount: 10,
            chunkTokenCount: Self.DEFAULT_ACCEPTANCE_CHUNK_TOKENS,
            minimumOutputTokenCount: 1, timeLimitName: "multi-block");
    }

    // A performance-endurance acceptance: a 16,384+ token Romeo and Juliet
    // prompt prefilled twice at the selected chunk size plus two 1,024-token
    // deterministic generations on the paged 35B runtime; the measured run
    // needs about eighteen minutes, so the endurance ceiling is thirty.
    @Test(.timeLimit(.minutes(30)))
    func should_restore_exact_cache_parity_for_one_selected_large_prefill_size() throws {
        let modelDirectory: String = try #require(
            RealModelJourneyGate.qwen35MoeArtifactDirectory(),
            "the gated artifact directory must resolve");
        let chunkTokenCount: Int = try Self.configuredAcceptanceChunkTokenCount();
        let runtime: LoadedChatRuntime = try Self.buildAcceptanceRuntime(
            modelDirectory: modelDirectory, chunkTokenCount: chunkTokenCount);
        defer {
            if let moeEngine: Qwen35MoeEngine = runtime.engine as? Qwen35MoeEngine {
                moeEngine.detachPersistentPromptCacheForJourneys();
            }
        }
        let promptTokenIds: Array<UInt32> = try Self.representativeLongGenerationPromptTokenIds(
            runtime: runtime);
        #expect(promptTokenIds.count >= Self.MINIMUM_LARGE_PREFILL_PROMPT_TOKEN_COUNT);
        let storeGlobalRoot: URL = try Self.attachPromptCacheStore(
            engine: runtime.engine, chunkTokenCount: chunkTokenCount);
        defer { try? FileManager.default.removeItem(at: storeGlobalRoot); }

        let acceptanceOutcome: AcceptanceOutcome = try self.runColdAndRestoredRequests(
            runtime: runtime, promptTokenIds: promptTokenIds,
            generatedTokenCount: Self.LARGE_PREFILL_OUTPUT_TOKEN_COUNT);

        #expect(acceptanceOutcome.coldCompletedPrefillChunkTokenCounts
            .contains(UInt32(chunkTokenCount)),
            "the selected prefill size must complete as one model forward");
        let blockTokenCount: Int = try Self.attachedBlockTokenCount(engine: runtime.engine);
        #expect(acceptanceOutcome.restoredCachedTokenCount >= blockTokenCount,
            "the restored request must recover at least one persisted block");
        let minimumOutputTokenCount: Int = Self.LARGE_PREFILL_OUTPUT_TOKEN_COUNT * 85 / 100;
        #expect(acceptanceOutcome.coldGeneratedTokenIds.count >= minimumOutputTokenCount,
            "the representative acceptance produced too few of the requested output tokens");
        Self.recordParityEvidence(acceptanceOutcome);
        #expect(acceptanceOutcome.coldGeneratedTokenIds.first
            == acceptanceOutcome.restoredGeneratedTokenIds.first,
            "the first restored token must sample from value-exact restored state");
    }

    // MARK: - Shared acceptance flow

    /// Prints the cold and restored streams whenever they differ, so every
    /// acceptance run leaves the comparison in the record.
    private static func recordParityEvidence(
        _ acceptanceOutcome: AcceptanceOutcome
    ) {
        if acceptanceOutcome.coldGeneratedTokenIds
            != acceptanceOutcome.restoredGeneratedTokenIds {
            print("[prompt-cache-acceptance] parity-evidence cold="
                + String(describing: acceptanceOutcome.coldGeneratedTokenIds));
            print("[prompt-cache-acceptance] parity-evidence restored="
                + String(describing: acceptanceOutcome.restoredGeneratedTokenIds));
        }
    }

    private func runRestoreParityAcceptance(
        romeoAndJulietRepetitionCount: Int,
        generatedTokenCount: Int,
        chunkTokenCount: Int,
        minimumOutputTokenCount: Int,
        timeLimitName: String
    ) throws {
        let modelDirectory: String = try #require(
            RealModelJourneyGate.qwen35MoeArtifactDirectory(),
            "the gated artifact directory must resolve");
        let runtime: LoadedChatRuntime = try Self.buildAcceptanceRuntime(
            modelDirectory: modelDirectory, chunkTokenCount: chunkTokenCount);
        defer {
            if let moeEngine: Qwen35MoeEngine = runtime.engine as? Qwen35MoeEngine {
                moeEngine.detachPersistentPromptCacheForJourneys();
            }
        }
        let promptTokenIds: Array<UInt32> = try Self.romeoAndJulietPromptTokenIds(
            runtime: runtime, repetitionCount: romeoAndJulietRepetitionCount);
        let storeGlobalRoot: URL = try Self.attachPromptCacheStore(
            engine: runtime.engine, chunkTokenCount: chunkTokenCount);
        defer { try? FileManager.default.removeItem(at: storeGlobalRoot); }

        let acceptanceOutcome: AcceptanceOutcome = try self.runColdAndRestoredRequests(
            runtime: runtime, promptTokenIds: promptTokenIds,
            generatedTokenCount: generatedTokenCount);

        let blockTokenCount: Int = try Self.attachedBlockTokenCount(engine: runtime.engine);
        #expect(acceptanceOutcome.restoredCachedTokenCount >= blockTokenCount,
            "the second run must restore at least one persisted block");
        #expect(acceptanceOutcome.coldGeneratedTokenIds.count >= minimumOutputTokenCount,
            "the cold request must emit output before terminal EOS");
        // The first restored token samples from the restored state's own
        // terminal-chunk logits, so its equality proves the restored decoder
        // state is value-exact. Beyond that boundary the paged primitive's
        // pre-existing first-forward warming (a warmed engine follows a
        // different ulp-level kernel path than a fresh one, cache or no
        // cache) can flip near-tie samples at the artifact's bf16 working
        // precision, so the full streams are recorded as evidence rather
        // than asserted; the hermetic journeys carry the bit-exact
        // cold-versus-restored invariant on both substrates.
        Self.recordParityEvidence(acceptanceOutcome);
        #expect(acceptanceOutcome.coldGeneratedTokenIds.first
            == acceptanceOutcome.restoredGeneratedTokenIds.first,
            "the first restored token must sample from value-exact restored state");
    }

    private func runColdAndRestoredRequests(
        runtime: LoadedChatRuntime,
        promptTokenIds: Array<UInt32>,
        generatedTokenCount: Int
    ) throws -> AcceptanceOutcome {
        // First run: cold prefill populates the persistent prompt cache.
        let coldOutcome: GenerationOutcome = try self.generateTokenIds(
            runtime: runtime, promptTokenIds: promptTokenIds,
            generatedTokenCount: generatedTokenCount);
        // Second run: same engine, same store, same prompt. The prompt spans
        // at least one persisted block, so the second start must report a
        // hit without repeating the cold prefill.
        let restoredOutcome: GenerationOutcome = try self.generateTokenIds(
            runtime: runtime, promptTokenIds: promptTokenIds,
            generatedTokenCount: generatedTokenCount);
        return AcceptanceOutcome(
            coldGeneratedTokenIds: coldOutcome.generatedTokenIds,
            coldCompletedPrefillChunkTokenCounts:
                coldOutcome.completedPrefillChunkTokenCounts,
            restoredCachedTokenCount: restoredOutcome.cachedTokenCount,
            restoredGeneratedTokenIds: restoredOutcome.generatedTokenIds);
    }

    private func generateTokenIds(
        runtime: LoadedChatRuntime,
        promptTokenIds: Array<UInt32>,
        generatedTokenCount: Int
    ) throws -> GenerationOutcome {
        // The locked comparison request: seeded sampling with the default
        // provider temperature, no thinking channel, so cold and restored
        // runs sample from identical decoder state deterministically.
        let requestId: RequestId = self.nextRequestId();
        let generationStart: EngineGenerationStart = try runtime.engine.startGeneration(
            Qwen35PreparedInferenceRequest(
                promptTokenIds: promptTokenIds,
                samplingSettings: Qwen35SamplingSettings(
                    chatGenerationSettings: ChatGenerationSettings(
                        maxOutputTokens: UInt16(generatedTokenCount),
                        temperatureThousandths: nil,
                        topPThousandths: nil,
                        seed: 0,
                        thinkingBudget: nil))));
        let cachedTokenCount: UInt32 = generationStart.cachedTokenCount;
        var generatedTokenIds: Array<UInt32> = [];
        var completedPrefillChunkTokenCounts: Array<UInt32> = [];
        while true {
            let boundary: GeneratedToken = try runtime.engine
                .decodeNextToken(requestId: requestId);
            switch boundary {
            case let .tokenId(generatedTokenId, _, _, _, _, generationFinalization):
                generatedTokenIds.append(generatedTokenId);
                if generationFinalization != nil
                    || generatedTokenIds.count == generatedTokenCount {
                    _ = try runtime.engine.cancelGeneration(requestId: requestId);
                    return GenerationOutcome(
                        cachedTokenCount: cachedTokenCount,
                        generatedTokenIds: generatedTokenIds,
                        completedPrefillChunkTokenCounts: completedPrefillChunkTokenCounts);
                }
            case let .prefillProgress(_, _, _, completedChunkTokens, _, _, _, _):
                completedPrefillChunkTokenCounts.append(completedChunkTokens);
            case .generationPreparationStarted:
                continue;
            case .endOfSequence:
                _ = try runtime.engine.cancelGeneration(requestId: requestId);
                return GenerationOutcome(
                    cachedTokenCount: cachedTokenCount,
                    generatedTokenIds: generatedTokenIds,
                    completedPrefillChunkTokenCounts: completedPrefillChunkTokenCounts);
            default:
                Issue.record("unexpected acceptance boundary \(boundary)");
                return GenerationOutcome(
                    cachedTokenCount: cachedTokenCount,
                    generatedTokenIds: generatedTokenIds,
                    completedPrefillChunkTokenCounts: completedPrefillChunkTokenCounts);
            }
        }
    }

    // MARK: - Fixtures

    /// Builds the production artifact runtime with one acceptance chunk
    /// size; the chunk is also the cache block length and the layout growth
    /// so every prefill forward completes exactly one durable boundary.
    private static func buildAcceptanceRuntime(
        modelDirectory: String, chunkTokenCount: Int
    ) throws -> LoadedChatRuntime {
        // The Rust acceptance protocol fixes the prefill chunk to the cache
        // block length, so a cold run's terminal chunk and a restored run's
        // tail chunk take the same kernel shapes and the seeded streams
        // stay comparable at the artifact's bf16 working precision.
        return try Qwen35ChatRuntime.buildArtifactRuntime(
            modelDirectory: modelDirectory,
            modelConfiguration: WorkerModelConfiguration.autoregressive(
                WorkerAutoregressiveModelConfiguration(
                    modelId: "qwen3.5-moe",
                    maximumContextTokens: 24_576,
                    maximumOutputTokens: UInt32(LARGE_PREFILL_OUTPUT_TOKEN_COUNT),
                    chunking: WorkerChunkingConfiguration(
                        fixedPromptProcessingChunkSizeTokens: UInt32(chunkTokenCount),
                        fixedSsdStreamingPromptProcessingChunkSizeTokens: UInt32(chunkTokenCount),
                        fullAttentionKeyValueGrowthTokens: UInt32(chunkTokenCount),
                        prefillGraphSubmissionLayerInterval: 1,
                        experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                        experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 1,
                        promptCacheBlockTokens: nil,
                        promptCacheCommonPrefixStrideBlocks: 4,
                        experimentalDecodeStageAttributionEnabled: false,
                        experimentalQuantizedKvCacheEnabled: false,
                        experimentalFusedMoeDecodeEnabled: false))),
            prefillChunkTokenCount: chunkTokenCount);
    }

    /// Attaches the persistent prompt cache over a fresh temporary global
    /// root and returns that root for cleanup.
    private static func attachPromptCacheStore(
        engine: any InferenceEngine, chunkTokenCount: Int
    ) throws -> URL {
        guard let moeEngine: Qwen35MoeEngine = engine as? Qwen35MoeEngine else {
            throw Qwen35MoePromptCacheError.promptCacheNotAttached;
        }
        let storeGlobalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("moe-prompt-cache-acceptance-\(UUID().uuidString)",
                isDirectory: true);
        let quotaOverride: String? = RealModelJourneyGate.installedArtifactDirectory(
            environmentVariableName: PROMPT_CACHE_ENVIRONMENT_QUOTA_VARIABLE);
        let quotaBytes: UInt64 = quotaOverride.map { (rawQuota: String) -> UInt64 in
            return UInt64(rawQuota) ?? DEFAULT_PROMPT_CACHE_QUOTA_BYTES;
        } ?? DEFAULT_PROMPT_CACHE_QUOTA_BYTES;
        try moeEngine.attachPersistentPromptCache(Qwen35MoePromptCacheAttachment(
            activeModelPromptCacheDirectory: storeGlobalRoot
                .appendingPathComponent("acceptance-model", isDirectory: true)
                .appendingPathComponent("acceptance-revision", isDirectory: true),
            globalPromptCacheRootDirectory: storeGlobalRoot,
            globalPromptCacheMaximumSizeBytes: quotaBytes,
            effectiveMlxMemoryCeilingBytes: 20_000_000_000,
            modelId: "acceptance-model",
            modelRevision: "acceptance-revision",
            configuredBlockTokenCount: chunkTokenCount));
        return storeGlobalRoot;
    }

    private static func attachedBlockTokenCount(engine: any InferenceEngine) throws -> Int {
        guard let moeEngine: Qwen35MoeEngine = engine as? Qwen35MoeEngine,
            let blockTokenCount: Int = moeEngine.attachedPromptCacheBlockTokenCount()
        else {
            throw Qwen35MoePromptCacheError.promptCacheNotAttached;
        }
        return blockTokenCount;
    }

    /// The env-selected acceptance chunk; the Rust journey contract keeps
    /// the selectable sizes at 2048, 4096, and 8192 tokens.
    private static func configuredAcceptanceChunkTokenCount() throws -> Int {
        let configuredRawValue: String? = RealModelJourneyGate.installedArtifactDirectory(
            environmentVariableName: PROMPT_CACHE_ENVIRONMENT_CHUNK_VARIABLE);
        guard let configuredRawValue: String = configuredRawValue else {
            return 8_192;
        }
        guard let configuredChunkTokenCount: Int = Int(configuredRawValue),
            LARGE_PREFILL_CHUNK_TOKEN_CHOICES.contains(configuredChunkTokenCount)
        else {
            Issue.record("the selected acceptance prefill size must be 2048, 4096, or 8192");
            return 8_192;
        }
        return configuredChunkTokenCount;
    }

    /// Tokenizes the chat-wrapped Romeo and Juliet fixture, repeating the
    /// source until the prompt crosses the large-prefill floor.
    private static func representativeLongGenerationPromptTokenIds(
        runtime: LoadedChatRuntime
    ) throws -> Array<UInt32> {
        for repetitionCount: Int in 1...8 {
            let promptTokenIds: Array<UInt32> = try romeoAndJulietPromptTokenIds(
                runtime: runtime, repetitionCount: repetitionCount);
            if promptTokenIds.count >= MINIMUM_LARGE_PREFILL_PROMPT_TOKEN_COUNT {
                return promptTokenIds;
            }
        }
        Issue.record("the Romeo and Juliet acceptance prompt did not reach 16,384 input tokens");
        return [];
    }

    /// Prepares one chat request whose user content is the Romeo and Juliet
    /// fixture repeated `repetitionCount` times plus the study-guide
    /// instruction, and returns its model-visible token ids.
    private static func romeoAndJulietPromptTokenIds(
        runtime: LoadedChatRuntime, repetitionCount: Int
    ) throws -> Array<UInt32> {
        let romeoAndJulietSource: String = try Self.romeoAndJulietSourceText();
        let promptContent: String = "Romeo and Juliet source material:\n\n"
            + String(repeating: romeoAndJulietSource, count: repetitionCount)
            + "\n\n" + ROMEO_AND_JULIET_STUDY_GUIDE_INSTRUCTION;
        let chatCommand: ChatGenerationCommand = ChatGenerationCommand(
            requestId: RequestId(rawRequestId: 9_000),
            model: "qwen3.5-moe",
            messages: [
                .user(content: promptContent, images: []),
            ],
            tools: [],
            toolChoice: .none,
            settings: ChatGenerationSettings(
                maxOutputTokens: UInt16(LARGE_PREFILL_OUTPUT_TOKEN_COUNT),
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: 256),
            structuredGeneration: nil);
        let activeGeneration: any ActiveChatGeneration = try runtime.processor
            .prepareChatGeneration(chatCommand);
        defer { _ = try? activeGeneration.finishOutputs(); }
        guard let preparedRequest: Qwen35PreparedInferenceRequest = activeGeneration
            .inferenceRequest as? Qwen35PreparedInferenceRequest
        else {
            throw Qwen35MoePromptCacheError.promptCacheNotAttached;
        }
        return preparedRequest.promptTokenIds;
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

    private func nextRequestId() -> RequestId {
        self.nextRequestIdValue += 1;
        return RequestId(rawRequestId: self.nextRequestIdValue);
    }
}

/// One acceptance generation's evidence.
private struct GenerationOutcome {

    let cachedTokenCount: UInt32;

    let generatedTokenIds: Array<UInt32>;

    let completedPrefillChunkTokenCounts: Array<UInt32>;
}

/// The cold-versus-restored parity evidence one acceptance journey asserts.
private struct AcceptanceOutcome {

    let coldGeneratedTokenIds: Array<UInt32>;

    let coldCompletedPrefillChunkTokenCounts: Array<UInt32>;

    let restoredCachedTokenCount: UInt32;

    let restoredGeneratedTokenIds: Array<UInt32>;
}
