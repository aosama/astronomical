import Foundation;

import Testing;

import MLX;
import MLXLMCommon;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing;

/**
 * Hermetic Mixture-of-Experts engine journeys: a real Qwen3.5 MoE forward
 * pass over an in-memory tiny model (the upstream mlx-swift-lm CI pattern),
 * the resident-expert residency contract derived from the configuration,
 * and the fail-closed load paths. The engine serves the resident execution
 * mode; every routed expert sits in memory and the telemetry says so. No
 * downloads; the suite is serialized so the MLX journeys never overlap (the
 * repository's one-model-at-a-time rule).
 */
extension HermeticMlxJourneyContainer {

    @Suite(.tags(.hermeticMlxJourney))
    final class Qwen35MoeEngineTests {

    private static let ROMEO_AND_JULIET_PROMPT: String = "What is the play about?";

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    @Test(.timeLimit(.minutes(1)))
    func should_stream_prefill_boundaries_and_resident_expert_telemetry_from_a_real_moe_forward_pass() throws {
        let engine: Qwen35MoeEngine = try Self.makePinnedEngine();
        let loadResult: EngineLoadResult = try engine.load();
        #expect(loadResult.minimumMlxMemoryCeilingBytes == 1);
        #expect(loadResult.expertMemoryMode == .resident);

        let promptTokenIds: Array<UInt32> = Self.fixturePromptTokenIds();
        let requestId: RequestId = RequestId(rawRequestId: 93);
        let generationStart: EngineGenerationStart = try engine.startGeneration(
            Qwen35PreparedInferenceRequest(
                promptTokenIds: promptTokenIds,
                samplingSettings: Qwen35SamplingSettings(
                    chatGenerationSettings: Self.settings(seed: 7))));

        #expect(generationStart.cachedTokenCount == 0);
        #expect(generationStart.expertMemoryMode == .resident);
        #expect(generationStart.promptProcessingPhase == .target);

        var processedTokenCount: UInt32 = 0;
        var preparationSeen: Bool = false;
        var generatedTokenIds: Array<UInt32> = [];
        var firstDecodeElapsedSeen: Bool = false;
        while generatedTokenIds.count < 5 {
            let boundary: GeneratedToken = try engine.decodeNextToken(requestId: requestId);
            switch boundary {
            case let .prefillProgress(chunkProcessed, _, _, completedChunkTokens, _, _, tokenExpertMode, promptWorkReuse):
                processedTokenCount += chunkProcessed;
                #expect(chunkProcessed > 0);
                #expect(completedChunkTokens == chunkProcessed);
                #expect(tokenExpertMode == .resident);
                #expect(promptWorkReuse.targetEligibleTokenCount == 0);
            case let .generationPreparationStarted(totalLayerCount, residentExpertCount, residentExpertPayloadBytes, _):
                preparationSeen = true;
                #expect(processedTokenCount == UInt32(promptTokenIds.count));
                #expect(totalLayerCount == Qwen35MoeInMemoryEngineFixture.FIXTURE_LAYER_COUNT);
                #expect(residentExpertCount == Qwen35MoeInMemoryEngineFixture.residentExpertCount());
                #expect(residentExpertPayloadBytes == Qwen35MoeInMemoryEngineFixture.residentExpertPayloadBytes());
            case let .tokenId(generatedTokenId, false, tokenExpertMode, _, firstDecodeElapsed, nil):
                #expect(generatedTokenId < 512);
                #expect(tokenExpertMode == .resident);
                if firstDecodeElapsed != nil {
                    #expect(firstDecodeElapsedSeen == false);
                    firstDecodeElapsedSeen = true;
                }
                generatedTokenIds.append(generatedTokenId);
            default:
                Issue.record("unexpected MoE engine boundary \(boundary)");
                return;
            }
        }
        #expect(processedTokenCount == UInt32(promptTokenIds.count));
        #expect(preparationSeen);
        #expect(firstDecodeElapsedSeen);

        // One seeded request samples from one sampler; the engine must
        // reject a second concurrent start.
        #expect(throws: InferenceEngineError.engineBusy) {
            _ = try engine.startGeneration(Qwen35PreparedInferenceRequest(
                promptTokenIds: promptTokenIds,
                samplingSettings: Qwen35SamplingSettings(
                    chatGenerationSettings: Qwen35MoeEngineTests.settings(seed: 7))));
        }

        let finalization: GenerationFinalization = try engine.cancelGeneration(
            requestId: requestId);
        #expect(finalization.expertMemoryMode == .resident);
        #expect(finalization.expertResidencyTelemetry == ExpertResidencyTelemetry(
            totalLayerCount: Qwen35MoeInMemoryEngineFixture.FIXTURE_LAYER_COUNT,
            residentExpertCount: Qwen35MoeInMemoryEngineFixture.residentExpertCount(),
            residentExpertPayloadBytes: Qwen35MoeInMemoryEngineFixture.residentExpertPayloadBytes()));

        let idleSnapshot: WorkerMlxMemorySnapshot? = engine.collectMlxMemorySnapshot();
        #expect(idleSnapshot?.expertPayloadBytes == Qwen35MoeInMemoryEngineFixture.residentExpertPayloadBytes());
    }

    @Test(.timeLimit(.minutes(1)))
    func should_reproduce_the_same_token_stream_from_identical_seeded_moe_engines() throws {
        let firstEngine: Qwen35MoeEngine = try Self.makePinnedEngine();
        let secondEngine: Qwen35MoeEngine = try Self.makePinnedEngine();
        let promptTokenIds: Array<UInt32> = Self.fixturePromptTokenIds();

        let firstTokenIds: Array<UInt32> = try Self.collectTokenIds(
            engine: firstEngine, promptTokenIds: promptTokenIds, seed: 42, tokenBudget: 5);
        let secondTokenIds: Array<UInt32> = try Self.collectTokenIds(
            engine: secondEngine, promptTokenIds: promptTokenIds, seed: 42, tokenBudget: 5);
        #expect(firstTokenIds == secondTokenIds);
        #expect(firstTokenIds.isEmpty == false);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_fail_the_load_when_no_moe_model_was_loaded() throws {
        let engine: Qwen35MoeEngine = Qwen35MoeEngine();
        do {
            _ = try engine.load();
            Issue.record("an engine without a loaded model must fail the load");
        } catch let loadError as InferenceEngineError {
            guard case let .modelLoad(failureReason) = loadError else {
                Issue.record("expected a model load failure, got \(loadError)");
                return;
            }
            #expect(failureReason == "no MoE model is loaded");
        }
    }

    @Test(.timeLimit(.minutes(1)))
    func should_fail_closed_when_a_dense_checkpoint_reaches_the_moe_engine() throws {
        let engine: Qwen35MoeEngine = Qwen35MoeEngine();
        do {
            try withRandomState(MLXRandom.RandomState(seed: 3)) {
                try engine.loadInMemoryModel(configBytes: Qwen35MoeInMemoryEngineFixture.denseTwinConfigBytes());
            }
            Issue.record("a dense checkpoint must not load into the MoE engine");
        } catch let loadError as InferenceEngineError {
            guard case let .modelLoad(failureReason) = loadError else {
                Issue.record("expected a model load failure, got \(loadError)");
                return;
            }
            #expect(failureReason == "the Qwen3.5 checkpoint is not a Mixture of Experts configuration");
        }
    }

    @Test(.timeLimit(.minutes(1)))
    func should_reject_decode_without_an_active_moe_request() throws {
        let engine: Qwen35MoeEngine = Qwen35MoeEngine();
        #expect(throws: InferenceEngineError.invalidRequest(
            reason: "the engine holds no active MoE request")) {
            _ = try engine.decodeNextToken(requestId: RequestId(rawRequestId: 1));
        }
    }

    // MARK: - Fixtures

    /// The pinned in-memory engine comes from the shared MoE fixture.
    private static func makePinnedEngine() throws -> Qwen35MoeEngine {
        return try Qwen35MoeInMemoryEngineFixture.makePinnedEngine();
    }

    /// Romeo and Juliet prompt bytes are the model-visible token ids; every
    /// byte value stays inside the tiny vocabulary.
    private static func fixturePromptTokenIds() -> Array<UInt32> {
        return ROMEO_AND_JULIET_PROMPT.utf8.map { (promptByte: UInt8) -> UInt32 in
            return UInt32(promptByte) % 512;
        };
    }

    private static func settings(
        seed: UInt64?,
        temperatureThousandths: UInt16? = nil,
        topPThousandths: UInt16? = nil
    ) -> ChatGenerationSettings {
        return ChatGenerationSettings(
            maxOutputTokens: 16,
            temperatureThousandths: temperatureThousandths,
            topPThousandths: topPThousandths,
            seed: seed,
            thinkingBudget: nil);
    }

    private static func collectTokenIds(
        engine: Qwen35MoeEngine,
        promptTokenIds: Array<UInt32>,
        seed: UInt64,
        tokenBudget: Int
    ) throws -> Array<UInt32> {
        let requestId: RequestId = RequestId(rawRequestId: 99);
        _ = try engine.startGeneration(Qwen35PreparedInferenceRequest(
            promptTokenIds: promptTokenIds,
            samplingSettings: Qwen35SamplingSettings(
                chatGenerationSettings: Qwen35MoeEngineTests.settings(seed: seed))));
        var collectedTokenIds: Array<UInt32> = [];
        while collectedTokenIds.count < tokenBudget {
            let boundary: GeneratedToken = try engine.decodeNextToken(requestId: requestId);
            if case let .tokenId(generatedTokenId, _, _, _, _, _) = boundary {
                collectedTokenIds.append(generatedTokenId);
            }
        }
        _ = try engine.cancelGeneration(requestId: requestId);
        return collectedTokenIds;
    }
}

}
