import Foundation;

import Testing;

import MLX;
import MLXLMCommon;

import IpcProtocol;
import ModelServing;

@testable import ModelServing;

/**
 * Hermetic dense-engine journeys: a real Qwen3.5 dense forward pass over an
 * in-memory tiny model (the upstream mlx-swift-lm CI pattern), the full
 * prefill-to-decode boundary contract, seeded reproducibility, and the
 * fail-closed load path. No downloads; the suite is serialized so the MLX
 * journeys never overlap (the repository's one-model-at-a-time rule).
 */
@Suite(.serialized)
final class Qwen35DenseEngineTests {

    private static let ROMEO_AND_JULIET_PROMPT: String = "What is the play about?";

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    @Test(.timeLimit(.minutes(1)))
    func should_stream_prefill_boundaries_and_tokens_from_a_real_dense_forward_pass() throws {
        let engine: Qwen35DenseEngine = try Self.makePinnedEngine();
        let loadResult: EngineLoadResult = try engine.load();
        #expect(loadResult.minimumMlxMemoryCeilingBytes == 1);
        #expect(loadResult.expertMemoryMode == nil);

        let promptTokenIds: Array<UInt32> = Self.fixturePromptTokenIds();
        let requestId: RequestId = RequestId(rawRequestId: 91);
        let generationStart: EngineGenerationStart = try engine.startGeneration(
            Qwen35PreparedInferenceRequest(
                promptTokenIds: promptTokenIds,
                samplingSettings: Qwen35SamplingSettings(
                    chatGenerationSettings: Self.settings(seed: 7))));

        #expect(generationStart.cachedTokenCount == 0);
        #expect(generationStart.promptProcessingPhase == .target);

        var processedTokenCount: UInt32 = 0;
        var preparationSeen: Bool = false;
        var generatedTokenIds: Array<UInt32> = [];
        var firstDecodeElapsedSeen: Bool = false;
        while generatedTokenIds.count < 5 {
            let boundary: GeneratedToken = try engine.decodeNextToken(requestId: requestId);
            switch boundary {
            case let .prefillProgress(chunkProcessed, _, _, completedChunkTokens, _, _, _, promptWorkReuse):
                processedTokenCount += chunkProcessed;
                #expect(chunkProcessed > 0);
                #expect(completedChunkTokens == chunkProcessed);
                #expect(promptWorkReuse.targetEligibleTokenCount == 0);
            case .generationPreparationStarted:
                preparationSeen = true;
                #expect(processedTokenCount == UInt32(promptTokenIds.count));
            case let .tokenId(generatedTokenId, false, _, _, firstDecodeElapsed, nil):
                #expect(generatedTokenId < 512);
                if firstDecodeElapsed != nil {
                    #expect(firstDecodeElapsedSeen == false);
                    firstDecodeElapsedSeen = true;
                }
                generatedTokenIds.append(generatedTokenId);
            default:
                Issue.record("unexpected dense engine boundary \(boundary)");
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
                    chatGenerationSettings: Qwen35DenseEngineTests.settings(seed: 7))));
        }

        let finalization: GenerationFinalization = try engine.cancelGeneration(
            requestId: requestId);
        #expect(finalization.mlxMemorySnapshot != nil);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_reproduce_the_same_token_stream_from_identical_seeded_engines() throws {
        let firstEngine: Qwen35DenseEngine = try Self.makePinnedEngine();
        let secondEngine: Qwen35DenseEngine = try Self.makePinnedEngine();
        let promptTokenIds: Array<UInt32> = Self.fixturePromptTokenIds();

        let firstTokenIds: Array<UInt32> = try Self.collectTokenIds(
            engine: firstEngine, promptTokenIds: promptTokenIds, seed: 42, tokenBudget: 5);
        let secondTokenIds: Array<UInt32> = try Self.collectTokenIds(
            engine: secondEngine, promptTokenIds: promptTokenIds, seed: 42, tokenBudget: 5);
        #expect(firstTokenIds == secondTokenIds);
        #expect(firstTokenIds.isEmpty == false);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_fail_the_load_while_the_artifact_weight_path_is_pending() throws {
        let engine: Qwen35DenseEngine = Qwen35DenseEngine();
        do {
            _ = try engine.load();
            Issue.record("the pending artifact weight path must fail closed");
        } catch let loadError as InferenceEngineError {
            guard case let .modelLoad(failureReason) = loadError else {
                Issue.record("expected a model load failure, got \(loadError)");
                return;
            }
            #expect(failureReason
                .contains("artifact streaming slice") == true);
        }
    }

    @Test(.timeLimit(.minutes(1)))
    func should_reject_decode_without_an_active_dense_request() throws {
        let engine: Qwen35DenseEngine = Qwen35DenseEngine();
        #expect(throws: InferenceEngineError.invalidRequest(
            reason: "the engine holds no active dense request")) {
            _ = try engine.decodeNextToken(requestId: RequestId(rawRequestId: 1));
        }
    }

    // MARK: - Sampling settings mapping

    @Test
    func should_default_omitted_sampling_settings_to_the_provider_defaults() throws {
        let mapping: Qwen35SamplingSettings = Qwen35SamplingSettings(
            chatGenerationSettings: Qwen35DenseEngineTests.settings(seed: nil));
        #expect(mapping.temperature == Qwen35SamplingSettings.defaultTemperature);
        #expect(mapping.topP == Qwen35SamplingSettings.defaultTopP);
        #expect(mapping.seed == nil);
    }

    @Test
    func should_convert_thousandths_wire_settings_into_sampler_scale() throws {
        let mapping: Qwen35SamplingSettings = Qwen35SamplingSettings(
            chatGenerationSettings: Qwen35DenseEngineTests.settings(
                seed: 5, temperatureThousandths: 750, topPThousandths: 900));
        #expect(mapping.temperature == 0.75);
        #expect(mapping.topP == 0.9);
        #expect(mapping.seed == 5);

        // Identical seeds over identical logits sample identically.
        let logits: MLXArray = MLXArray([Float]([-1.0, -1.5, 2.0, 0.5]));
        let firstSampled: Int = mapping.makeSampler().sample(logits: logits).item(Int.self);
        let secondSampled: Int = mapping.makeSampler().sample(logits: logits).item(Int.self);
        #expect(firstSampled == secondSampled);
    }

    @Test
    func should_sample_the_maximum_logit_under_an_explicit_zero_temperature() throws {
        let mapping: Qwen35SamplingSettings = Qwen35SamplingSettings(
            chatGenerationSettings: Qwen35DenseEngineTests.settings(seed: nil, temperatureThousandths: 0));
        let logits: MLXArray = MLXArray([Float]([-1.0, -1.5, 2.0, 0.5]));
        #expect(mapping.makeSampler().sample(logits: logits).item(Int.self) == 2);
    }

    // MARK: - Fixtures

    /**
     * Builds the dense engine from the tiny upstream-shaped config with the
     * initializer weights pinned, so the journey is reproducible without a
     * checkpoint.
     */
    private static func makePinnedEngine() throws -> Qwen35DenseEngine {
        let engine: Qwen35DenseEngine = Qwen35DenseEngine(prefillChunkTokenCount: 8);
        try withRandomState(MLXRandom.RandomState(seed: 3)) {
            try engine.loadInMemoryModel(configBytes: Data(tinyDenseConfigJson.utf8));
        }
        return engine;
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
        engine: Qwen35DenseEngine,
        promptTokenIds: Array<UInt32>,
        seed: UInt64,
        tokenBudget: Int
    ) throws -> Array<UInt32> {
        let requestId: RequestId = RequestId(rawRequestId: 97);
        _ = try engine.startGeneration(Qwen35PreparedInferenceRequest(
            promptTokenIds: promptTokenIds,
            samplingSettings: Qwen35SamplingSettings(
                chatGenerationSettings: Qwen35DenseEngineTests.settings(seed: seed))));
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

    /**
     * The tiny dense config mirrors the upstream mlx-swift-lm CI fixture:
     * two hybrid decoder layers whose dimensions stay divisible, with the
     * M-RoPE parameters the dense text model expects.
     */
    private static let tinyDenseConfigJson: String = """
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "dtype": "bfloat16",
            "eos_token_id": [3, 4],
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "attention_bias": false,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "layer_types": ["full_attention", "full_attention"],
                "vocab_size": 512,
                "full_attention_interval": 2,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "max_position_embeddings": 4096,
                "rope_parameters": {
                    "type": "default",
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 1.0
                }
            }
        }
        """;
}
