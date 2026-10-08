import Foundation;

import MLX;

import ModelServing;

@testable import ModelServing;

/**
 * The pinned in-memory Qwen3.5-MoE engine fixture: the tiny upstream-shaped
 * config with initializer weights fixed by seed, so every hermetic MoE
 * journey runs a real forward pass without a packaged checkpoint or a
 * download. The geometry constants and the residency expectations they
 * imply live here so the engine journeys and the worker journeys derive
 * every expectation from one source instead of drifting apart.
 */
enum Qwen35MoeInMemoryEngineFixture {

    static let FIXTURE_LAYER_COUNT: UInt32 = 2;
    static let FIXTURE_EXPERT_COUNT: UInt32 = 4;
    static let FIXTURE_HIDDEN_SIZE: UInt32 = 64;
    static let FIXTURE_EXPERT_INTERMEDIATE_SIZE: UInt32 = 64;
    static let FIXTURE_END_TOKEN_IDS: Array<UInt32> = [3, 4];

    /// Every routed expert across every decoder layer sits resident, so the
    /// resident count is the config product.
    static func residentExpertCount() -> UInt32 {
        return FIXTURE_LAYER_COUNT * FIXTURE_EXPERT_COUNT;
    }

    /// The paging plan the paged journeys install: the first half of each
    /// layer's experts stays resident while misses page through the seam.
    static func retainedExpertIdsPerLayer() -> Array<Array<Int>> {
        return (0..<Int(FIXTURE_LAYER_COUNT)).map({ (_: Int) -> Array<Int> in
            return [0, 1];
        });
    }

    /// The full-retention plan the bit-identity journey uses as its zero-
    /// paging twin: every expert installed at setup, so a runtime page read
    /// never happens and any stream difference is attributable to paging.
    static func fullyRetainedExpertIdsPerLayer() -> Array<Array<Int>> {
        return (0..<Int(FIXTURE_LAYER_COUNT)).map({ (_: Int) -> Array<Int> in
            return Array(0..<Int(FIXTURE_EXPERT_COUNT));
        });
    }

    /// The retained set is half of each layer's routed experts.
    static func retainedExpertCount() -> UInt32 {
        return FIXTURE_LAYER_COUNT * 2;
    }

    /// The retained payload is the bf16 gate, up, and down matrices of one
    /// SwitchGLU expert times the retained set.
    static func retainedExpertPayloadBytes() -> UInt64 {
        let projectionElementCount: UInt64 = 3
            * UInt64(FIXTURE_HIDDEN_SIZE)
            * UInt64(FIXTURE_EXPERT_INTERMEDIATE_SIZE);
        let bfloat16BytesPerElement: UInt64 = 2;
        return UInt64(retainedExpertCount())
            * projectionElementCount
            * bfloat16BytesPerElement;
    }

    /// The resident payload is the bf16 gate, up, and down matrices of one
    /// SwitchGLU expert times the resident set.
    static func residentExpertPayloadBytes() -> UInt64 {
        let projectionElementCount: UInt64 = 3
            * UInt64(FIXTURE_HIDDEN_SIZE)
            * UInt64(FIXTURE_EXPERT_INTERMEDIATE_SIZE);
        let bfloat16BytesPerElement: UInt64 = 2;
        return UInt64(residentExpertCount())
            * projectionElementCount
            * bfloat16BytesPerElement;
    }

    static func mixtureOfExpertsConfigBytes() -> Data {
        return Data(mixtureOfExpertsJson.utf8);
    }

    static func denseTwinConfigBytes() -> Data {
        return Data(denseTwinJson.utf8);
    }

    /**
     * Builds the MoE engine from the tiny config with the initializer
     * weights pinned, so the journey is reproducible without a checkpoint.

     * - Parameter prefillChunkTokenCount: the prompt chunk the engine
       processes per forward pass; the journeys pick chunk sizes that
       produce deterministic boundary counts.
     * - Returns: the engine with its tiny MoE model loaded in memory.
     * - Throws: the model load failure when the config or the model
       construction is rejected.
     */
    static func makePinnedEngine(prefillChunkTokenCount: Int = 8) throws -> Qwen35MoeEngine {
        let engine: Qwen35MoeEngine = Qwen35MoeEngine(prefillChunkTokenCount: prefillChunkTokenCount);
        try withRandomState(MLXRandom.RandomState(seed: 3)) {
            try engine.loadInMemoryModel(configBytes: mixtureOfExpertsConfigBytes());
        }
        return engine;
    }

    /**
     * Builds the MoE engine from the pinned fixture and installs paged
     * expert execution over it: every layer's upstream SwitchGLU is swapped
     * for the paged primitive, the plan's retained experts are installed as
     * the layer's resident payload, and every routed miss pages through the
     * given materializer.
     */
    static func makePinnedPagedEngine(
        retainedExpertIdsPerLayer: Array<Array<Int>>,
        expertPageMaterializer: any Qwen35MoeExpertPageMaterializing
    ) throws -> Qwen35MoeEngine {
        let engine: Qwen35MoeEngine = try makePinnedEngine();
        try engine.installPagedExpertExecution(
            retainedExpertIdsPerLayer: retainedExpertIdsPerLayer,
            expertPageMaterializer: expertPageMaterializer);
        return engine;
    }

    private static let mixtureOfExpertsJson: String = """
        {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            \(sharedDenseSection(textModelType: "qwen3_5_moe_text")),
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64
            }
        }
        """
    private static let denseTwinJson: String = """
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            \(sharedDenseSection(textModelType: "qwen3_5_text"))
            }
        }
        """

    /// The text_config half shared by both variants; the repo validator
    /// pins its model_type per architecture (qwen3_5_moe_text vs
    /// qwen3_5_text), so the fixture carries it as a parameter.
    private static func sharedDenseSection(textModelType: String) -> String {
        return """
            "dtype": "bfloat16",
            "eos_token_id": [3, 4],
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "\(textModelType)",
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
        """
    }
}
