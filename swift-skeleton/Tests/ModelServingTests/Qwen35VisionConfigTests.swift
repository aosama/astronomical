import XCTest;
import ModelServing;

/// The frozen trimmed Ornith vision fixture, twin-porting
/// vision_config_test_support's FROZEN_VISION_CONFIG_JSON.
enum Qwen35VisionConfigFixtures {
    static let FROZEN_VISION_CONFIG_JSON: String = """
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "dtype": "bfloat16",
        "eos_token_id": [248046, 248044],
        "tie_word_embeddings": false,
        "quantization": {"config_file": "optiq/optiq.safetensors", "n_tensors": 333, "base_model": "example-org/Example-1.0-35B"},
        "quantization_config": {"config_file": "optiq/optiq.safetensors", "n_tensors": 333, "base_model": "example-org/Example-1.0-35B"},
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_act": "silu",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "attention_bias": false,
            "mlp_bias": false,
            "norm_topk_prob": true,
            "output_router_logits": false,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        },
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 27,
            "dtype": "bfloat16",
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4304,
            "model_type": "qwen3_5_moe_vision",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        }
    }
    """;
}

/// Behavioral journeys for the accepted Qwen3.5 vision configuration,
/// twin-porting crates/model-serving/tests/qwen3_5_hermetic/vision_config.rs.
final class Qwen35VisionConfigTests: XCTestCase {

    func testShouldParseTheFrozenOrnithVisionConfig() throws {
        let visionConfig: Qwen3_5VisionConfig = try Qwen3_5VisionConfig.fromJsonBytes(
            configBytes: Data(Qwen35VisionConfigFixtures.FROZEN_VISION_CONFIG_JSON.utf8));

        XCTAssertEqual(visionConfig.depth, 27);
        XCTAssertEqual(visionConfig.hiddenSize, 1152);
        XCTAssertEqual(visionConfig.inChannels, 3);
        XCTAssertEqual(visionConfig.intermediateSize, 4304);
        XCTAssertEqual(visionConfig.headCount, 16);
        XCTAssertEqual(visionConfig.positionEmbeddingCount, 2304);
        XCTAssertEqual(visionConfig.patchSize, 16);
        XCTAssertEqual(visionConfig.spatialMergeSize, 2);
        XCTAssertEqual(visionConfig.temporalPatchSize, 2);
        XCTAssertEqual(visionConfig.outHiddenSize, 2048);
        XCTAssertEqual(visionConfig.hiddenActivation, "gelu_pytorch_tanh");
    }

    func testShouldAllowATextOnlyQwenConfigWithoutVisionConfig() throws {
        let textOnlyConfigJson: String = """
        {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text"
            }
        }
        """;

        let visionConfig: Qwen3_5VisionConfig? = try Qwen3_5VisionConfig.fromOptionalJsonBytes(
            configBytes: Data(textOnlyConfigJson.utf8));

        XCTAssertNil(visionConfig);
    }

    func testShouldAcceptAnOrnithVisionConfigWithADifferentDepth() throws {
        let configBytes: String = Qwen35VisionConfigFixtures.FROZEN_VISION_CONFIG_JSON
            .replacingOccurrences(of: "\"depth\": 27", with: "\"depth\": 32");
        let visionConfig: Qwen3_5VisionConfig = try Qwen3_5VisionConfig.fromJsonBytes(
            configBytes: Data(configBytes.utf8));
        XCTAssertEqual(visionConfig.depth, 32);
    }

    func testShouldAcceptAnOrnithVisionConfigWithADifferentHiddenSize() throws {
        let configBytes: String = Qwen35VisionConfigFixtures.FROZEN_VISION_CONFIG_JSON
            .replacingOccurrences(of: "\"hidden_size\": 1152", with: "\"hidden_size\": 1024");
        let visionConfig: Qwen3_5VisionConfig = try Qwen3_5VisionConfig.fromJsonBytes(
            configBytes: Data(configBytes.utf8));
        XCTAssertEqual(visionConfig.hiddenSize, 1024);
    }

    func testShouldAcceptAnOrnithVisionConfigWithADifferentPatchSize() throws {
        let configBytes: String = Qwen35VisionConfigFixtures.FROZEN_VISION_CONFIG_JSON
            .replacingOccurrences(of: "\"patch_size\": 16", with: "\"patch_size\": 14");
        let visionConfig: Qwen3_5VisionConfig = try Qwen3_5VisionConfig.fromJsonBytes(
            configBytes: Data(configBytes.utf8));
        XCTAssertEqual(visionConfig.patchSize, 14);
    }

    func testShouldRejectAVisionHiddenSizeThatCannotFormEqualAttentionHeads() throws {
        let configBytes: String = Qwen35VisionConfigFixtures.FROZEN_VISION_CONFIG_JSON
            .replacingOccurrences(of: "\"hidden_size\": 1152", with: "\"hidden_size\": 1153");

        XCTAssertThrowsError(try Qwen3_5VisionConfig.fromJsonBytes(configBytes: Data(configBytes.utf8)));
    }

    func testShouldRejectAVisionActivationOutsideTheQwen35ExecutionGraph() throws {
        let configBytes: String = Qwen35VisionConfigFixtures.FROZEN_VISION_CONFIG_JSON
            .replacingOccurrences(of: "\"hidden_act\": \"gelu_pytorch_tanh\"", with: "\"hidden_act\": \"silu\"");

        XCTAssertThrowsError(try Qwen3_5VisionConfig.fromJsonBytes(configBytes: Data(configBytes.utf8)));
    }

    func testShouldRejectAnOrnithVisionConfigWithTheWrongModelType() throws {
        let configBytes: String = Qwen35VisionConfigFixtures.FROZEN_VISION_CONFIG_JSON
            .replacingOccurrences(
                of: "\"model_type\": \"qwen3_5_moe_vision\"", with: "\"model_type\": \"wrong_vision\"");
        XCTAssertThrowsError(
            try Qwen3_5VisionConfig.fromJsonBytes(configBytes: Data(configBytes.utf8)),
            "vision config with wrong model_type should be rejected");
    }
}
