import Foundation;

import ModelServing;

/// Synthesizes a complete tiny MoE Qwen3.5 artifact on disk: one decoder
/// layer whose feed-forward block is the config-driven switch MLP inventory
/// (router, per-expert switch projections, shared expert), a real
/// safetensors shard mirroring the config-derived tensor profiles, and the
/// shard index.
///
/// The synthesis flow is the dense fixture's: tensor names and shapes come
/// from `Qwen3_5TensorSpec.qwen3_5LanguageTensorProfiles` over the MoE
/// config — never hand-enumerated — so the fixture stays coupled to the
/// config contract instead of a golden inventory.
public enum TinyMoeArtifactFixture {

    public static let MODEL_DIRECTORY_LEAF_NAME: String = "example-tiny-moe-qwen35";
    public static let MAX_OUTPUT_TOKENS: UInt32 = 256;

    public struct SynthesizedLayout {
        public var configBytes: Array<UInt8>;
        public var totalPayloadBytes: UInt64;
        public var tensorProfileCount: Int;
        public var expertCount: Int;
        public var layerCount: Int;
    }

    /// Writes the full MoE artifact directory and returns the layout facts
    /// the journeys assert against.
    ///
    /// - Parameter includeTokenizerFiles: writes the tiny working tokenizer
    ///   pair in place of the placeholder `tokenizer.json` bytes; family
    ///   classification journeys need the working pair.
    public static func writeModelDirectory(
        includeTokenizerFiles: Bool = false
    ) throws -> (modelDirectoryUrl: URL, layout: SynthesizedLayout) {
        let configBytes: Array<UInt8> = try TinyMoeArtifactFixture.moeSingleLayerConfigBytes();
        var config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        let unfilteredProfiles: Array<TensorProfile> = Qwen3_5TensorSpec
            .qwen3_5LanguageTensorProfiles(qwen3_5Config: config);
        let shardTensorNames: Set<String> = Set(unfilteredProfiles.map { (tensorProfile: TensorProfile) -> String in
            return tensorProfile.name;
        });
        config.resolveUnquantizedModulesFromShardIndex(shardTensorNames: shardTensorNames);
        let tensorProfiles: Array<TensorProfile> = Qwen3_5TensorSpec
            .qwen3_5LanguageTensorProfiles(qwen3_5Config: config);

        let framedShardBytes: TinyDenseArtifactFixture.SynthesizedShardBytes = TinyDenseArtifactFixture
            .synthesizedShardBytes(tensorProfiles: tensorProfiles);
        let indexBytes: Array<UInt8> = TinyDenseArtifactFixture.synthesizedIndexBytes(
            tensorNames: shardTensorNames, totalPayloadBytes: framedShardBytes.payloadBytes);


        let modelDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "\(TinyMoeArtifactFixture.MODEL_DIRECTORY_LEAF_NAME)-\(UUID().uuidString)");
        try FileManager.default.createDirectory(at: modelDirectoryUrl, withIntermediateDirectories: true);
        try writeFixtureData(Data(configBytes), to: modelDirectoryUrl.appendingPathComponent("config.json"));
        if includeTokenizerFiles {
            try TinyTokenizerFixture.writeFiles(modelDirectoryUrl: modelDirectoryUrl);
        } else {
            try Data(TinyDenseArtifactFixture.PLACEHOLDER_TOKENIZER_BYTES).write(
                to: modelDirectoryUrl.appendingPathComponent("tokenizer.json"));
        }
        try Data(indexBytes).write(
            to: modelDirectoryUrl.appendingPathComponent("model.safetensors.index.json"));
        try framedShardBytes.fileBytes.write(
            to: modelDirectoryUrl.appendingPathComponent(TinyDenseArtifactFixture.SHARD_FILE_NAME));
        return (
            modelDirectoryUrl: modelDirectoryUrl,
            layout: SynthesizedLayout(
                configBytes: configBytes,
                totalPayloadBytes: framedShardBytes.payloadBytes,
                tensorProfileCount: tensorProfiles.count,
                expertCount: TinyMoeArtifactFixture.EXPERT_COUNT,
                layerCount: TinyMoeArtifactFixture.LAYER_COUNT)
        );
    }

    private static let EXPERT_COUNT: Int = 4;
    private static let LAYER_COUNT: Int = 1;

    /**
     * The tiny MoE config: the dense twin's attention block with the
     * switch-MLP feed-forward contract — four routed experts, top-2
     * routing, and a shared expert — whose every dimension stays divisible
     * by the affine group size.
     */
    private static func moeSingleLayerConfigBytes() throws -> Array<UInt8> {
        let configJsonText: String = """
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "dtype": "bfloat16",
                "eos_token_id": [3, 4],
                "pad_token_id": 4,
                "tie_word_embeddings": false,
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "hidden_act": "silu",
                    "hidden_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 64,
                    "rms_norm_eps": 0.000001,
                    "attention_bias": false,
                    "mlp_bias": false,
                    "norm_topk_prob": true,
                    "output_router_logits": false,
                    "vocab_size": 256,
                    "intermediate_size": 64,
                    "max_position_embeddings": 4096,
                    "full_attention_interval": 1,
                    "linear_conv_kernel_dim": 4,
                    "linear_num_key_heads": 1,
                    "linear_num_value_heads": 1,
                    "linear_key_head_dim": 64,
                    "linear_value_head_dim": 64,
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 64,
                    "shared_expert_intermediate_size": 64,
                    "rope_parameters": {
                        "type": "default",
                        "mrope_interleaved": true,
                        "mrope_section": [11, 11, 10],
                        "rope_theta": 100000.0,
                        "partial_rotary_factor": 1.0
                    },
                    "layer_types": ["full_attention"]
                },
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine", "language_model.model.embed_tokens": {"group_size": 64, "bits": 8}, "language_model.lm_head": {"group_size": 64, "bits": 8}},
                "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine", "language_model.model.embed_tokens": {"group_size": 64, "bits": 8}, "language_model.lm_head": {"group_size": 64, "bits": 8}}
            }
            """;
        return Array(configJsonText.utf8);
    }
}
