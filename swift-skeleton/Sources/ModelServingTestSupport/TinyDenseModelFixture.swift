import Foundation;

import IpcProtocol;

import ModelServing;

/// Synthesizes the tiny dense Qwen3.5 model directory shared by the serving
/// journeys: the minimal dense config plus the tiny BPE tokenizer files. No
/// shard files by design; the in-memory path constructs the engine from the
/// config document.
public enum TinyDenseModelFixture {

    /** The model configuration matching the tiny dense config document. */
    public static func autoregressiveConfiguration() -> WorkerAutoregressiveModelConfiguration {
        return WorkerAutoregressiveModelConfiguration(
            modelId: "qwen3.5",
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

    /** Writes the config and tokenizer files into a fresh directory. */
    public static func writeModelDirectory(modelDirectoryUrl: URL) throws -> Void {
        try Data(tinyDenseConfigJson.utf8).write(
            to: modelDirectoryUrl.appendingPathComponent("config.json"));
        try TinyTokenizerFixture.writeFiles(modelDirectoryUrl: modelDirectoryUrl);
    }

    /** The same tiny dense config as the dense engine journeys, with the
    end-of-sequence ids matching the fixture tokenizer's control marker. */
    private static let tinyDenseConfigJson: String = """
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "dtype": "bfloat16",
            "eos_token_id": [3],
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
