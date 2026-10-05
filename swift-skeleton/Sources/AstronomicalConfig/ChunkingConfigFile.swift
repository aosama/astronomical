import Foundation;

/**
 * `chunking` section of the v1 user configuration document.
 *
 * MIGRATION MARKER — deferred from this slice: `prompt_cache_block_tokens`
 * is Option<Option<u32>> in Rust, where JSON null means "explicitly unset"
 * and absence means "inherit the resolution default". This port collapses
 * both to nil; the chunking-config slice restores the distinction alongside
 * ChunkingConfig resolution.
 */
internal struct ChunkingConfigFile: Equatable {
    internal let fixedPromptProcessingChunkSizeTokens: UInt32?;
    internal let fixedSsdStreamingPromptProcessingChunkSizeTokens: UInt32?;
    internal let fullAttentionKeyValueGrowthTokens: UInt32?;
    internal let prefillGraphSubmissionLayerInterval: UInt32?;
    internal let experimentalSsdPagingPrefillGraphSubmissionLayerInterval: UInt32?;
    internal let experimentalSsdPagingGenerationGraphSubmissionLayerInterval: UInt32?;
    internal let promptCacheBlockTokens: UInt32?;
    internal let promptCacheCommonPrefixStrideBlocks: UInt32?;
    internal let experimentalDecodeStageAttributionEnabled: Bool?;
    internal let experimentalQuantizedKvCacheEnabled: Bool?;
    internal let experimentalFusedMoeDecodeEnabled: Bool?;

    internal init(
        fixedPromptProcessingChunkSizeTokens: UInt32?,
        fixedSsdStreamingPromptProcessingChunkSizeTokens: UInt32?,
        fullAttentionKeyValueGrowthTokens: UInt32?,
        prefillGraphSubmissionLayerInterval: UInt32?,
        experimentalSsdPagingPrefillGraphSubmissionLayerInterval: UInt32?,
        experimentalSsdPagingGenerationGraphSubmissionLayerInterval: UInt32?,
        promptCacheBlockTokens: UInt32?,
        promptCacheCommonPrefixStrideBlocks: UInt32?,
        experimentalDecodeStageAttributionEnabled: Bool?,
        experimentalQuantizedKvCacheEnabled: Bool?,
        experimentalFusedMoeDecodeEnabled: Bool?
    ) {
        self.fixedPromptProcessingChunkSizeTokens = fixedPromptProcessingChunkSizeTokens;
        self.fixedSsdStreamingPromptProcessingChunkSizeTokens = fixedSsdStreamingPromptProcessingChunkSizeTokens;
        self.fullAttentionKeyValueGrowthTokens = fullAttentionKeyValueGrowthTokens;
        self.prefillGraphSubmissionLayerInterval = prefillGraphSubmissionLayerInterval;
        self.experimentalSsdPagingPrefillGraphSubmissionLayerInterval = experimentalSsdPagingPrefillGraphSubmissionLayerInterval;
        self.experimentalSsdPagingGenerationGraphSubmissionLayerInterval = experimentalSsdPagingGenerationGraphSubmissionLayerInterval;
        self.promptCacheBlockTokens = promptCacheBlockTokens;
        self.promptCacheCommonPrefixStrideBlocks = promptCacheCommonPrefixStrideBlocks;
        self.experimentalDecodeStageAttributionEnabled = experimentalDecodeStageAttributionEnabled;
        self.experimentalQuantizedKvCacheEnabled = experimentalQuantizedKvCacheEnabled;
        self.experimentalFusedMoeDecodeEnabled = experimentalFusedMoeDecodeEnabled;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> ChunkingConfigFile {
        try StrictJson.requireKnownKeys(
            object: jsonObject,
            knownKeys: [
                "fixed_prompt_processing_chunk_size_tokens",
                "fixed_ssd_streaming_prompt_processing_chunk_size_tokens",
                "full_attention_key_value_growth_tokens",
                "prefill_graph_submission_layer_interval",
                "experimental_ssd_paging_prefill_graph_submission_layer_interval",
                "experimental_ssd_paging_generation_graph_submission_layer_interval",
                "prompt_cache_block_tokens",
                "prompt_cache_common_prefix_stride_blocks",
                "experimental_decode_stage_attribution_enabled",
                "experimental_quantized_kv_cache_enabled",
                "experimental_fused_moe_decode_enabled"
            ],
            fieldName: "chunking"
        );
        return ChunkingConfigFile(
            fixedPromptProcessingChunkSizeTokens: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "fixed_prompt_processing_chunk_size_tokens"
            ),
            fixedSsdStreamingPromptProcessingChunkSizeTokens: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "fixed_ssd_streaming_prompt_processing_chunk_size_tokens"
            ),
            fullAttentionKeyValueGrowthTokens: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "full_attention_key_value_growth_tokens"
            ),
            prefillGraphSubmissionLayerInterval: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "prefill_graph_submission_layer_interval"
            ),
            experimentalSsdPagingPrefillGraphSubmissionLayerInterval: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "experimental_ssd_paging_prefill_graph_submission_layer_interval"
            ),
            experimentalSsdPagingGenerationGraphSubmissionLayerInterval: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "experimental_ssd_paging_generation_graph_submission_layer_interval"
            ),
            promptCacheBlockTokens: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "prompt_cache_block_tokens"
            ),
            promptCacheCommonPrefixStrideBlocks: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "prompt_cache_common_prefix_stride_blocks"
            ),
            experimentalDecodeStageAttributionEnabled: try StrictJson.optionalBoolean(
                object: jsonObject,
                fieldName: "experimental_decode_stage_attribution_enabled"
            ),
            experimentalQuantizedKvCacheEnabled: try StrictJson.optionalBoolean(
                object: jsonObject,
                fieldName: "experimental_quantized_kv_cache_enabled"
            ),
            experimentalFusedMoeDecodeEnabled: try StrictJson.optionalBoolean(
                object: jsonObject,
                fieldName: "experimental_fused_moe_decode_enabled"
            )
        );
    }

    internal func toJsonObject() -> Any {
        var jsonObject: Dictionary<String, Any> = Dictionary<String, Any>();
        if let fixedPromptProcessingChunkSizeTokens: UInt32 = self.fixedPromptProcessingChunkSizeTokens {
            jsonObject["fixed_prompt_processing_chunk_size_tokens"] = fixedPromptProcessingChunkSizeTokens;
        }
        if let fixedSsdStreamingPromptProcessingChunkSizeTokens: UInt32 = self.fixedSsdStreamingPromptProcessingChunkSizeTokens {
            jsonObject["fixed_ssd_streaming_prompt_processing_chunk_size_tokens"] = fixedSsdStreamingPromptProcessingChunkSizeTokens;
        }
        if let fullAttentionKeyValueGrowthTokens: UInt32 = self.fullAttentionKeyValueGrowthTokens {
            jsonObject["full_attention_key_value_growth_tokens"] = fullAttentionKeyValueGrowthTokens;
        }
        if let prefillGraphSubmissionLayerInterval: UInt32 = self.prefillGraphSubmissionLayerInterval {
            jsonObject["prefill_graph_submission_layer_interval"] = prefillGraphSubmissionLayerInterval;
        }
        if let experimentalSsdPagingPrefillGraphSubmissionLayerInterval: UInt32 = self.experimentalSsdPagingPrefillGraphSubmissionLayerInterval {
            jsonObject["experimental_ssd_paging_prefill_graph_submission_layer_interval"] = experimentalSsdPagingPrefillGraphSubmissionLayerInterval;
        }
        if let experimentalSsdPagingGenerationGraphSubmissionLayerInterval: UInt32 = self.experimentalSsdPagingGenerationGraphSubmissionLayerInterval {
            jsonObject["experimental_ssd_paging_generation_graph_submission_layer_interval"] = experimentalSsdPagingGenerationGraphSubmissionLayerInterval;
        }
        if let promptCacheBlockTokens: UInt32 = self.promptCacheBlockTokens {
            jsonObject["prompt_cache_block_tokens"] = promptCacheBlockTokens;
        }
        if let promptCacheCommonPrefixStrideBlocks: UInt32 = self.promptCacheCommonPrefixStrideBlocks {
            jsonObject["prompt_cache_common_prefix_stride_blocks"] = promptCacheCommonPrefixStrideBlocks;
        }
        if let experimentalDecodeStageAttributionEnabled: Bool = self.experimentalDecodeStageAttributionEnabled {
            jsonObject["experimental_decode_stage_attribution_enabled"] = experimentalDecodeStageAttributionEnabled;
        }
        if let experimentalQuantizedKvCacheEnabled: Bool = self.experimentalQuantizedKvCacheEnabled {
            jsonObject["experimental_quantized_kv_cache_enabled"] = experimentalQuantizedKvCacheEnabled;
        }
        if let experimentalFusedMoeDecodeEnabled: Bool = self.experimentalFusedMoeDecodeEnabled {
            jsonObject["experimental_fused_moe_decode_enabled"] = experimentalFusedMoeDecodeEnabled;
        }
        return jsonObject;
    }
}
