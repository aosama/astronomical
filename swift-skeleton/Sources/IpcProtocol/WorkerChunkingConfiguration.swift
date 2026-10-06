import Foundation;

/// Complete resolved model-serving work-partition contract supplied by the supervisor.
///
/// The user-facing configuration is validated before this data transfer object
/// is created. Keeping these values together prevents the worker, model, cache,
/// and model families from independently restoring hidden defaults.
public struct WorkerChunkingConfiguration: Equatable, Sendable {
    /// Fixed prompt work while sparse experts are fully resident.
    public let fixedPromptProcessingChunkSizeTokens: UInt32;
    /// Fixed prompt work while sparse experts stream from storage.
    public let fixedSsdStreamingPromptProcessingChunkSizeTokens: UInt32;
    /// Capacity added when append-only attention state outgrows its current slab.
    public let fullAttentionKeyValueGrowthTokens: UInt32;
    /// Decoder-layer interval between multi-token prefill command buffers while
    /// sparse experts are fully resident. Zero keeps one lazy tape per chunk.
    public let prefillGraphSubmissionLayerInterval: UInt32;
    /// Decoder-layer interval between multi-token prefill command buffers while
    /// sparse experts stream from SSD (solid-state drive). Zero keeps one lazy
    /// tape per chunk.
    public let experimentalSsdPagingPrefillGraphSubmissionLayerInterval: UInt32;
    /// Experimental decoder-layer interval for one-token solid-state-drive paging.
    ///
    /// Zero disables intermediate generation submission. Ignored while experts
    /// are fully memory resident.
    public let experimentalSsdPagingGenerationGraphSubmissionLayerInterval: UInt32;
    /// Exact persistent-cache block length, or `nil` for model-derived sizing.
    public let promptCacheBlockTokens: UInt32?;
    /// Number of cache blocks between retained branch restart checkpoints.
    public let promptCacheCommonPrefixStrideBlocks: UInt32;
    /// Diagnostic mode that runs decode forwards as stage-split evaluations
    /// for deep attribution. Roughly doubles decode time; never for serving.
    public let experimentalDecodeStageAttributionEnabled: Bool;
    /// Diagnostic/quality-gated mode that stores the KV (key-value) slab quantized.
    public let experimentalQuantizedKvCacheEnabled: Bool;
    /// Fused single-token quantized expert decode kernels for capable MoE
    /// (mixture-of-experts) families.
    public let experimentalFusedMoeDecodeEnabled: Bool;

    public init(
        fixedPromptProcessingChunkSizeTokens: UInt32,
        fixedSsdStreamingPromptProcessingChunkSizeTokens: UInt32,
        fullAttentionKeyValueGrowthTokens: UInt32,
        prefillGraphSubmissionLayerInterval: UInt32,
        experimentalSsdPagingPrefillGraphSubmissionLayerInterval: UInt32,
        experimentalSsdPagingGenerationGraphSubmissionLayerInterval: UInt32,
        promptCacheBlockTokens: UInt32?,
        promptCacheCommonPrefixStrideBlocks: UInt32,
        experimentalDecodeStageAttributionEnabled: Bool,
        experimentalQuantizedKvCacheEnabled: Bool,
        experimentalFusedMoeDecodeEnabled: Bool
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

    internal static let wireFieldNames: Array<String> = [
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
        "experimental_fused_moe_decode_enabled",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "fixed_prompt_processing_chunk_size_tokens", value: .unsignedInteger(UInt64(self.fixedPromptProcessingChunkSizeTokens)));
        wireObject.appendEntry(key: "fixed_ssd_streaming_prompt_processing_chunk_size_tokens", value: .unsignedInteger(UInt64(self.fixedSsdStreamingPromptProcessingChunkSizeTokens)));
        wireObject.appendEntry(key: "full_attention_key_value_growth_tokens", value: .unsignedInteger(UInt64(self.fullAttentionKeyValueGrowthTokens)));
        wireObject.appendEntry(key: "prefill_graph_submission_layer_interval", value: .unsignedInteger(UInt64(self.prefillGraphSubmissionLayerInterval)));
        wireObject.appendEntry(key: "experimental_ssd_paging_prefill_graph_submission_layer_interval", value: .unsignedInteger(UInt64(self.experimentalSsdPagingPrefillGraphSubmissionLayerInterval)));
        wireObject.appendEntry(key: "experimental_ssd_paging_generation_graph_submission_layer_interval", value: .unsignedInteger(UInt64(self.experimentalSsdPagingGenerationGraphSubmissionLayerInterval)));
        wireObject.appendEntry(key: "prompt_cache_block_tokens", value: WorkerChunkingConfiguration.optionalUInt32WireValue(self.promptCacheBlockTokens));
        wireObject.appendEntry(key: "prompt_cache_common_prefix_stride_blocks", value: .unsignedInteger(UInt64(self.promptCacheCommonPrefixStrideBlocks)));
        wireObject.appendEntry(key: "experimental_decode_stage_attribution_enabled", value: .boolean(self.experimentalDecodeStageAttributionEnabled));
        wireObject.appendEntry(key: "experimental_quantized_kv_cache_enabled", value: .boolean(self.experimentalQuantizedKvCacheEnabled));
        wireObject.appendEntry(key: "experimental_fused_moe_decode_enabled", value: .boolean(self.experimentalFusedMoeDecodeEnabled));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerChunkingConfiguration {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedConfiguration = WorkerChunkingConfiguration(
            fixedPromptProcessingChunkSizeTokens: try wireObject.decodeUInt32(fieldName: "fixed_prompt_processing_chunk_size_tokens"),
            fixedSsdStreamingPromptProcessingChunkSizeTokens: try wireObject.decodeUInt32(fieldName: "fixed_ssd_streaming_prompt_processing_chunk_size_tokens"),
            fullAttentionKeyValueGrowthTokens: try wireObject.decodeUInt32(fieldName: "full_attention_key_value_growth_tokens"),
            prefillGraphSubmissionLayerInterval: try wireObject.decodeUInt32(fieldName: "prefill_graph_submission_layer_interval"),
            experimentalSsdPagingPrefillGraphSubmissionLayerInterval: try wireObject.decodeUInt32(fieldName: "experimental_ssd_paging_prefill_graph_submission_layer_interval"),
            experimentalSsdPagingGenerationGraphSubmissionLayerInterval: try wireObject.decodeUInt32(fieldName: "experimental_ssd_paging_generation_graph_submission_layer_interval"),
            promptCacheBlockTokens: try wireObject.decodeOptionalUInt32(fieldName: "prompt_cache_block_tokens"),
            promptCacheCommonPrefixStrideBlocks: try wireObject.decodeUInt32(fieldName: "prompt_cache_common_prefix_stride_blocks"),
            experimentalDecodeStageAttributionEnabled: try wireObject.decodeBoolAllowingAbsent(fieldName: "experimental_decode_stage_attribution_enabled"),
            experimentalQuantizedKvCacheEnabled: try wireObject.decodeBoolAllowingAbsent(fieldName: "experimental_quantized_kv_cache_enabled"),
            experimentalFusedMoeDecodeEnabled: try wireObject.decodeBoolAllowingAbsent(fieldName: "experimental_fused_moe_decode_enabled"));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerChunkingConfiguration.wireFieldNames);
        return parsedConfiguration;
    }

    /// Returns the command-buffer submission interval for one forward.
    ///
    /// Resident multi-token prefill uses the prefill interval (default zero:
    /// one lazy tape). Resident one-token generation stays at zero. SSD-paged
    /// prefill and decode use the experimental SSD-paging intervals.
    public func graphSubmissionLayerInterval(tokenCount: Int32, sparseExpertsArePaged: Bool) -> UInt32 {
        return WorkerChunkingConfiguration.graphSubmissionLayerInterval(
            tokenCount: tokenCount,
            sparseExpertsArePaged: sparseExpertsArePaged,
            prefillGraphSubmissionLayerInterval: self.prefillGraphSubmissionLayerInterval,
            ssdPagingPrefillGraphSubmissionLayerInterval: self.experimentalSsdPagingPrefillGraphSubmissionLayerInterval,
            ssdPagingGenerationGraphSubmissionLayerInterval: self.experimentalSsdPagingGenerationGraphSubmissionLayerInterval);
    }

    /// Returns the command-buffer submission interval for one forward,
    /// matching the standalone Rust helper of the same purpose.
    public static func graphSubmissionLayerInterval(
        tokenCount: Int32,
        sparseExpertsArePaged: Bool,
        prefillGraphSubmissionLayerInterval: UInt32,
        ssdPagingPrefillGraphSubmissionLayerInterval: UInt32,
        ssdPagingGenerationGraphSubmissionLayerInterval: UInt32
    ) -> UInt32 {
        if !sparseExpertsArePaged {
            if tokenCount == 1 {
                return 0;
            }
            return prefillGraphSubmissionLayerInterval;
        }
        if tokenCount == 1 {
            return ssdPagingGenerationGraphSubmissionLayerInterval;
        }
        return ssdPagingPrefillGraphSubmissionLayerInterval;
    }

    private static func optionalUInt32WireValue(_ optionalValue: UInt32?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .unsignedInteger(UInt64(unwrappedValue));
    }
}
