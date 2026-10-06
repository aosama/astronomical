import Foundation;

/// Which chunking fields the operator explicitly configured, porting
/// ConfiguredChunkingFields from crates/config/src/chunking_config.rs. Legacy
/// migration and precedence merging need the distinction between "explicitly
/// set" and "left at the default".
public struct ConfiguredChunkingFields: Equatable, Sendable {

    public let fixedPromptProcessingChunkSizeTokens: Bool;
    public let fixedSsdStreamingPromptProcessingChunkSizeTokens: Bool;
    public let fullAttentionKeyValueGrowthTokens: Bool;
    public let prefillGraphSubmissionLayerInterval: Bool;
    public let experimentalSsdPagingPrefillGraphSubmissionLayerInterval: Bool;
    public let experimentalSsdPagingGenerationGraphSubmissionLayerInterval: Bool;
    public let promptCacheBlockTokens: Bool;
    public let promptCacheCommonPrefixStrideBlocks: Bool;
    public let experimentalDecodeStageAttributionEnabled: Bool;
    public let experimentalQuantizedKvCacheEnabled: Bool;
    public let experimentalFusedMoeDecodeEnabled: Bool;

    public init(chunkingConfigFile: ChunkingConfigFile) {
        self.fixedPromptProcessingChunkSizeTokens = chunkingConfigFile.fixedPromptProcessingChunkSizeTokens != nil;
        self.fixedSsdStreamingPromptProcessingChunkSizeTokens = chunkingConfigFile.fixedSsdStreamingPromptProcessingChunkSizeTokens != nil;
        self.fullAttentionKeyValueGrowthTokens = chunkingConfigFile.fullAttentionKeyValueGrowthTokens != nil;
        self.prefillGraphSubmissionLayerInterval = chunkingConfigFile.prefillGraphSubmissionLayerInterval != nil;
        self.experimentalSsdPagingPrefillGraphSubmissionLayerInterval = chunkingConfigFile.experimentalSsdPagingPrefillGraphSubmissionLayerInterval != nil;
        self.experimentalSsdPagingGenerationGraphSubmissionLayerInterval = chunkingConfigFile.experimentalSsdPagingGenerationGraphSubmissionLayerInterval != nil;
        // The wire struct collapses Option<Option<u32>> into UInt32?; the
        // per-occurrence distinction is restored with the wire-struct slice.
        self.promptCacheBlockTokens = chunkingConfigFile.promptCacheBlockTokens != nil;
        self.promptCacheCommonPrefixStrideBlocks = chunkingConfigFile.promptCacheCommonPrefixStrideBlocks != nil;
        self.experimentalDecodeStageAttributionEnabled = chunkingConfigFile.experimentalDecodeStageAttributionEnabled != nil;
        self.experimentalQuantizedKvCacheEnabled = chunkingConfigFile.experimentalQuantizedKvCacheEnabled != nil;
        self.experimentalFusedMoeDecodeEnabled = chunkingConfigFile.experimentalFusedMoeDecodeEnabled != nil;
    }
}

internal extension ChunkingConfigFile {

    func configuredFields() -> ConfiguredChunkingFields {
        return ConfiguredChunkingFields(chunkingConfigFile: self);
    }

    /// Per-model override merging: model-level fields win, global fields fill
    /// the gaps, and a model with no overrides clones the global config,
    /// mirroring ChunkingConfigFile::merged.
    static func merged(globalChunkingFile: ChunkingConfigFile, modelChunkingFile: ChunkingConfigFile?) -> ChunkingConfigFile {
        guard let unwrappedModelChunkingFile = modelChunkingFile else {
            return globalChunkingFile;
        }
        return ChunkingConfigFile(
            fixedPromptProcessingChunkSizeTokens: unwrappedModelChunkingFile.fixedPromptProcessingChunkSizeTokens ?? globalChunkingFile.fixedPromptProcessingChunkSizeTokens,
            fixedSsdStreamingPromptProcessingChunkSizeTokens: unwrappedModelChunkingFile.fixedSsdStreamingPromptProcessingChunkSizeTokens ?? globalChunkingFile.fixedSsdStreamingPromptProcessingChunkSizeTokens,
            fullAttentionKeyValueGrowthTokens: unwrappedModelChunkingFile.fullAttentionKeyValueGrowthTokens ?? globalChunkingFile.fullAttentionKeyValueGrowthTokens,
            prefillGraphSubmissionLayerInterval: unwrappedModelChunkingFile.prefillGraphSubmissionLayerInterval ?? globalChunkingFile.prefillGraphSubmissionLayerInterval,
            experimentalSsdPagingPrefillGraphSubmissionLayerInterval: unwrappedModelChunkingFile.experimentalSsdPagingPrefillGraphSubmissionLayerInterval ?? globalChunkingFile.experimentalSsdPagingPrefillGraphSubmissionLayerInterval,
            experimentalSsdPagingGenerationGraphSubmissionLayerInterval: unwrappedModelChunkingFile.experimentalSsdPagingGenerationGraphSubmissionLayerInterval ?? globalChunkingFile.experimentalSsdPagingGenerationGraphSubmissionLayerInterval,
            promptCacheBlockTokens: unwrappedModelChunkingFile.promptCacheBlockTokens ?? globalChunkingFile.promptCacheBlockTokens,
            promptCacheCommonPrefixStrideBlocks: unwrappedModelChunkingFile.promptCacheCommonPrefixStrideBlocks ?? globalChunkingFile.promptCacheCommonPrefixStrideBlocks,
            experimentalDecodeStageAttributionEnabled: unwrappedModelChunkingFile.experimentalDecodeStageAttributionEnabled ?? globalChunkingFile.experimentalDecodeStageAttributionEnabled,
            experimentalQuantizedKvCacheEnabled: unwrappedModelChunkingFile.experimentalQuantizedKvCacheEnabled ?? globalChunkingFile.experimentalQuantizedKvCacheEnabled,
            experimentalFusedMoeDecodeEnabled: unwrappedModelChunkingFile.experimentalFusedMoeDecodeEnabled ?? globalChunkingFile.experimentalFusedMoeDecodeEnabled
        );
    }
}

/// Fully resolved chunking configuration, porting ChunkingConfig from
/// crates/config/src/chunking_config.rs. Every field carries the repository
/// default when the operator left it unset, and resolve() validates the
/// combined result before serving.
public struct ChunkingConfig: Equatable, Sendable {

    /// KV growth slack before a full-attention layer stops growing its cache.
    internal static let defaultFullAttentionKeyValueGrowthTokens: UInt32 = 256;
    /// Resident prefill works in chunks of this many tokens.
    internal static let defaultFixedPromptProcessingChunkSizeTokens: UInt32 = 2_048;
    /// Legacy config files were written when the resident default was larger;
    /// the value is kept only so legacy documents stay semantically stable.
    internal static let legacyDefaultFixedPromptProcessingChunkSizeTokens: UInt32 = 4_096;
    /// SSD streaming prefill chunks independently of the resident chunk.
    internal static let defaultFixedSsdStreamingPromptProcessingChunkSizeTokens: UInt32 = 2_048;
    internal static let defaultExperimentalSsdPagingPrefillGraphSubmissionLayerInterval: UInt32 = 1;
    internal static let defaultExperimentalSsdPagingGenerationGraphSubmissionLayerInterval: UInt32 = 3;
    internal static let defaultPrefillGraphSubmissionLayerInterval: UInt32 = 0;
    /// Common-prefix scanning advances prompt-cache block strides of this size.
    internal static let defaultPromptCacheCommonPrefixStrideBlocks: UInt32 = 4;

    private let fixedPromptProcessingChunkSizeTokensValue: UInt32;
    private let fixedSsdStreamingPromptProcessingChunkSizeTokensValue: UInt32;
    private let fullAttentionKeyValueGrowthTokensValue: UInt32;
    private let prefillGraphSubmissionLayerIntervalValue: UInt32;
    private let experimentalSsdPagingPrefillGraphSubmissionLayerIntervalValue: UInt32;
    private let experimentalSsdPagingGenerationGraphSubmissionLayerIntervalValue: UInt32;
    private let promptCacheBlockTokensValue: UInt32?;
    private let promptCacheCommonPrefixStrideBlocksValue: UInt32;
    private let experimentalDecodeStageAttributionEnabledValue: Bool;
    private let experimentalQuantizedKvCacheEnabledValue: Bool;
    private let experimentalFusedMoeDecodeEnabledValue: Bool;

    private init() {
        self.fixedPromptProcessingChunkSizeTokensValue = ChunkingConfig.defaultFixedPromptProcessingChunkSizeTokens;
        self.fixedSsdStreamingPromptProcessingChunkSizeTokensValue = ChunkingConfig.defaultFixedSsdStreamingPromptProcessingChunkSizeTokens;
        self.fullAttentionKeyValueGrowthTokensValue = ChunkingConfig.defaultFullAttentionKeyValueGrowthTokens;
        self.prefillGraphSubmissionLayerIntervalValue = ChunkingConfig.defaultPrefillGraphSubmissionLayerInterval;
        self.experimentalSsdPagingPrefillGraphSubmissionLayerIntervalValue = ChunkingConfig.defaultExperimentalSsdPagingPrefillGraphSubmissionLayerInterval;
        self.experimentalSsdPagingGenerationGraphSubmissionLayerIntervalValue = ChunkingConfig.defaultExperimentalSsdPagingGenerationGraphSubmissionLayerInterval;
        self.promptCacheBlockTokensValue = nil;
        self.promptCacheCommonPrefixStrideBlocksValue = ChunkingConfig.defaultPromptCacheCommonPrefixStrideBlocks;
        self.experimentalDecodeStageAttributionEnabledValue = false;
        self.experimentalQuantizedKvCacheEnabledValue = false;
        self.experimentalFusedMoeDecodeEnabledValue = false;
    }

    private init(resolvedChunkingFile: ChunkingConfigFile) {
        self.fixedPromptProcessingChunkSizeTokensValue = resolvedChunkingFile.fixedPromptProcessingChunkSizeTokens ?? ChunkingConfig.defaultFixedPromptProcessingChunkSizeTokens;
        self.fixedSsdStreamingPromptProcessingChunkSizeTokensValue = resolvedChunkingFile.fixedSsdStreamingPromptProcessingChunkSizeTokens ?? ChunkingConfig.defaultFixedSsdStreamingPromptProcessingChunkSizeTokens;
        self.fullAttentionKeyValueGrowthTokensValue = resolvedChunkingFile.fullAttentionKeyValueGrowthTokens ?? ChunkingConfig.defaultFullAttentionKeyValueGrowthTokens;
        self.prefillGraphSubmissionLayerIntervalValue = resolvedChunkingFile.prefillGraphSubmissionLayerInterval ?? ChunkingConfig.defaultPrefillGraphSubmissionLayerInterval;
        self.experimentalSsdPagingPrefillGraphSubmissionLayerIntervalValue = resolvedChunkingFile.experimentalSsdPagingPrefillGraphSubmissionLayerInterval ?? ChunkingConfig.defaultExperimentalSsdPagingPrefillGraphSubmissionLayerInterval;
        self.experimentalSsdPagingGenerationGraphSubmissionLayerIntervalValue = resolvedChunkingFile.experimentalSsdPagingGenerationGraphSubmissionLayerInterval ?? ChunkingConfig.defaultExperimentalSsdPagingGenerationGraphSubmissionLayerInterval;
        self.promptCacheBlockTokensValue = resolvedChunkingFile.promptCacheBlockTokens;
        self.promptCacheCommonPrefixStrideBlocksValue = resolvedChunkingFile.promptCacheCommonPrefixStrideBlocks ?? ChunkingConfig.defaultPromptCacheCommonPrefixStrideBlocks;
        self.experimentalDecodeStageAttributionEnabledValue = resolvedChunkingFile.experimentalDecodeStageAttributionEnabled ?? false;
        self.experimentalQuantizedKvCacheEnabledValue = resolvedChunkingFile.experimentalQuantizedKvCacheEnabled ?? false;
        self.experimentalFusedMoeDecodeEnabledValue = resolvedChunkingFile.experimentalFusedMoeDecodeEnabled ?? false;
    }

    public static func defaultConfig() -> ChunkingConfig {
        return ChunkingConfig();
    }

    public static func resolve(configuredChunkingFile: ChunkingConfigFile) throws -> ChunkingConfig {
        let resolvedChunkingConfig = ChunkingConfig(resolvedChunkingFile: configuredChunkingFile);
        try resolvedChunkingConfig.validate();
        return resolvedChunkingConfig;
    }

    private func validate() throws {
        let validatedFields: Array<(fieldName: String, fieldValue: UInt64)> = [
            (fieldName: "chunking.fixed_prompt_processing_chunk_size_tokens", fieldValue: UInt64(self.fixedPromptProcessingChunkSizeTokensValue)),
            (fieldName: "chunking.full_attention_key_value_growth_tokens", fieldValue: UInt64(self.fullAttentionKeyValueGrowthTokensValue)),
            (fieldName: "chunking.prompt_cache_common_prefix_stride_blocks", fieldValue: UInt64(self.promptCacheCommonPrefixStrideBlocksValue)),
            (fieldName: "chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens", fieldValue: UInt64(self.fixedSsdStreamingPromptProcessingChunkSizeTokensValue))
        ];
        for validatedField in validatedFields {
            if validatedField.fieldValue == 0 {
                throw ConfigResolutionError.invalidChunkingValue(fieldName: validatedField.fieldName, description: "must be positive");
            }
        }
        if let promptCacheBlockTokensValue: UInt32 = self.promptCacheBlockTokensValue {
            if promptCacheBlockTokensValue == 0 {
                throw ConfigResolutionError.invalidChunkingValue(
                    fieldName: "chunking.prompt_cache_block_tokens",
                    description: "must be null for automatic sizing or a positive token count"
                );
            }
        }
        // The MLX key-value growth path stores the value in a signed 32-bit
        // dimension, so a value past Int32.max cannot be represented.
        if self.fullAttentionKeyValueGrowthTokensValue > UInt32(Int32.max) {
            throw ConfigResolutionError.invalidChunkingValue(
                fieldName: "chunking.full_attention_key_value_growth_tokens",
                description: "must fit the signed 32-bit MLX dimension range"
            );
        }
    }

    public func fixedPromptProcessingChunkSizeTokens() -> UInt32 {
        return self.fixedPromptProcessingChunkSizeTokensValue;
    }

    public func fixedSsdStreamingPromptProcessingChunkSizeTokens() -> UInt32 {
        return self.fixedSsdStreamingPromptProcessingChunkSizeTokensValue;
    }

    public func fullAttentionKeyValueGrowthTokens() -> UInt32 {
        return self.fullAttentionKeyValueGrowthTokensValue;
    }

    public func prefillGraphSubmissionLayerInterval() -> UInt32 {
        return self.prefillGraphSubmissionLayerIntervalValue;
    }

    public func experimentalSsdPagingPrefillGraphSubmissionLayerInterval() -> UInt32 {
        return self.experimentalSsdPagingPrefillGraphSubmissionLayerIntervalValue;
    }

    public func experimentalSsdPagingGenerationGraphSubmissionLayerInterval() -> UInt32 {
        return self.experimentalSsdPagingGenerationGraphSubmissionLayerIntervalValue;
    }

    public func promptCacheBlockTokens() -> UInt32? {
        return self.promptCacheBlockTokensValue;
    }

    public func promptCacheCommonPrefixStrideBlocks() -> UInt32 {
        return self.promptCacheCommonPrefixStrideBlocksValue;
    }

    public func experimentalDecodeStageAttributionEnabled() -> Bool {
        return self.experimentalDecodeStageAttributionEnabledValue;
    }

    public func experimentalQuantizedKvCacheEnabled() -> Bool {
        return self.experimentalQuantizedKvCacheEnabledValue;
    }

    public func experimentalFusedMoeDecodeEnabled() -> Bool {
        return self.experimentalFusedMoeDecodeEnabledValue;
    }
}
