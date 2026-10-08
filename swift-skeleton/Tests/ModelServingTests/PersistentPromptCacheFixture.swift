import Foundation;

import ModelServing;

@testable import ModelServing;

/// The frozen Ornith 1.0 persistent prompt-cache fixture, port of the
/// contract half of crates/model-serving/tests/common/qwen3_5_moe.rs. This
/// hermetic fixture binds no MLX arrays: it supplies the frozen BF16 state
/// contract explicitly; real engine acceptance uses load-derived dtypes and
/// reads the resulting block geometry back from the engine.
enum PersistentPromptCacheFixture {

    static let ORNITH_MODEL_ID: String = "Ornith-1.0-35B-OptiQ-4bit";
    static let ORNITH_MODEL_REVISION: String = "ce62c23d34b91d84f838e0b292d517dbe4b9b60f";

    /// The frozen storage contract shared by the lookup and block-format
    /// journeys; resolved once per process.
    static func ornithModelContract() throws -> PersistentPromptCacheModelContract {
        let frozenConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: try Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        // This hermetic fixture has no bound MLX arrays. Supply its frozen
        // BF16 state contract explicitly; real engine acceptance uses
        // load-derived dtypes and reads the resulting block geometry back
        // from the engine.
        let decoderLayerCacheDtypes: [Qwen35DecoderLayerCacheDtypes] =
            try bfloat16DecoderLayerCacheDtypes(qwen35Config: frozenConfig);
        return try PersistentPromptCacheModelContract.resolve(
            modelId: PersistentPromptCacheFixture.ORNITH_MODEL_ID,
            modelRevision: PersistentPromptCacheFixture.ORNITH_MODEL_REVISION,
            decoderCacheLayout: try Qwen35DecoderCacheLayoutBuilder.buildDecoderCacheLayout(
                qwen35Config: frozenConfig,
                fullAttentionKeyValueGrowthTokens: 256,
                decoderLayerCacheDtypes: decoderLayerCacheDtypes),
            maximumContextTokenCount: Int(frozenConfig.maximumPositionCount()),
            effectiveMlxMemoryCeilingBytes: 20_000_000_000,
            globalSsdQuotaBytes: 50_000_000_000,
            configuredBlockTokenCount: nil,
            commonPrefixCheckpointStrideBlocks: 4);
    }

    /// Builds the per-layer BF16 state dtypes the frozen config's layer
    /// families demand, mirroring the Rust `decoder_layer_cache_dtypes`.
    static func bfloat16DecoderLayerCacheDtypes(
        qwen35Config: Qwen3_5Config
    ) throws -> [Qwen35DecoderLayerCacheDtypes] {
        return try decoderLayerCacheDtypes(
            qwen35Config: qwen35Config, activationStateDtype: .bfloat16);
    }

    private static func decoderLayerCacheDtypes(
        qwen35Config: Qwen3_5Config, activationStateDtype: DecoderCacheTensorDtype
    ) throws -> [Qwen35DecoderLayerCacheDtypes] {
        return (0..<Int(qwen35Config.layerCount())).map({ (decoderLayerIndex: Int) in
            if qwen35Config.decoderLayerIsFullAttention(decoderLayerIndex: decoderLayerIndex) {
                return .fullAttention(
                    keys: activationStateDtype, values: activationStateDtype);
            }
            return .linearAttention(convolution: activationStateDtype);
        });
    }

    /// The prompt `0..<tokenCount` covering `completeBlockCount` full blocks
    /// plus `trailingTokenCount` trailing tokens.
    static func promptTokensWithCompleteBlocksAndTrailingTokens(
        modelContract: PersistentPromptCacheModelContract,
        completeBlockCount: Int,
        trailingTokenCount: Int
    ) -> [UInt32] {
        let promptTokenCount: Int =
            completeBlockCount * modelContract.blockTokenCount + trailingTokenCount;
        return (0..<promptTokenCount).map({ (tokenIndex: Int) -> UInt32 in
            return UInt32(tokenIndex);
        });
    }

    /// Hashes the prompt's first `requestedBlockCount` complete blocks in
    /// chain order, mirroring the Rust key-walk helper.
    static func blockKeysForPrompt(
        modelContract: PersistentPromptCacheModelContract,
        promptTokens: [UInt32],
        requestedBlockCount: Int
    ) throws -> [PersistentPromptCacheBlockKey] {
        var blockKeys: [PersistentPromptCacheBlockKey] = [];
        blockKeys.reserveCapacity(requestedBlockCount);
        var parentBlockKey: PersistentPromptCacheBlockKey? = nil;
        for blockIndex: Int in 0..<requestedBlockCount {
            let blockStart: Int = blockIndex * modelContract.blockTokenCount;
            let blockEnd: Int = blockStart + modelContract.blockTokenCount;
            let blockKey: PersistentPromptCacheBlockKey;
            if let parentBlockKey: PersistentPromptCacheBlockKey = parentBlockKey {
                blockKey = try parentBlockKey.forChildBlock(
                    blockTokens: Array(promptTokens[blockStart..<blockEnd]));
            } else {
                blockKey = try PersistentPromptCacheBlockKey.forRootBlock(
                    modelContract: modelContract,
                    blockTokens: Array(promptTokens[blockStart..<blockEnd]));
            }
            parentBlockKey = blockKey;
            blockKeys.append(blockKey);
        }
        return blockKeys;
    }
}
