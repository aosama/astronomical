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

    /// Synthetic tail tokens in a high range so they never collide with the
    /// sequential fixture tokens, mirroring the Rust journey generators.
    static func syntheticTailTokens(tokenCount: Int, tokenSeed: UInt32) -> [UInt32] {
        return (0..<tokenCount).map({ (tokenOffset: Int) -> UInt32 in
            return tokenSeed &+ UInt32(tokenOffset);
        });
    }

    /// Writes real header-shaped state files, mirroring the contract's
    /// exact-size geometry so the transaction's size validation holds.
    final class SyntheticStateFileStaging: PersistentPromptCacheStateFileStaging {

        func stageStateFile(
            stateFileName: String, stagingBlockDirectory: URL, blockTokenCount: Int,
            modelContract: PersistentPromptCacheModelContract
        ) throws -> UInt64 {
            let stateLayouts: [DecoderCachePersistedTensorLayout] = stateFileName
                == PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME
                ? modelContract.decoderCacheLayout.sequenceTensorLayouts()
                : modelContract.decoderCacheLayout.boundaryTensorLayouts();
            // The native writer materializes tensors largest-first (name
            // tie-break) so the exact-size geometry stays predictive; the
            // stager must assign payload offsets in that same order, because
            // cumulative offsets pass through different values — and
            // therefore different digit counts — in any other order.
            var tensorEntries: [(
                tensorName: String, dtypeName: String, dimensions: [Int], payloadBytes: UInt64
            )] = [];
            for persistedTensorLayout: DecoderCachePersistedTensorLayout in stateLayouts {
                let tensorLayout: DecoderCacheTensorLayout = persistedTensorLayout.tensorLayout;
                let tensorShape: [Int] = tensorLayout.dimensions.enumerated().map(
                    { (dimensionEntry: (offset: Int, element: Int)) -> Int in
                        if dimensionEntry.offset == tensorLayout.sequenceAxis {
                            return blockTokenCount;
                        }
                        return dimensionEntry.element;
                    });
                var tensorPayloadByteCount: UInt64 = UInt64(tensorLayout.dtype.scalarByteCount);
                for tensorDimension: Int in tensorShape {
                    tensorPayloadByteCount = tensorPayloadByteCount
                        &* UInt64(max(tensorDimension, 0));
                }
                tensorEntries.append((
                    persistedTensorLayout.persistentTensorName,
                    tensorLayout.dtype.safetensorsDtypeName,
                    tensorShape,
                    tensorPayloadByteCount));
            }
            tensorEntries.sort(by: { (leftTensor, rightTensor) -> Bool in
                if leftTensor.payloadBytes != rightTensor.payloadBytes {
                    return leftTensor.payloadBytes > rightTensor.payloadBytes;
                }
                return leftTensor.tensorName < rightTensor.tensorName;
            });
            var headerObject: [String: Any] = [:];
            var payloadOffsetBytes: UInt64 = 0;
            for tensorEntry in tensorEntries {
                headerObject[tensorEntry.tensorName] = [
                    "dtype": tensorEntry.dtypeName,
                    "shape": tensorEntry.dimensions,
                    "data_offsets": [payloadOffsetBytes, payloadOffsetBytes &+ tensorEntry.payloadBytes],
                ];
                payloadOffsetBytes = payloadOffsetBytes &+ tensorEntry.payloadBytes;
            }
            headerObject["__metadata__"] = [
                "format_version": PersistentPromptCacheBlockHeader.FORMAT_VERSION,
                "block_token_count": String(blockTokenCount),
                "storage_contract_fingerprint": modelContract.storageContractFingerprintHex(),
            ];
            var fileBytes: Data = Data();
            let headerData: Data = try JSONSerialization.data(
                withJSONObject: headerObject, options: [.sortedKeys]);
            var littleEndianHeaderLength: UInt64 = UInt64(headerData.count);
            withUnsafeBytes(of: &littleEndianHeaderLength) { (valueBuffer: UnsafeRawBufferPointer) in
                fileBytes.append(contentsOf: valueBuffer);
            };
            fileBytes.append(headerData);
            fileBytes.append(Data(count: Int(payloadOffsetBytes)));
            try fileBytes.write(
                to: stagingBlockDirectory.appendingPathComponent(stateFileName));
            return UInt64(fileBytes.count);
        }
    }

    /// Opens a hermetic disk store under a unique temporary root with the
    /// frozen fixture contract.
    static func openDiskStore(
        modelContract: PersistentPromptCacheModelContract,
        globalPromptCacheMaximumSizeBytes: UInt64 = 50_000_000_000
    ) throws -> PersistentPromptCacheDiskStore {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalRoot, withIntermediateDirectories: true);
        return try Self.openDiskStore(
            globalRoot: globalRoot, modelContract: modelContract,
            globalPromptCacheMaximumSizeBytes: globalPromptCacheMaximumSizeBytes);
    }

    /// Opens (or rescans) the store under an existing fixture root so a
    /// journey can reopen the same directory across publication and restart.
    static func openDiskStore(
        globalRoot: URL,
        activeModelPromptCacheDirectory: URL? = nil,
        modelContract: PersistentPromptCacheModelContract,
        globalPromptCacheMaximumSizeBytes: UInt64 = 50_000_000_000
    ) throws -> PersistentPromptCacheDiskStore {
        return try PersistentPromptCacheDiskStore.open(
            diskStoreConfig: PersistentPromptCacheDiskStoreConfig(
                activeModelPromptCacheDirectory: activeModelPromptCacheDirectory
                    ?? globalRoot.appendingPathComponent("org/model-a/rev-1", isDirectory: true),
                globalPromptCacheRootDirectory: globalRoot,
                globalPromptCacheMaximumSizeBytes: globalPromptCacheMaximumSizeBytes),
            modelContract: modelContract);
    }

    /// Tiny synthetic contracts exercising the storage-layout variants the
    /// frozen Ornith fixture cannot: one append-only attention layer with a
    /// 16-token block (sequence state only), and one fixed recurrent tensor
    /// (boundary state only).
    static func syntheticSequenceOnlyContract() throws -> PersistentPromptCacheModelContract {
        let decoderCacheLayout: DecoderCacheLayout = try DecoderCacheLayout(layers: [
            .appendOnlyAttention(
                keys: .sequence(
                    tensorRoleName: "attention.keys",
                    dtype: .float16,
                    dimensions: [1, 0, 4],
                    sequenceAxis: 1),
                values: .sequence(
                    tensorRoleName: "attention.values",
                    dtype: .float16,
                    dimensions: [1, 0, 4],
                    sequenceAxis: 1),
                capacityGrowthTokens: 16),
        ]);
        return try PersistentPromptCacheModelContract.resolve(
            modelId: "fictional-sequence-only-model",
            modelRevision: "fictional-revision",
            decoderCacheLayout: decoderCacheLayout,
            maximumContextTokenCount: 128,
            effectiveMlxMemoryCeilingBytes: 1_000_000,
            globalSsdQuotaBytes: 1_000_000,
            configuredBlockTokenCount: nil,
            commonPrefixCheckpointStrideBlocks: 4);
    }

    static func syntheticBoundaryOnlyContract() throws -> PersistentPromptCacheModelContract {
        let decoderCacheLayout: DecoderCacheLayout = try DecoderCacheLayout(layers: [
            .recurrentTensor(tensor: .fixed(
                tensorRoleName: "recurrent.state",
                dtype: .float32,
                dimensions: [25])),
        ]);
        return try PersistentPromptCacheModelContract.resolve(
            modelId: "fictional-boundary-only-model",
            modelRevision: "fictional-revision",
            decoderCacheLayout: decoderCacheLayout,
            maximumContextTokenCount: 100,
            effectiveMlxMemoryCeilingBytes: 1_000_000,
            globalSsdQuotaBytes: 10_000,
            configuredBlockTokenCount: nil,
            commonPrefixCheckpointStrideBlocks: 4);
    }

    /// A boundary-state-only contract whose two fixed tensors carry the
    /// hybrid layer's composite role vocabulary — `linear.convolution` and
    /// `linear.gated_delta_recurrent` — so published snapshots feed the
    /// state bridge's absorb directly.
    static func syntheticCompositeBoundaryContract() throws -> PersistentPromptCacheModelContract {
        let decoderCacheLayout: DecoderCacheLayout = try DecoderCacheLayout(layers: [
            .composite(components: [
                .recurrentTensor(tensor: .fixed(
                    tensorRoleName: "linear.convolution",
                    dtype: .float32,
                    dimensions: [1, 1, 4])),
                .recurrentTensor(tensor: .fixed(
                    tensorRoleName: "linear.gated_delta_recurrent",
                    dtype: .float32,
                    dimensions: [1, 2, 2, 2])),
            ]),
        ]);
        return try PersistentPromptCacheModelContract.resolve(
            modelId: "fictional-composite-model",
            modelRevision: "fictional-revision",
            decoderCacheLayout: decoderCacheLayout,
            maximumContextTokenCount: 100,
            effectiveMlxMemoryCeilingBytes: 1_000_000,
            globalSsdQuotaBytes: 100_000,
            configuredBlockTokenCount: nil,
            commonPrefixCheckpointStrideBlocks: 4);
    }

    /// A sequence-only contract whose full-attention tensors match the live
    /// decoder state's rank-four slab shape [batch, heads, tokens, head
    /// dimension], so published blocks feed the state bridge's restore
    /// directly: keys and values are [1, 2, 0, 4] with the token axis at
    /// position two.
    static func syntheticRankFourSequenceContract() throws -> PersistentPromptCacheModelContract {
        let decoderCacheLayout: DecoderCacheLayout = try DecoderCacheLayout(layers: [
            .appendOnlyAttention(
                keys: .sequence(
                    tensorRoleName: "attention.keys",
                    dtype: .float16,
                    dimensions: [1, 2, 0, 4],
                    sequenceAxis: 2),
                values: .sequence(
                    tensorRoleName: "attention.values",
                    dtype: .float16,
                    dimensions: [1, 2, 0, 4],
                    sequenceAxis: 2),
                capacityGrowthTokens: 16),
        ]);
        return try PersistentPromptCacheModelContract.resolve(
            modelId: "fictional-rank-four-model",
            modelRevision: "fictional-revision",
            decoderCacheLayout: decoderCacheLayout,
            maximumContextTokenCount: 128,
            effectiveMlxMemoryCeilingBytes: 1_000_000,
            globalSsdQuotaBytes: 1_000_000,
            configuredBlockTokenCount: nil,
            commonPrefixCheckpointStrideBlocks: 4);
    }

    /// Sums regular-file bytes under one directory tree, mirroring the Rust
    /// journeys' directory size helper.
    static func directoryFileSizeBytes(directoryPath: URL) throws -> UInt64 {
        var pendingDirectories: [URL] = [directoryPath];
        var totalByteCount: UInt64 = 0;
        while let pendingDirectory: URL = pendingDirectories.popLast() {
            for enumeratedEntry: URL in try FileManager.default.contentsOfDirectory(
                at: pendingDirectory, includingPropertiesForKeys: [.fileSizeKey],
                options: []) {
                let entryPath: URL = pendingDirectory.appendingPathComponent(
                    enumeratedEntry.lastPathComponent);
                var isDirectory: ObjCBool = ObjCBool(false);
                FileManager.default.fileExists(atPath: entryPath.path, isDirectory: &isDirectory);
                if isDirectory.boolValue {
                    pendingDirectories.append(entryPath);
                } else {
                    let fileAttributes: [FileAttributeKey: Any] = try FileManager.default
                        .attributesOfItem(atPath: entryPath.path);
                    totalByteCount = totalByteCount &+ ((fileAttributes[.size] as? NSNumber)?
                        .uint64Value ?? 0);
                }
            }
        }
        return totalByteCount;
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
