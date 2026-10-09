import Foundation;

import MLX;
import MLXLMCommon;

import RuntimeIntegration;


/// The MoE engine's persistent prompt-cache owner: opens the disk store from
/// the executed cache dtypes, restores the longest prompt prefix, and hands
/// restored state to the engine, port of the Rust engine-state restore half
/// of `decoder_state_reuse` without the request-stack machinery the Swift
/// single-request engine does not carry. Capture publication lives in the
/// capture owner.
public final class Qwen35MoePromptCacheController {

    /// The append granularity the layout's admission projection assumes;
    /// full-attention slabs grow by prefill chunk appends, so the contract's
    /// persistence alignment follows the chunk the engine serves.
    private let fullAttentionKeyValueGrowthTokens: Int;

    var diskStoreValue: PersistentPromptCacheDiskStore?;

    private(set) var counters: PersistentPromptCacheCounters = PersistentPromptCacheCounters();

    init(fullAttentionKeyValueGrowthTokens: Int) {
        self.fullAttentionKeyValueGrowthTokens = fullAttentionKeyValueGrowthTokens;
    }

    /// Whether the store is open and capture/restore may run.
    public var isOpen: Bool {
        return self.diskStoreValue != nil;
    }

    /// The resolved storage contract; valid only after a successful attach.
    public var modelContract: PersistentPromptCacheModelContract {
        guard let diskStore: PersistentPromptCacheDiskStore = self.diskStoreValue else {
            preconditionFailure("the persistent prompt cache was consulted before attach");
        }
        return diskStore.modelContract;
    }

    /// The open disk store, or nil while the cache stays detached.
    var diskStore: PersistentPromptCacheDiskStore? {
        return self.diskStoreValue;
    }

    /// Resolves the storage contract from the executed cache dtypes and
    /// opens the disk store, port of the Rust startup order that resolves
    /// the contract once from the bound graph before any request runs.
    func openStore(
        attachment: Qwen35MoePromptCacheAttachment,
        repositoryConfiguration: Qwen3_5Config,
        executedLayerCacheDtypes: [Qwen35DecoderLayerCacheDtypes]
    ) throws {
        let decoderCacheLayout: DecoderCacheLayout = try Qwen35DecoderCacheLayoutBuilder
            .buildDecoderCacheLayout(
                qwen35Config: repositoryConfiguration,
                fullAttentionKeyValueGrowthTokens: self.fullAttentionKeyValueGrowthTokens,
                decoderLayerCacheDtypes: executedLayerCacheDtypes);
        let modelContract: PersistentPromptCacheModelContract = try
            PersistentPromptCacheModelContract.resolve(
                modelId: attachment.modelId,
                modelRevision: attachment.modelRevision,
                decoderCacheLayout: decoderCacheLayout,
                maximumContextTokenCount: Int(repositoryConfiguration.maximumPositionCount()),
                effectiveMlxMemoryCeilingBytes: attachment.effectiveMlxMemoryCeilingBytes,
                globalSsdQuotaBytes: attachment.globalPromptCacheMaximumSizeBytes,
                configuredBlockTokenCount: attachment.configuredBlockTokenCount,
                commonPrefixCheckpointStrideBlocks: attachment
                    .commonPrefixCheckpointStrideBlocks);
        let storeConfig: PersistentPromptCacheDiskStoreConfig =
            PersistentPromptCacheDiskStoreConfig(
                activeModelPromptCacheDirectory: attachment.activeModelPromptCacheDirectory,
                globalPromptCacheRootDirectory: attachment.globalPromptCacheRootDirectory,
                globalPromptCacheMaximumSizeBytes: attachment
                    .globalPromptCacheMaximumSizeBytes)
                .forModel(modelId: attachment.modelId, modelRevision: attachment.modelRevision);
        self.diskStoreValue = try PersistentPromptCacheDiskStore.open(
            diskStoreConfig: storeConfig, modelContract: modelContract);
    }

    /// Restores the longest cached prompt prefix into a fresh state stack,
    /// port of the Rust restore sequence: decide from the small index first,
    /// load every block of the proven chain, reconstruct each full-attention
    /// layer with one concatenation, then absorb the newest snapshot at the
    /// restored end. On a miss the outcome stays cold and the request
    /// prefills the complete prompt.
    func restorePromptPrefix(
        promptTokenIds: [UInt32]
    ) throws -> Qwen35MoePromptCacheRestoreOutcome {
        guard let diskStore: PersistentPromptCacheDiskStore = self.diskStoreValue else {
            throw Qwen35MoePromptCacheError.promptCacheNotAttached;
        }
        let modelContract: PersistentPromptCacheModelContract = diskStore.modelContract;
        let lookupResult: PersistentPromptCachePrefixLookupResult =
            PersistentPromptCachePrefixLookup.forPromptWithPartialTailBlockRecovery(
                modelContract: modelContract,
                promptTokens: promptTokenIds,
                blockCausalInputs: [],
                kvBlockExists: { (blockHash: Data) -> Bool in
                    return diskStore.hasKvBlock(blockHash: blockHash);
                },
                recurrentSnapshotExists: { (blockHash: Data) -> Bool in
                    return diskStore.hasRecurrentSnapshot(blockHash: blockHash);
                });
        let restoredTokenCount: Int = lookupResult.restoredTokenCount;
        if restoredTokenCount == 0 {
            self.counters.recordCacheMiss();
            return Qwen35MoePromptCacheRestoreOutcome(lookupResult: lookupResult);
        }
        let blockTokenCount: Int = modelContract.blockTokenCount;
        let completeBlockCount: Int = restoredTokenCount / blockTokenCount;
        var chainBlockKeys: [PersistentPromptCacheBlockKey] = [];
        chainBlockKeys.reserveCapacity(completeBlockCount);
        var chainParentKey: PersistentPromptCacheBlockKey? = nil;
        for blockIndex: Int in 0..<completeBlockCount {
            let blockStart: Int = blockIndex * blockTokenCount;
            let blockEnd: Int = blockStart + blockTokenCount;
            let reconstructedBlockKey: PersistentPromptCacheBlockKey = try
                Self.reconstructedBlockKey(
                    modelContract: modelContract, promptTokenIds: promptTokenIds,
                    blockStart: blockStart, blockEnd: blockEnd,
                    parentBlockKey: chainParentKey);
            chainParentKey = reconstructedBlockKey;
            chainBlockKeys.append(reconstructedBlockKey);
        }
        var loadBlockKeys: [PersistentPromptCacheBlockKey] = chainBlockKeys;
        if let restoredTailBlockKey: PersistentPromptCacheBlockKey =
            lookupResult.restoredPartialTailBlockKey {
            // The tail loads like any other sequence block but never joins
            // the capture-parent chain.
            loadBlockKeys.append(restoredTailBlockKey);
        }
        var restoredKvBlockTensorMaps: [[String: MLXArray]] = [];
        restoredKvBlockTensorMaps.reserveCapacity(loadBlockKeys.count);
        for (loadIndex, loadBlockKey): (Int, PersistentPromptCacheBlockKey) in
            loadBlockKeys.enumerated() {
            guard let loadedKvTensors: [String: MLXArray] = try diskStore
                .loadKvBlock(blockKey: loadBlockKey)
            else {
                throw Qwen35MoePromptCacheError.restoredBlockVanished(
                    blockIndex: UInt32(clamping: loadIndex));
            }
            restoredKvBlockTensorMaps.append(loadedKvTensors);
        }
        // The fresh stack mirrors the validated layout's layer families:
        // full-attention owners take the concatenated slabs and composite
        // owners absorb the recurrent snapshot.
        var decoderLayerStates: [DecoderCacheState] = [];
        decoderLayerStates.reserveCapacity(modelContract.decoderCacheLayout.layerCount);
        for layerIndex: Int in 0..<modelContract.decoderCacheLayout.layerCount {
            switch modelContract.decoderCacheLayout.layer(layerIndex) {
            case .composite:
                decoderLayerStates.append(.composite(
                    convolution: ConvolutionState(),
                    recurrent: GatedDeltaRecurrentState()));
            case .some:
                decoderLayerStates.append(
                    .appendOnlyAttention(FullAttentionKeyValueState()));
            case .none:
                throw Qwen35MoePromptCacheError.liveCacheFamilyMismatch(layerIndex: layerIndex);
            }
        }
        let restoredStateStack: RequestDecoderStateStack =
            RequestDecoderStateStack(decoderLayerStates: decoderLayerStates);
        try restoredStateStack.restoreFullAttentionKvConcat(
            blockTensorMaps: restoredKvBlockTensorMaps, restoredTokenCount: restoredTokenCount);
        guard let newestCompleteChainKey: PersistentPromptCacheBlockKey = chainBlockKeys.last
        else {
            throw Qwen35MoePromptCacheError.restoredBlockVanished(blockIndex: 0);
        }
        let recurrentSnapshotBlockKey: PersistentPromptCacheBlockKey =
            lookupResult.restoredPartialTailBlockKey
            ?? lookupResult.lastRestoredBlockKey
            ?? newestCompleteChainKey;
        guard var recurrentSnapshotTensors: [String: MLXArray] = try diskStore
            .loadRecurrentSnapshot(blockKey: recurrentSnapshotBlockKey)
        else {
            throw Qwen35MoePromptCacheError.restoredBlockVanished(
                blockIndex: recurrentSnapshotBlockKey.blockIndex());
        }
        try restoredStateStack.absorbPersistentPromptCacheRecurrentSnapshot(
            &recurrentSnapshotTensors);
        restoredStateStack.materializeRestoredPersistentPromptCacheState();
        self.counters.recordCacheHit(restoredTokenCount: restoredTokenCount);
        if lookupResult.restoredPartialTailBlockKey != nil {
            self.counters.recordPartialTailHit();
        }
        return Qwen35MoePromptCacheRestoreOutcome(
            lookupResult: lookupResult, restoredStateStack: restoredStateStack);
    }

    /// Rebuilds one block key from the prompt's own tokens so a restored
    /// chain continues under the same content-addressed identity capture
    /// would have produced.
    private static func reconstructedBlockKey(
        modelContract: PersistentPromptCacheModelContract,
        promptTokenIds: [UInt32],
        blockStart: Int,
        blockEnd: Int,
        parentBlockKey: PersistentPromptCacheBlockKey?
    ) throws -> PersistentPromptCacheBlockKey {
        let blockTokens: [UInt32] = Array(promptTokenIds[blockStart..<blockEnd]);
        let blockCausalInput: PersistentPromptCacheBlockCausalInput =
            PersistentPromptCacheBlockCausalInput.empty();
        if let parentBlockKey: PersistentPromptCacheBlockKey = parentBlockKey {
            return try parentBlockKey.forChildBlockWithCausalInput(
                blockTokens: blockTokens, blockCausalInput: blockCausalInput);
        }
        return try PersistentPromptCacheBlockKey.forRootBlockWithCausalInput(
            modelContract: modelContract, blockTokens: blockTokens,
            blockCausalInput: blockCausalInput);
    }
}

/// The engine-visible result of one restore attempt: the cold or hit lookup
/// plus, on a hit, the state stack the engine mirrors into its live caches.
public struct Qwen35MoePromptCacheRestoreOutcome {

    public let lookupResult: PersistentPromptCachePrefixLookupResult;

    /// The restored state owners; nil when the lookup missed and the request
    /// prefills cold.
    public let restoredStateStack: RequestDecoderStateStack?;

    init(
        lookupResult: PersistentPromptCachePrefixLookupResult,
        restoredStateStack: RequestDecoderStateStack? = nil
    ) {
        self.lookupResult = lookupResult;
        self.restoredStateStack = restoredStateStack;
    }

    /// The number of prompt tokens restored from the persistent cache.
    public var restoredTokenCount: Int {
        return self.lookupResult.restoredTokenCount;
    }
}
