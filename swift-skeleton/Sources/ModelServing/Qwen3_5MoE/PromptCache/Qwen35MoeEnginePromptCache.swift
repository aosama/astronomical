import Foundation;

import MLX;
import MLXLMCommon;

import IpcProtocol;


/// The MoE engine's persistent prompt-cache behavior: store attachment from
/// executed cache dtypes, prefix restore at request start, and boundary
/// capture during prompt processing, port of the Rust engine's cache wiring
/// across `start_generation`, `decoder_state_reuse`, and the capture owner.
extension Qwen35MoeEngine {

    /// Attaches the persistent prompt cache: one probe forward materializes
    /// the executed cache dtypes, the storage contract resolves from them,
    /// and the disk store opens. Fails closed while a request is active or
    /// an earlier attachment exists.
    public func attachPersistentPromptCache(
        _ promptCacheAttachment: Qwen35MoePromptCacheAttachment
    ) throws {
        guard self.promptCacheController == nil else {
            throw InferenceEngineError.invalidRequest(
                reason: "the persistent prompt cache is already attached");
        }
        guard let moeModel: any LanguageModel = self.moeModel,
            let repositoryConfiguration: Qwen3_5Config = self.moeRepositoryConfiguration
        else {
            throw InferenceEngineError.invalidRequest(
                reason: "no MoE model is loaded");
        }
        guard self.activeCache == nil else {
            throw InferenceEngineError.engineBusy;
        }
        let promptCacheController: Qwen35MoePromptCacheController =
            Qwen35MoePromptCacheController(
                fullAttentionKeyValueGrowthTokens: self.prefillChunkTokenCount);
        // The probe state is disposable: only the executed scalar types it
        // reveals feed the contract.
        let probeCaches: [KVCache] = try moeModel.newCache(parameters: nil);
        _ = moeModel(
            LMInput.Text(tokens: MLXArray([UInt32(1)], [1, 1])),
            cache: probeCaches,
            state: nil);
        let executedLayerCacheDtypes: [Qwen35DecoderLayerCacheDtypes] = try
            Qwen35MoeLiveCacheStateMirror.executedLayerCacheDtypes(probeCaches: probeCaches);
        try promptCacheController.openStore(
            attachment: promptCacheAttachment,
            repositoryConfiguration: repositoryConfiguration,
            executedLayerCacheDtypes: executedLayerCacheDtypes);
        self.promptCacheController = promptCacheController;
    }

    /// Restores the longest cached prompt prefix into the fresh request
    /// caches before the first prefill chunk and returns the reported
    /// cached-token count. A miss leaves the request cold.
    func restorePersistentPromptCachePrefixForActiveRequest() throws -> UInt32 {
        guard let promptCacheController: Qwen35MoePromptCacheController =
            self.promptCacheController,
            promptCacheController.isOpen,
            let activeCache: [KVCache] = self.activeCache
        else {
            return 0;
        }
        let restoreOutcome: Qwen35MoePromptCacheRestoreOutcome;
        do {
            restoreOutcome = try promptCacheController.restorePromptPrefix(
                promptTokenIds: self.promptTokenIds);
        } catch {
            // A restore that lookup proved and loading then broke is a
            // fatal engine fault, mirroring the Rust fatal translation.
            throw InferenceEngineError.fatalExecution(
                reason: "the persistent prompt-cache restore failed: \(error)");
        }
        guard let restoredStateStack: RequestDecoderStateStack =
            restoreOutcome.restoredStateStack
        else {
            self.captureChainParentBlockKey = nil;
            return 0;
        }
        try Qwen35MoeLiveCacheStateMirror.applyRestoredState(
            restoredStateStack, toLiveCaches: activeCache,
            restoredTokenCount: restoreOutcome.restoredTokenCount,
            slabGrowthTokens: self.prefillChunkTokenCount);
        self.prefillNextTokenOffset = restoreOutcome.restoredTokenCount;
        self.captureChainParentBlockKey = restoreOutcome.lookupResult.lastRestoredBlockKey;
        return UInt32(clamping: restoreOutcome.restoredTokenCount);
    }

    /// Captures every boundary the just-finished prefill chunk completed and
    /// the prompt's partial tail when the chunk was the terminal one. The
    /// chunk cursor may advance only after publication proved durable.
    func capturePersistentPromptCacheAfterPrefillChunk(
        activeCache: [KVCache],
        chunkStart: Int,
        chunkEnd: Int
    ) throws {
        guard let promptCacheController: Qwen35MoePromptCacheController =
            self.promptCacheController,
            promptCacheController.isOpen
        else {
            return;
        }
        do {
            try promptCacheController.captureBlocksAtBoundaries(
                liveCaches: activeCache,
                promptTokenIds: self.promptTokenIds,
                prefillChunkStart: chunkStart,
                prefillChunkEnd: chunkEnd,
                chainParentBlockKey: &self.captureChainParentBlockKey);
            promptCacheController.capturePartialTailBlock(
                liveCaches: activeCache,
                promptTokenIds: self.promptTokenIds,
                prefillEnd: chunkEnd,
                chainParentBlockKey: self.captureChainParentBlockKey);
        } catch {
            // Required persistence failures stop the request with the
            // user-visible translation the Rust engine reports.
            throw Self.requiredPersistenceFailure(error);
        }
    }

    /// The prompt-work reuse this request's prefill boundaries report.
    func promptWorkReuseForActiveRequest() -> WorkerPromptWorkReuse {
        return WorkerPromptWorkReuse(
            targetEligibleTokenCount: UInt64(clamping: self.activeRestoredPromptCacheTokenCount),
            targetRestoredTokenCount: UInt64(clamping: self.activeRestoredPromptCacheTokenCount));
    }

    /// The attached disk store for journey evidence; nil while detached.
    var attachedPromptCacheDiskStore: PersistentPromptCacheDiskStore? {
        return self.promptCacheController?.diskStore;
    }

    /// The attached cache's cumulative counters for journey evidence.
    var attachedPromptCacheCounters: PersistentPromptCacheCounters? {
        return self.promptCacheController?.counters;
    }

    /// The attached contract's block token count; nil while detached.
    func attachedPromptCacheBlockTokenCount() -> Int? {
        guard let promptCacheController: Qwen35MoePromptCacheController =
            self.promptCacheController,
            promptCacheController.isOpen
        else {
            return nil;
        }
        return promptCacheController.modelContract.blockTokenCount;
    }

    /// Releases the attached persistent prompt cache; the next request runs
    /// cold until a new attachment. Journeys and shutdown paths drop the
    /// store this way before removing its directory.
    public func detachPersistentPromptCacheForJourneys() {
        self.promptCacheController = nil;
    }

    /// The cumulative prompt-cache evidence for the worker status surface,
    /// or nil while no cache is attached. Mirrors the Rust engine's
    /// `collect_persistent_prompt_cache_stats`.
    public func collectPersistentPromptCacheStats() -> WorkerPersistentPromptCacheStats? {
        guard let promptCacheController: Qwen35MoePromptCacheController =
            self.promptCacheController,
            promptCacheController.isOpen,
            let diskStore: PersistentPromptCacheDiskStore =
                promptCacheController.diskStoreValue
        else {
            return nil;
        }
        let counters: PersistentPromptCacheCounters = promptCacheController.counters;
        return WorkerPersistentPromptCacheStats(
            persistentPromptCacheHits: counters.persistentPromptCacheHits,
            persistentPromptCacheMisses: counters.persistentPromptCacheMisses,
            persistentPromptCacheTokensSaved: counters.persistentPromptCacheTokensSaved,
            persistentPromptCachePartialTailHits: counters.persistentPromptCachePartialTailHits,
            persistentPromptCacheBlockTokenCount:
                UInt64(promptCacheController.modelContract.blockTokenCount),
            persistentPromptCacheSequenceStateBlockCount:
                UInt64(diskStore.sequenceStateBlockCount()),
            persistentPromptCacheBoundaryStateSnapshotCount:
                UInt64(diskStore.boundaryStateSnapshotCount()),
            persistentPromptCacheVisualEmbeddingCount:
                UInt64(diskStore.visualEmbeddingCount()),
            persistentPromptCacheTotalSizeBytes: diskStore.totalSizeBytes(),
            persistentPromptCacheVisualEmbeddingTotalSizeBytes:
                diskStore.visualEmbeddingTotalSizeBytes(),
            persistentPromptCacheMaximumSizeBytes: diskStore.globalPromptCacheMaximumSizeBytes,
            persistentPromptCacheVisualEmbeddingHits:
                counters.persistentPromptCacheVisualEmbeddingHits,
            persistentPromptCacheVisualEmbeddingMisses:
                counters.persistentPromptCacheVisualEmbeddingMisses,
            persistentPromptCacheVisualEmbeddingRowsLoaded:
                counters.persistentPromptCacheVisualEmbeddingRowsLoaded);
    }

    /// The attached contract's per-token sequence-state charge; nil while
    /// no cache is attached.
    public func contextWorkspaceBytesPerToken() -> Int? {
        guard let promptCacheController: Qwen35MoePromptCacheController =
            self.promptCacheController,
            promptCacheController.isOpen
        else {
            return nil;
        }
        return promptCacheController.modelContract.sequenceStatePayloadBytesPerToken;
    }

    /// Clears the persisted prompt-cache state for one model, or for the
    /// active model when the identifier is nil. Mirrors the Rust engine's
    /// store-owned clear; the outcome carries the removed counts.
    public func clearPersistentPromptCache(
        modelId: String?
    ) throws -> PersistentPromptCacheClearOutcome {
        guard let promptCacheController: Qwen35MoePromptCacheController =
            self.promptCacheController,
            promptCacheController.isOpen,
            let diskStore: PersistentPromptCacheDiskStore =
                promptCacheController.diskStoreValue
        else {
            return PersistentPromptCacheClearOutcome(
                modelId: modelId, blocksRemoved: 0, bytesFreed: 0);
        }
        return try diskStore.clearPromptCache(modelId: modelId);
    }

    /// Translates one cache failure into the engine error surface, port of
    /// the Rust `required_prompt_state_persistence_failure`.
    private static func requiredPersistenceFailure(
        _ promptCacheError: Error
    ) -> InferenceEngineError {
        return InferenceEngineError.invalidRequest(
            reason: "persistent prompt cache failed during prompt processing; "
                + "the request was stopped: \(promptCacheError)");
    }
}
