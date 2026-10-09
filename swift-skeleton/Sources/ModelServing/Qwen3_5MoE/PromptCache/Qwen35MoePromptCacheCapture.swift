import Foundation;

import MLX;
import MLXLMCommon;

import RuntimeIntegration;


/// The request-side capture half of the engine's persistent prompt-cache
/// owner, port of the Rust `persistent_prompt_cache_capture` and
/// `persistent_prompt_cache_tail_capture`. The request cursor advances only
/// after the matching block is durably published or proven already present;
/// full-block capture failures stop the request, partial-tail capture stays
/// best-effort.
extension Qwen35MoePromptCacheController {

    /// Captures and publishes one block per validated boundary the prefill
    /// chunk completed, chaining keys from the request's restored or already
    /// published parent. `chainParentBlockKey` advances to each durable
    /// block so the next boundary extends the same identity chain.
    func captureBlocksAtBoundaries(
        liveCaches: [KVCache],
        promptTokenIds: [UInt32],
        prefillChunkStart: Int,
        prefillChunkEnd: Int,
        chainParentBlockKey: inout PersistentPromptCacheBlockKey?
    ) throws {
        guard let diskStore: PersistentPromptCacheDiskStore = self.diskStoreValue else {
            throw Qwen35MoePromptCacheError.promptCacheNotAttached;
        }
        let modelContract: PersistentPromptCacheModelContract = diskStore.modelContract;
        let completedBoundaryTokens: [Int] = PersistentPromptCachePrefillBoundary
            .completedPrefillChunkTokens(
                prefillChunkStart: prefillChunkStart, prefillChunkEnd: prefillChunkEnd,
                persistentPromptCacheBlockTokenCount: modelContract.blockTokenCount);
        for completedBoundaryTokensEntry: Int in completedBoundaryTokens {
            let absoluteBoundary: Int = prefillChunkStart + completedBoundaryTokensEntry;
            if absoluteBoundary % modelContract.blockTokenCount != 0
                || absoluteBoundary > prefillChunkEnd {
                throw Qwen35MoePromptCacheError.requiredCaptureFailure(
                    stage: "required persistent prompt-state capture",
                    description: "prompt-cache boundary position is invalid");
            }
            let blockStart: Int = absoluteBoundary - modelContract.blockTokenCount;
            let blockEnd: Int = absoluteBoundary;
            let blockTokens: [UInt32] = Array(promptTokenIds[blockStart..<blockEnd]);
            let blockKey: PersistentPromptCacheBlockKey = try Self.chainedBlockKey(
                modelContract: modelContract, blockTokens: blockTokens,
                parentBlockKey: chainParentBlockKey);
            let liveStateStack: RequestDecoderStateStack = try
                Qwen35MoeLiveCacheStateMirror.mirroredStateStack(liveCaches: liveCaches);
            let kvBlockTensors: [String: MLXArray] = try liveStateStack
                .extractPersistentPromptCacheKvBlockTensors(
                    blockStartTokens: blockStart, blockEndTokens: blockEnd,
                    contractBlockTokenCount: modelContract.blockTokenCount);
            let recurrentSnapshotTensors: [String: MLXArray] = try liveStateStack
                .extractPersistentPromptCacheRecurrentSnapshotTensors();
            try self.publishBlockWithReclamationRetry(
                diskStore: diskStore, blockKey: blockKey,
                parentBlockKey: chainParentBlockKey,
                kvBlockTensors: kvBlockTensors,
                recurrentSnapshotTensors: recurrentSnapshotTensors);
            // Both publication outcomes prove durable availability, so the
            // next block may safely chain from this key.
            chainParentBlockKey = blockKey;
        }
    }

    /// Best-effort capture of the prompt's final partial block after the
    /// terminal prefill chunk. Every failure is skipped: the request never
    /// depends on tail reuse.
    func capturePartialTailBlock(
        liveCaches: [KVCache],
        promptTokenIds: [UInt32],
        prefillEnd: Int,
        chainParentBlockKey: PersistentPromptCacheBlockKey?
    ) {
        guard let diskStore: PersistentPromptCacheDiskStore = self.diskStoreValue else {
            return;
        }
        let modelContract: PersistentPromptCacheModelContract = diskStore.modelContract;
        let promptTokenCount: Int = promptTokenIds.count;
        // Only the terminal chunk holds every prompt token in decoder state.
        if prefillEnd != promptTokenCount {
            return;
        }
        let tailTokenCount: Int = promptTokenCount % modelContract.blockTokenCount;
        if tailTokenCount == 0 {
            return;
        }
        let tailBlockStart: Int = promptTokenCount - tailTokenCount;
        let tailTokens: [UInt32] = Array(promptTokenIds[tailBlockStart..<promptTokenCount]);
        let tailBlockKey: PersistentPromptCacheBlockKey;
        do {
            tailBlockKey = try Self.chainedBlockKey(
                modelContract: modelContract, blockTokens: tailTokens,
                parentBlockKey: chainParentBlockKey);
        } catch {
            return;
        }
        let liveStateStack: RequestDecoderStateStack;
        let kvBlockTensors: [String: MLXArray];
        let recurrentSnapshotTensors: [String: MLXArray];
        do {
            liveStateStack = try Qwen35MoeLiveCacheStateMirror.mirroredStateStack(
                liveCaches: liveCaches);
            kvBlockTensors = try liveStateStack.extractPersistentPromptCacheKvBlockTensors(
                blockStartTokens: tailBlockStart, blockEndTokens: promptTokenCount,
                contractBlockTokenCount: modelContract.blockTokenCount);
            recurrentSnapshotTensors = try liveStateStack
                .extractPersistentPromptCacheRecurrentSnapshotTensors();
        } catch {
            return;
        }
        do {
            try self.publishBlockWithReclamationRetry(
                diskStore: diskStore, blockKey: tailBlockKey,
                parentBlockKey: chainParentBlockKey,
                kvBlockTensors: kvBlockTensors,
                recurrentSnapshotTensors: recurrentSnapshotTensors);
        } catch {
            // Tail reuse is an optimization; a failed publication never
            // blocks the request that produced it.
        }
    }

    /// Publishes one block, retrying exactly once after an allocator-cache
    /// clear when the typed MLX active-memory limit blocked serialization.
    /// Reusing the caller's extracted tensors is essential: this is resource
    /// reclamation, not a second logical capture.
    private func publishBlockWithReclamationRetry(
        diskStore: PersistentPromptCacheDiskStore,
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?,
        kvBlockTensors: [String: MLXArray],
        recurrentSnapshotTensors: [String: MLXArray]
    ) throws {
        do {
            _ = try self.publishStagedBlock(
                diskStore: diskStore, blockKey: blockKey, parentBlockKey: parentBlockKey,
                kvBlockTensors: kvBlockTensors,
                recurrentSnapshotTensors: recurrentSnapshotTensors);
            return;
        } catch let publicationError as PersistentPromptCacheDiskStoreError {
            guard publicationError.activeMemoryDeficitBytes() != nil else {
                throw publicationError;
            }
            Memory.clearCache();
            _ = try self.publishStagedBlock(
                diskStore: diskStore, blockKey: blockKey, parentBlockKey: parentBlockKey,
                kvBlockTensors: kvBlockTensors,
                recurrentSnapshotTensors: recurrentSnapshotTensors);
        }
    }

    private func publishStagedBlock(
        diskStore: PersistentPromptCacheDiskStore,
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?,
        kvBlockTensors: [String: MLXArray],
        recurrentSnapshotTensors: [String: MLXArray]
    ) throws -> PersistentPromptCachePublicationOutcome {
        let blockStaging: PersistentPromptCacheStateFileMlxStaging =
            PersistentPromptCacheStateFileMlxStaging(
                sequenceStateTensors: kvBlockTensors,
                boundaryStateTensors: recurrentSnapshotTensors);
        return try diskStore.publishBlock(
            staging: blockStaging, blockKey: blockKey, parentBlockKey: parentBlockKey);
    }

    private static func chainedBlockKey(
        modelContract: PersistentPromptCacheModelContract,
        blockTokens: [UInt32],
        parentBlockKey: PersistentPromptCacheBlockKey?
    ) throws -> PersistentPromptCacheBlockKey {
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
