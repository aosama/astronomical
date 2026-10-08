import Foundation;

import MLX;

import RuntimeIntegration;

/// Loads published state files back as live MLX tensors, port of the Rust
/// `load_kv_block` and `load_recurrent_snapshot`. Header validation runs
/// before any tensor payload is mapped, so wrong-model geometry and
/// malformed offsets are rejected without allocating decoder arrays; a
/// corrupt file is deleted and untracked before the typed failure is
/// raised, so the next publication writes a fresh copy.
extension PersistentPromptCacheDiskStore {

    /// Loads one published full-attention KV block's tensors by key, or nil
    /// when the key is not tracked.
    public func loadKvBlock(
        blockKey: PersistentPromptCacheBlockKey
    ) throws -> [String: MLXArray]? {
        return try self.loadFileKind(sequenceState: true, blockKey: blockKey);
    }

    /// Loads one published recurrent snapshot's tensors by key, or nil when
    /// the key is not tracked.
    public func loadRecurrentSnapshot(
        blockKey: PersistentPromptCacheBlockKey
    ) throws -> [String: MLXArray]? {
        return try self.loadFileKind(sequenceState: false, blockKey: blockKey);
    }

    private func loadFileKind(
        sequenceState: Bool,
        blockKey: PersistentPromptCacheBlockKey
    ) throws -> [String: MLXArray]? {
        // Hold the short index lock only to clone the path; disk and MLX
        // work run unlocked so unrelated cache lookups are not serialized.
        let trackedFilePath: String?;
        self.stateLock.lock();
        trackedFilePath = self.trackedFiles.file(
            sequenceState: sequenceState, fileHash: blockKey.blockHash())?.filePath;
        self.stateLock.unlock();
        guard let blockFilePath: String = trackedFilePath
        else {
            return nil;
        }
        let blockFileUrl: URL = URL(fileURLWithPath: blockFilePath);
        do {
            let storedHeader: PersistentPromptCacheBlockHeader = sequenceState
                ? try PersistentPromptCacheBlockHeader.readKvBlock(
                    blockFileUrl: blockFileUrl, modelContract: self.modelContract)
                : try PersistentPromptCacheBlockHeader.readRecurrentSnapshot(
                    snapshotFileUrl: blockFileUrl, modelContract: self.modelContract);
            // Stamping writes each file's real token count, so a header count
            // that disagrees with the addressing key proves the index points
            // at the wrong file: a partial tail must never be served for a
            // full-block key or the reverse.
            if storedHeader.blockTokenCount != blockKey.tokenCount() {
                throw PersistentPromptCacheBlockError.blockTokenCountMismatch(
                    actualBlockTokenCount: storedHeader.blockTokenCount,
                    expectedBlockTokenCount: blockKey.tokenCount());
            }
        } catch {
            return try self.discardCorruptBlockFileAndRaise(
                sequenceState: sequenceState, blockKey: blockKey,
                blockFilePath: blockFilePath, validationProblem: String(describing: error));
        }
        let loadedArrays: [String: MLXArray];
        do {
            loadedArrays = try MLX.loadArrays(url: blockFileUrl);
        } catch let loadError as MlxRuntimeError {
            throw PersistentPromptCacheDiskStoreError.loadSafetensors(source: loadError);
        } catch {
            throw PersistentPromptCacheDiskStoreError.loadSafetensors(
                source: .runtimeOperation(
                    operation: "load persistent prompt-cache state file",
                    description: String(describing: error)));
        }
        let expectedTensorLayouts: [DecoderCachePersistedTensorLayout] = sequenceState
            ? self.modelContract.decoderCacheLayout.sequenceTensorLayouts()
            : self.modelContract.decoderCacheLayout.boundaryTensorLayouts();
        var loadedTensors: [String: MLXArray] = [:];
        loadedTensors.reserveCapacity(expectedTensorLayouts.count);
        for persistedTensorLayout: DecoderCachePersistedTensorLayout in expectedTensorLayouts {
            let tensorName: String = persistedTensorLayout.persistentTensorName;
            guard let loadedTensor: MLXArray = loadedArrays[tensorName]
            else {
                throw PersistentPromptCacheDiskStoreError.loadSafetensors(
                    source: .tensorLookupFailed(tensorName: tensorName));
            }
            loadedTensors[tensorName] = loadedTensor;
        }
        return loadedTensors;
    }

    /// Deletes a corrupt published file first (absent already counts as
    /// deleted), then untracks it only after deletion succeeded, then raises
    /// the typed validation failure.
    private func discardCorruptBlockFileAndRaise(
        sequenceState: Bool,
        blockKey: PersistentPromptCacheBlockKey,
        blockFilePath: String,
        validationProblem: String
    ) throws -> [String: MLXArray] {
        try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
            filePath: URL(fileURLWithPath: blockFilePath));
        self.untrackFileAndSubtractGlobalAccounting(
            sequenceState: sequenceState, fileHash: blockKey.blockHash());
        throw PersistentPromptCacheDiskStoreError.validateBlock(
            blockFilePath: blockFilePath, problem: validationProblem);
    }
}
