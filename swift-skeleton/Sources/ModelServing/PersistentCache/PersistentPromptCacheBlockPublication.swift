import Foundation;

/// The synchronous publication entry point and idempotency rules, port of
/// the Rust `disk_store_write`. A successful return means the requested
/// block is durably available now: the caller may advance its parent cursor
/// for both outcomes; there is no queued, skipped, or eventually-written
/// state.
enum PersistentPromptCachePublicationOutcome: Equatable {

    /// This call durably committed missing state.
    case published;

    /// An exact, fully validated block was already durable.
    case alreadyPublished;
}

extension PersistentPromptCacheDiskStore {

    /// Publishes one block through the staged-payload seam. The seam stands
    /// in for the MLX direct writer until the capture/restore slice wires
    /// tensor payloads; every orchestration rule around it already runs.
    func publishBlock(
        staging: PersistentPromptCacheStateFileStaging,
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?
    ) throws -> PersistentPromptCachePublicationOutcome {
        self.writeOperationsLock.lock();
        defer { self.writeOperationsLock.unlock(); }
        try self.prepareActiveModelStorageDirectories();
        try Self.validateRequestedBlockAncestry(
            blockKey: blockKey, parentBlockKey: parentBlockKey);
        let blockHash: Data = blockKey.blockHash();
        // The index is an acceleration structure, never authority. If its
        // path disappeared, discard the stale entry and account from disk
        // again.
        var existingBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock?;
        self.stateLock.lock();
        existingBlock = self.trackedFiles.block(blockHash: blockHash);
        self.stateLock.unlock();
        if let trackedBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock = existingBlock,
            FileManager.default.fileExists(atPath: trackedBlock.blockDirectoryPath) == false {
            self.stateLock.lock();
            self.trackedFiles.removeBlock(blockHash: blockHash);
            self.stateLock.unlock();
            try self.refreshGlobalPromptCacheAccounting();
            existingBlock = nil;
        }
        if let existingBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock = existingBlock {
            try Self.validateExistingBlockForPublication(
                blockKey: blockKey, parentBlockKey: parentBlockKey,
                existingBlock: existingBlock, modelContract: self.modelContract);
            if existingBlock.blockIndex != blockKey.blockIndex()
                || existingBlock.parentBlockHash != parentBlockKey?.blockHash() {
                throw PersistentPromptCacheDiskStoreError.existingBlockTopologyMismatch(
                    blockHash: blockHash);
            }
            let sequenceStateIsComplete: Bool = self.modelContract.hasSequenceState == false
                || existingBlock.sequenceStateFile != nil;
            let boundaryStateIsComplete: Bool = self.modelContract.hasBoundaryState == false
                || existingBlock.boundaryStateFile != nil;
            // Idempotency is granted only after manifest and present state
            // files pass full validation. Hash equality alone cannot prove
            // topology.
            if sequenceStateIsComplete && boundaryStateIsComplete {
                return .alreadyPublished;
            }
            if sequenceStateIsComplete == false {
                throw PersistentPromptCacheDiskStoreError.existingBlockTopologyMismatch(
                    blockHash: blockHash);
            }
            // Sequence state may remain valid after retention compacts a
            // parent boundary. Reaching the same block as a leaf restores
            // that single missing file without replacing the sequence state.
            try self.publishMissingBoundaryStateTransaction(
                staging: staging, blockKey: blockKey);
            return .published;
        }
        // Children are never admitted speculatively. Requiring the parent in
        // the validated index guarantees every published non-root remains
        // restorable.
        if let parentBlockKey: PersistentPromptCacheBlockKey = parentBlockKey {
            self.stateLock.lock();
            let parentIsTracked: Bool = self.trackedFiles.block(
                blockHash: parentBlockKey.blockHash()) != nil;
            self.stateLock.unlock();
            if parentIsTracked == false {
                throw PersistentPromptCacheDiskStoreError.parentStateNotPublished(
                    blockIndex: blockKey.blockIndex());
            }
        }
        try self.publishNewBlockTransaction(
            staging: staging, blockKey: blockKey, parentBlockKey: parentBlockKey);
        try self.supersedeShorterPartialTailSiblings(
            blockKey: blockKey, parentBlockKey: parentBlockKey);
        return .published;
    }

    /// Removes strictly shorter tail siblings sharing the new tail's chain
    /// position. Prompts grow monotonically within a conversation, so once a
    /// longer tail is durable at a chain position, every shorter tail written
    /// earlier by that same conversation can never be probed again. Siblings
    /// with equal or larger sequence files stay: equal-length tails serve
    /// divergent conversations that share the parent prefix, and longer
    /// tails remain the growth path of their own conversations. Full blocks
    /// are never touched because their sequence files hold more tokens than
    /// any tail.
    private func supersedeShorterPartialTailSiblings(
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?
    ) throws {
        if self.modelContract.hasSequenceState == false
            || blockKey.tokenCount() >= self.modelContract.blockTokenCount {
            return;
        }
        // Sequence-state file bytes grow strictly with token count, so a
        // smaller file proves a strictly shorter tail.
        let newTailSequenceFileSizeBytes: UInt64 = try self.modelContract
            .sequenceStateFileBytesForBlockTokenCount(
                blockTokenCount: blockKey.tokenCount());
        let parentBlockHash: Data? = parentBlockKey?.blockHash();
        let blockHash: Data = blockKey.blockHash();
        var supersededSiblingDirectoryPaths: [String] = [];
        self.stateLock.lock();
        for trackedEntry: (blockHash: Data, trackedBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock)
            in self.trackedFiles.trackedBlocks() {
            let trackedSibling: PersistentPromptCacheDiskStoreIndex.TrackedBlock =
                trackedEntry.trackedBlock;
            let siblingIsShorterTail: Bool = trackedEntry.blockHash != blockHash
                && trackedSibling.blockIndex == blockKey.blockIndex()
                && trackedSibling.parentBlockHash == parentBlockHash
                && trackedSibling.sequenceStateFile.map(
                    { (sequenceStateFile: PersistentPromptCacheDiskStoreIndex.TrackedFile) -> Bool in
                        return sequenceStateFile.fileSizeBytes < newTailSequenceFileSizeBytes;
                    }) == true;
            if siblingIsShorterTail {
                supersededSiblingDirectoryPaths.append(trackedSibling.blockDirectoryPath);
            }
        }
        self.stateLock.unlock();
        if supersededSiblingDirectoryPaths.isEmpty {
            return;
        }
        for supersededDirectoryPath: String in supersededSiblingDirectoryPaths {
            try PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
                directoryPath: URL(fileURLWithPath: supersededDirectoryPath));
        }
        self.stateLock.lock();
        self.trackedFiles.removeBlocksByDirectoryPaths(
            directoryPaths: supersededSiblingDirectoryPaths);
        self.stateLock.unlock();
        try self.refreshGlobalPromptCacheAccounting();
    }

    private static func validateExistingBlockForPublication(
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?,
        existingBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock,
        modelContract: PersistentPromptCacheModelContract
    ) throws {
        let blockHash: Data = blockKey.blockHash();
        let blockManifest: PersistentPromptCacheBlockManifest = try
            PersistentPromptCacheBlockManifest.readFromBlockDirectory(
                blockDirectory: URL(fileURLWithPath: existingBlock.blockDirectoryPath),
                modelContract: modelContract);
        if (try? blockManifest.blockHash()) != blockHash
            || blockManifest.blockIndex != blockKey.blockIndex()
            || blockManifest.parentBlockHash() != parentBlockKey?.blockHash() {
            throw PersistentPromptCacheDiskStoreError.existingBlockTopologyMismatch(
                blockHash: blockHash);
        }
        try Self.validateExistingStateFile(
            trackedFile: existingBlock.sequenceStateFile, sequenceState: true,
            modelContract: modelContract);
        try Self.validateExistingStateFile(
            trackedFile: existingBlock.boundaryStateFile, sequenceState: false,
            modelContract: modelContract);
    }

    private static func validateExistingStateFile(
        trackedFile: PersistentPromptCacheDiskStoreIndex.TrackedFile?,
        sequenceState: Bool,
        modelContract: PersistentPromptCacheModelContract
    ) throws {
        guard let trackedFile: PersistentPromptCacheDiskStoreIndex.TrackedFile = trackedFile
        else {
            return;
        }
        let stateFileUrl: URL = URL(fileURLWithPath: trackedFile.filePath);
        do {
            if sequenceState {
                _ = try PersistentPromptCacheBlockHeader.readKvBlock(
                    blockFileUrl: stateFileUrl, modelContract: modelContract);
            } else {
                _ = try PersistentPromptCacheBlockHeader.readRecurrentSnapshot(
                    snapshotFileUrl: stateFileUrl, modelContract: modelContract);
            }
        } catch {
            throw PersistentPromptCacheDiskStoreError.validateBlock(
                blockFilePath: trackedFile.filePath,
                problem: String(describing: error));
        }
    }

    private static func validateRequestedBlockAncestry(
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?
    ) throws {
        // Root is exactly index zero with no parent. Every other block must
        // advance one ordinal from a supplied parent; gaps and alternate
        // roots fail closed.
        let ancestryIsValid: Bool;
        switch (blockKey.blockIndex(), parentBlockKey) {
        case (0, nil):
            ancestryIsValid = true;
        case (0, .some), (_, nil):
            ancestryIsValid = false;
        case (let blockIndex, .some(let suppliedParentBlockKey)):
            let (advancedParentIndex, parentIndexOverflowed) = suppliedParentBlockKey
                .blockIndex().addingReportingOverflow(1);
            ancestryIsValid = parentIndexOverflowed == false
                && advancedParentIndex == blockIndex;
        }
        if ancestryIsValid == false {
            throw PersistentPromptCacheDiskStoreError.invalidRequestedBlockAncestry(
                blockIndex: blockKey.blockIndex());
        }
    }
}
