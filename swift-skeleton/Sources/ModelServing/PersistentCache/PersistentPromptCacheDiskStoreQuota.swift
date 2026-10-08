import Foundation;

import ModelServing;

/// The quota-enforcement and retention-reconciliation half of
/// `PersistentPromptCacheDiskStore`: pressure relief ordered unconditional
/// cleanup first, oldest written next, protecting the active publication
/// ancestry, with startup-only compaction of crash-interrupted parent
/// boundaries.
extension PersistentPromptCacheDiskStore {

    func enforceGlobalPromptCacheQuota() throws {
        return try self.enforceGlobalPromptCacheQuotaForCommit(
            additionalCommittedSizeBytes: 0,
            postCommitReclaimableSizeBytes: 0,
            protectedBlockDirectoryPaths: [],
            excludedDirectory: nil);
    }

    func enforceStartupGlobalPromptCacheQuota(
        protectedBlockDirectoryPaths: [String]
    ) throws {
        return try self.enforceGlobalPromptCacheQuotaInternal(
            additionalCommittedSizeBytes: 0,
            postCommitReclaimableSizeBytes: 0,
            protectedBlockDirectoryPaths: protectedBlockDirectoryPaths,
            excludedDirectory: nil,
            shouldRecordStartupCleanup: true);
    }

    func enforceGlobalPromptCacheQuotaForCommit(
        additionalCommittedSizeBytes: UInt64,
        postCommitReclaimableSizeBytes: UInt64,
        protectedBlockDirectoryPaths: [String],
        excludedDirectory: URL?
    ) throws {
        return try self.enforceGlobalPromptCacheQuotaInternal(
            additionalCommittedSizeBytes: additionalCommittedSizeBytes,
            postCommitReclaimableSizeBytes: postCommitReclaimableSizeBytes,
            protectedBlockDirectoryPaths: protectedBlockDirectoryPaths,
            excludedDirectory: excludedDirectory,
            shouldRecordStartupCleanup: false);
    }

    private func enforceGlobalPromptCacheQuotaInternal(
        additionalCommittedSizeBytes: UInt64,
        postCommitReclaimableSizeBytes: UInt64,
        protectedBlockDirectoryPaths: [String],
        excludedDirectory: URL?,
        shouldRecordStartupCleanup: Bool
    ) throws {
        // Exclude the current staging directory because
        // `additionalCommittedSizeBytes` already accounts for it. Counting
        // both would charge the transaction twice.
        let globalQuotaScan: PersistentPromptCacheQuotaScan = try
            PersistentPromptCacheQuotaScanEngine.scanGlobalPromptCacheQuota(
                globalPromptCacheRootDirectory: self.globalPromptCacheRootDirectory,
                excludedDirectory: excludedDirectory);
        var globalPromptCacheTotalSizeBytes: UInt64 = globalQuotaScan.totalSizeBytes;
        var globalVisualEmbeddingTotalSizeBytes: UInt64 = globalQuotaScan
            .visualEmbeddingTotalSizeBytes;
        self.stateLock.lock();
        self.globalPromptCacheTotalSizeBytes = globalPromptCacheTotalSizeBytes;
        self.globalVisualEmbeddingTotalSizeBytes = globalVisualEmbeddingTotalSizeBytes;
        self.stateLock.unlock();
        var removedEvictionPaths: Set<String> = [];
        var startupCleanupEvidence: PersistentPromptCacheStartupCleanupEvidence =
            PersistentPromptCacheStartupCleanupEvidence();
        for evictionCandidate: PersistentPromptCacheEvictionCandidate in globalQuotaScan
            .evictionCandidatesOldestWrittenFirst {
            if Self.evictionCandidateWasAlreadyRemoved(
                evictionCandidate, removedEvictionPaths: removedEvictionPaths) {
                continue;
            }
            // Interrupted transactions and obsolete formats are removed even
            // when the committed projection already fits.
            if evictionCandidate.isUnconditionallyRemovable == false
                && evictionCandidate.containsProtectedBlockDirectory(
                    protectedBlockDirectoryPaths: protectedBlockDirectoryPaths) {
                continue;
            }
            // Stop deleting committed value once the post-commit projection
            // fits. `postCommitReclaimableSizeBytes` is subtracted only in
            // arithmetic; its actual file remains until commit becomes
            // durable.
            if evictionCandidate.isUnconditionallyRemovable == false {
                let (projectedBytes, projectionOverflow) = globalPromptCacheTotalSizeBytes
                    .addingReportingOverflow(additionalCommittedSizeBytes);
                let postCommitBytes: UInt64 = projectionOverflow
                    ? UInt64.max
                    : projectedBytes.subtractingReportingOverflow(
                        postCommitReclaimableSizeBytes).partialValue;
                if postCommitBytes <= self.globalPromptCacheMaximumSizeBytes {
                    continue;
                }
            }
            try Self.removeEvictionCandidate(evictionCandidate);
            if shouldRecordStartupCleanup {
                Self.recordRemovedStartupCandidate(
                    startupCleanupEvidence: &startupCleanupEvidence,
                    cleanupClassification: evictionCandidate.unconditionalCleanupClassification
                        ?? .quotaEviction,
                    removedCandidate: evictionCandidate);
            }
            self.stateLock.lock();
            self.trackedFiles.removeFilesByPath(filePaths: evictionCandidate.trackedFilePaths);
            self.trackedFiles.removeBlocksByDirectoryPaths(
                directoryPaths: evictionCandidate.blockDirectoryPaths);
            self.stateLock.unlock();
            globalPromptCacheTotalSizeBytes = globalPromptCacheTotalSizeBytes
                .subtractingReportingOverflow(evictionCandidate.fileSizeBytes).partialValue;
            globalVisualEmbeddingTotalSizeBytes = globalVisualEmbeddingTotalSizeBytes
                .subtractingReportingOverflow(evictionCandidate.visualEmbeddingSizeBytes)
                .partialValue;
            removedEvictionPaths.insert(evictionCandidate.tieBreakerPath);
            for removedBlockDirectoryPath: String in evictionCandidate.blockDirectoryPaths {
                removedEvictionPaths.insert(removedBlockDirectoryPath);
            }
            self.stateLock.lock();
            self.globalPromptCacheTotalSizeBytes = globalPromptCacheTotalSizeBytes;
            self.globalVisualEmbeddingTotalSizeBytes = globalVisualEmbeddingTotalSizeBytes;
            self.stateLock.unlock();
        }
        let (addedBytes, addOverflow) = globalPromptCacheTotalSizeBytes
            .addingReportingOverflow(additionalCommittedSizeBytes);
        let finalCommittedSizeBytes: UInt64 = addOverflow
            ? UInt64.max
            : addedBytes.subtractingReportingOverflow(postCommitReclaimableSizeBytes)
                .partialValue;
        if finalCommittedSizeBytes > self.globalPromptCacheMaximumSizeBytes {
            throw PersistentPromptCacheDiskStoreError.globalPromptCacheQuotaNotSatisfied(
                maximumSizeBytes: self.globalPromptCacheMaximumSizeBytes,
                remainingSizeBytes: finalCommittedSizeBytes);
        }
        if shouldRecordStartupCleanup {
            self.recordStartupCleanupEvidence(startupCleanupEvidence);
        }
    }

    /// Startup-only repair of retention state after an interrupted
    /// publication. Every block validated into the active model index is
    /// protected during the first quota pass so unrelated models and stale
    /// artifacts absorb pressure first; then the only safe in-chain bytes —
    /// redundant non-checkpoint parent boundaries — compact, and quota is
    /// retried while protecting the chain.

    func reconcileStartupRetentionAndGlobalQuota() throws {
        let protectedActiveBlockDirectoryPaths: [String];
        self.stateLock.lock();
        protectedActiveBlockDirectoryPaths = self.trackedFiles.trackedBlocks().map(
            { (trackedEntry: (blockHash: Data, trackedBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock)) in
                return trackedEntry.trackedBlock.blockDirectoryPath;
            });
        self.stateLock.unlock();
        do {
            return try self.enforceStartupGlobalPromptCacheQuota(
                protectedBlockDirectoryPaths: protectedActiveBlockDirectoryPaths);
        } catch PersistentPromptCacheDiskStoreError.globalPromptCacheQuotaNotSatisfied {
            // Unprotected eviction was insufficient; compact the chain below.
        }
        try self.compactActiveModelParentBoundariesForStartupQuota();
        return try self.enforceStartupGlobalPromptCacheQuota(
            protectedBlockDirectoryPaths: protectedActiveBlockDirectoryPaths);
    }

    /// A boundary is redundant only when a durable child exists: leaf
    /// boundaries remain the required restart point for that prompt prefix.
    /// Common-prefix checkpoints deliberately retain intermediate restart
    /// points even when they have children.
    private func compactActiveModelParentBoundariesForStartupQuota() throws {
        if self.modelContract.hasSequenceState == false
            || self.modelContract.hasBoundaryState == false {
            return;
        }
        var committedSizeBytes: UInt64 = self.totalSizeBytes();
        if committedSizeBytes <= self.globalPromptCacheMaximumSizeBytes {
            return;
        }
        var startupCleanupEvidence: PersistentPromptCacheStartupCleanupEvidence =
            PersistentPromptCacheStartupCleanupEvidence();
        var trackedBlocks: [(blockHash: Data, trackedBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock)];
        self.stateLock.lock();
        trackedBlocks = self.trackedFiles.trackedBlocks();
        self.stateLock.unlock();
        let blockHashesWithCommittedChildren: Set<Data> = Set(
            trackedBlocks.compactMap({ (trackedEntry) -> Data? in
                return trackedEntry.trackedBlock.parentBlockHash;
            }));
        struct StartupBoundaryReclaimCandidate {
            var blockHash: Data;
            var blockDirectoryPath: String;
            var boundaryFilePath: String;
            var boundaryFileSizeBytes: UInt64;
            var modifiedAt: Date;
        }
        var reclaimableBoundaries: [StartupBoundaryReclaimCandidate] = [];
        for trackedEntry: (blockHash: Data, trackedBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock)
            in trackedBlocks {
            guard let boundaryStateFile: PersistentPromptCacheDiskStoreIndex.TrackedFile =
                trackedEntry.trackedBlock.boundaryStateFile else {
                continue;
            };
            if blockHashesWithCommittedChildren.contains(trackedEntry.blockHash) == false
                || PersistentPromptCacheRetentionPolicy.boundaryIsCommonPrefixCheckpoint(
                    blockIndex: trackedEntry.trackedBlock.blockIndex,
                    commonPrefixCheckpointStrideBlocks: self.modelContract
                        .commonPrefixCheckpointStrideBlocks) == false {
                continue;
            }
            let modifiedAt: Date;
            do {
                let fileAttributes: [FileAttributeKey: Any] = try FileManager.default
                    .attributesOfItem(atPath: boundaryStateFile.filePath);
                modifiedAt = (fileAttributes[.modificationDate] as? Date)
                    ?? Date(timeIntervalSince1970: 0);
            } catch {
                throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                    blockFilePath: boundaryStateFile.filePath,
                    problem: String(describing: error));
            }
            reclaimableBoundaries.append(StartupBoundaryReclaimCandidate(
                blockHash: trackedEntry.blockHash,
                blockDirectoryPath: trackedEntry.trackedBlock.blockDirectoryPath,
                boundaryFilePath: boundaryStateFile.filePath,
                boundaryFileSizeBytes: boundaryStateFile.fileSizeBytes,
                modifiedAt: modifiedAt));
        }
        reclaimableBoundaries.sort(by: { (leftCandidate, rightCandidate) -> Bool in
            if leftCandidate.modifiedAt != rightCandidate.modifiedAt {
                return leftCandidate.modifiedAt < rightCandidate.modifiedAt;
            }
            return leftCandidate.boundaryFilePath.utf8
                .lexicographicallyPrecedes(rightCandidate.boundaryFilePath.utf8);
        });
        for reclaimableBoundary: StartupBoundaryReclaimCandidate in reclaimableBoundaries {
            if committedSizeBytes <= self.globalPromptCacheMaximumSizeBytes {
                break;
            }
            try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: URL(fileURLWithPath: reclaimableBoundary.boundaryFilePath));
            try PersistentPromptCacheStoreFile.synchronizeDirectory(directoryPath: URL(
                fileURLWithPath: reclaimableBoundary.blockDirectoryPath));
            self.stateLock.lock();
            self.trackedFiles.removeBlock(blockHash: reclaimableBoundary.blockHash);
            self.stateLock.unlock();
            committedSizeBytes = committedSizeBytes
                .subtractingReportingOverflow(reclaimableBoundary.boundaryFileSizeBytes)
                .partialValue;
            startupCleanupEvidence.recordArtifact(
                reason: .interruptedTransactionRecovery,
                removedByteCount: reclaimableBoundary.boundaryFileSizeBytes);
        }
        self.recordStartupCleanupEvidence(startupCleanupEvidence);
        try self.refreshGlobalPromptCacheAccounting();
    }

    static func removeEvictionCandidate(
        _ evictionCandidate: PersistentPromptCacheEvictionCandidate
    ) throws {
        switch evictionCandidate {
        case .standaloneFile(let filePath, _, _, _, _):
            try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: URL(fileURLWithPath: filePath));
        case .staleDirectory(let directoryPath, _, _, _):
            try PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
                directoryPath: URL(fileURLWithPath: directoryPath));
        case .blockSubtree(_, _, _, let blockDirectoryPaths, _):
            for blockDirectoryPath: String in blockDirectoryPaths {
                try PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
                    directoryPath: URL(fileURLWithPath: blockDirectoryPath));
            }
        }
    }

    static func recordRemovedStartupCandidate(
        startupCleanupEvidence: inout PersistentPromptCacheStartupCleanupEvidence,
        cleanupClassification: PersistentPromptCacheEvictionCandidate.CleanupClassification,
        removedCandidate: PersistentPromptCacheEvictionCandidate
    ) {
        let reason: PersistentPromptCacheStartupCleanupEvidence.Reason;
        switch cleanupClassification {
        case .interruptedTransactionRecovery:
            reason = .interruptedTransactionRecovery;
        case .obsoleteFormat:
            reason = .obsoleteFormat;
        case .quotaEviction:
            reason = .quotaEviction;
        }
        for _ in 0..<removedCandidate.removedArtifactCount {
            startupCleanupEvidence.recordArtifact(reason: reason, removedByteCount: 0);
        }
        startupCleanupEvidence.recordBlock(
            reason: reason,
            removedByteCount: removedCandidate.fileSizeBytes);
    }

    /// Subtree candidates overlap by construction: a removed ancestor can
    /// cover a later candidate even when their root paths differ, hence the
    /// member scan.
    static func evictionCandidateWasAlreadyRemoved(
        _ evictionCandidate: PersistentPromptCacheEvictionCandidate,
        removedEvictionPaths: Set<String>
    ) -> Bool {
        return removedEvictionPaths.contains(evictionCandidate.tieBreakerPath)
            || evictionCandidate.blockDirectoryPaths.contains(
                where: { (blockDirectoryPath: String) -> Bool in
                    return removedEvictionPaths.contains(blockDirectoryPath);
                });
    }

}
