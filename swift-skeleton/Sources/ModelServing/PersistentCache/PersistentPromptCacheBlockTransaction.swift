import Foundation;

import ModelServing;

/// The staging seam the publication transaction writes state files
/// through. The MLX direct writer provides the production implementation
/// with the capture/restore slice; hermetic journeys provide synthetic
/// writers producing real header-shaped files.
public protocol PersistentPromptCacheStateFileStaging: AnyObject {

    /// Stages one contract-named state file into `stagingBlockDirectory`
    /// and returns its exact written byte size.
    func stageStateFile(
        stateFileName: String,
        stagingBlockDirectory: URL,
        blockTokenCount: Int,
        modelContract: PersistentPromptCacheModelContract
    ) throws -> UInt64;
}

/// One staged block's measured publication inputs.
struct PersistentPromptCacheStagedBlockFiles {
    var sequenceStateFileSizeBytes: UInt64?;
    var boundaryStateFileSizeBytes: UInt64?;
    var totalSizeBytes: UInt64;
}

struct PersistentPromptCacheParentBoundaryReclaim {
    var blockHash: Data;
    var blockDirectoryPath: String;
    var filePath: String;
    var fileSizeBytes: UInt64;
}

/// Crash-safe publication transactions for block directories, port of the
/// Rust `disk_store_block_transaction`. A complete block follows this
/// order: stage every state file and the manifest into a unique staging
/// directory, synchronize it, reserve global quota while protecting the
/// chain being extended, atomically rename the staging directory to its
/// content hash, synchronize `blocks/`, expose the block through the
/// in-memory index, and reclaim a redundant parent boundary only after the
/// child is durable. Reordering those steps can expose half-written state,
/// break ancestry, or delete the only restorable boundary during an
/// interrupted publication.
extension PersistentPromptCacheDiskStore {

    func publishNewBlockTransaction(
        staging: PersistentPromptCacheStateFileStaging,
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?
    ) throws {
        // The final name is deterministic, but staging must be unique so a
        // crashed writer cannot be mistaken for a committed block on restart.
        let blockDirectoryName: String = PersistentPromptCacheStoreFile.hexEncode(
            blockKey.blockHash());
        let finalBlockDirectory: URL = self.blocksDirectory
            .appendingPathComponent(blockDirectoryName, isDirectory: true);
        let stagingBlockDirectory: URL = Self.uniqueStagingBlockDirectory(
            blocksDirectory: self.blocksDirectory, blockName: blockDirectoryName);
        try Self.createStagingDirectory(stagingDirectory: stagingBlockDirectory);
        let stagedFiles: PersistentPromptCacheStagedBlockFiles;
        do {
            stagedFiles = try self.stageCompleteBlock(
                staging: staging,
                blockKey: blockKey,
                parentBlockKey: parentBlockKey,
                stagingBlockDirectory: stagingBlockDirectory);
        } catch {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
                directoryPath: stagingBlockDirectory);
            throw error;
        }
        do {
            try PersistentPromptCacheStoreFile.synchronizeDirectory(
                directoryPath: stagingBlockDirectory);
        } catch let syncError as PersistentPromptCacheDiskStoreError {
            throw Self.cleanupStagingAfterError(
                stagingDirectory: stagingBlockDirectory, originalError: syncError);
        }
        // Quota projection may subtract this boundary because it becomes
        // redundant after commit. The file is not physically removed until
        // the child rename and parent-directory synchronization have succeeded.
        let parentBoundaryReclaim: PersistentPromptCacheParentBoundaryReclaim? = self
            .parentBoundaryReclaimAfterCommit(
                parentBlockKey: parentBlockKey, childSizeBytes: stagedFiles.totalSizeBytes);
        var protectedBlockDirectoryPaths: [String] = self.protectedAncestryForCommit(
            parentBlockKey: parentBlockKey);
        protectedBlockDirectoryPaths.append(finalBlockDirectory.path);
        do {
            try self.enforceGlobalPromptCacheQuotaForCommit(
                additionalCommittedSizeBytes: stagedFiles.totalSizeBytes,
                postCommitReclaimableSizeBytes: parentBoundaryReclaim?.fileSizeBytes ?? 0,
                protectedBlockDirectoryPaths: protectedBlockDirectoryPaths,
                excludedDirectory: stagingBlockDirectory);
        } catch let quotaError as PersistentPromptCacheDiskStoreError {
            throw Self.cleanupStagingAfterError(
                stagingDirectory: stagingBlockDirectory, originalError: quotaError);
        }
        // A writer holding the process-local publication lock should not
        // race itself. Presence here therefore means disk topology changed
        // outside this transaction; do not replace content we did not
        // validate.
        if FileManager.default.fileExists(atPath: finalBlockDirectory.path) {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
                directoryPath: stagingBlockDirectory);
            throw PersistentPromptCacheDiskStoreError.existingBlockTopologyMismatch(
                blockHash: blockKey.blockHash());
        }
        do {
            try FileManager.default.moveItem(
                at: stagingBlockDirectory, to: finalBlockDirectory);
        } catch {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
                directoryPath: stagingBlockDirectory);
            throw PersistentPromptCacheDiskStoreError.renameTempFile(
                tempFilePath: stagingBlockDirectory.path,
                blockFilePath: finalBlockDirectory.path,
                problem: String(describing: error));
        }
        try PersistentPromptCacheStoreFile.synchronizeDirectory(
            directoryPath: self.blocksDirectory);
        // Index publication occurs after durable filesystem publication. A
        // lookup can therefore never observe an indexed staging transaction.
        self.trackCommittedBlock(
            blockKey: blockKey,
            parentBlockKey: parentBlockKey,
            finalBlockDirectory: finalBlockDirectory,
            stagedFiles: stagedFiles);
        if let parentBoundaryReclaim: PersistentPromptCacheParentBoundaryReclaim = parentBoundaryReclaim {
            try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: URL(fileURLWithPath: parentBoundaryReclaim.filePath));
            try PersistentPromptCacheStoreFile.synchronizeDirectory(directoryPath: URL(
                fileURLWithPath: parentBoundaryReclaim.blockDirectoryPath));
            self.stateLock.lock();
            self.trackedFiles.removeBlock(blockHash: parentBoundaryReclaim.blockHash);
            self.stateLock.unlock();
        }
        try self.refreshGlobalPromptCacheAccounting();
    }

    /// Startup compaction may legally leave sequence state on a
    /// non-checkpoint parent without its boundary snapshot. If that exact
    /// block is reached again as a leaf, publication restores only the
    /// missing boundary; it never rewrites the already durable sequence
    /// state or manifest.
    func publishMissingBoundaryStateTransaction(
        staging: PersistentPromptCacheStateFileStaging,
        blockKey: PersistentPromptCacheBlockKey
    ) throws {
        let blockHash: Data = blockKey.blockHash();
        self.stateLock.lock();
        let existingBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock? = self.trackedFiles
            .block(blockHash: blockHash);
        self.stateLock.unlock();
        guard let existingBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock = existingBlock
        else {
            throw PersistentPromptCacheDiskStoreError.existingBlockTopologyMismatch(
                blockHash: blockHash);
        }
        let stagingDirectory: URL = Self.uniqueStagingBlockDirectory(
            blocksDirectory: self.blocksDirectory,
            blockName: PersistentPromptCacheStoreFile.hexEncode(blockHash) + "-boundary");
        try Self.createStagingDirectory(stagingDirectory: stagingDirectory);
        let stagedBoundaryFileSizeBytes: UInt64
        do {
            stagedBoundaryFileSizeBytes = try staging.stageStateFile(
                stateFileName: PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME,
                stagingBlockDirectory: stagingDirectory,
                blockTokenCount: blockKey.tokenCount(),
                modelContract: self.modelContract);
        } catch let stagingError as PersistentPromptCacheDiskStoreError {
            throw Self.cleanupStagingAfterError(
                stagingDirectory: stagingDirectory, originalError: stagingError);
        }
        do {
            try Self.validateStagedFileSize(
                stagedFileSizeBytes: stagedBoundaryFileSizeBytes,
                stagedFilePath: stagingDirectory.appendingPathComponent(
                    PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME).path,
                expectedFileSizeBytes: try self.modelContract
                    .boundaryStateFileBytesForBlockTokenCount(
                        blockTokenCount: blockKey.tokenCount()));
        } catch let sizeError as PersistentPromptCacheDiskStoreError {
            throw Self.cleanupStagingAfterError(
                stagingDirectory: stagingDirectory, originalError: sizeError);
        }
        do {
            try PersistentPromptCacheStoreFile.synchronizeDirectory(
                directoryPath: stagingDirectory);
        } catch let syncError as PersistentPromptCacheDiskStoreError {
            throw Self.cleanupStagingAfterError(
                stagingDirectory: stagingDirectory, originalError: syncError);
        }
        self.stateLock.lock();
        let protectedBlockDirectoryPaths: [String] = self.trackedFiles
            .protectedAncestryDirectoryPaths(chainTipBlockHash: blockHash);
        self.stateLock.unlock();
        try self.enforceGlobalPromptCacheQuotaForCommit(
            additionalCommittedSizeBytes: stagedBoundaryFileSizeBytes,
            postCommitReclaimableSizeBytes: 0,
            protectedBlockDirectoryPaths: protectedBlockDirectoryPaths,
            excludedDirectory: stagingDirectory);
        // Renaming one file within an already committed block is atomic. The
        // subsequent block-directory sync makes the new directory entry durable.
        let finalBoundaryFilePath: URL = URL(fileURLWithPath: existingBlock.blockDirectoryPath)
            .appendingPathComponent(PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME);
        do {
            try FileManager.default.moveItem(
                at: stagingDirectory.appendingPathComponent(
                    PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME),
                to: finalBoundaryFilePath);
        } catch {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
                directoryPath: stagingDirectory);
            throw PersistentPromptCacheDiskStoreError.renameTempFile(
                tempFilePath: stagingDirectory.path,
                blockFilePath: finalBoundaryFilePath.path,
                problem: String(describing: error));
        }
        try? PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
            directoryPath: stagingDirectory);
        try PersistentPromptCacheStoreFile.synchronizeDirectory(directoryPath: URL(
            fileURLWithPath: existingBlock.blockDirectoryPath));
        try PersistentPromptCacheStoreFile.synchronizeDirectory(
            directoryPath: self.blocksDirectory);
        self.stateLock.lock();
        self.trackedFiles.insertBlock(
            blockHash: blockHash,
            trackedBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock(
                blockDirectoryPath: existingBlock.blockDirectoryPath,
                blockIndex: existingBlock.blockIndex,
                parentBlockHash: existingBlock.parentBlockHash,
                sequenceStateFile: existingBlock.sequenceStateFile,
                boundaryStateFile: PersistentPromptCacheDiskStoreIndex.TrackedFile(
                    filePath: finalBoundaryFilePath.path,
                    fileSizeBytes: stagedBoundaryFileSizeBytes)));
        self.stateLock.unlock();
        try self.refreshGlobalPromptCacheAccounting();
    }

    /// State-kind presence comes from the immutable model contract: empty
    /// placeholder files are forbidden because they would make a block look
    /// complete while carrying no restorable model state. Exact-size checks
    /// prove the written bytes match the predicted geometry, and quota
    /// admission uses the actual written sizes.
    private func stageCompleteBlock(
        staging: PersistentPromptCacheStateFileStaging,
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?,
        stagingBlockDirectory: URL
    ) throws -> PersistentPromptCacheStagedBlockFiles {
        var sequenceStateFileSizeBytes: UInt64? = nil;
        var boundaryStateFileSizeBytes: UInt64? = nil;
        if self.modelContract.hasSequenceState {
            let stagedSize: UInt64 = try staging.stageStateFile(
                stateFileName: PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME,
                stagingBlockDirectory: stagingBlockDirectory,
                blockTokenCount: blockKey.tokenCount(),
                modelContract: self.modelContract);
            try Self.validateStagedFileSize(
                stagedFileSizeBytes: stagedSize,
                stagedFilePath: stagingBlockDirectory.appendingPathComponent(
                    PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME).path,
                expectedFileSizeBytes: try self.modelContract
                    .sequenceStateFileBytesForBlockTokenCount(
                        blockTokenCount: blockKey.tokenCount()));
            sequenceStateFileSizeBytes = stagedSize;
        }
        if self.modelContract.hasBoundaryState {
            let stagedSize: UInt64 = try staging.stageStateFile(
                stateFileName: PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME,
                stagingBlockDirectory: stagingBlockDirectory,
                blockTokenCount: blockKey.tokenCount(),
                modelContract: self.modelContract);
            try Self.validateStagedFileSize(
                stagedFileSizeBytes: stagedSize,
                stagedFilePath: stagingBlockDirectory.appendingPathComponent(
                    PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME).path,
                expectedFileSizeBytes: try self.modelContract
                    .boundaryStateFileBytesForBlockTokenCount(
                        blockTokenCount: blockKey.tokenCount()));
            boundaryStateFileSizeBytes = stagedSize;
        }
        try PersistentPromptCacheBlockManifest(
            blockKey: blockKey, parentBlockKey: parentBlockKey, modelContract: self.modelContract)
            .writeToStagingDirectory(stagingBlockDirectory: stagingBlockDirectory);
        let manifestFileSizeBytes: UInt64 = try PersistentPromptCacheDiskStoreScan
            .cacheOwnedFileByteCount(filePath: stagingBlockDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.BLOCK_MANIFEST_FILE_NAME));
        if manifestFileSizeBytes > self.modelContract.maximumBlockManifestFileBytes {
            throw PersistentPromptCacheDiskStoreError.sizeBoundExceeded(
                maximumSizeBytes: self.modelContract.maximumBlockManifestFileBytes,
                estimatedBlockBytes: manifestFileSizeBytes);
        }
        let (stateSizeBytes, stateOverflow) = (sequenceStateFileSizeBytes ?? 0)
            .addingReportingOverflow(boundaryStateFileSizeBytes ?? 0);
        let (totalSizeBytes, totalOverflow) = stateSizeBytes
            .addingReportingOverflow(manifestFileSizeBytes);
        if stateOverflow || totalOverflow {
            throw PersistentPromptCacheDiskStoreError.globalPromptCacheSizeOverflow(
                rootDirectory: stagingBlockDirectory.path);
        }
        if totalSizeBytes > self.globalPromptCacheMaximumSizeBytes {
            throw PersistentPromptCacheDiskStoreError.sizeBoundExceeded(
                maximumSizeBytes: self.globalPromptCacheMaximumSizeBytes,
                estimatedBlockBytes: totalSizeBytes);
        }
        return PersistentPromptCacheStagedBlockFiles(
            sequenceStateFileSizeBytes: sequenceStateFileSizeBytes,
            boundaryStateFileSizeBytes: boundaryStateFileSizeBytes,
            totalSizeBytes: totalSizeBytes);
    }

    private func trackCommittedBlock(
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?,
        finalBlockDirectory: URL,
        stagedFiles: PersistentPromptCacheStagedBlockFiles
    ) {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        self.trackedFiles.insertBlock(
            blockHash: blockKey.blockHash(),
            trackedBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock(
                blockDirectoryPath: finalBlockDirectory.path,
                blockIndex: blockKey.blockIndex(),
                parentBlockHash: parentBlockKey?.blockHash(),
                sequenceStateFile: stagedFiles.sequenceStateFileSizeBytes.map(
                    { (stagedSizeBytes: UInt64) -> PersistentPromptCacheDiskStoreIndex.TrackedFile in
                        return PersistentPromptCacheDiskStoreIndex.TrackedFile(
                            filePath: finalBlockDirectory.appendingPathComponent(
                                PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME).path,
                            fileSizeBytes: stagedSizeBytes);
                    }),
                boundaryStateFile: stagedFiles.boundaryStateFileSizeBytes.map(
                    { (stagedSizeBytes: UInt64) -> PersistentPromptCacheDiskStoreIndex.TrackedFile in
                        return PersistentPromptCacheDiskStoreIndex.TrackedFile(
                            filePath: finalBlockDirectory.appendingPathComponent(
                                PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME).path,
                            fileSizeBytes: stagedSizeBytes);
                    })));
    }

    private func protectedAncestryForCommit(
        parentBlockKey: PersistentPromptCacheBlockKey?
    ) -> [String] {
        guard let parentBlockKey: PersistentPromptCacheBlockKey = parentBlockKey else {
            return [];
        }
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.trackedFiles.protectedAncestryDirectoryPaths(
            chainTipBlockHash: parentBlockKey.blockHash());
    }

    private func parentBoundaryReclaimAfterCommit(
        parentBlockKey: PersistentPromptCacheBlockKey?,
        childSizeBytes: UInt64
    ) -> PersistentPromptCacheParentBoundaryReclaim? {
        if self.modelContract.hasSequenceState == false
            || self.modelContract.hasBoundaryState == false {
            return nil;
        }
        guard let parentBlockKey: PersistentPromptCacheBlockKey = parentBlockKey else {
            return nil;
        }
        if PersistentPromptCacheRetentionPolicy.boundaryIsCommonPrefixCheckpoint(
            blockIndex: parentBlockKey.blockIndex(),
            commonPrefixCheckpointStrideBlocks: self.modelContract
                .commonPrefixCheckpointStrideBlocks)
            || self.totalSizeBytes().addingReportingOverflow(childSizeBytes).partialValue
                <= self.globalPromptCacheMaximumSizeBytes {
            return nil;
        }
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        guard let parentBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock = self.trackedFiles
            .block(blockHash: parentBlockKey.blockHash()),
            let parentBoundaryFile: PersistentPromptCacheDiskStoreIndex.TrackedFile = parentBlock
                .boundaryStateFile
        else {
            return nil;
        }
        return PersistentPromptCacheParentBoundaryReclaim(
            blockHash: parentBlockKey.blockHash(),
            blockDirectoryPath: parentBlock.blockDirectoryPath,
            filePath: parentBoundaryFile.filePath,
            fileSizeBytes: parentBoundaryFile.fileSizeBytes);
    }

    static func validateStagedFileSize(
        stagedFileSizeBytes: UInt64,
        stagedFilePath: String,
        expectedFileSizeBytes: UInt64
    ) throws {
        if stagedFileSizeBytes == expectedFileSizeBytes {
            return;
        }
        throw PersistentPromptCacheDiskStoreError.writtenFileSizeMismatch(
            filePath: stagedFilePath,
            reportedSizeBytes: expectedFileSizeBytes,
            actualSizeBytes: stagedFileSizeBytes);
    }

    static func createStagingDirectory(
        stagingDirectory: URL
    ) throws {
        do {
            try FileManager.default.createDirectory(
                at: stagingDirectory, withIntermediateDirectories: false);
        } catch {
            throw PersistentPromptCacheDiskStoreError.createPromptCacheDirectory(
                directoryPath: stagingDirectory.path, problem: String(describing: error));
        }
    }

    static func uniqueStagingBlockDirectory(
        blocksDirectory: URL, blockName: String
    ) -> URL {
        let currentNanos: UInt64 = UInt64(Date().timeIntervalSince1970 * 1_000_000_000);
        return blocksDirectory.appendingPathComponent(
            "\(blockName).staging-\(getpid())-\(currentNanos)", isDirectory: true);
    }

    static func cleanupStagingAfterError(
        stagingDirectory: URL, originalError: PersistentPromptCacheDiskStoreError
    ) -> PersistentPromptCacheDiskStoreError {
        // A cleanup failure wins because it leaves bytes that startup must
        // recover and may prevent the quota from being satisfied. Otherwise
        // retain the operation's original, more useful failure.
        do {
            try PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
                directoryPath: stagingDirectory);
        } catch let cleanupError as PersistentPromptCacheDiskStoreError {
            return cleanupError;
        } catch {
            return originalError;
        }
        return originalError;
    }
}
