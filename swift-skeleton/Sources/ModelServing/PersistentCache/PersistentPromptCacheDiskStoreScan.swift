import Foundation;


/// Startup scan: discovers, validates, and cleans persistent prompt-cache
/// files while the disk store opens, port of the Rust `disk_store_scan`.
/// Phase one validates each block directory in isolation and builds
/// candidates; phase two validates cross-directory ancestry to a fixed
/// point before indexing any of them, preventing lookup from observing a
/// partially accepted graph.
enum PersistentPromptCacheDiskStoreScan {

    private struct BlockDirectoryCandidate {
        var blockDirectoryPath: String;
        var blockIndex: UInt32;
        var parentBlockHash: Data?;
        var sequenceStateFile: PersistentPromptCacheDiskStoreIndex.TrackedFile?;
        var boundaryStateFile: PersistentPromptCacheDiskStoreIndex.TrackedFile?;
    }

    /// Scans one flat directory of hash-named `.safetensors` files (the
    /// visual-embedding layer), removing stale temporaries and corrupt
    /// entries while recording reason-separated cleanup evidence.
    static func scanCurrentFormatDirectory(
        directory: URL,
        trackedFiles: inout PersistentPromptCacheDiskStoreIndex,
        startupCleanupEvidence: inout PersistentPromptCacheStartupCleanupEvidence,
        headerValidator: (URL) throws -> Bool
    ) throws {
        let directoryEntries: [URL];
        do {
            directoryEntries = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
                options: []);
        } catch {
            throw PersistentPromptCacheDiskStoreError.readPromptCacheDirectory(
                directoryPath: directory.path, problem: String(describing: error));
        }
        for enumeratedEntryPath: URL in directoryEntries {
            let entryPath: URL = PersistentPromptCacheStoreFile.storeFormEntryURL(
                directory: directory, enumeratedEntry: enumeratedEntryPath);
            let entryIsRegularFile: Bool = (try? entryPath.resourceValues(
                forKeys: [.isRegularFileKey]))?.isRegularFile ?? false;
            if entryPath.pathExtension == "tmp" {
                let removedByteCount: UInt64 = try Self.cacheOwnedFileByteCount(
                    filePath: entryPath);
                try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                    filePath: entryPath);
                startupCleanupEvidence.recordArtifact(
                    reason: .interruptedTransactionRecovery, removedByteCount: removedByteCount);
                continue;
            }
            if entryIsRegularFile == false || entryPath.pathExtension != "safetensors" {
                continue;
            }
            guard let persistentPromptCacheFileHash: Data =
                PersistentPromptCacheStoreFile.parseBlockHashHex(entryPath.deletingPathExtension()
                    .lastPathComponent)
            else {
                let removedByteCount: UInt64 = try Self.cacheOwnedFileByteCount(
                    filePath: entryPath);
                try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                    filePath: entryPath);
                startupCleanupEvidence.recordArtifact(
                    reason: .corruptCurrentFormat, removedByteCount: removedByteCount);
                continue;
            };
            let fileSizeBytes: UInt64 = try Self.cacheOwnedFileByteCount(filePath: entryPath);
            let headerAccepted: Bool;
            do {
                headerAccepted = try headerValidator(entryPath);
            } catch {
                throw PersistentPromptCacheDiskStoreError.openBlockFile(
                    blockFilePath: entryPath.path, problem: String(describing: error));
            }
            if headerAccepted == false {
                try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                    filePath: entryPath);
                startupCleanupEvidence.recordArtifact(
                    reason: .corruptCurrentFormat, removedByteCount: fileSizeBytes);
                continue;
            }
            trackedFiles.insertVisualEmbeddingFile(
                fileHash: persistentPromptCacheFileHash,
                trackedFile: PersistentPromptCacheDiskStoreIndex.TrackedFile(
                    filePath: entryPath.path, fileSizeBytes: fileSizeBytes));
        }
    }

    /// Scans the committed block directories, validates every manifest and
    /// state header, reconciles sequence ancestry and boundary retention
    /// topology to a fixed point, and indexes only the surviving graph.
    static func scanCurrentFormatBlockDirectories(
        blocksDirectory: URL,
        trackedFiles: inout PersistentPromptCacheDiskStoreIndex,
        modelContract: PersistentPromptCacheModelContract,
        startupCleanupEvidence: inout PersistentPromptCacheStartupCleanupEvidence
    ) throws {
        var blockCandidateByHash: [Data: BlockDirectoryCandidate] = [:];
        let directoryEntries: [URL];
        do {
            directoryEntries = try FileManager.default.contentsOfDirectory(
                at: blocksDirectory, includingPropertiesForKeys: [.isDirectoryKey],
                options: []);
        } catch {
            throw PersistentPromptCacheDiskStoreError.readPromptCacheDirectory(
                directoryPath: blocksDirectory.path, problem: String(describing: error));
        }
        for enumeratedEntryPath: URL in directoryEntries {
            let blockDirectoryPath: URL = PersistentPromptCacheStoreFile.storeFormEntryURL(
                directory: blocksDirectory, enumeratedEntry: enumeratedEntryPath);
            let entryIsDirectory: Bool = (try? blockDirectoryPath.resourceValues(
                forKeys: [.isDirectoryKey]))?.isDirectory ?? false;
            if entryIsDirectory == false {
                continue;
            }
            let blockDirectoryName: String = blockDirectoryPath.lastPathComponent;
            // A staging name proves publication never reached its atomic
            // directory rename. Remove the whole transaction; individual
            // files are not salvageable. The staging check runs before the
            // hash parse because a staging name is not hash-shaped.
            if blockDirectoryName.contains(".staging-") {
                try Self.removeCandidateDirectory(
                    blockDirectoryPath, into: &startupCleanupEvidence,
                    reason: .interruptedTransactionRecovery);
                continue;
            }
            guard let blockHashFromDirectory: Data =
                PersistentPromptCacheStoreFile.parseBlockHashHex(blockDirectoryName)
            else {
                try Self.removeCandidateDirectory(
                    blockDirectoryPath, into: &startupCleanupEvidence,
                    reason: .corruptCurrentFormat);
                continue;
            };
            let blockManifest: PersistentPromptCacheBlockManifest;
            do {
                blockManifest = try PersistentPromptCacheBlockManifest.readFromBlockDirectory(
                    blockDirectory: blockDirectoryPath, modelContract: modelContract);
            } catch {
                try Self.removeCandidateDirectory(
                    blockDirectoryPath, into: &startupCleanupEvidence,
                    reason: .corruptCurrentFormat);
                continue;
            }
            if (try? blockManifest.blockHash()) != blockHashFromDirectory {
                try Self.removeCandidateDirectory(
                    blockDirectoryPath, into: &startupCleanupEvidence,
                    reason: .corruptCurrentFormat);
                continue;
            }
            // Missing or invalid state remains nil until topology
            // reconciliation. That later phase can distinguish a legal
            // compacted parent boundary from an incomplete leaf or a block
            // missing required sequence state.
            let sequenceStateFile: PersistentPromptCacheDiskStoreIndex.TrackedFile? =
                blockManifest.hasSequenceState
                ? try Self.validateBlockStateFile(
                    stateFileUrl: blockDirectoryPath.appendingPathComponent(
                        PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME),
                    sequenceState: true, modelContract: modelContract)
                : nil;
            let boundaryStateFile: PersistentPromptCacheDiskStoreIndex.TrackedFile? =
                blockManifest.hasBoundaryState
                ? try Self.validateBlockStateFile(
                    stateFileUrl: blockDirectoryPath.appendingPathComponent(
                        PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME),
                    sequenceState: false, modelContract: modelContract)
                : nil;
            blockCandidateByHash[blockHashFromDirectory] = BlockDirectoryCandidate(
                blockDirectoryPath: blockDirectoryPath.path,
                blockIndex: blockManifest.blockIndex,
                parentBlockHash: blockManifest.parentBlockHash(),
                sequenceStateFile: sequenceStateFile,
                boundaryStateFile: boundaryStateFile);
        }
        try Self.reconcileBlockTopology(
            blockCandidateByHash: &blockCandidateByHash,
            trackedFiles: &trackedFiles,
            modelContract: modelContract,
            startupCleanupEvidence: &startupCleanupEvidence);
    }

    /// Sequence ancestry is transitive: removing one invalid parent can
    /// orphan a child that looked valid in the previous pass, so iterate to
    /// a fixed point. Boundary retention is intentionally asymmetric — a
    /// non-checkpoint parent may omit its boundary once a child exists, but
    /// leaves and checkpoints may not. Boundary removals can create new
    /// sequence orphans, so prune those descendants to a fixed point before
    /// exposing the surviving graph to lookup.
    private static func reconcileBlockTopology(
        blockCandidateByHash: inout [Data: BlockDirectoryCandidate],
        trackedFiles: inout PersistentPromptCacheDiskStoreIndex,
        modelContract: PersistentPromptCacheModelContract,
        startupCleanupEvidence: inout PersistentPromptCacheStartupCleanupEvidence
    ) throws {
        while true {
            let invalidBlockHashes: [Data] = blockCandidateByHash
                .filter({ (_, candidate: BlockDirectoryCandidate) in
                    return Self.blockCandidateHasValidSequenceAncestry(
                        candidate,
                        blockCandidateByHash: blockCandidateByHash,
                        modelContract: modelContract) == false;
                })
                .map({ (blockHash: Data, _: BlockDirectoryCandidate) in
                    return blockHash;
                });
            if invalidBlockHashes.isEmpty {
                break;
            }
            try Self.removeInvalidBlockCandidates(
                invalidBlockHashes, blockCandidateByHash: &blockCandidateByHash,
                startupCleanupEvidence: &startupCleanupEvidence);
        }
        let blockHashesWithChildren: Set<Data> = Set(
            blockCandidateByHash.values.compactMap({ (candidate: BlockDirectoryCandidate) in
                return candidate.parentBlockHash;
            }));
        let invalidBoundaryBlockHashes: [Data] = blockCandidateByHash
            .filter({ (blockHash: Data, candidate: BlockDirectoryCandidate) in
                return Self.blockCandidateHasValidBoundaryTopology(
                    blockHash,
                    candidate: candidate,
                    blockHashesWithChildren: blockHashesWithChildren,
                    modelContract: modelContract) == false;
            })
            .map({ (blockHash: Data, _: BlockDirectoryCandidate) in
                return blockHash;
            });
        try Self.removeInvalidBlockCandidates(
            invalidBoundaryBlockHashes, blockCandidateByHash: &blockCandidateByHash,
            startupCleanupEvidence: &startupCleanupEvidence);
        while true {
            let orphanBlockHashes: [Data] = blockCandidateByHash
                .filter({ (blockHash: Data, candidate: BlockDirectoryCandidate) in
                    return candidate.blockIndex > 0
                        && (candidate.parentBlockHash.map(
                            { (parentHash: Data) -> Bool in
                                return blockCandidateByHash[parentHash] != nil;
                            }) ?? false) == false;
                })
                .map({ (blockHash: Data, _: BlockDirectoryCandidate) in
                    return blockHash;
                });
            if orphanBlockHashes.isEmpty {
                break;
            }
            try Self.removeInvalidBlockCandidates(
                orphanBlockHashes, blockCandidateByHash: &blockCandidateByHash,
                startupCleanupEvidence: &startupCleanupEvidence);
        }
        for candidateEntry: (key: Data, value: BlockDirectoryCandidate)
            in blockCandidateByHash {
            trackedFiles.insertBlock(
                blockHash: candidateEntry.key,
                trackedBlock: PersistentPromptCacheDiskStoreIndex.TrackedBlock(
                    blockDirectoryPath: candidateEntry.value.blockDirectoryPath,
                    blockIndex: candidateEntry.value.blockIndex,
                    parentBlockHash: candidateEntry.value.parentBlockHash,
                    sequenceStateFile: candidateEntry.value.sequenceStateFile,
                    boundaryStateFile: candidateEntry.value.boundaryStateFile));
        }
    }

    /// A child edge is valid only when indices are consecutive: content
    /// hashes do not encode an independently inspectable ordinal, so the
    /// manifest supplies it.
    private static func blockCandidateHasValidSequenceAncestry(
        _ blockCandidate: BlockDirectoryCandidate,
        blockCandidateByHash: [Data: BlockDirectoryCandidate],
        modelContract: PersistentPromptCacheModelContract
    ) -> Bool {
        if modelContract.hasSequenceState && blockCandidate.sequenceStateFile == nil {
            return false;
        }
        switch (blockCandidate.blockIndex, blockCandidate.parentBlockHash) {
        case (0, .none):
            return true;
        case (0, .some), (_, .none):
            return false;
        case (let blockIndex, .some(let parentBlockHash)):
            guard let parentCandidate: BlockDirectoryCandidate =
                blockCandidateByHash[parentBlockHash] else {
                return false;
            }
            let (nextParentIndex, overflow) = parentCandidate.blockIndex
                .addingReportingOverflow(1);
            return overflow == false && nextParentIndex == blockIndex;
        }
    }

    /// Only hybrid state can reconstruct a compacted parent's boundary from
    /// a later boundary plus the complete sequence chain; boundary-only
    /// models must retain a snapshot at every indexed block.
    private static func blockCandidateHasValidBoundaryTopology(
        _ blockHash: Data,
        candidate: BlockDirectoryCandidate,
        blockHashesWithChildren: Set<Data>,
        modelContract: PersistentPromptCacheModelContract
    ) -> Bool {
        if modelContract.hasBoundaryState == false || candidate.boundaryStateFile != nil {
            return true;
        }
        return modelContract.hasSequenceState
            && blockHashesWithChildren.contains(blockHash)
            && PersistentPromptCacheRetentionPolicy.boundaryIsCommonPrefixCheckpoint(
                blockIndex: candidate.blockIndex,
                commonPrefixCheckpointStrideBlocks:
                    modelContract.commonPrefixCheckpointStrideBlocks) == false;
    }

    private static func removeInvalidBlockCandidates(
        _ invalidBlockHashes: [Data],
        blockCandidateByHash: inout [Data: BlockDirectoryCandidate],
        startupCleanupEvidence: inout PersistentPromptCacheStartupCleanupEvidence
    ) throws {
        for invalidBlockHash: Data in invalidBlockHashes {
            guard let invalidBlockCandidate: BlockDirectoryCandidate =
                blockCandidateByHash.removeValue(forKey: invalidBlockHash) else {
                continue;
            }
            try Self.removeCandidateDirectory(
                URL(fileURLWithPath: invalidBlockCandidate.blockDirectoryPath),
                into: &startupCleanupEvidence, reason: .corruptCurrentFormat);
        }
    }

    private static func removeCandidateDirectory(
        _ blockDirectoryPath: URL,
        into startupCleanupEvidence: inout PersistentPromptCacheStartupCleanupEvidence,
        reason: PersistentPromptCacheStartupCleanupEvidence.Reason
    ) throws {
        let removedByteCount: UInt64 = try Self.cacheOwnedDirectoryByteCount(
            directoryPath: blockDirectoryPath);
        try PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
            directoryPath: blockDirectoryPath);
        startupCleanupEvidence.recordBlock(reason: reason, removedByteCount: removedByteCount);
    }

    /// Loads nothing: returns the validated tracked file, or nil when the
    /// state file is absent, not a regular file, or fails its header
    /// contract.
    private static func validateBlockStateFile(
        stateFileUrl: URL,
        sequenceState: Bool,
        modelContract: PersistentPromptCacheModelContract
    ) throws -> PersistentPromptCacheDiskStoreIndex.TrackedFile? {
        let stateFileSizeBytes: UInt64;
        do {
            let fileAttributes: [FileAttributeKey: Any] = try FileManager.default
                .attributesOfItem(atPath: stateFileUrl.path);
            guard (fileAttributes[.type] as? FileAttributeType) == .typeRegular else {
                return nil;
            }
            stateFileSizeBytes = (fileAttributes[.size] as? NSNumber)?.uint64Value ?? 0;
        } catch let statError as NSError
            where statError.code == NSFileNoSuchFileError
                || statError.code == NSFileReadNoSuchFileError {
            return nil;
        } catch {
            throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                blockFilePath: stateFileUrl.path, problem: String(describing: error));
        }
        let blockHeader: PersistentPromptCacheBlockHeader;
        do {
            blockHeader = sequenceState
                ? try PersistentPromptCacheBlockHeader.readKvBlock(
                    blockFileUrl: stateFileUrl, modelContract: modelContract)
                : try PersistentPromptCacheBlockHeader.readRecurrentSnapshot(
                    snapshotFileUrl: stateFileUrl, modelContract: modelContract);
        } catch {
            return nil;
        }
        _ = blockHeader;
        return PersistentPromptCacheDiskStoreIndex.TrackedFile(
            filePath: stateFileUrl.path, fileSizeBytes: stateFileSizeBytes);
    }

    static func cacheOwnedFileByteCount(filePath: URL) throws -> UInt64 {
        do {
            let fileAttributes: [FileAttributeKey: Any] = try FileManager.default
                .attributesOfItem(atPath: filePath.path);
            return (fileAttributes[.size] as? NSNumber)?.uint64Value ?? 0;
        } catch {
            throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                blockFilePath: filePath.path, problem: String(describing: error));
        }
    }

    static func cacheOwnedDirectoryByteCount(directoryPath: URL) throws -> UInt64 {
        var pendingDirectories: [URL] = [directoryPath];
        var totalByteCount: UInt64 = 0;
        while let pendingDirectory: URL = pendingDirectories.popLast() {
            let directoryEntries: [URL];
            do {
                directoryEntries = try FileManager.default.contentsOfDirectory(
                    at: pendingDirectory, includingPropertiesForKeys: nil, options: []);
            } catch {
                throw PersistentPromptCacheDiskStoreError.readPromptCacheDirectory(
                    directoryPath: pendingDirectory.path,
                    problem: String(describing: error));
            }
            for entryPath: URL in directoryEntries {
                var isDirectory: ObjCBool = ObjCBool(false);
                FileManager.default.fileExists(
                    atPath: entryPath.path, isDirectory: &isDirectory);
                if isDirectory.boolValue {
                    pendingDirectories.append(entryPath);
                } else {
                    let fileByteCount: UInt64 = try Self.cacheOwnedFileByteCount(
                        filePath: entryPath);
                    totalByteCount = totalByteCount &+ fileByteCount;
                }
            }
        }
        return totalByteCount;
    }
}
