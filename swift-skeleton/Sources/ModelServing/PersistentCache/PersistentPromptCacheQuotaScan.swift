import Foundation;

import ModelServing;

/// Recursive global prompt-cache quota discovery, port of the Rust
/// `disk_store_global_quota_scan`. This type has no deletion authority: it
/// takes one filesystem snapshot, counts every owned byte, reconstructs
/// committed block ancestry, and returns deterministic eviction units.
/// Keeping discovery separate from deletion avoids rescanning and
/// resorting the whole cache after every removed block.
public struct PersistentPromptCacheQuotaScan: Sendable {

    public var evictionCandidatesOldestWrittenFirst:
        [PersistentPromptCacheEvictionCandidate];

    public var totalSizeBytes: UInt64;

    public var visualEmbeddingTotalSizeBytes: UInt64;
}

enum PersistentPromptCacheQuotaScanEngine {

    /// A hash alone is not globally unique: separate model/revision
    /// directories can contain the same token hash, and the same directory
    /// can retain stale files from another tensor layout. All three fields
    /// are required before a manifest parent edge may connect two blocks.
    private struct BlockIdentity: Hashable {
        var blocksDirectoryPath: String;
        var storageContractFingerprint: String;
        var blockHash: Data;
    }

    private struct BlockDirectory {
        var identity: BlockIdentity;
        var parentBlockHash: Data?;
        var blockDirectoryPath: String;
        var fileSizeBytes: UInt64;
        var modifiedAt: Date;
        var trackedFilePaths: [String];
    }

    static func scanGlobalPromptCacheQuota(
        globalPromptCacheRootDirectory: URL,
        excludedDirectory: URL?
    ) throws -> PersistentPromptCacheQuotaScan {
        var standaloneFiles: [PersistentPromptCacheEvictionCandidate] = [];
        var staleDirectories: [PersistentPromptCacheEvictionCandidate] = [];
        var blockDirectories: [BlockDirectory] = [];
        try scanGlobalPromptCacheEntries(
            globalPromptCacheRootDirectory: globalPromptCacheRootDirectory,
            excludedDirectory: excludedDirectory,
            standaloneFiles: &standaloneFiles,
            staleDirectories: &staleDirectories,
            blockDirectories: &blockDirectories);
        var totalSizeBytes: UInt64 = 0;
        var visualEmbeddingTotalSizeBytes: UInt64 = 0;
        for standaloneFile: PersistentPromptCacheEvictionCandidate in standaloneFiles {
            totalSizeBytes = try checkedAddGlobalSize(
                globalPromptCacheRootDirectory: globalPromptCacheRootDirectory,
                accumulatedSizeBytes: totalSizeBytes,
                additionalSizeBytes: standaloneFile.fileSizeBytes);
            if standaloneFile.visualEmbeddingSizeBytes > 0 {
                visualEmbeddingTotalSizeBytes = try checkedAddGlobalSize(
                    globalPromptCacheRootDirectory: globalPromptCacheRootDirectory,
                    accumulatedSizeBytes: visualEmbeddingTotalSizeBytes,
                    additionalSizeBytes: standaloneFile.visualEmbeddingSizeBytes);
            }
        }
        for staleDirectory: PersistentPromptCacheEvictionCandidate in staleDirectories {
            totalSizeBytes = try checkedAddGlobalSize(
                globalPromptCacheRootDirectory: globalPromptCacheRootDirectory,
                accumulatedSizeBytes: totalSizeBytes,
                additionalSizeBytes: staleDirectory.fileSizeBytes);
        }
        for blockDirectory: BlockDirectory in blockDirectories {
            totalSizeBytes = try checkedAddGlobalSize(
                globalPromptCacheRootDirectory: globalPromptCacheRootDirectory,
                accumulatedSizeBytes: totalSizeBytes,
                additionalSizeBytes: blockDirectory.fileSizeBytes);
        }
        var evictionCandidates: [PersistentPromptCacheEvictionCandidate] = standaloneFiles;
        evictionCandidates.append(contentsOf: staleDirectories);
        evictionCandidates.append(
            contentsOf: try buildBlockSubtreeCandidates(blockDirectories: blockDirectories));
        // Stale transaction artifacts sort before durable content regardless
        // of age. Within each class, oldest-write-first gives predictable
        // LRU-like pressure relief without maintaining another persistent
        // access database.
        evictionCandidates.sort(by: { (leftCandidate, rightCandidate) -> Bool in
            let leftUnconditional: Bool = leftCandidate.isUnconditionallyRemovable;
            let rightUnconditional: Bool = rightCandidate.isUnconditionallyRemovable;
            if leftUnconditional != rightUnconditional {
                return leftUnconditional;
            }
            if leftCandidate.modifiedAt != rightCandidate.modifiedAt {
                return leftCandidate.modifiedAt < rightCandidate.modifiedAt;
            }
            return leftCandidate.tieBreakerPath.utf8
                .lexicographicallyPrecedes(rightCandidate.tieBreakerPath.utf8);
        });
        return PersistentPromptCacheQuotaScan(
            evictionCandidatesOldestWrittenFirst: evictionCandidates,
            totalSizeBytes: totalSizeBytes,
            visualEmbeddingTotalSizeBytes: visualEmbeddingTotalSizeBytes);
    }

    private static func scanGlobalPromptCacheEntries(
        globalPromptCacheRootDirectory: URL,
        excludedDirectory: URL?,
        standaloneFiles: inout [PersistentPromptCacheEvictionCandidate],
        staleDirectories: inout [PersistentPromptCacheEvictionCandidate],
        blockDirectories: inout [BlockDirectory]
    ) throws {
        var pendingDirectories: [URL] = [globalPromptCacheRootDirectory];
        while let pendingDirectory: URL = pendingDirectories.popLast() {
            // URL equality fails across trailing-slash variants (the
            // `isDirectory:` appender adds one, enumeration does not), so the
            // exclusion compares resolved path strings.
            if let excludedDirectory: URL = excludedDirectory,
                pendingDirectory.path == excludedDirectory.path {
                continue;
            }
            let directoryEntries: [URL];
            do {
                directoryEntries = try FileManager.default.contentsOfDirectory(
                    at: pendingDirectory, includingPropertiesForKeys: nil, options: []);
            } catch {
                throw PersistentPromptCacheDiskStoreError.readPromptCacheDirectory(
                    directoryPath: pendingDirectory.path,
                    problem: String(describing: error));
            }
            for enumeratedEntryPath: URL in directoryEntries {
                let entryPath: URL = PersistentPromptCacheStoreFile.storeFormEntryURL(
                    directory: pendingDirectory, enumeratedEntry: enumeratedEntryPath);
                // The store form now matches the caller-provided exclusion,
                // so string path comparison covers both trailing-slash
                // variants (URL equality fails across them).
                if let excludedDirectory: URL = excludedDirectory,
                    entryPath.path == excludedDirectory.path {
                    continue;
                }
                // Symlink classification must use lstat semantics like
                // Rust's read_dir: a symlink pointing at a directory is a
                // removable standalone entry, never a subtree to walk into
                // and never followed outside the cache root.
                let entryType: FileAttributeType? = (try? FileManager.default
                    .attributesOfItem(atPath: entryPath.path))?[.type]
                    as? FileAttributeType;
                let isDirectory: Bool = entryType == .typeDirectory;
                if isDirectory {
                    // A valid block directory is a leaf for this traversal:
                    // its contents are counted together by
                    // `scanBlockDirectory`.
                    if isStaleBlockStagingDirectory(directoryPath: entryPath) {
                        staleDirectories.append(try scanStaleDirectory(directoryPath: entryPath));
                    } else if let blockDirectory: BlockDirectory = try scanBlockDirectory(
                        directoryPath: entryPath) {
                        blockDirectories.append(blockDirectory);
                    } else {
                        pendingDirectories.append(entryPath);
                    }
                } else {
                    standaloneFiles.append(try scanStandaloneFile(filePath: entryPath));
                }
            }
        }
    }

    private static func scanStandaloneFile(
        filePath: URL
    ) throws -> PersistentPromptCacheEvictionCandidate {
        let fileAttributes: [FileAttributeKey: Any];
        do {
            fileAttributes = try FileManager.default.attributesOfItem(atPath: filePath.path);
        } catch {
            throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                blockFilePath: filePath.path, problem: String(describing: error));
        }
        let modifiedAt: Date = (fileAttributes[.modificationDate] as? Date) ?? Date(timeIntervalSince1970: 0);
        let parentDirectoryName: String = filePath.deletingLastPathComponent().lastPathComponent;
        let isVisualEmbedding: Bool = parentDirectoryName == "visual_embeddings";
        // `kv_blocks` and `recurrent_snapshots` are retired pre-format-11
        // storage trees. Their files are recoverable cache artifacts, not
        // valid committed blocks, so startup may reclaim them before
        // current-format content.
        let obsoleteDirectoryNames: Set<String> = [
            "kv_blocks", "recurrent_snapshots",
            "speculative_prefill_selections", "speculative_prefill_target_states",
        ];
        var isObsoleteFormatArtifact: Bool = false;
        var ancestorDirectory: URL = filePath.deletingLastPathComponent();
        while ancestorDirectory.path != "/" {
            if obsoleteDirectoryNames.contains(ancestorDirectory.lastPathComponent) {
                isObsoleteFormatArtifact = true;
                break;
            }
            ancestorDirectory = ancestorDirectory.deletingLastPathComponent();
        }
        let cleanupClassification: PersistentPromptCacheEvictionCandidate
            .StandaloneFileClassification;
        if filePath.pathExtension == "tmp" {
            cleanupClassification = .abandonedTransaction;
        } else if isObsoleteFormatArtifact {
            cleanupClassification = .obsoleteFormat;
        } else {
            cleanupClassification = .committedArtifact;
        }
        return .standaloneFile(
            filePath: filePath.path,
            fileSizeBytes: (fileAttributes[.size] as? NSNumber)?.uint64Value ?? 0,
            modifiedAt: modifiedAt,
            isVisualEmbedding: isVisualEmbedding,
            cleanupClassification: cleanupClassification);
    }

    private static func scanStaleDirectory(
        directoryPath: URL
    ) throws -> PersistentPromptCacheEvictionCandidate {
        let modifiedAt: Date = try Self.blockDirectoryModifiedAt(directoryPath: directoryPath);
        let (fileSizeBytes, trackedFilePaths) = try Self.directoryFileSizeAndPaths(
            directoryPath: directoryPath);
        return .staleDirectory(
            directoryPath: directoryPath.path,
            fileSizeBytes: fileSizeBytes,
            modifiedAt: modifiedAt,
            trackedFilePaths: trackedFilePaths);
    }

    private static func scanBlockDirectory(
        directoryPath: URL
    ) throws -> BlockDirectory? {
        let parentDirectory: URL = directoryPath.deletingLastPathComponent();
        guard parentDirectory.lastPathComponent == "blocks" else {
            return nil;
        }
        guard let blockHash: Data = PersistentPromptCacheStoreFile.parseBlockHashHex(
            directoryPath.lastPathComponent) else {
            return nil;
        };
        let modifiedAt: Date = try Self.blockDirectoryModifiedAt(directoryPath: directoryPath);
        // Global quota scans every model and revision, so it cannot validate
        // a foreign manifest against the active model contract. It reads
        // only the topology fields needed for safe grouping. Invalid
        // manifests receive a path-unique synthetic fingerprint, preventing
        // accidental ancestry joins.
        let parsedManifest: PersistentPromptCacheBlockManifest? = try?
            PersistentPromptCacheBlockManifest.readUnvalidatedFromBlockDirectory(
                blockDirectory: directoryPath);
        let blockManifest: PersistentPromptCacheBlockManifest? = parsedManifest.flatMap(
            { (manifest: PersistentPromptCacheBlockManifest) in
                return (try? manifest.blockHash()) == blockHash ? manifest : nil;
            });
        let parentBlockHash: Data? = blockManifest?.parentBlockHash();
        let storageContractFingerprint: String = blockManifest?.storageContractFingerprint
            ?? "invalid-manifest:\(directoryPath.path)";
        let (fileSizeBytes, trackedFilePaths) = try Self.directoryFileSizeAndPaths(
            directoryPath: directoryPath);
        return BlockDirectory(
            identity: BlockIdentity(
                blocksDirectoryPath: parentDirectory.path,
                storageContractFingerprint: storageContractFingerprint,
                blockHash: blockHash),
            parentBlockHash: parentBlockHash,
            blockDirectoryPath: directoryPath.path,
            fileSizeBytes: fileSizeBytes,
            modifiedAt: modifiedAt,
            trackedFilePaths: trackedFilePaths);
    }

    /// Builds one ancestry-closed subtree candidate per committed block:
    /// parent edges exist only inside one blocks directory and one storage
    /// fingerprint, which is the core guard against cross-model eviction.
    /// Every block can be a candidate root; overlapping candidates are
    /// expected and the deletion owner skips later overlaps by path.
    private static func buildBlockSubtreeCandidates(
        blockDirectories: [BlockDirectory]
    ) throws -> [PersistentPromptCacheEvictionCandidate] {
        var blockDirectoryByIdentity: [BlockIdentity: BlockDirectory] = [:];
        for blockDirectory: BlockDirectory in blockDirectories {
            blockDirectoryByIdentity[blockDirectory.identity] = blockDirectory;
        }
        var childrenByParentIdentity: [BlockIdentity: [BlockIdentity]] = [:];
        for blockDirectory: BlockDirectory in blockDirectoryByIdentity.values {
            guard let parentBlockHash: Data = blockDirectory.parentBlockHash else {
                continue;
            }
            let parentIdentity: BlockIdentity = BlockIdentity(
                blocksDirectoryPath: blockDirectory.identity.blocksDirectoryPath,
                storageContractFingerprint: blockDirectory.identity.storageContractFingerprint,
                blockHash: parentBlockHash);
            childrenByParentIdentity[parentIdentity, default: []]
                .append(blockDirectory.identity);
        }
        var evictionCandidates: [PersistentPromptCacheEvictionCandidate] = [];
        for blockIdentity: BlockIdentity in blockDirectoryByIdentity.keys {
            var subtreeBlockIdentities: [BlockIdentity] = [];
            collectSubtreeBlockIdentities(
                rootBlockIdentity: blockIdentity,
                childrenByParentIdentity: childrenByParentIdentity,
                subtreeBlockIdentities: &subtreeBlockIdentities);
            var subtreeFileSizeBytes: UInt64 = 0;
            var subtreeBlockDirectoryPaths: [String] = [];
            var subtreeTrackedFilePaths: [String] = [];
            for subtreeBlockIdentity: BlockIdentity in subtreeBlockIdentities {
                guard let subtreeBlockDirectory: BlockDirectory =
                    blockDirectoryByIdentity[subtreeBlockIdentity] else {
                    continue;
                }
                let (addedBytes, overflow) = subtreeFileSizeBytes
                    .addingReportingOverflow(subtreeBlockDirectory.fileSizeBytes);
                if overflow {
                    throw PersistentPromptCacheDiskStoreError.globalPromptCacheSizeOverflow(
                        rootDirectory: subtreeBlockDirectory.identity.blocksDirectoryPath);
                }
                subtreeFileSizeBytes = addedBytes;
                subtreeBlockDirectoryPaths.append(
                    subtreeBlockDirectory.blockDirectoryPath);
                subtreeTrackedFilePaths.append(
                    contentsOf: subtreeBlockDirectory.trackedFilePaths);
            }
            guard let rootBlockDirectory: BlockDirectory = blockDirectoryByIdentity[blockIdentity]
            else {
                continue;
            }
            evictionCandidates.append(.blockSubtree(
                rootBlockDirectoryPath: rootBlockDirectory.blockDirectoryPath,
                fileSizeBytes: subtreeFileSizeBytes,
                modifiedAt: rootBlockDirectory.modifiedAt,
                blockDirectoryPaths: subtreeBlockDirectoryPaths,
                trackedFilePaths: subtreeTrackedFilePaths));
        }
        return evictionCandidates;
    }

    /// Uses an explicit stack and visited set: corrupt manifests can form
    /// cycles, and quota recovery must terminate rather than recurse
    /// forever.
    private static func collectSubtreeBlockIdentities(
        rootBlockIdentity: BlockIdentity,
        childrenByParentIdentity: [BlockIdentity: [BlockIdentity]],
        subtreeBlockIdentities: inout [BlockIdentity]
    ) {
        var pendingBlockIdentities: [BlockIdentity] = [rootBlockIdentity];
        var visitedBlockIdentities: Set<BlockIdentity> = [];
        while let blockIdentity: BlockIdentity = pendingBlockIdentities.popLast() {
            if visitedBlockIdentities.contains(blockIdentity) {
                continue;
            }
            visitedBlockIdentities.insert(blockIdentity);
            subtreeBlockIdentities.append(blockIdentity);
            if let childBlockIdentities: [BlockIdentity] =
                childrenByParentIdentity[blockIdentity] {
                pendingBlockIdentities.append(contentsOf: childBlockIdentities);
            }
        }
    }

    static func directoryFileSizeAndPaths(
        directoryPath: URL
    ) throws -> (fileSizeBytes: UInt64, trackedFilePaths: [String]) {
        var pendingDirectories: [URL] = [directoryPath];
        var totalSizeBytes: UInt64 = 0;
        var trackedFilePaths: [String] = [];
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
                    continue;
                }
                let (fileByteCount, addOverflow) = totalSizeBytes.addingReportingOverflow(
                    try PersistentPromptCacheDiskStoreScan.cacheOwnedFileByteCount(
                        filePath: entryPath));
                if addOverflow {
                    throw PersistentPromptCacheDiskStoreError.globalPromptCacheSizeOverflow(
                        rootDirectory: directoryPath.path);
                }
                totalSizeBytes = fileByteCount;
                trackedFilePaths.append(entryPath.path);
            }
        }
        return (totalSizeBytes, trackedFilePaths);
    }

    private static func checkedAddGlobalSize(
        globalPromptCacheRootDirectory: URL,
        accumulatedSizeBytes: UInt64,
        additionalSizeBytes: UInt64
    ) throws -> UInt64 {
        let (addedBytes, overflow) = accumulatedSizeBytes.addingReportingOverflow(
            additionalSizeBytes);
        if overflow {
            throw PersistentPromptCacheDiskStoreError.globalPromptCacheSizeOverflow(
                rootDirectory: globalPromptCacheRootDirectory.path);
        }
        return addedBytes;
    }

    private static func blockDirectoryModifiedAt(
        directoryPath: URL
    ) throws -> Date {
        let fileAttributes: [FileAttributeKey: Any];
        do {
            fileAttributes = try FileManager.default.attributesOfItem(
                atPath: directoryPath.path);
        } catch {
            throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                blockFilePath: directoryPath.path, problem: String(describing: error));
        }
        return (fileAttributes[.modificationDate] as? Date) ?? Date(timeIntervalSince1970: 0);
    }

    private static func isStaleBlockStagingDirectory(
        directoryPath: URL
    ) -> Bool {
        return directoryPath.deletingLastPathComponent().lastPathComponent == "blocks"
            && directoryPath.lastPathComponent.contains(".staging-");
    }
}
