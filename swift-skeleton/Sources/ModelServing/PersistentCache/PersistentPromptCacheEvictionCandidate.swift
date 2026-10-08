import Foundation;

/// Typed eviction units produced by the global cache scan, port of the
/// Rust `disk_store_global_quota_candidate`. Committed prompt-cache blocks
/// are never represented as independent files: they are grouped into
/// ancestry-closed subtrees so eviction cannot leave a descendant whose
/// required sequence-state parent has disappeared.
public enum PersistentPromptCacheEvictionCandidate: Sendable {

    /// The per-reason category a cleanup reclamation is recorded under.
    public enum CleanupClassification: Equatable, Sendable {
        case interruptedTransactionRecovery;
        case obsoleteFormat;
        case quotaEviction;
    }

    public enum StandaloneFileClassification: Equatable, Sendable {
        case abandonedTransaction;
        case obsoleteFormat;
        case committedArtifact;
    }

    /// A non-block artifact such as a visual embedding or legacy temporary file.
    case standaloneFile(
        filePath: String,
        fileSizeBytes: UInt64,
        modifiedAt: Date,
        isVisualEmbedding: Bool,
        cleanupClassification: StandaloneFileClassification);

    /// An abandoned `.staging-*` transaction, always removable before content.
    case staleDirectory(
        directoryPath: String,
        fileSizeBytes: UInt64,
        modifiedAt: Date,
        trackedFilePaths: [String]);

    /// One committed block and every descendant in the same storage namespace.
    case blockSubtree(
        rootBlockDirectoryPath: String,
        fileSizeBytes: UInt64,
        modifiedAt: Date,
        blockDirectoryPaths: [String],
        trackedFilePaths: [String]);

    /// The modification timestamp ordering evictions least-recent-use first.
    var modifiedAt: Date {
        switch self {
        case .standaloneFile(_, _, let modifiedAt, _, _): return modifiedAt;
        case .staleDirectory(_, _, let modifiedAt, _): return modifiedAt;
        case .blockSubtree(_, _, let modifiedAt, _, _): return modifiedAt;
        }
    }

    /// Filesystem timestamps may have coarse resolution; a stable path tie
    /// breaker makes eviction deterministic across scans of identical state.
    var tieBreakerPath: String {
        switch self {
        case .standaloneFile(let filePath, _, _, _, _): return filePath;
        case .staleDirectory(let directoryPath, _, _, _): return directoryPath;
        case .blockSubtree(let rootBlockDirectoryPath, _, _, _, _): return rootBlockDirectoryPath;
        }
    }

    /// The reason an unconditionally removable candidate records, when its
    /// classification proves it was never valid committed content.
    var unconditionalCleanupClassification: CleanupClassification? {
        switch self {
        case .standaloneFile(_, _, _, _, let cleanupClassification):
            switch cleanupClassification {
            case .abandonedTransaction: return .interruptedTransactionRecovery;
            case .obsoleteFormat: return .obsoleteFormat;
            case .committedArtifact: return nil;
            }
        case .staleDirectory: return .interruptedTransactionRecovery;
        case .blockSubtree: return nil;
        }
    }

    var isUnconditionallyRemovable: Bool {
        return self.unconditionalCleanupClassification != nil;
    }

    var removedArtifactCount: Int {
        if case .standaloneFile = self {
            return 1;
        }
        return 0;
    }

    var removedBlockCount: Int {
        switch self {
        case .standaloneFile: return 0;
        case .staleDirectory: return 1;
        case .blockSubtree(_, _, _, let blockDirectoryPaths, _): return blockDirectoryPaths.count;
        }
    }

    var fileSizeBytes: UInt64 {
        switch self {
        case .standaloneFile(_, let fileSizeBytes, _, _, _): return fileSizeBytes;
        case .staleDirectory(_, let fileSizeBytes, _, _): return fileSizeBytes;
        case .blockSubtree(_, let fileSizeBytes, _, _, _): return fileSizeBytes;
        }
    }

    var visualEmbeddingSizeBytes: UInt64 {
        guard case let .standaloneFile(_, fileSizeBytes, _, isVisualEmbedding, _) = self,
            isVisualEmbedding else {
            return 0;
        }
        return fileSizeBytes;
    }

    var trackedFilePaths: [String] {
        switch self {
        case .standaloneFile(let filePath, _, _, _, _): return [filePath];
        case .staleDirectory(_, _, _, let trackedFilePaths): return trackedFilePaths;
        case .blockSubtree(_, _, _, _, let trackedFilePaths): return trackedFilePaths;
        }
    }

    var blockDirectoryPaths: [String] {
        guard case .blockSubtree(_, _, _, let blockDirectoryPaths, _) = self else {
            return [];
        }
        return blockDirectoryPaths;
    }

    /// Protects the whole candidate when any member intersects the active
    /// chain: deleting only its unprotected members would violate subtree
    /// atomicity.
    func containsProtectedBlockDirectory(
        protectedBlockDirectoryPaths: [String]
    ) -> Bool {
        guard case .blockSubtree(_, _, _, let blockDirectoryPaths, _) = self else {
            return false;
        }
        return blockDirectoryPaths.contains(where: { (blockDirectoryPath: String) -> Bool in
            return protectedBlockDirectoryPaths.contains(blockDirectoryPath);
        });
    }
}
