import Foundation;

import ModelServing;

/// The kind of one cache-owned file the store tracks.
enum PersistentPromptCacheFileKind {
    case sequenceStateBlock;
    case boundaryStateSnapshot;
    case visualEmbedding;
}

/// Valid enabled prompt-cache filesystem state for one active model under
/// one global quota, port of the Rust `PersistentPromptCacheDiskStoreConfig`.
public struct PersistentPromptCacheDiskStoreConfig: Sendable {

    let activeModelPromptCacheDirectory: URL;

    let globalPromptCacheRootDirectory: URL;

    let globalPromptCacheMaximumSizeBytes: UInt64;

    public init(
        activeModelPromptCacheDirectory: URL,
        globalPromptCacheRootDirectory: URL,
        globalPromptCacheMaximumSizeBytes: UInt64
    ) {
        self.activeModelPromptCacheDirectory = activeModelPromptCacheDirectory;
        self.globalPromptCacheRootDirectory = globalPromptCacheRootDirectory;
        self.globalPromptCacheMaximumSizeBytes = globalPromptCacheMaximumSizeBytes;
    }

    /// Derives an isolated cache namespace while retaining the shared global quota.
    public func forModel(modelId: String, modelRevision: String)
        -> PersistentPromptCacheDiskStoreConfig {
        return PersistentPromptCacheDiskStoreConfig(
            activeModelPromptCacheDirectory: self.globalPromptCacheRootDirectory
                .appendingPathComponent(modelId, isDirectory: true)
                .appendingPathComponent(modelRevision, isDirectory: true),
            globalPromptCacheRootDirectory: self.globalPromptCacheRootDirectory,
            globalPromptCacheMaximumSizeBytes: self.globalPromptCacheMaximumSizeBytes);
    }
}

/// Persistent, SSD-backed prompt-cache store, port of the Rust
/// `PersistentPromptCacheDiskStore`. Current-format state is published as
/// complete atomic block directories with manifest-bound ancestry and
/// contract-derived sequence and boundary files. The engine-facing
/// tensor load and direct-writer save paths land with the capture/restore
/// slice; this type owns the filesystem state, quota, retention, and
/// startup recovery around them.
public final class PersistentPromptCacheDiskStore {

    private static let BLOCKS_DIRECTORY_NAME: String = "blocks";
    private static let VISUAL_EMBEDDINGS_DIRECTORY_NAME: String = "visual_embeddings";

    let activeModelPromptCacheDirectory: URL;
    let blocksDirectory: URL;
    let visualEmbeddingsDirectory: URL;
    let globalPromptCacheRootDirectory: URL;
    let globalPromptCacheMaximumSizeBytes: UInt64;
    let modelContract: PersistentPromptCacheModelContract;

    let stateLock: NSLock;
    var trackedFiles: PersistentPromptCacheDiskStoreIndex;
    var globalPromptCacheTotalSizeBytes: UInt64;
    var globalVisualEmbeddingTotalSizeBytes: UInt64;
    var pendingStartupCleanupEvidence: PersistentPromptCacheStartupCleanupEvidence?;

    /// Opens (or creates) the prompt-cache directory and scans current-format files.
    ///
    /// Open order is a recovery protocol: establish trusted directories,
    /// rebuild the active-model index from valid committed artifacts, remove
    /// abandoned global transactions, then reconcile retention and quota.
    public static func open(
        diskStoreConfig: PersistentPromptCacheDiskStoreConfig,
        modelContract: PersistentPromptCacheModelContract
    ) throws -> PersistentPromptCacheDiskStore {
        let persistentPromptCacheDirectory: URL = diskStoreConfig.activeModelPromptCacheDirectory;
        let blocksDirectory: URL = persistentPromptCacheDirectory.appendingPathComponent(
            PersistentPromptCacheDiskStore.BLOCKS_DIRECTORY_NAME, isDirectory: true);
        let visualEmbeddingsDirectory: URL = persistentPromptCacheDirectory
            .appendingPathComponent(
                PersistentPromptCacheDiskStore.VISUAL_EMBEDDINGS_DIRECTORY_NAME,
                isDirectory: true);
        try PersistentPromptCacheGlobalQuota.preparePromptCacheDirectoryTree(
            globalPromptCacheRootDirectory: diskStoreConfig.globalPromptCacheRootDirectory,
            activeModelPromptCacheDirectory: persistentPromptCacheDirectory,
            activeModelStorageDirectories: [blocksDirectory, visualEmbeddingsDirectory]);
        var startupCleanupEvidence: PersistentPromptCacheStartupCleanupEvidence =
            PersistentPromptCacheStartupCleanupEvidence();
        try PersistentPromptCacheGlobalQuota.removeRetiredSpeculativePrefillCacheDirectories(
            activeModelPromptCacheDirectory: persistentPromptCacheDirectory,
            startupCleanupEvidence: &startupCleanupEvidence);
        var trackedFiles: PersistentPromptCacheDiskStoreIndex = PersistentPromptCacheDiskStoreIndex();
        try PersistentPromptCacheDiskStoreScan.scanCurrentFormatBlockDirectories(
            blocksDirectory: blocksDirectory,
            trackedFiles: &trackedFiles,
            modelContract: modelContract,
            startupCleanupEvidence: &startupCleanupEvidence);
        let diskStore: PersistentPromptCacheDiskStore = PersistentPromptCacheDiskStore(
            activeModelPromptCacheDirectory: persistentPromptCacheDirectory,
            blocksDirectory: blocksDirectory,
            visualEmbeddingsDirectory: visualEmbeddingsDirectory,
            globalPromptCacheRootDirectory: diskStoreConfig.globalPromptCacheRootDirectory,
            globalPromptCacheMaximumSizeBytes: diskStoreConfig.globalPromptCacheMaximumSizeBytes,
            modelContract: modelContract,
            trackedFiles: trackedFiles,
            startupCleanupEvidence: startupCleanupEvidence.intoNonEmpty());
        // Stale bytes must disappear before quota considers deleting useful
        // content. Retention reconciliation then protects the validated
        // active chain while evicting unrelated content and completing crash
        // cleanup.
        try diskStore.removeUnconditionallyReclaimableStartupArtifacts();
        try diskStore.reconcileStartupRetentionAndGlobalQuota();
        return diskStore;
    }

    private init(
        activeModelPromptCacheDirectory: URL,
        blocksDirectory: URL,
        visualEmbeddingsDirectory: URL,
        globalPromptCacheRootDirectory: URL,
        globalPromptCacheMaximumSizeBytes: UInt64,
        modelContract: PersistentPromptCacheModelContract,
        trackedFiles: PersistentPromptCacheDiskStoreIndex,
        startupCleanupEvidence: PersistentPromptCacheStartupCleanupEvidence?
    ) {
        self.activeModelPromptCacheDirectory = activeModelPromptCacheDirectory;
        self.blocksDirectory = blocksDirectory;
        self.visualEmbeddingsDirectory = visualEmbeddingsDirectory;
        self.globalPromptCacheRootDirectory = globalPromptCacheRootDirectory;
        self.globalPromptCacheMaximumSizeBytes = globalPromptCacheMaximumSizeBytes;
        self.modelContract = modelContract;
        self.stateLock = NSLock();
        self.trackedFiles = trackedFiles;
        self.globalPromptCacheTotalSizeBytes = 0;
        self.globalVisualEmbeddingTotalSizeBytes = 0;
        self.pendingStartupCleanupEvidence = startupCleanupEvidence;
    }

    public var blockTokenCount: Int {
        return self.modelContract.blockTokenCount;
    }

    public func sequenceStateBlockCount() -> Int {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.trackedFiles.sequenceStateBlockCount;
    }

    public func boundaryStateSnapshotCount() -> Int {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.trackedFiles.boundaryStateSnapshotCount;
    }

    public func visualEmbeddingCount() -> Int {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.trackedFiles.visualEmbeddingCount;
    }

    /// The tracked on-disk size of one sequence-state block file.
    public func sequenceStateBlockFileSizeBytes(blockHash: Data) -> UInt64? {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.trackedFiles.file(sequenceState: true, fileHash: blockHash)?
            .fileSizeBytes;
    }

    /// The tracked on-disk size of one recurrent-boundary snapshot file.
    public func recurrentSnapshotFileSizeBytes(blockHash: Data) -> UInt64? {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.trackedFiles.file(sequenceState: false, fileHash: blockHash)?
            .fileSizeBytes;
    }

    /// Whether one tracked sequence-state file is still present; a vanished
    /// file untracks itself and subtracts its bytes immediately.
    public func hasKvBlock(blockHash: Data) -> Bool {
        return self.trackedFileStillExists(sequenceState: true, fileHash: blockHash);
    }

    /// Whether one tracked recurrent-boundary snapshot is still present.
    public func hasRecurrentSnapshot(blockHash: Data) -> Bool {
        return self.trackedFileStillExists(sequenceState: false, fileHash: blockHash);
    }

    public func totalSizeBytes() -> UInt64 {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.globalPromptCacheTotalSizeBytes;
    }

    public func visualEmbeddingTotalSizeBytes() -> UInt64 {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.globalVisualEmbeddingTotalSizeBytes;
    }

    public func startupCleanupEvidence() -> PersistentPromptCacheStartupCleanupEvidence? {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.pendingStartupCleanupEvidence;
    }

    public func takeStartupCleanupEvidence() -> PersistentPromptCacheStartupCleanupEvidence? {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        let takenEvidence: PersistentPromptCacheStartupCleanupEvidence? =
            self.pendingStartupCleanupEvidence;
        self.pendingStartupCleanupEvidence = nil;
        return takenEvidence;
    }

    func recordStartupCleanupEvidence(
        _ additionalEvidence: PersistentPromptCacheStartupCleanupEvidence
    ) {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        if additionalEvidence.isEmpty() {
            return;
        }
        if var existingEvidence: PersistentPromptCacheStartupCleanupEvidence =
            self.pendingStartupCleanupEvidence {
            existingEvidence.merge(additionalEvidence);
            self.pendingStartupCleanupEvidence = existingEvidence;
        } else {
            self.pendingStartupCleanupEvidence = additionalEvidence;
        }
    }

    private func trackedFileStillExists(
        sequenceState: Bool, fileHash: Data
    ) -> Bool {
        let trackedFilePath: String?;
        self.stateLock.lock();
        trackedFilePath = self.trackedFiles.file(
            sequenceState: sequenceState, fileHash: fileHash)?.filePath;
        self.stateLock.unlock();
        guard let trackedFilePath: String = trackedFilePath else {
            return false;
        }
        // `NotFound` is authoritative and updates telemetry immediately.
        // Other metadata errors are left for the actual load path to report
        // with full typed context rather than turning a permission fault
        // into a cache miss.
        if FileManager.default.fileExists(atPath: trackedFilePath) {
            return true;
        }
        self.untrackFileAndSubtractGlobalAccounting(
            sequenceState: sequenceState, fileHash: fileHash);
        return false;
    }

    func untrackFileAndSubtractGlobalAccounting(
        sequenceState: Bool, fileHash: Data
    ) {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        guard let removedTrackedFile: PersistentPromptCacheDiskStoreIndex.TrackedFile =
            sequenceState
            ? self.trackedFiles.block(blockHash: fileHash)?.sequenceStateFile
            : self.trackedFiles.block(blockHash: fileHash)?.boundaryStateFile
        else {
            return;
        }
        self.trackedFiles.removeBlock(blockHash: fileHash);
        self.globalPromptCacheTotalSizeBytes = self.globalPromptCacheTotalSizeBytes
            .subtractingReportingOverflow(removedTrackedFile.fileSizeBytes).partialValue;
        if sequenceState == false {
            self.globalVisualEmbeddingTotalSizeBytes = self
                .globalVisualEmbeddingTotalSizeBytes
                .subtractingReportingOverflow(removedTrackedFile.fileSizeBytes).partialValue;
        }
    }

    /// Serializes deletion against publication and keeps the active index valid.
    public func clearPromptCache(modelId: String?) throws
        -> PersistentPromptCacheClearOutcome {
        self.stateLock.lock();
        let clearTargetDirectory: URL? = modelId.map(
            { (modelId: String) -> URL in
                return self.globalPromptCacheRootDirectory.appendingPathComponent(modelId);
            });
        let activeModelCacheWasCleared: Bool = clearTargetDirectory.map(
            { (targetDirectory: URL) -> Bool in
                return self.activeModelPromptCacheDirectory.path
                    .hasPrefix(targetDirectory.path);
            }) ?? true;
        self.stateLock.unlock();
        let clearOutcome: PersistentPromptCacheClearOutcome = try
            PersistentPromptCacheDirectoryClear.clearPersistentPromptCacheDirectory(
                globalPromptCacheRootDirectory: self.globalPromptCacheRootDirectory,
                modelId: modelId);
        if activeModelCacheWasCleared {
            try self.prepareActiveModelStorageDirectories();
        }
        try self.refreshGlobalPromptCacheAccounting();
        if activeModelCacheWasCleared {
            self.stateLock.lock();
            defer { self.stateLock.unlock(); }
            self.trackedFiles = PersistentPromptCacheDiskStoreIndex();
        }
        return clearOutcome;
    }

    func prepareActiveModelStorageDirectories() throws {
        try PersistentPromptCacheGlobalQuota.preparePromptCacheDirectoryTree(
            globalPromptCacheRootDirectory: self.globalPromptCacheRootDirectory,
            activeModelPromptCacheDirectory: self.activeModelPromptCacheDirectory,
            activeModelStorageDirectories: [
                self.blocksDirectory, self.visualEmbeddingsDirectory,
            ]);
    }

    /// Interrupted transactions and obsolete formats have no readable
    /// current-format value, so cleanup is not pressure-dependent.
    func removeUnconditionallyReclaimableStartupArtifacts() throws {
        let globalQuotaScan: PersistentPromptCacheQuotaScan = try
            PersistentPromptCacheQuotaScanEngine.scanGlobalPromptCacheQuota(
                globalPromptCacheRootDirectory: self.globalPromptCacheRootDirectory,
                excludedDirectory: nil);
        var startupCleanupEvidence: PersistentPromptCacheStartupCleanupEvidence =
            PersistentPromptCacheStartupCleanupEvidence();
        for staleTransactionArtifact: PersistentPromptCacheEvictionCandidate in globalQuotaScan
            .evictionCandidatesOldestWrittenFirst
            .filter({ (candidate: PersistentPromptCacheEvictionCandidate) -> Bool in
                return candidate.isUnconditionallyRemovable;
            }) {
            guard let cleanupClassification: PersistentPromptCacheEvictionCandidate
                .CleanupClassification = staleTransactionArtifact
                .unconditionalCleanupClassification else {
                continue;
            }
            try Self.removeEvictionCandidate(staleTransactionArtifact);
            Self.recordRemovedStartupCandidate(
                startupCleanupEvidence: &startupCleanupEvidence,
                cleanupClassification: cleanupClassification,
                removedCandidate: staleTransactionArtifact);
            self.stateLock.lock();
            self.trackedFiles.removeFilesByPath(
                filePaths: staleTransactionArtifact.trackedFilePaths);
            self.trackedFiles.removeBlocksByDirectoryPaths(
                directoryPaths: staleTransactionArtifact.blockDirectoryPaths);
            self.stateLock.unlock();
        }
        self.recordStartupCleanupEvidence(startupCleanupEvidence);
        try self.refreshGlobalPromptCacheAccounting();
    }

    /// Atomic counters are telemetry snapshots, not the source of truth:
    /// re-scan after recovery or rollback because another cleanup path may
    /// have changed disk state without incrementally updating every counter.
    func refreshGlobalPromptCacheAccounting() throws {
        let globalQuotaScan: PersistentPromptCacheQuotaScan = try
            PersistentPromptCacheQuotaScanEngine.scanGlobalPromptCacheQuota(
                globalPromptCacheRootDirectory: self.globalPromptCacheRootDirectory,
                excludedDirectory: nil);
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        self.globalPromptCacheTotalSizeBytes = globalQuotaScan.totalSizeBytes;
        self.globalVisualEmbeddingTotalSizeBytes = globalQuotaScan
            .visualEmbeddingTotalSizeBytes;
    }

}
