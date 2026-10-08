import Foundation;

/// Count and byte total for one startup-cleanup reason, port of the Rust
/// `PersistentPromptCacheStartupCleanupCategory`.
public struct PersistentPromptCacheStartupCleanupCategory: Equatable, Sendable {

    public private(set) var artifactCount: UInt64;

    public private(set) var blockCount: UInt64;

    public private(set) var byteCount: UInt64;

    public init() {
        self.artifactCount = 0;
        self.blockCount = 0;
        self.byteCount = 0;
    }

    mutating func recordArtifact(removedByteCount: UInt64) {
        self.artifactCount = self.artifactCount &+ 1;
        self.byteCount = self.byteCount &+ removedByteCount;
    }

    mutating func recordBlock(removedByteCount: UInt64) {
        self.blockCount = self.blockCount &+ 1;
        self.byteCount = self.byteCount &+ removedByteCount;
    }

    mutating func recordBlocks(removedBlockCount: Int, removedByteCount: UInt64) {
        self.blockCount = self.blockCount &+ UInt64(max(removedBlockCount, 0));
        self.byteCount = self.byteCount &+ removedByteCount;
    }

    /// Adds one candidate's total bytes to this category after its artifacts
    /// and blocks have been counted separately.
    mutating func addRemovedBytes(_ removedByteCount: UInt64) {
        self.byteCount = self.byteCount &+ removedByteCount;
    }

    /// Counts one removed eviction candidate's artifacts and blocks and adds
    /// its total bytes once.
    mutating func recordCandidate(
        removedArtifactCount: Int, removedBlockCount: Int, removedByteCount: UInt64
    ) {
        for _ in 0..<max(removedArtifactCount, 0) {
            self.recordArtifact(removedByteCount: 0);
        }
        self.recordBlocks(
            removedBlockCount: removedBlockCount, removedByteCount: 0);
        self.addRemovedBytes(removedByteCount);
    }

    func isEmpty() -> Bool {
        return self.artifactCount == 0 && self.blockCount == 0 && self.byteCount == 0;
    }

    mutating func merge(_ additionalCategory: PersistentPromptCacheStartupCleanupCategory) {
        self.artifactCount = self.artifactCount &+ additionalCategory.artifactCount;
        self.blockCount = self.blockCount &+ additionalCategory.blockCount;
        self.byteCount = self.byteCount &+ additionalCategory.byteCount;
    }
}

/// Bounded reason-separated evidence for cleanup performed while opening
/// the cache, retained until the first structural zero-restoration miss.
/// Port of the Rust `PersistentPromptCacheStartupCleanupEvidence`.
public struct PersistentPromptCacheStartupCleanupEvidence: Equatable, Sendable {

    public private(set) var interruptedTransactionRecovery:
        PersistentPromptCacheStartupCleanupCategory;

    public private(set) var obsoleteFormat: PersistentPromptCacheStartupCleanupCategory;

    public private(set) var corruptCurrentFormat: PersistentPromptCacheStartupCleanupCategory;

    public private(set) var quotaEviction: PersistentPromptCacheStartupCleanupCategory;

    public init() {
        self.interruptedTransactionRecovery = PersistentPromptCacheStartupCleanupCategory();
        self.obsoleteFormat = PersistentPromptCacheStartupCleanupCategory();
        self.corruptCurrentFormat = PersistentPromptCacheStartupCleanupCategory();
        self.quotaEviction = PersistentPromptCacheStartupCleanupCategory();
    }

    mutating func merge(_ additionalEvidence: PersistentPromptCacheStartupCleanupEvidence) {
        self.interruptedTransactionRecovery.merge(
            additionalEvidence.interruptedTransactionRecovery);
        self.obsoleteFormat.merge(additionalEvidence.obsoleteFormat);
        self.corruptCurrentFormat.merge(additionalEvidence.corruptCurrentFormat);
        self.quotaEviction.merge(additionalEvidence.quotaEviction);
    }

    /// The cleanup reasons the startup scan distinguishes.
    enum Reason {
        case interruptedTransactionRecovery;
        case obsoleteFormat;
        case corruptCurrentFormat;
        case quotaEviction;
    }

    mutating func recordArtifact(
        reason: Reason, removedByteCount: UInt64
    ) {
        switch reason {
        case .interruptedTransactionRecovery:
            self.interruptedTransactionRecovery.recordArtifact(removedByteCount: removedByteCount);
        case .obsoleteFormat:
            self.obsoleteFormat.recordArtifact(removedByteCount: removedByteCount);
        case .corruptCurrentFormat:
            self.corruptCurrentFormat.recordArtifact(removedByteCount: removedByteCount);
        case .quotaEviction:
            self.quotaEviction.recordArtifact(removedByteCount: removedByteCount);
        }
    }

    mutating func recordBlock(
        reason: Reason, removedByteCount: UInt64
    ) {
        switch reason {
        case .interruptedTransactionRecovery:
            self.interruptedTransactionRecovery.recordBlock(removedByteCount: removedByteCount);
        case .obsoleteFormat:
            self.obsoleteFormat.recordBlock(removedByteCount: removedByteCount);
        case .corruptCurrentFormat:
            self.corruptCurrentFormat.recordBlock(removedByteCount: removedByteCount);
        case .quotaEviction:
            self.quotaEviction.recordBlock(removedByteCount: removedByteCount);
        }
    }

    /// Records one removed eviction candidate under its cleanup reason,
    /// counting its artifacts and blocks separately and adding its total
    /// bytes once, port of the Rust `record_removed_startup_candidate`.
    mutating func recordCandidate(
        reason: Reason, removedArtifactCount: Int, removedBlockCount: Int,
        removedByteCount: UInt64
    ) {
        switch reason {
        case .interruptedTransactionRecovery:
            self.interruptedTransactionRecovery.recordCandidate(
                removedArtifactCount: removedArtifactCount,
                removedBlockCount: removedBlockCount,
                removedByteCount: removedByteCount);
        case .obsoleteFormat:
            self.obsoleteFormat.recordCandidate(
                removedArtifactCount: removedArtifactCount,
                removedBlockCount: removedBlockCount,
                removedByteCount: removedByteCount);
        case .corruptCurrentFormat:
            self.corruptCurrentFormat.recordCandidate(
                removedArtifactCount: removedArtifactCount,
                removedBlockCount: removedBlockCount,
                removedByteCount: removedByteCount);
        case .quotaEviction:
            self.quotaEviction.recordCandidate(
                removedArtifactCount: removedArtifactCount,
                removedBlockCount: removedBlockCount,
                removedByteCount: removedByteCount);
        }
    }

    func intoNonEmpty() -> PersistentPromptCacheStartupCleanupEvidence? {
        return self.isEmpty() ? nil : self;
    }

    func isEmpty() -> Bool {
        return self.interruptedTransactionRecovery.isEmpty()
            && self.obsoleteFormat.isEmpty()
            && self.corruptCurrentFormat.isEmpty()
            && self.quotaEviction.isEmpty();
    }
}
