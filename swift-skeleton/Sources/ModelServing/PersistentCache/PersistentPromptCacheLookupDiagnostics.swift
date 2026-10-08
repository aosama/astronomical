import Foundation;

/// Evidence gathered while looking for the longest restorable persistent
/// prompt-cache prefix, port of the Rust
/// `PersistentPromptCacheLookupDiagnostics`.
public struct PersistentPromptCacheLookupDiagnostics: Equatable, Sendable {

    private let completePromptBlockCountValue: Int;
    private let maximumRestorableBlockCountValue: Int;
    private var matchedSequenceStateBlockCountValue: Int;
    private var firstMissingSequenceStateBlockIndexValue: Int?;
    private var firstMissingSequenceStateBlockHashValue: Data?;
    private var newestBoundaryStateSnapshotBlockIndexValue: Int?;
    private var restoredPartialTailBlockTokenCountValue: Int?;
    private var missReasonValue: PersistentPromptCacheMissReason?;

    init(
        completePromptBlockCount: Int,
        maximumRestorableBlockCount: Int
    ) {
        self.completePromptBlockCountValue = completePromptBlockCount;
        self.maximumRestorableBlockCountValue = maximumRestorableBlockCount;
        self.matchedSequenceStateBlockCountValue = 0;
        self.firstMissingSequenceStateBlockIndexValue = nil;
        self.firstMissingSequenceStateBlockHashValue = nil;
        self.newestBoundaryStateSnapshotBlockIndexValue = nil;
        self.restoredPartialTailBlockTokenCountValue = nil;
        self.missReasonValue = nil;
    }

    /// The number of complete persistent prompt-cache blocks in the prompt.
    public var completePromptBlockCount: Int {
        return self.completePromptBlockCountValue;
    }

    /// The number of complete blocks that are eligible for restore.
    public var maximumRestorableBlockCount: Int {
        return self.maximumRestorableBlockCountValue;
    }

    /// How many contiguous KV blocks matched before lookup stopped.
    public var matchedSequenceStateBlockCount: Int {
        return self.matchedSequenceStateBlockCountValue;
    }

    /// The first missing KV block position, when lookup stopped at a gap.
    public var firstMissingSequenceStateBlockIndex: Int? {
        return self.firstMissingSequenceStateBlockIndexValue;
    }

    /// The expected block hash for the first missing KV block, when available.
    public var firstMissingSequenceStateBlockHash: Data? {
        return self.firstMissingSequenceStateBlockHashValue;
    }

    /// The recurrent snapshot position used as the restore boundary.
    public var newestBoundaryStateSnapshotBlockIndex: Int? {
        return self.newestBoundaryStateSnapshotBlockIndexValue;
    }

    /// The token count of the restored partial tail block, when one matched.
    public var restoredPartialTailBlockTokenCount: Int? {
        return self.restoredPartialTailBlockTokenCountValue;
    }

    /// Why lookup failed, when it failed.
    public var missReason: PersistentPromptCacheMissReason? {
        return self.missReasonValue;
    }

    internal func recordingMatchedSequenceStateBlockCount(
        _ matchedSequenceStateBlockCount: Int
    ) -> PersistentPromptCacheLookupDiagnostics {
        var updated = self;
        updated.matchedSequenceStateBlockCountValue = matchedSequenceStateBlockCount;
        return updated;
    }

    internal func recordingFirstMissingSequenceStateBlock(
        index missingSequenceStateBlockIndex: Int,
        hash missingSequenceStateBlockHash: Data?
    ) -> PersistentPromptCacheLookupDiagnostics {
        var updated = self;
        updated.firstMissingSequenceStateBlockIndexValue = missingSequenceStateBlockIndex;
        updated.firstMissingSequenceStateBlockHashValue = missingSequenceStateBlockHash;
        return updated;
    }

    internal func recordingNewestBoundaryStateSnapshotBlockIndex(
        _ boundaryStateSnapshotBlockIndex: Int
    ) -> PersistentPromptCacheLookupDiagnostics {
        var updated = self;
        updated.newestBoundaryStateSnapshotBlockIndexValue = boundaryStateSnapshotBlockIndex;
        return updated;
    }

    internal func recordingRestoredPartialTailBlockTokenCount(
        _ restoredPartialTailBlockTokenCount: Int
    ) -> PersistentPromptCacheLookupDiagnostics {
        var updated = self;
        updated.restoredPartialTailBlockTokenCountValue = restoredPartialTailBlockTokenCount;
        return updated;
    }

    internal func recordingMissReason(
        _ missReason: PersistentPromptCacheMissReason
    ) -> PersistentPromptCacheLookupDiagnostics {
        var updated = self;
        updated.missReasonValue = missReason;
        return updated;
    }
}
