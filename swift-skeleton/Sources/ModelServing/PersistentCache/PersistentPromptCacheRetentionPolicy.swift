import Foundation;

/// Sparse boundary-retention policy for branched prompt reuse, port of the
/// Rust `retention_policy`. Sequence state remains append-only at every
/// block. Boundary state is larger fixed restart state: keep the root and
/// every stride-th boundary so common-prefix branches retain useful restart
/// points while linear parents can be compacted.
enum PersistentPromptCacheRetentionPolicy {

    /// Returns whether a block-boundary snapshot remains available for
    /// branched prompts. The stride is validated as positive when the
    /// immutable storage contract is resolved; keeping it contract-owned
    /// makes retention, startup reconciliation, and topology validation
    /// apply exactly the same branch restart policy.
    static func boundaryIsCommonPrefixCheckpoint(
        blockIndex: UInt32,
        commonPrefixCheckpointStrideBlocks: UInt32
    ) -> Bool {
        if blockIndex == 0 {
            return true;
        }
        let (nextBlockIndex, overflow) = blockIndex.addingReportingOverflow(1);
        if overflow {
            return false;
        }
        return commonPrefixCheckpointStrideBlocks != 0
            && nextBlockIndex % commonPrefixCheckpointStrideBlocks == 0;
    }
}
