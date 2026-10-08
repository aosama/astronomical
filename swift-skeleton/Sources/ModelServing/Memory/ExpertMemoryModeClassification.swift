import Foundation

import IpcProtocol

/// Single classification of expert ownership for status and execution.
///
/// Families supply whether a complete owner exists, whether a pager exists,
/// and how many paged bytes are retained. They must not invent a second
/// hybrid / paged / resident answer.
public enum ExpertMemoryModeClassification {

    /// Classifies expert RAM ownership from structural facts.
    public static func classify(
        completeSparseOwnerIsInstalled: Bool,
        sparseExpertPagingIsConfigured: Bool,
        retainedPagedExpertPayloadBytes: UInt64
    ) -> ExpertMemoryMode {
        if completeSparseOwnerIsInstalled || !sparseExpertPagingIsConfigured {
            return .resident
        }
        if retainedPagedExpertPayloadBytes > 0 {
            return .hybrid
        }
        return .paged
    }
}
