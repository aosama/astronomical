import Foundation

/// Decision reached when execution offers one materialized page to the cache.
public enum RetainedExpertLayerCommitOutcome: Equatable {

    /// Ownership transferred atomically with the exact byte delta.
    case committed(RetainedExpertLayerCommitDelta)

    /// The existing owner stayed because it already covers the need.
    case preservedExisting

    /// A live ceiling rejected the candidate; prior ownership untouched.
    case rejectedByCurrentCeiling
}
