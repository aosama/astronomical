import Foundation

/// Typed sequencing advice for a live ceiling change.
public enum MemoryCeilingChangeDecision: Equatable, Hashable, Sendable {

    /// The requested ceiling already matches the installed one.
    case unchanged

    /// Raising lets MLX accept capacity before the budget is published.
    case raise(mayAttemptCompleteResidency: Bool)

    /// Lowering reclaims before MLX enforces the smaller limit.
    case lower(
        mustDemoteCompleteResidency: Bool,
        retainedPagedExpertReclamationBytes: UInt64)

    /// The requested ceiling cannot preserve non-evictable ownership.
    case reject(boundary: MemoryBoundary, shortfallBytes: UInt64)
}
