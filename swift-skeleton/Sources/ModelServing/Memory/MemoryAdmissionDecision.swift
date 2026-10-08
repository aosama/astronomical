import Foundation

/// Shared context-admission result: admit, demote complete experts first,
/// reclaim elastic paged payload, or reject at the named boundary.
public enum MemoryAdmissionDecision: Equatable, Hashable, Sendable {

    /// Projected active memory fits behind the ceiling.
    case admit

    /// The indivisible complete owner must demote before this request fits.
    case demoteCompleteResidency(reassessAfterDemotion: Bool)

    /// Elastic paged-expert payload can cover the named overflow.
    case reclaim(requiredBytes: UInt64)

    /// Even full reclamation cannot make the request fit.
    case reject(boundary: MemoryBoundary, shortfallBytes: UInt64)
}
