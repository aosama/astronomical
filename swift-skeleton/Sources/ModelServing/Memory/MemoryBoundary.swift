import Foundation

/**
 * The memory boundary that prevented an operation from being admitted.
 *
 * Every admission decision in this package carries a boundary when it
 * rejects: the named wall that stopped the operation. Execution owners
 * surface that boundary to users and logs; they never invent a second
 * reason vocabulary.
 */
public enum MemoryBoundary: Equatable, Hashable, Sendable {

    /// Stable ownership would exceed the configured active-memory ceiling.
    case stableActiveCeiling

    /// Expected temporary work would exceed the approved transient allowance.
    case transientPeakAllowance

    /// One pending allocation cannot fit beside current active ownership.
    case allocationProjection

    /// Complete experts plus required request headroom cannot coexist.
    case completeResidency

    /// Retained expert payload exceeds the composed capacity left for experts.
    case retainedExpertPayload

    /// A requested live ceiling cannot preserve non-evictable ownership.
    case liveCeilingMinimum
}
