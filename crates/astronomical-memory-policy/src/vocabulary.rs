//! Shared vocabulary for cross-component memory-policy decisions.
//!
//! Every admission decision in this package carries a `MemoryBoundary` when it
//! rejects: the named wall that stopped the operation plus the byte shortfall.
//! Execution owners surface that boundary to users and logs; they never invent
//! a second reason vocabulary.

/// The memory boundary that prevented an operation from being admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryBoundary {
    /// Stable ownership would exceed the configured active-memory ceiling.
    StableActiveCeiling,
    /// Expected temporary work would exceed the approved transient allowance.
    TransientPeakAllowance,
    /// One pending allocation cannot fit beside current active ownership.
    AllocationProjection,
    /// Complete experts plus required request headroom cannot coexist.
    CompleteResidency,
    /// Retained expert payload exceeds the composed capacity left for experts.
    RetainedExpertPayload,
    /// A requested live ceiling cannot preserve non-evictable ownership.
    LiveCeilingMinimum,
}

/// Common request admission result consumed by execution owners.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryAdmissionDecision {
    /// The operation may proceed without changing expert ownership.
    Admit,
    /// Paged experts must release at least the stated number of bytes.
    Reclaim { required_bytes: u64 },
    /// Complete experts are indivisible; execution must demote and reassess.
    DemoteCompleteResidency { reassess_after_demotion: bool },
    /// No legal expert transition can satisfy the named boundary.
    Reject {
        boundary: MemoryBoundary,
        shortfall_bytes: u64,
    },
}
