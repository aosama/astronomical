//! Admission: does one pending allocation fit beside current ownership?
//!
//! This module decides what should happen. Callers remain responsible for
//! synchronizing streams, clearing allocator storage, and performing allocations.

use crate::vocabulary::MemoryBoundary;

/// One internally consistent observation at an allocation boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationAdmissionObservation {
    /// Live MLX active bytes before the pending allocation.
    pub active_memory_bytes: u64,
    /// Reclaimable allocator storage, excluded from active-limit enforcement.
    pub mlx_buffer_cache_bytes: u64,
    /// Exact byte count of the allocation about to be created.
    pub pending_allocation_bytes: u64,
    /// Worker-resolved stable MLX active-memory ceiling.
    pub active_memory_ceiling_bytes: u64,
}

impl AllocationAdmissionObservation {
    #[must_use]
    pub const fn new(
        active_memory_bytes: u64,
        mlx_buffer_cache_bytes: u64,
        pending_allocation_bytes: u64,
        active_memory_ceiling_bytes: u64,
    ) -> Self {
        Self {
            active_memory_bytes,
            mlx_buffer_cache_bytes,
            pending_allocation_bytes,
            active_memory_ceiling_bytes,
        }
    }

    /// Decides active-memory fit independently from MLX buffer-cache ownership.
    #[must_use]
    pub const fn decide(self) -> AllocationAdmissionDecision {
        let Some(projected_active_memory_bytes) = self
            .active_memory_bytes
            .checked_add(self.pending_allocation_bytes)
        else {
            return AllocationAdmissionDecision::Reject {
                boundary: MemoryBoundary::AllocationProjection,
                shortfall_bytes: u64::MAX,
            };
        };
        if projected_active_memory_bytes > self.active_memory_ceiling_bytes {
            return AllocationAdmissionDecision::Reject {
                boundary: MemoryBoundary::AllocationProjection,
                shortfall_bytes: projected_active_memory_bytes
                    .saturating_sub(self.active_memory_ceiling_bytes),
            };
        }
        let total_memory_after_allocation =
            projected_active_memory_bytes.checked_add(self.mlx_buffer_cache_bytes);
        if self.mlx_buffer_cache_bytes > 0
            && match total_memory_after_allocation {
                Some(total_memory_bytes) => total_memory_bytes > self.active_memory_ceiling_bytes,
                None => true,
            }
        {
            return AllocationAdmissionDecision::ClearMlxBufferCacheThenAdmit;
        }
        AllocationAdmissionDecision::Admit
    }
}

/// Typed policy result for one pending MLX allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationAdmissionDecision {
    /// Active and total memory both fit without cleanup.
    Admit,
    /// Active ownership fits, but reclaiming allocator storage avoids total pressure.
    ClearMlxBufferCacheThenAdmit,
    /// Active ownership cannot fit even if allocator storage is cleared.
    Reject {
        boundary: MemoryBoundary,
        shortfall_bytes: u64,
    },
}
