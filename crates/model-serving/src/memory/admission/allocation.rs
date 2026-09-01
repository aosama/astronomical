//! Admission: does one pending allocation fit beside current ownership?
//!
//! This module decides what should happen. Callers remain responsible for
//! synchronizing streams, clearing allocator storage, and performing allocations.

use crate::memory::MemoryBoundary;

/// One internally consistent observation at an allocation boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationAdmissionObservation {
    /// Live MLX active bytes before the pending allocation.
    pub active_memory_bytes: u64,
    /// Reclaimable allocator storage, excluded from active-limit enforcement.
    pub allocator_cache_bytes: u64,
    /// Exact byte count of the allocation about to be created.
    pub pending_allocation_bytes: u64,
    /// Worker-resolved stable MLX active-memory ceiling.
    pub active_memory_ceiling_bytes: u64,
}

impl AllocationAdmissionObservation {
    #[must_use]
    pub const fn new(
        active_memory_bytes: u64,
        allocator_cache_bytes: u64,
        pending_allocation_bytes: u64,
        active_memory_ceiling_bytes: u64,
    ) -> Self {
        Self {
            active_memory_bytes,
            allocator_cache_bytes,
            pending_allocation_bytes,
            active_memory_ceiling_bytes,
        }
    }

    /// Decides active-memory fit independently from allocator-cache ownership.
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
            projected_active_memory_bytes.checked_add(self.allocator_cache_bytes);
        if self.allocator_cache_bytes > 0
            && match total_memory_after_allocation {
                Some(total_memory_bytes) => total_memory_bytes > self.active_memory_ceiling_bytes,
                None => true,
            }
        {
            return AllocationAdmissionDecision::ClearAllocatorCacheThenAdmit;
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
    ClearAllocatorCacheThenAdmit,
    /// Active ownership cannot fit even if allocator storage is cleared.
    Reject {
        boundary: MemoryBoundary,
        shortfall_bytes: u64,
    },
}

/// Computes retained-expert capacity while preserving future page and transient work.
#[must_use]
pub const fn retained_expert_payload_capacity_bytes(
    active_memory_bytes: u64,
    active_memory_ceiling_bytes: u64,
    maximum_expert_page_bytes: u64,
    pending_allocation_bytes: u64,
    observed_transient_high_water_bytes: u64,
    current_retained_expert_payload_bytes: u64,
    pending_retained_expert_payload_bytes: u64,
) -> u64 {
    let future_page_reserve_bytes = if pending_retained_expert_payload_bytes == 0 {
        if pending_allocation_bytes > maximum_expert_page_bytes {
            pending_allocation_bytes
        } else {
            maximum_expert_page_bytes
        }
    } else {
        let Some(combined_reserve_bytes) =
            pending_allocation_bytes.checked_add(maximum_expert_page_bytes)
        else {
            return 0;
        };
        combined_reserve_bytes
    };
    let Some(post_load_retained_payload_bytes) =
        current_retained_expert_payload_bytes.checked_add(pending_retained_expert_payload_bytes)
    else {
        return 0;
    };
    let live_reserved_bytes = match active_memory_bytes.checked_add(future_page_reserve_bytes) {
        Some(live_reserved_bytes) => live_reserved_bytes,
        None => u64::MAX,
    };
    let effective_ceiling_bytes =
        active_memory_ceiling_bytes.saturating_sub(observed_transient_high_water_bytes);
    if live_reserved_bytes <= effective_ceiling_bytes {
        post_load_retained_payload_bytes
            .saturating_add(effective_ceiling_bytes - live_reserved_bytes)
    } else {
        post_load_retained_payload_bytes
            .saturating_sub(live_reserved_bytes - effective_ceiling_bytes)
    }
}
