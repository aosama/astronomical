//! Beside-source unit tests for the allocator-cache reclaim threshold policy.
//!
//! The threshold boundary (exactly-at-threshold counts as reclaimable) is the
//! contract the prefill cadence depends on; these tests pin it so a future
//! refactor of the comparison cannot quietly change the reclamation
//! behavior.

use crate::mlx_runtime_types::{
    ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES, allocator_cache_exceeds_reclaim_threshold,
};

#[test]
fn should_leave_small_allocator_cache_pooled_after_a_prefill_chunk() {
    assert!(!allocator_cache_exceeds_reclaim_threshold(
        ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES - 1,
        ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES,
    ));
}

#[test]
fn should_reclaim_allocator_cache_at_the_policy_threshold() {
    assert!(allocator_cache_exceeds_reclaim_threshold(
        ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES,
        ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES,
    ));
}
