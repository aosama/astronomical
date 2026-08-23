use astronomical_runtime_integration::{
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
