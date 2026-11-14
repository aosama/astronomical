use astronomical_model_serving::{
    AllocationAdmissionDecision, AllocationAdmissionObservation, MlxAllocationAdmission,
    safe_minimum_mlx_memory_ceiling_bytes,
};

#[test]
fn should_recommend_cache_cleanup_when_active_allocation_fits_but_total_memory_does_not() {
    assert_eq!(
        AllocationAdmissionObservation::new(9_000, 1_000, 500, 9_500).decide(),
        AllocationAdmissionDecision::ClearAllocatorCacheThenAdmit,
    );
}

#[test]
fn should_calculate_safe_minimum_from_non_evictable_memory_and_one_expert_page() {
    assert_eq!(
        safe_minimum_mlx_memory_ceiling_bytes(10_000, 4_000, 1_500),
        7_500
    );
}

#[test]
fn should_use_the_updated_ceiling_for_later_allocation_decisions() {
    let mut allocation_budget = MlxAllocationAdmission::new(1_500, 10_000);

    allocation_budget.update_active_memory_ceiling_bytes(8_000);

    assert_eq!(allocation_budget.active_memory_ceiling_bytes(), 8_000);
}
