use astronomical_model_serving::{
    AllocationAdmissionDecision, AllocationAdmissionObservation, MlxAllocationBudget,
    retained_expert_payload_capacity_bytes, safe_minimum_mlx_memory_ceiling_bytes,
};

#[test]
fn should_size_expert_retention_against_the_configured_mlx_ceiling() {
    assert_eq!(
        retained_expert_payload_capacity_bytes(
            33_000_000_000,
            40_200_896_512,
            1_000_000_000,
            1_000_000_000,
            0,
            0,
            0,
        ),
        6_200_896_512,
    );
}

#[test]
fn should_count_an_incoming_complete_layer_once_while_reserving_a_future_page() {
    let incoming_layer_bytes = 855_638_016;
    let current_retained_bytes = 27_380_416_512;
    let retained_capacity_bytes = retained_expert_payload_capacity_bytes(
        29_982_882_054,
        32_212_254_720,
        26_738_688,
        26_738_688,
        0,
        current_retained_bytes,
        incoming_layer_bytes,
    );

    assert!(retained_capacity_bytes >= current_retained_bytes + incoming_layer_bytes);
}

#[test]
fn should_recommend_cache_cleanup_when_active_allocation_fits_but_total_memory_does_not() {
    assert_eq!(
        AllocationAdmissionObservation::new(9_000, 1_000, 500, 9_500).decide(),
        AllocationAdmissionDecision::ClearAllocatorCacheThenAdmit,
    );
}

#[test]
fn should_reduce_retention_capacity_by_observed_transient_headroom() {
    let without_transient =
        retained_expert_payload_capacity_bytes(4_000, 10_000, 1_500, 1_500, 0, 2_000, 0);
    let with_transient =
        retained_expert_payload_capacity_bytes(4_000, 10_000, 1_500, 1_500, 2_000, 2_000, 0);

    assert_eq!(without_transient, 6_500);
    assert_eq!(with_transient, 4_500);
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
    let mut allocation_budget = MlxAllocationBudget::new(1_500, 10_000);

    allocation_budget.update_active_memory_ceiling_bytes(8_000);

    assert_eq!(allocation_budget.active_memory_ceiling_bytes(), 8_000);
}
