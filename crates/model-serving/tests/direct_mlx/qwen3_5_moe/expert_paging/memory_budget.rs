use astronomical_model_serving::{
    LiveMetalBudget, MemoryBudgetSnapshot, automatic_expert_weight_memory_cache_maximum_size_bytes,
    maximum_possible_expert_route_payload_bytes, safe_minimum_mlx_memory_ceiling_bytes,
};

#[test]
fn should_size_optional_complete_layer_growth_only_against_the_configured_mlx_ceiling() {
    let memory_budget_snapshot = memory_budget_snapshot(10_000, 8_000, 0, 500);

    assert!(memory_budget_snapshot.within_cap());
}

#[test]
fn should_size_expert_retention_against_the_configured_mlx_ceiling() {
    let memory_budget_snapshot =
        memory_budget_snapshot(40_200_896_512, 33_000_000_000, 0, 1_000_000_000);

    assert_eq!(
        automatic_expert_weight_memory_cache_maximum_size_bytes(&memory_budget_snapshot, 0, 0,),
        6_200_896_512,
        "expert retention must use the configured MLX ceiling rather than a second driver recommendation",
    );
}

#[test]
fn should_add_local_capacity_to_the_current_expert_weight_memory_cache_payload() {
    let memory_budget_snapshot = memory_budget_snapshot(10_000, 4_000, 1_000, 1_500);

    assert_eq!(
        automatic_expert_weight_memory_cache_maximum_size_bytes(&memory_budget_snapshot, 2_000, 0,),
        6_500,
    );
}

#[test]
fn should_shrink_the_expert_weight_memory_cache_budget_when_local_usage_exceeds_the_cap() {
    let memory_budget_snapshot = memory_budget_snapshot(10_000, 9_000, 100, 500);

    assert_eq!(
        automatic_expert_weight_memory_cache_maximum_size_bytes(&memory_budget_snapshot, 2_000, 0,),
        2_500,
    );
}

#[test]
fn should_count_an_incoming_complete_layer_once_when_sizing_post_load_retention() {
    let incoming_complete_layer_payload_bytes = 855_638_016;
    let current_retained_expert_payload_bytes = 27_380_416_512;
    let memory_budget_snapshot =
        memory_budget_snapshot(32_212_254_720, 29_982_882_054, 0, 26_738_688);

    assert!(
        automatic_expert_weight_memory_cache_maximum_size_bytes(
            &memory_budget_snapshot,
            current_retained_expert_payload_bytes,
            incoming_complete_layer_payload_bytes,
        ) >= current_retained_expert_payload_bytes + incoming_complete_layer_payload_bytes,
    );
}

#[test]
fn should_compute_a_fail_closed_route_upper_bound_when_exact_evidence_is_unavailable() {
    assert_eq!(
        maximum_possible_expert_route_payload_bytes(7_077_888, 512, 180),
        Some(1_274_019_840),
    );
}

#[test]
fn should_cap_multi_token_route_reservation_at_the_layer_expert_capacity() {
    assert_eq!(
        maximum_possible_expert_route_payload_bytes(7_077_888, 512, 20_480),
        Some(3_623_878_656),
    );
}

#[test]
fn should_fail_closed_when_the_possible_route_payload_overflows() {
    assert_eq!(
        maximum_possible_expert_route_payload_bytes(u64::MAX, 2, 2),
        None,
    );
}

#[test]
fn should_preserve_warm_retention_when_exact_route_analysis_has_no_missing_pages() {
    let memory_budget_snapshot =
        memory_budget_snapshot_with_pending_allocation(10_000, 9_000, 0, 500, 0);

    assert_eq!(
        automatic_expert_weight_memory_cache_maximum_size_bytes(&memory_budget_snapshot, 7_000, 0,),
        7_500,
        "an exact warm route should reserve only the future page and must not evict as if assignments were missing",
    );
}

#[test]
fn should_retain_exact_missing_route_payload_while_reserving_one_future_page() {
    let exact_missing_route_payload_bytes = 2_000;
    let memory_budget_snapshot = memory_budget_snapshot_with_pending_allocation(
        10_000,
        9_000,
        0,
        500,
        exact_missing_route_payload_bytes,
    );

    assert_eq!(
        automatic_expert_weight_memory_cache_maximum_size_bytes(
            &memory_budget_snapshot,
            7_000,
            exact_missing_route_payload_bytes,
        ),
        7_500,
        "admission should evict only enough old retention to hold exact misses and one future page",
    );
}

#[test]
fn should_fail_closed_when_local_active_and_allocator_cache_bytes_overflow() {
    let memory_budget_snapshot = memory_budget_snapshot(u64::MAX, u64::MAX, u64::MAX, 1);

    assert!(!memory_budget_snapshot.within_cap());
}

#[test]
fn should_reclaim_allocator_cache_when_enforcement_passes_but_total_gpu_memory_exceeds_cap() {
    let memory_budget_snapshot = memory_budget_snapshot(9_500, 9_000, 1_000, 500);

    // The enforcement check passes: active + pending = 9500 <= cap = 9500.
    assert!(memory_budget_snapshot.within_cap());
    // But total GPU memory (active + cache + pending = 10500) exceeds the cap,
    // so the allocator cache should be cleared before proceeding.
    assert!(memory_budget_snapshot.should_clear_allocator_cache_to_reduce_total_gpu_memory());
}

#[test]
fn should_reduce_retention_ceiling_when_observed_transient_buffers_need_headroom() {
    // At 10 GB cap, 4 GB active, 0 cache, 1.5 KB pending: without transient
    // reservation the retention ceiling is generous. With 2 GB of transient
    // headroom, the ceiling shrinks by exactly 2 GB.
    let mut memory_budget_snapshot = memory_budget_snapshot(10_000, 4_000, 0, 1_500);
    let retention_without_transient =
        automatic_expert_weight_memory_cache_maximum_size_bytes(&memory_budget_snapshot, 2_000, 0);
    // Without transient reservation: cap - (active + pending) = 10000 - 5500 = 4500
    // plus post_load_retained = 2000, so max = 2000 + 4500 = 6500
    assert_eq!(retention_without_transient, 6_500);

    memory_budget_snapshot.observed_transient_high_water_bytes = 2_000;
    let retention_with_transient =
        automatic_expert_weight_memory_cache_maximum_size_bytes(&memory_budget_snapshot, 2_000, 0);
    // With 2 GB transient: effective_cap = 10000 - 2000 = 8000
    // cap - (active + pending) = 8000 - 5500 = 2500
    // plus post_load_retained = 2000, so max = 2000 + 2500 = 4500
    assert_eq!(retention_with_transient, 4_500);
    assert!(
        retention_with_transient < retention_without_transient,
        "transient reservation must shrink the retention ceiling"
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
fn should_not_count_evictable_complete_layers_in_the_safe_minimum() {
    assert_eq!(
        safe_minimum_mlx_memory_ceiling_bytes(10_000, 8_000, 2_000),
        4_000
    );
}

#[test]
fn should_use_the_updated_cap_for_later_memory_budget_snapshots() {
    let mut live_metal_budget = LiveMetalBudget::new(1_500, 10_000);

    live_metal_budget.update_configured_cap_bytes(8_000);

    assert_eq!(live_metal_budget.configured_cap_bytes(), 8_000);
}

fn memory_budget_snapshot(
    configured_cap_bytes: u64,
    active_bytes: u64,
    allocator_cache_bytes: u64,
    maximum_expert_page_bytes: u64,
) -> MemoryBudgetSnapshot {
    memory_budget_snapshot_with_pending_allocation(
        configured_cap_bytes,
        active_bytes,
        allocator_cache_bytes,
        maximum_expert_page_bytes,
        maximum_expert_page_bytes,
    )
}

fn memory_budget_snapshot_with_pending_allocation(
    configured_cap_bytes: u64,
    active_bytes: u64,
    allocator_cache_bytes: u64,
    maximum_expert_page_bytes: u64,
    pending_allocation_bytes: u64,
) -> MemoryBudgetSnapshot {
    let projected_bytes = active_bytes.saturating_add(pending_allocation_bytes);
    MemoryBudgetSnapshot {
        stage: "expert_weight_memory_cache_budget_test".to_owned(),
        active_bytes,
        allocator_cache_bytes,
        pending_allocation_bytes,
        projected_bytes,
        configured_cap_bytes,
        maximum_expert_page_bytes,
        observed_transient_high_water_bytes: 0,
    }
}
