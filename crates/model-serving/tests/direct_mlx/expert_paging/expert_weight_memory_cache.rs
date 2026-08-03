use astronomical_model_serving::{ExpertWeightMemoryCache, MemoryBudgetSnapshot};

#[test]
fn should_preserve_one_decode_route_for_every_layer_left_paged_by_complete_layer_admission() {
    let expert_weight_memory_cache =
        expert_weight_memory_cache_with_maximum(4, vec![100, 100, 100, 100], 1_000);

    assert!(
        expert_weight_memory_cache.can_retain_complete_layer_expert_payload(0, 700),
        "the complete layer and three remaining decode routes should exactly fit"
    );
    assert!(
        !expert_weight_memory_cache.can_retain_complete_layer_expert_payload(0, 701),
        "complete-layer admission must preserve one decode route for every remaining paged layer"
    );
}

#[test]
fn should_admit_the_final_complete_layer_without_reserving_a_partial_route() {
    let expert_weight_memory_cache = expert_weight_memory_cache_with_maximum(1, vec![100], 500);

    assert!(
        expert_weight_memory_cache.can_retain_complete_layer_expert_payload(0, 500),
        "the final complete layer leaves no paged layer that needs a route floor"
    );
}

#[test]
fn should_fall_back_to_physical_complete_layer_admission_when_route_floors_are_unaffordable() {
    let expert_weight_memory_cache =
        expert_weight_memory_cache_with_maximum(4, vec![100, 100, 100, 100], 250);

    assert!(
        expert_weight_memory_cache.can_retain_complete_layer_expert_payload(0, 250),
        "a very small ceiling should retain useful complete payload instead of funding no strategy"
    );
}

#[test]
fn should_preserve_each_remaining_layers_exact_decode_route_payload() {
    let expert_weight_memory_cache =
        expert_weight_memory_cache_with_maximum(3, vec![100, 200, 300], 1_000);

    assert!(
        expert_weight_memory_cache.can_retain_complete_layer_expert_payload(0, 500),
        "the proposed complete layer plus heterogeneous remaining route floors should fit exactly"
    );
    assert!(
        !expert_weight_memory_cache.can_retain_complete_layer_expert_payload(0, 501),
        "admission must use exact per-layer route bytes rather than a uniform estimate"
    );
}

#[test]
fn should_reconcile_retention_before_a_temporary_page_uses_the_configured_mlx_ceiling() {
    let mut expert_weight_memory_cache = ExpertWeightMemoryCache::new(40, vec![0; 40]);
    let memory_budget_snapshot = MemoryBudgetSnapshot {
        stage: "temporary_expert_page_after_retained_request".to_owned(),
        active_bytes: 18_862_497_618,
        allocator_cache_bytes: 6_070_000,
        pending_allocation_bytes: 358_244_624,
        projected_bytes: 19_875_103_264,
        configured_cap_bytes: 20_401_094_656,
        maximum_expert_page_bytes: 855_638_016,
    };

    expert_weight_memory_cache.reconcile_retention_before_temporary_expert_page(
        &memory_budget_snapshot,
        0,
        &[],
    );

    assert_eq!(
        expert_weight_memory_cache
            .statistics()
            .maximum_resident_payload_byte_count,
        682_959_022,
        "the configured MLX ceiling must preserve only the exact capacity remaining beside the next page"
    );
}

#[test]
fn should_prevent_expert_repopulation_until_request_memory_pressure_ends() {
    let mut expert_weight_memory_cache = ExpertWeightMemoryCache::new(40, vec![0; 40]);

    expert_weight_memory_cache.limit_retention_for_request_memory_pressure(0);
    expert_weight_memory_cache.update_maximum_resident_payload_byte_count(8_000_000_000);

    assert_eq!(
        expert_weight_memory_cache
            .statistics()
            .maximum_resident_payload_byte_count,
        0,
        "live expert-page budget updates must not repopulate retention during request pressure"
    );

    assert!(
        expert_weight_memory_cache.resume_retention_after_request_memory_pressure(),
        "lifting a barrier that left complete layers missing should schedule recovery"
    );

    assert_eq!(
        expert_weight_memory_cache
            .statistics()
            .maximum_resident_payload_byte_count,
        u64::MAX,
        "automatic expert retention should resume after the pressured request finishes"
    );
}

#[test]
fn should_preserve_a_partial_retention_ceiling_during_request_memory_pressure() {
    let mut expert_weight_memory_cache = ExpertWeightMemoryCache::new(40, vec![0; 40]);

    expert_weight_memory_cache.limit_retention_for_request_memory_pressure(3_000_000_000);
    expert_weight_memory_cache.update_maximum_resident_payload_byte_count(8_000_000_000);

    assert_eq!(
        expert_weight_memory_cache
            .statistics()
            .maximum_resident_payload_byte_count,
        3_000_000_000,
        "live budget updates must preserve the request-scoped partial retention ceiling"
    );

    expert_weight_memory_cache.resume_retention_after_request_memory_pressure();

    assert_eq!(
        expert_weight_memory_cache
            .statistics()
            .maximum_resident_payload_byte_count,
        u64::MAX,
        "automatic expert retention should resume after the pressured request finishes"
    );
}

#[test]
fn should_only_tighten_repeated_request_memory_pressure_ceilings() {
    let mut expert_weight_memory_cache = ExpertWeightMemoryCache::new(40, vec![0; 40]);

    expert_weight_memory_cache.limit_retention_for_request_memory_pressure(3_000_000_000);
    expert_weight_memory_cache.limit_retention_for_request_memory_pressure(5_000_000_000);

    assert_eq!(
        expert_weight_memory_cache
            .statistics()
            .maximum_resident_payload_byte_count,
        3_000_000_000,
        "later pressure checks must not relax an earlier request-scoped ceiling"
    );
}

#[test]
fn should_freeze_missing_complete_layer_growth_during_soft_request_pressure() {
    let mut expert_weight_memory_cache = ExpertWeightMemoryCache::new(40, vec![0; 40]);
    let artificial_low_memory_barrier_bytes = 0;
    let recovered_live_memory_barrier_bytes = 8_000_000_000;
    let missing_complete_layer_payload_bytes = 1_000_000_000;

    expert_weight_memory_cache
        .limit_retention_for_request_memory_pressure(artificial_low_memory_barrier_bytes);
    expert_weight_memory_cache
        .update_maximum_resident_payload_byte_count(recovered_live_memory_barrier_bytes);
    assert!(
        !expert_weight_memory_cache
            .can_retain_complete_layer_expert_payload(0, missing_complete_layer_payload_bytes,),
        "the artificial low-memory barrier should prevent complete-layer restoration"
    );

    expert_weight_memory_cache.resume_retention_after_request_memory_pressure();

    assert!(
        expert_weight_memory_cache.freeze_retention_growth_for_request_memory_pressure(),
        "soft request pressure must freeze optional layer growth until finalization releases the request ceiling"
    );
    expert_weight_memory_cache
        .update_maximum_resident_payload_byte_count(recovered_live_memory_barrier_bytes);

    assert!(
        !expert_weight_memory_cache
            .can_retain_complete_layer_expert_payload(0, missing_complete_layer_payload_bytes,),
        "a later page budget must not repopulate retention during the same pressured request"
    );
}

#[test]
fn should_admit_complete_layers_against_physical_residency_without_a_virtual_partial_reserve() {
    let expert_weight_memory_cache = expert_weight_memory_cache_with_maximum(4, vec![0; 4], 1_000);

    assert!(
        expert_weight_memory_cache.can_retain_complete_layer_expert_payload(0, 1_000),
        "a complete layer may use physically available capacity"
    );
    assert!(
        !expert_weight_memory_cache.can_retain_complete_layer_expert_payload(0, 1_001),
        "a complete layer must not exceed the physical retention ceiling"
    );
}

fn expert_weight_memory_cache_with_maximum(
    layer_count: usize,
    minimum_decode_route_payload_byte_count_by_layer: Vec<u64>,
    maximum_resident_payload_byte_count: u64,
) -> ExpertWeightMemoryCache {
    let mut expert_weight_memory_cache = ExpertWeightMemoryCache::new(
        layer_count,
        minimum_decode_route_payload_byte_count_by_layer,
    );
    expert_weight_memory_cache
        .update_maximum_resident_payload_byte_count(maximum_resident_payload_byte_count);
    expert_weight_memory_cache
}
