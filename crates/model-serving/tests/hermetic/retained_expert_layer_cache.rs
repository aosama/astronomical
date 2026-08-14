use astronomical_model_serving::{
    ExpertWeightPage, RetainedExpertLayerCache, last_prefill_chunk_demand_weight,
};

#[derive(Debug)]
struct FakeExpertPage {
    payload_bytes: u64,
}

impl ExpertWeightPage for FakeExpertPage {
    fn resident_payload_byte_count(&self) -> u64 {
        self.payload_bytes
    }
}

#[test]
fn should_replace_one_page_without_exceeding_the_shared_ceiling() {
    let mut retained_pages = RetainedExpertLayerCache::new(2);
    retained_pages.update_maximum_resident_payload_bytes(220);
    assert!(retained_pages.replace_layer(0, FakeExpertPage { payload_bytes: 100 }));
    assert!(retained_pages.replace_layer(1, FakeExpertPage { payload_bytes: 100 }));

    assert!(retained_pages.replace_layer(1, FakeExpertPage { payload_bytes: 120 }));

    let statistics = retained_pages.statistics();
    assert_eq!(statistics.entry_count, 2);
    assert_eq!(statistics.resident_payload_byte_count, 220);
    assert_eq!(statistics.eviction_count, 1);
}

#[test]
fn should_reject_a_replacement_that_exceeds_the_effective_ceiling() {
    let mut retained_pages = RetainedExpertLayerCache::new(2);
    retained_pages.update_maximum_resident_payload_bytes(200);
    assert!(retained_pages.replace_layer(0, FakeExpertPage { payload_bytes: 100 }));
    assert!(retained_pages.replace_layer(1, FakeExpertPage { payload_bytes: 100 }));

    assert!(!retained_pages.replace_layer(1, FakeExpertPage { payload_bytes: 201 }));
    assert_eq!(retained_pages.statistics().resident_payload_byte_count, 200);
}

#[test]
fn should_remove_a_stale_page_before_loading_replacements() {
    let mut retained_pages = RetainedExpertLayerCache::new(2);
    retained_pages.update_maximum_resident_payload_bytes(200);
    assert!(retained_pages.replace_layer(0, FakeExpertPage { payload_bytes: 100 }));
    assert!(retained_pages.replace_layer(1, FakeExpertPage { payload_bytes: 100 }));

    assert!(retained_pages.remove_layer(1));
    assert!(retained_pages.replace_layer(0, FakeExpertPage { payload_bytes: 150 }));

    let statistics = retained_pages.statistics();
    assert_eq!(statistics.entry_count, 1);
    assert_eq!(statistics.resident_payload_byte_count, 150);
    assert_eq!(statistics.eviction_count, 2);
}

#[test]
fn should_reclaim_highest_layers_for_request_pressure() {
    let mut retained_pages = RetainedExpertLayerCache::new(4);
    retained_pages.update_maximum_resident_payload_bytes(400);
    for layer_index in 0..4 {
        assert!(retained_pages.replace_layer(layer_index, FakeExpertPage { payload_bytes: 100 }));
    }

    assert!(retained_pages.limit_for_request_pressure(150));

    let statistics = retained_pages.statistics();
    assert_eq!(statistics.entry_count, 2);
    assert_eq!(statistics.resident_payload_byte_count, 200);
    assert!(retained_pages.retained_layer(0).is_some());
    assert!(retained_pages.retained_layer(1).is_some());
    assert!(retained_pages.retained_layer(2).is_none());
    assert!(retained_pages.retained_layer(3).is_none());
}

#[test]
fn should_restore_the_normal_ceiling_after_request_pressure() {
    let mut retained_pages = RetainedExpertLayerCache::new(2);
    retained_pages.update_maximum_resident_payload_bytes(200);
    assert!(retained_pages.replace_layer(0, FakeExpertPage { payload_bytes: 100 }));
    assert!(retained_pages.replace_layer(1, FakeExpertPage { payload_bytes: 100 }));

    assert!(retained_pages.limit_for_request_pressure(100));
    assert!(retained_pages.resume_after_request_pressure());
    assert_eq!(
        retained_pages
            .statistics()
            .maximum_resident_payload_byte_count,
        200
    );
}

#[test]
fn should_spend_the_global_budget_on_highest_route_frequency() {
    let mut retained_pages = RetainedExpertLayerCache::<FakeExpertPage>::new(2);
    retained_pages.update_maximum_resident_payload_bytes(30);
    retained_pages.record_expert_demand(0, 2, &[0, 0, 0, 1]);
    retained_pages.record_expert_demand(1, 2, &[0, 0, 1, 1, 1, 1]);

    let selected_experts = retained_pages
        .preferred_expert_ids_for_global_budget(&[20, 40], &[2, 2])
        .expect("matching geometry should produce a plan");

    assert_eq!(selected_experts, vec![vec![0], vec![1]]);
}

#[test]
fn should_prefer_more_logical_bytes_saved_over_smaller_expert_payload() {
    let mut retained_pages = RetainedExpertLayerCache::<FakeExpertPage>::new(2);
    retained_pages.update_maximum_resident_payload_bytes(20);
    retained_pages.record_expert_demand(0, 1, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    retained_pages.record_expert_demand(1, 1, &[0, 0, 0, 0, 0, 0]);

    let selected_experts = retained_pages
        .preferred_expert_ids_for_global_budget(&[20, 10], &[1, 1])
        .expect("matching geometry should produce a plan");

    // Ten routes on the 20-byte expert outrank six routes on the 10-byte expert.
    assert_eq!(selected_experts, vec![vec![0], vec![]]);
}

#[test]
fn should_use_stable_identifiers_when_global_demand_per_byte_is_tied() {
    let mut retained_pages = RetainedExpertLayerCache::<FakeExpertPage>::new(2);
    retained_pages.update_maximum_resident_payload_bytes(20);

    let selected_experts = retained_pages
        .preferred_expert_ids_for_global_budget(&[20, 20], &[2, 2])
        .expect("matching geometry should produce a plan");

    assert_eq!(selected_experts, vec![vec![0, 1], vec![]]);
}

#[test]
fn should_ignore_invalid_route_identifiers_without_corrupting_demand() {
    let mut retained_pages = RetainedExpertLayerCache::<FakeExpertPage>::new(1);
    retained_pages.update_maximum_resident_payload_bytes(10);
    retained_pages.record_expert_demand(0, 2, &[1, 9, 1]);

    let selected_experts = retained_pages
        .preferred_expert_ids_for_global_budget(&[20], &[2])
        .expect("matching geometry should produce a plan");

    assert_eq!(selected_experts, vec![vec![1]]);
}

#[test]
fn should_reject_mismatched_or_zero_capacity_planner_geometry() {
    let retained_pages = RetainedExpertLayerCache::<FakeExpertPage>::new(2);

    assert!(
        retained_pages
            .preferred_expert_ids_for_global_budget(&[20], &[2, 2])
            .is_err()
    );
    assert!(
        retained_pages
            .preferred_expert_ids_for_global_budget(&[20, 20], &[2, 0])
            .is_err()
    );
    assert!(
        retained_pages
            .preferred_expert_ids_for_global_budget(&[20, 0], &[2, 2])
            .is_err()
    );
}

#[test]
fn should_start_a_fresh_demand_window_after_topology_planning() {
    let mut retained_pages = RetainedExpertLayerCache::<FakeExpertPage>::new(1);
    retained_pages.update_maximum_resident_payload_bytes(10);
    retained_pages.record_expert_demand(0, 2, &[1, 1, 1]);
    retained_pages.clear_expert_demand();
    retained_pages.record_expert_demand(0, 2, &[0]);

    let selected_experts = retained_pages
        .preferred_expert_ids_for_global_budget(&[20], &[2])
        .expect("matching geometry should produce a plan");

    assert_eq!(selected_experts, vec![vec![0]]);
}

#[test]
fn should_scale_last_prefill_chunk_demand_by_earlier_token_density() {
    assert_eq!(last_prefill_chunk_demand_weight(6, 3), 2);
    assert_eq!(last_prefill_chunk_demand_weight(5, 3), 1);
    assert_eq!(last_prefill_chunk_demand_weight(0, 8), 1);
    assert_eq!(last_prefill_chunk_demand_weight(8, 0), 1);
    assert_eq!(last_prefill_chunk_demand_weight(u64::MAX, 1), u64::MAX);
}

#[test]
fn should_prefer_last_chunk_routes_when_weighted_demand_exceeds_earlier_frequency() {
    let mut retained_pages = RetainedExpertLayerCache::<FakeExpertPage>::new(1);
    retained_pages.update_maximum_resident_payload_bytes(10);
    retained_pages.set_demand_assignment_weight(1);
    retained_pages.record_expert_demand(0, 2, &[0, 0, 0, 0, 0]);
    retained_pages.set_demand_assignment_weight(2);
    retained_pages.record_expert_demand(0, 2, &[1, 1, 1]);

    let selected_experts = retained_pages
        .preferred_expert_ids_for_global_budget(&[20], &[2])
        .expect("matching geometry should produce a last-chunk-weighted plan");

    // Five earlier routes lose to three last-chunk routes counted twice.
    assert_eq!(selected_experts, vec![vec![1]]);
}

#[test]
fn should_keep_raw_frequency_ranking_when_last_chunk_weight_is_one() {
    let mut retained_pages = RetainedExpertLayerCache::<FakeExpertPage>::new(1);
    retained_pages.update_maximum_resident_payload_bytes(10);
    retained_pages.set_demand_assignment_weight(1);
    retained_pages.record_expert_demand(0, 2, &[0, 0, 0, 0, 0]);
    retained_pages.record_expert_demand(0, 2, &[1, 1, 1]);

    let selected_experts = retained_pages
        .preferred_expert_ids_for_global_budget(&[20], &[2])
        .expect("matching geometry should produce an unweighted plan");

    assert_eq!(selected_experts, vec![vec![0]]);
}

#[test]
fn should_treat_a_zero_demand_weight_as_one_assignment() {
    let mut retained_pages = RetainedExpertLayerCache::<FakeExpertPage>::new(1);
    retained_pages.update_maximum_resident_payload_bytes(10);
    retained_pages.set_demand_assignment_weight(0);
    retained_pages.record_expert_demand(0, 2, &[0]);
    retained_pages.record_expert_demand(0, 2, &[1, 1]);

    let selected_experts = retained_pages
        .preferred_expert_ids_for_global_budget(&[20], &[2])
        .expect("matching geometry should produce a unit-weight plan");

    // A zero weight must not discard assignments; two routes still beat one.
    assert_eq!(selected_experts, vec![vec![1]]);
}
