use astronomical_model_serving::{
    ExpertWeightPage, RetainedExpertLayerCommitDelta, RetainedExpertLayerCommitError,
    RetainedExpertLayerCommitOutcome, RetainedExpertPageCache, last_prefill_chunk_demand_weight,
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

fn commit_partial_page(
    retained_pages: &mut RetainedExpertPageCache<FakeExpertPage>,
    layer_index: usize,
    expert_ids: &[usize],
    payload_bytes: u64,
) -> RetainedExpertLayerCommitOutcome {
    retained_pages
        .commit_materialized_routed_page(
            layer_index,
            4,
            expert_ids.to_vec(),
            FakeExpertPage { payload_bytes },
        )
        .expect("fictional routed-page metadata should be valid")
        .outcome
}

fn commit_complete_page(
    retained_pages: &mut RetainedExpertPageCache<FakeExpertPage>,
    layer_index: usize,
    payload_bytes: u64,
) -> RetainedExpertLayerCommitOutcome {
    retained_pages
        .commit_materialized_complete_layer(layer_index, 4, FakeExpertPage { payload_bytes })
        .expect("fictional complete-layer metadata should be valid")
        .outcome
}

#[test]
fn should_commit_complete_and_partial_pages_with_explicit_classes() {
    let mut retained_pages = RetainedExpertPageCache::new(2);
    retained_pages.update_maximum_resident_payload_bytes(60);

    assert!(matches!(
        commit_complete_page(&mut retained_pages, 0, 40),
        RetainedExpertLayerCommitOutcome::Committed(_)
    ));
    assert!(matches!(
        commit_partial_page(&mut retained_pages, 1, &[1, 3], 20),
        RetainedExpertLayerCommitOutcome::Committed(_)
    ));

    let statistics = retained_pages.statistics();
    assert_eq!(statistics.entry_count, 2);
    assert_eq!(statistics.resident_payload_byte_count, 60);
    assert_eq!(statistics.complete_layer_count, 1);
    assert_eq!(statistics.partial_layer_count, 1);
}

#[test]
fn should_reject_a_commit_above_the_ceiling_while_preserving_the_previous_owner() {
    let mut retained_pages = RetainedExpertPageCache::new(1);
    retained_pages.update_maximum_resident_payload_bytes(30);
    assert!(matches!(
        commit_partial_page(&mut retained_pages, 0, &[0, 1], 20),
        RetainedExpertLayerCommitOutcome::Committed(_)
    ));

    assert_eq!(
        commit_complete_page(&mut retained_pages, 0, 40),
        RetainedExpertLayerCommitOutcome::RejectedByCurrentCeiling
    );
    assert_eq!(retained_pages.statistics().resident_payload_byte_count, 20);
    assert_eq!(retained_pages.statistics().partial_layer_count, 1);
}

#[test]
fn should_replace_a_partial_owner_with_a_complete_mandatory_read() {
    let mut retained_pages = RetainedExpertPageCache::new(1);
    retained_pages.update_maximum_resident_payload_bytes(40);
    commit_partial_page(&mut retained_pages, 0, &[0, 2], 20);

    assert_eq!(
        commit_complete_page(&mut retained_pages, 0, 40),
        RetainedExpertLayerCommitOutcome::Committed(RetainedExpertLayerCommitDelta {
            released_payload_bytes: 20,
            committed_payload_bytes: 40,
        })
    );

    let statistics = retained_pages.statistics();
    assert_eq!(statistics.entry_count, 1);
    assert_eq!(statistics.complete_layer_count, 1);
    assert_eq!(statistics.partial_layer_count, 0);
    assert_eq!(statistics.mandatory_read_promotion_count, 1);
}

#[test]
fn should_preserve_a_useful_partial_owner_when_a_proposed_route_set_differs() {
    let mut retained_pages = RetainedExpertPageCache::new(1);
    retained_pages.update_maximum_resident_payload_bytes(40);
    commit_partial_page(&mut retained_pages, 0, &[0, 1], 20);

    assert_eq!(
        commit_partial_page(&mut retained_pages, 0, &[2, 3], 20),
        RetainedExpertLayerCommitOutcome::PreservedExisting
    );
    assert_eq!(
        retained_pages.topology_snapshot(0)[0].retained_expert_ids,
        vec![0, 1]
    );
}

#[test]
fn should_replace_a_partial_owner_with_a_strict_superset_routed_page() {
    let mut retained_pages = RetainedExpertPageCache::new(1);
    retained_pages.update_maximum_resident_payload_bytes(40);
    commit_partial_page(&mut retained_pages, 0, &[0, 1], 20);

    assert!(matches!(
        commit_partial_page(&mut retained_pages, 0, &[0, 1, 2], 30),
        RetainedExpertLayerCommitOutcome::Committed(_)
    ));
    assert_eq!(
        retained_pages.topology_snapshot(0)[0].retained_expert_ids,
        vec![0, 1, 2]
    );
    assert_eq!(retained_pages.statistics().resident_payload_byte_count, 30);
}

#[test]
fn should_reclaim_multiple_partial_pages_before_one_complete_page() {
    let mut retained_pages = RetainedExpertPageCache::new(3);
    retained_pages.update_maximum_resident_payload_bytes(80);
    commit_complete_page(&mut retained_pages, 0, 40);
    commit_partial_page(&mut retained_pages, 1, &[0, 1], 20);
    commit_partial_page(&mut retained_pages, 2, &[2, 3], 20);

    let reclamation = retained_pages.reclaim_for_request_pressure(30);
    let statistics = retained_pages.statistics();
    assert_eq!(reclamation.released_partial_layer_count, 2);
    assert_eq!(reclamation.released_partial_payload_bytes, 40);
    assert_eq!(reclamation.released_complete_layer_count, 0);
    assert_eq!(statistics.entry_count, 1);
    assert_eq!(statistics.resident_payload_byte_count, 40);
    assert!(retained_pages.retained_layer(0).is_some());
}

#[test]
fn should_resume_a_request_pressure_cap_without_loading_any_page() {
    let mut retained_pages = RetainedExpertPageCache::new(2);
    retained_pages.update_maximum_resident_payload_bytes(80);
    commit_complete_page(&mut retained_pages, 0, 40);
    commit_complete_page(&mut retained_pages, 1, 40);

    assert!(retained_pages.limit_for_request_pressure(40));
    assert!(retained_pages.resume_after_request_pressure());
    let statistics = retained_pages.statistics();
    assert_eq!(statistics.maximum_resident_payload_byte_count, 80);
    assert_eq!(statistics.disk_page_load_count, 0);
}

#[test]
fn should_apply_an_absolute_forward_pressure_cap_and_allow_safe_growth_to_it() {
    let mut retained_pages = RetainedExpertPageCache::new(3);
    retained_pages.update_maximum_resident_payload_bytes(120);
    commit_complete_page(&mut retained_pages, 0, 40);
    commit_complete_page(&mut retained_pages, 1, 40);

    assert!(retained_pages.limit_for_request_pressure_to_maximum(50));
    assert_eq!(retained_pages.statistics().resident_payload_byte_count, 40);
    assert_eq!(
        commit_partial_page(&mut retained_pages, 2, &[0], 10),
        RetainedExpertLayerCommitOutcome::Committed(RetainedExpertLayerCommitDelta {
            released_payload_bytes: 0,
            committed_payload_bytes: 10,
        })
    );
    assert_eq!(retained_pages.statistics().resident_payload_byte_count, 50);
    assert!(!retained_pages.can_commit_materialized_page(2, 20));
}

#[test]
fn should_apply_a_lower_normal_ceiling_with_partial_first_reclamation() {
    let mut retained_pages = RetainedExpertPageCache::new(3);
    retained_pages.update_maximum_resident_payload_bytes(80);
    commit_complete_page(&mut retained_pages, 0, 40);
    commit_partial_page(&mut retained_pages, 1, &[0, 1], 20);
    commit_partial_page(&mut retained_pages, 2, &[2, 3], 20);

    let reclamation = retained_pages.update_maximum_resident_payload_bytes(50);

    let statistics = retained_pages.statistics();
    assert_eq!(reclamation.released_partial_layer_count, 2);
    assert_eq!(reclamation.released_partial_payload_bytes, 40);
    assert_eq!(reclamation.released_complete_layer_count, 0);
    assert_eq!(statistics.complete_layer_count, 1);
    assert_eq!(statistics.partial_layer_count, 0);
    assert_eq!(statistics.partial_layer_eviction_count, 2);
}

#[test]
fn should_ignore_invalid_route_identifiers_without_corrupting_demand() {
    let mut retained_pages = RetainedExpertPageCache::<FakeExpertPage>::new(1);
    retained_pages.update_maximum_resident_payload_bytes(10);
    retained_pages.record_expert_demand(0, 2, &[1, 9, 1]);
    commit_partial_page(&mut retained_pages, 0, &[1], 10);

    assert_eq!(
        retained_pages.topology_snapshot(0)[0].covered_weighted_demand,
        2
    );
}

#[test]
fn should_start_a_fresh_demand_window_after_topology_planning() {
    let mut retained_pages = RetainedExpertPageCache::<FakeExpertPage>::new(1);
    retained_pages.update_maximum_resident_payload_bytes(20);
    retained_pages.record_expert_demand(0, 2, &[1, 1, 1]);
    retained_pages.clear_expert_demand();
    retained_pages.record_expert_demand(0, 2, &[0]);
    commit_partial_page(&mut retained_pages, 0, &[0, 1], 20);

    assert_eq!(
        retained_pages.topology_snapshot(0)[0].covered_weighted_demand,
        1
    );
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
    let mut retained_pages = RetainedExpertPageCache::<FakeExpertPage>::new(2);
    retained_pages.update_maximum_resident_payload_bytes(20);
    retained_pages.set_demand_assignment_weight(1);
    retained_pages.record_expert_demand(0, 2, &[0, 0, 0, 0, 0]);
    retained_pages.set_demand_assignment_weight(2);
    retained_pages.record_expert_demand(1, 2, &[1, 1, 1]);
    commit_partial_page(&mut retained_pages, 0, &[0], 10);
    commit_partial_page(&mut retained_pages, 1, &[1], 10);

    // Five earlier routes lose to three last-chunk routes counted twice.
    let topology = retained_pages.topology_snapshot(0);
    assert_eq!(topology[0].covered_weighted_demand, 5);
    assert_eq!(topology[1].covered_weighted_demand, 6);
}

#[test]
fn should_keep_raw_frequency_ranking_when_last_chunk_weight_is_one() {
    let mut retained_pages = RetainedExpertPageCache::<FakeExpertPage>::new(2);
    retained_pages.update_maximum_resident_payload_bytes(20);
    retained_pages.set_demand_assignment_weight(1);
    retained_pages.record_expert_demand(0, 2, &[0, 0, 0, 0, 0]);
    retained_pages.record_expert_demand(1, 2, &[1, 1, 1]);
    commit_partial_page(&mut retained_pages, 0, &[0], 10);
    commit_partial_page(&mut retained_pages, 1, &[1], 10);

    let topology = retained_pages.topology_snapshot(0);
    assert_eq!(topology[0].covered_weighted_demand, 5);
    assert_eq!(topology[1].covered_weighted_demand, 3);
}

#[test]
fn should_treat_a_zero_demand_weight_as_one_assignment() {
    let mut retained_pages = RetainedExpertPageCache::<FakeExpertPage>::new(2);
    retained_pages.update_maximum_resident_payload_bytes(20);
    retained_pages.set_demand_assignment_weight(0);
    retained_pages.record_expert_demand(0, 2, &[0]);
    retained_pages.record_expert_demand(1, 2, &[1, 1]);
    commit_partial_page(&mut retained_pages, 0, &[0], 10);
    commit_partial_page(&mut retained_pages, 1, &[1], 10);

    // A zero weight must not discard assignments; two routes still beat one.
    let topology = retained_pages.topology_snapshot(0);
    assert_eq!(topology[0].covered_weighted_demand, 1);
    assert_eq!(topology[1].covered_weighted_demand, 2);
}

#[test]
fn should_reject_zero_or_overflowing_payload_accounting_without_mutating_ownership() {
    let mut retained_pages = RetainedExpertPageCache::<FakeExpertPage>::new(3);
    retained_pages.update_maximum_resident_payload_bytes(u64::MAX);

    assert!(!retained_pages.can_commit_materialized_page(0, 0));
    assert_eq!(
        retained_pages
            .commit_materialized_complete_layer(0, 4, FakeExpertPage { payload_bytes: 0 })
            .expect_err("zero-byte ownership must be rejected"),
        RetainedExpertLayerCommitError::ZeroPayload { layer_index: 0 }
    );
    commit_complete_page(&mut retained_pages, 1, u64::MAX);
    assert!(!retained_pages.can_commit_materialized_page(2, 1));
    assert_eq!(
        retained_pages
            .commit_materialized_complete_layer(2, 4, FakeExpertPage { payload_bytes: 1 })
            .expect_err("overflowing ownership must be rejected"),
        RetainedExpertLayerCommitError::PayloadByteCountOverflow { layer_index: 2 }
    );
    assert_eq!(
        retained_pages.statistics().resident_payload_byte_count,
        u64::MAX
    );
    assert!(retained_pages.retained_layer(2).is_none());
}
