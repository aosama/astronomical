use super::WarmExpertSlotTableCache;
use super::warm_expert_slot_table_cache_tests::{
    EXPERT_CAPACITY, WARM_SLOT_COUNT, streamed_weights_for_expert_count, test_runtime,
};
use serial_test::serial;

#[test]
#[serial]
fn should_report_route_missing_non_resident_and_occupied_experts_per_table() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(2);
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    let weights = streamed_weights_for_expert_count(&runtime, 2);
    cache
        .queue_pending_routed_expert_insert(0, &[7, 9], &weights, WARM_SLOT_COUNT)
        .expect("the warm insert should queue");
    cache
        .flush_pending_inserts(&runtime)
        .expect("the first warm flush should succeed");

    // A layer without a table answers None for the table-shaped queries and
    // zero occupancy, never an empty table claim.
    assert_eq!(cache.missing_route_expert_ids(1, &[3]), None);
    assert_eq!(cache.non_resident_expert_ids(1, EXPERT_CAPACITY), None);
    assert_eq!(cache.occupied_slot_count(1), 0);

    // A partial table reports exactly the route experts it lacks, the full
    // non-resident read set, and its occupancy.
    assert_eq!(cache.missing_route_expert_ids(0, &[7, 9]), Some(vec![]));
    assert_eq!(cache.missing_route_expert_ids(0, &[9, 12]), Some(vec![12]));
    assert_eq!(
        cache
            .non_resident_expert_ids(0, EXPERT_CAPACITY)
            .expect("a seated layer reports its read set"),
        (0..EXPERT_CAPACITY)
            .filter(|expert_id| ![7, 9].contains(expert_id))
            .collect::<Vec<_>>()
    );
    assert_eq!(cache.occupied_slot_count(0), 2);

    // Inputs are normalized: duplicates and unsorted routes answer in stable
    // sorted order.
    assert_eq!(
        cache.missing_route_expert_ids(0, &[12, 7, 12]),
        Some(vec![12])
    );
}

#[test]
#[serial]
fn should_report_real_demand_coverage_from_the_recorded_routing_evidence() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(1);
    // Production ceilings are set by the residency plan refresh; the tests
    // set a generous ceiling except the budget-refusal case.
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    let weights = streamed_weights_for_expert_count(&runtime, 2);

    cache
        .queue_pending_routed_expert_insert(0, &[7, 9], &weights, WARM_SLOT_COUNT)
        .expect("the warm insert should queue");
    cache
        .flush_pending_inserts(&runtime)
        .expect("the warm flush should succeed");

    // With no routing evidence the coverage is zero; recorded demand for
    // retained experts makes the residency planner frequency-aware.
    assert_eq!(
        cache.topology_snapshot(EXPERT_CAPACITY)[0].covered_weighted_demand,
        0
    );
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[7, 9, 9]);
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[11]);
    let snapshot = cache.topology_snapshot(EXPERT_CAPACITY);
    // Experts 7 and 9 are retained; 11 is routed but not retained.
    assert_eq!(snapshot[0].covered_weighted_demand, 3);
}
