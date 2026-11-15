use super::AstroExpertCache;
use super::astro_expert_cache_tests::{
    EXPERT_CAPACITY, WARM_SLOT_COUNT, streamed_weights_for_expert_count, test_runtime,
};
use serial_test::serial;

#[test]
#[serial]
fn should_report_route_missing_non_resident_and_occupied_experts_per_table() {
    let runtime = test_runtime();
    let mut astro_expert_cache = AstroExpertCache::new(2);
    astro_expert_cache.set_capacity(1024 * 1024 * 1024);
    let weights = streamed_weights_for_expert_count(&runtime, 2);
    astro_expert_cache
        .queue_pending_routed_expert_insert(0, &[7, 9], &weights, WARM_SLOT_COUNT)
        .expect("the warm insert should queue");
    astro_expert_cache
        .flush_pending_inserts(&runtime)
        .expect("the first warm flush should succeed");

    // A layer without a table answers None for the table-shaped queries and
    // zero occupancy, never an empty table claim.
    assert_eq!(astro_expert_cache.missing_route_expert_ids(1, &[3]), None);
    assert_eq!(
        astro_expert_cache.non_resident_expert_ids(1, EXPERT_CAPACITY),
        None
    );
    assert_eq!(astro_expert_cache.occupied_slot_count(1), 0);

    // A partial table reports exactly the route experts it lacks, the full
    // non-resident read set, and its occupancy.
    assert_eq!(
        astro_expert_cache.missing_route_expert_ids(0, &[7, 9]),
        Some(vec![])
    );
    assert_eq!(
        astro_expert_cache.missing_route_expert_ids(0, &[9, 12]),
        Some(vec![12])
    );
    assert_eq!(
        astro_expert_cache
            .non_resident_expert_ids(0, EXPERT_CAPACITY)
            .expect("a seated layer reports its read set"),
        (0..EXPERT_CAPACITY)
            .filter(|expert_id| ![7, 9].contains(expert_id))
            .collect::<Vec<_>>()
    );
    assert_eq!(astro_expert_cache.occupied_slot_count(0), 2);

    // Inputs are normalized: duplicates and unsorted routes answer in stable
    // sorted order.
    assert_eq!(
        astro_expert_cache.missing_route_expert_ids(0, &[12, 7, 12]),
        Some(vec![12])
    );
}

#[test]
#[serial]
fn should_report_real_demand_coverage_from_the_recorded_routing_evidence() {
    let runtime = test_runtime();
    let mut astro_expert_cache = AstroExpertCache::new(1);
    // Production ceilings are set by the residency plan refresh; the tests
    // set a generous ceiling except the budget-refusal case.
    astro_expert_cache.set_capacity(1024 * 1024 * 1024);
    let weights = streamed_weights_for_expert_count(&runtime, 2);

    astro_expert_cache
        .queue_pending_routed_expert_insert(0, &[7, 9], &weights, WARM_SLOT_COUNT)
        .expect("the warm insert should queue");
    astro_expert_cache
        .flush_pending_inserts(&runtime)
        .expect("the warm flush should succeed");

    // With no routing evidence the coverage is zero; recorded demand for
    // retained experts makes the residency planner frequency-aware.
    assert_eq!(
        astro_expert_cache.topology_snapshot(EXPERT_CAPACITY)[0].covered_weighted_demand,
        0
    );
    astro_expert_cache.record_expert_demand(0, EXPERT_CAPACITY, &[7, 9, 9]);
    astro_expert_cache.record_expert_demand(0, EXPERT_CAPACITY, &[11]);
    let snapshot = astro_expert_cache.topology_snapshot(EXPERT_CAPACITY);
    // Experts 7 and 9 are retained; 11 is routed but not retained.
    assert_eq!(snapshot[0].covered_weighted_demand, 3);
}

#[test]
#[serial]
fn should_prepare_no_optimistic_lookup_for_a_layer_without_a_table() {
    let mut astro_expert_cache = AstroExpertCache::new(2);
    astro_expert_cache.set_capacity(1024 * 1024 * 1024);

    assert!(
        astro_expert_cache
            .optimistic_table_page(1, EXPERT_CAPACITY)
            .is_none()
    );
}

#[test]
#[serial]
fn should_mark_every_expert_a_partial_table_lacks_in_its_optimistic_lookup() {
    let runtime = test_runtime();
    let mut astro_expert_cache = AstroExpertCache::new(1);
    astro_expert_cache.set_capacity(1024 * 1024 * 1024);
    let weights = streamed_weights_for_expert_count(&runtime, 2);
    astro_expert_cache
        .queue_pending_routed_expert_insert(0, &[7, 9], &weights, WARM_SLOT_COUNT)
        .expect("the warm insert should queue");
    astro_expert_cache
        .flush_pending_inserts(&runtime)
        .expect("the warm flush should succeed");

    let table_page = astro_expert_cache
        .optimistic_table_page(0, EXPERT_CAPACITY)
        .expect("a table that holds experts can be looked up");

    let missing_expert_mask = table_page
        .missing_expert_mask
        .expect("a partial table can miss");
    assert_eq!(missing_expert_mask.len(), EXPERT_CAPACITY);
    assert_eq!(missing_expert_mask[7], 0);
    assert_eq!(missing_expert_mask[9], 0);
    assert_eq!(
        missing_expert_mask.iter().sum::<u32>() as usize,
        EXPERT_CAPACITY - 2
    );
    assert_eq!(table_page.manifest.expert_ids, vec![7, 9]);
    // An expert the table lacks is pointed at a slot inside the table, so
    // the lookup stays in bounds while the mask reports the miss.
    let slot_count =
        u32::try_from(astro_expert_cache.occupied_slot_count(0)).expect("a small table");
    assert!(table_page.manifest.page_slot_by_global_expert_id[3] < slot_count.max(1));
    assert_ne!(
        table_page.manifest.page_slot_by_global_expert_id[7],
        table_page.manifest.page_slot_by_global_expert_id[9]
    );
}
