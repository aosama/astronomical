use super::WarmExpertSlotTableCache;
use super::warm_expert_slot_table_cache_tests::{
    EXPERT_CAPACITY, streamed_weights_for_expert_count, test_runtime,
};
use serial_test::serial;

#[test]
#[serial]
fn should_complete_a_cold_prefill_union_seed_using_only_missing_experts() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);

    let seeded_expert_ids: Vec<usize> = (0..3).collect();
    let seeded_weights = streamed_weights_for_expert_count(&runtime, seeded_expert_ids.len());
    let seeded_expert_count = cache
        .insert_streamed_experts(
            &runtime,
            0,
            &seeded_expert_ids,
            &seeded_weights,
            &seeded_expert_ids,
            EXPERT_CAPACITY,
        )
        .expect("the initial prefill union should seed a table");
    assert_eq!(seeded_expert_count, seeded_expert_ids.len());
    assert!(!cache.has_complete_layer(0, EXPERT_CAPACITY));

    let missing_expert_ids = cache
        .non_resident_expert_ids(0, EXPERT_CAPACITY)
        .expect("the seeded table should identify its missing experts");
    assert_eq!(
        missing_expert_ids,
        (seeded_expert_ids.len()..EXPERT_CAPACITY).collect::<Vec<_>>()
    );
    let completion_weights = streamed_weights_for_expert_count(&runtime, missing_expert_ids.len());
    let inserted_expert_count = cache
        .insert_streamed_experts(
            &runtime,
            0,
            &missing_expert_ids,
            &completion_weights,
            &missing_expert_ids,
            EXPERT_CAPACITY,
        )
        .expect("the missing experts should complete the table");

    assert_eq!(
        inserted_expert_count,
        EXPERT_CAPACITY - seeded_expert_ids.len()
    );
    assert!(cache.has_complete_layer(0, EXPERT_CAPACITY));
    let complete_page = cache
        .full_coverage_page(
            0,
            &(0..EXPERT_CAPACITY).collect::<Vec<_>>(),
            EXPERT_CAPACITY,
        )
        .expect("the completed table should cover every expert");
    assert_eq!(
        complete_page.1.expert_ids,
        (0..EXPERT_CAPACITY).collect::<Vec<_>>()
    );
}
