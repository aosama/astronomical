//! Unit tests for the plan that gives back the least-read experts across
//! every layer when the astro expert cache's capacity falls.

use super::super::astro_expert_cache_tests::{
    EXPERT_CAPACITY, STREAMED_PER_EXPERT_PAYLOAD_BYTES, WARM_SLOT_COUNT,
    streamed_weights_for_expert_count, test_runtime,
};
use super::{HeldSlotDemand, LayerTrim, TableTrimView, plan_hot_expert_trim, rebuild_fits_within};

const SLOT_BYTES: u64 = 100;

fn table_view(slot_count: usize, read_count_by_held_slot: &[(usize, u64)]) -> TableTrimView {
    let mut held_slot_demand = Vec::new();
    for &(slot, read_count) in read_count_by_held_slot {
        held_slot_demand.push(HeldSlotDemand {
            slot,
            decode_demand_count: 0,
            prefill_demand_count: 0,
            read_count,
        });
    }
    TableTrimView {
        slot_bytes: SLOT_BYTES,
        slot_count,
        held_slot_demand,
    }
}

#[test]
fn should_keep_every_layer_hot_experts_instead_of_dropping_whole_layers() {
    // Two complete layers whose least-read experts total 200 bytes: the trim
    // takes one expert from each, not the whole of either layer.
    let table_views = vec![
        Some(table_view(3, &[(0, 9), (1, 1), (2, 7)])),
        Some(table_view(3, &[(0, 2), (1, 8), (2, 6)])),
    ];

    let layer_trims = plan_hot_expert_trim(&table_views, 2 * SLOT_BYTES);

    assert_eq!(
        layer_trims,
        vec![
            LayerTrim::KeepSlots(vec![0, 2]),
            LayerTrim::KeepSlots(vec![1, 2])
        ]
    );
}

#[test]
fn should_give_back_unused_slots_before_any_held_expert() {
    let table_views = vec![
        Some(table_view(4, &[(0, 1), (1, 1)])),
        Some(table_view(2, &[(0, 0), (1, 0)])),
    ];

    let layer_trims = plan_hot_expert_trim(&table_views, 2 * SLOT_BYTES);

    assert_eq!(
        layer_trims,
        vec![LayerTrim::KeepSlots(vec![0, 1]), LayerTrim::Unchanged]
    );
}

#[test]
fn should_drop_a_layer_only_when_all_its_experts_are_the_least_read() {
    let table_views = vec![
        Some(table_view(2, &[(0, 1), (1, 1)])),
        Some(table_view(2, &[(0, 50), (1, 60)])),
    ];

    let layer_trims = plan_hot_expert_trim(&table_views, 2 * SLOT_BYTES);

    assert_eq!(layer_trims, vec![LayerTrim::Drop, LayerTrim::Unchanged]);
}

#[test]
fn should_rank_reply_demand_above_the_reads_a_whole_prompt_layer_collected() {
    // Layer 0 was seated whole by the prompt: many reads, no reply demand.
    // Layer 1 holds the two experts the reply keeps routing to.
    let mut prompt_seated_layer = table_view(2, &[(0, 500), (1, 400)]);
    let mut reply_hot_layer = table_view(2, &[(0, 1), (1, 1)]);
    reply_hot_layer.held_slot_demand[0].decode_demand_count = 8;
    reply_hot_layer.held_slot_demand[1].decode_demand_count = 6;
    prompt_seated_layer.held_slot_demand[1].prefill_demand_count = 900;

    let layer_trims = plan_hot_expert_trim(
        &[Some(prompt_seated_layer), Some(reply_hot_layer)],
        SLOT_BYTES,
    );

    // Among the experts no reply token uses, the prompt's own demand keeps
    // slot 1 and gives back slot 0.
    assert_eq!(
        layer_trims,
        vec![LayerTrim::KeepSlots(vec![1]), LayerTrim::Unchanged]
    );
}

#[test]
fn should_rebuild_a_table_that_fits_beside_the_old_one_exactly() {
    assert!(rebuild_fits_within(900, 100, 1_000));
    assert!(rebuild_fits_within(100, 200, 1_000));
}

#[test]
fn should_drop_instead_of_rebuilding_a_table_that_would_cross_the_limit() {
    assert!(!rebuild_fits_within(901, 100, 1_000));
    assert!(!rebuild_fits_within(u64::MAX, 1, u64::MAX));
}

#[test]
fn should_change_nothing_when_nothing_has_to_be_released() {
    let table_views = vec![Some(table_view(2, &[(0, 1), (1, 1)])), None];

    let layer_trims = plan_hot_expert_trim(&table_views, 0);

    assert_eq!(
        layer_trims,
        vec![LayerTrim::Unchanged, LayerTrim::Unchanged]
    );
}

#[test]
fn should_release_at_least_the_bytes_asked_for_even_inside_one_slot() {
    // A single byte still costs a whole slot: the table cannot free less.
    let table_views = vec![Some(table_view(2, &[(0, 3), (1, 5)]))];

    let layer_trims = plan_hot_expert_trim(&table_views, 1);

    assert_eq!(layer_trims, vec![LayerTrim::KeepSlots(vec![1])]);
}

#[test]
fn should_drop_every_table_when_more_is_asked_than_every_table_holds() {
    let table_views = vec![
        Some(table_view(2, &[(0, 3), (1, 5)])),
        None,
        Some(table_view(1, &[(0, 9)])),
    ];

    let layer_trims = plan_hot_expert_trim(&table_views, 100 * SLOT_BYTES);

    assert_eq!(
        layer_trims,
        vec![LayerTrim::Drop, LayerTrim::Unchanged, LayerTrim::Drop]
    );
}

#[test]
fn should_leave_layers_without_a_table_unchanged() {
    let table_views = vec![None, None];

    let layer_trims = plan_hot_expert_trim(&table_views, 5 * SLOT_BYTES);

    assert_eq!(
        layer_trims,
        vec![LayerTrim::Unchanged, LayerTrim::Unchanged]
    );
}

/// Two layers of four experts each, with experts 1 and 2 of layer 0 and 5
/// and 6 of layer 1 read far more often than the rest.
fn astro_expert_cache_with_two_warm_layers(
    runtime: &runtime_integration::MlxRuntime,
) -> super::AstroExpertCache {
    let mut astro_expert_cache = super::AstroExpertCache::new(2);
    astro_expert_cache.set_capacity(1024 * 1024 * 1024);
    for (layer_index, routed_expert_ids) in [(0, vec![1, 2, 3, 4]), (1, vec![5, 6, 7, 8])] {
        let streamed_weights = streamed_weights_for_expert_count(runtime, routed_expert_ids.len());
        astro_expert_cache
            .queue_pending_routed_expert_insert(
                layer_index,
                &routed_expert_ids,
                &streamed_weights,
                WARM_SLOT_COUNT,
            )
            .expect("the warm insert should queue");
        astro_expert_cache
            .flush_pending_inserts(runtime)
            .expect("the warm insert should flush");
    }
    for _ in 0..5 {
        astro_expert_cache.record_routed_reads(0, &[1, 2]);
        astro_expert_cache.record_routed_reads(1, &[5, 6]);
    }
    astro_expert_cache
}

fn sum_of_table_bytes(astro_expert_cache: &super::AstroExpertCache) -> u64 {
    let mut table_bytes = 0_u64;
    for table in astro_expert_cache.tables_by_layer.iter().flatten() {
        table_bytes = table_bytes.saturating_add(table.full_padded_payload_bytes);
    }
    table_bytes
}

#[test]
#[serial_test::file_serial]
fn should_rebuild_each_layer_with_its_most_read_experts_when_the_capacity_falls() {
    let runtime = test_runtime();
    let mut astro_expert_cache = astro_expert_cache_with_two_warm_layers(&runtime);
    let lowered_capacity_bytes = 4 * STREAMED_PER_EXPERT_PAYLOAD_BYTES;

    let reclamation =
        astro_expert_cache.set_capacity_keeping_hot_experts(&runtime, lowered_capacity_bytes);

    assert!(reclamation.released_payload_bytes() > 0);
    for (layer_index, hot_expert_ids, cold_expert_ids) in [(0, [1, 2], [3, 4]), (1, [5, 6], [7, 8])]
    {
        let coverage = astro_expert_cache.routed_expert_coverage(layer_index, &hot_expert_ids);
        assert_eq!(coverage.retained_expert_ids, hot_expert_ids.to_vec());
        let cold_coverage =
            astro_expert_cache.routed_expert_coverage(layer_index, &cold_expert_ids);
        assert_eq!(cold_coverage.missing_expert_ids, cold_expert_ids.to_vec());
        assert_eq!(astro_expert_cache.occupied_slot_count(layer_index), 2);
    }
    let resident_payload_bytes = astro_expert_cache.statistics().resident_payload_byte_count;
    assert!(resident_payload_bytes <= lowered_capacity_bytes);
    assert_eq!(
        resident_payload_bytes,
        sum_of_table_bytes(&astro_expert_cache)
    );
}

#[test]
#[serial_test::file_serial]
fn should_serve_a_kept_expert_from_the_rebuilt_table() {
    let runtime = test_runtime();
    let mut astro_expert_cache = astro_expert_cache_with_two_warm_layers(&runtime);

    astro_expert_cache
        .set_capacity_keeping_hot_experts(&runtime, 4 * STREAMED_PER_EXPERT_PAYLOAD_BYTES);

    let (_, kept_page_manifest) = astro_expert_cache
        .full_coverage_page(0, &[1, 2], EXPERT_CAPACITY)
        .expect("the rebuilt table should still serve its kept experts");
    assert_eq!(kept_page_manifest.expert_ids, vec![1, 2]);
    assert!(
        astro_expert_cache
            .full_coverage_page(0, &[3], EXPERT_CAPACITY)
            .is_none()
    );
}

#[test]
#[serial_test::file_serial]
fn should_leave_the_tables_alone_when_the_capacity_still_holds_them() {
    let runtime = test_runtime();
    let mut astro_expert_cache = astro_expert_cache_with_two_warm_layers(&runtime);
    let resident_payload_bytes = astro_expert_cache.statistics().resident_payload_byte_count;

    let reclamation =
        astro_expert_cache.set_capacity_keeping_hot_experts(&runtime, resident_payload_bytes);

    assert_eq!(reclamation.released_payload_bytes(), 0);
    assert_eq!(astro_expert_cache.occupied_slot_count(0), 4);
    assert_eq!(astro_expert_cache.occupied_slot_count(1), 4);
}

#[test]
#[serial_test::file_serial]
fn should_drop_every_table_when_the_capacity_falls_to_zero() {
    let runtime = test_runtime();
    let mut astro_expert_cache = astro_expert_cache_with_two_warm_layers(&runtime);

    astro_expert_cache.set_capacity_keeping_hot_experts(&runtime, 0);

    assert_eq!(
        astro_expert_cache.statistics().resident_payload_byte_count,
        0
    );
    assert_eq!(astro_expert_cache.occupied_slot_count(0), 0);
    assert_eq!(astro_expert_cache.occupied_slot_count(1), 0);
}
