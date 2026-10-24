//! Direct-MLX unit tests for the retained expert slot-table cache: padded
//! warm-table creation, partial-hit serving, least-frequently-used eviction,
//! demand-coverage reporting, budget refusal, and warm-insert evidence scoping.

use super::*;
use crate::memory::RetainedExpertPageClass;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

pub(super) const WARM_SLOT_COUNT: usize = 8;
pub(super) const EXPERT_CAPACITY: usize = 16;
/// One expert of `streamed_weights`: 2x4 float32 arrays x 3 projections.
pub(super) const STREAMED_PER_EXPERT_PAYLOAD_BYTES: u64 = 48;

pub(super) fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the warm-cache test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Two-row native projections: the streamed routed-set shape.
fn streamed_weights(runtime: &MlxRuntime) -> Qwen3_5PagedExpertWeights {
    streamed_weights_for_expert_count(runtime, 2)
}

/// Native projections sized to one routed set so expert rows match ids.
pub(super) fn streamed_weights_for_expert_count(
    runtime: &MlxRuntime,
    expert_count: usize,
) -> Qwen3_5PagedExpertWeights {
    let projection = |fill: f32| {
        runtime
            .array_from_f32(
                &vec![fill; expert_count * 4],
                &[
                    i32::try_from(expert_count).expect("the expert count fits i32"),
                    4,
                ],
            )
            .expect("the test projection should be valid")
    };
    Qwen3_5PagedExpertWeights {
        gate_projection: Qwen3_5AffineWeights::NativeBfloat16 {
            weight: projection(1.0),
        },
        up_projection: Qwen3_5AffineWeights::NativeBfloat16 {
            weight: projection(2.0),
        },
        down_projection: Qwen3_5AffineWeights::NativeBfloat16 {
            weight: projection(3.0),
        },
        gate_activation_rotation:
            astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::NoRotation,
        up_activation_rotation:
            astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::NoRotation,
        down_activation_rotation:
            astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::NoRotation,
    }
}

#[test]
#[serial]
fn should_grow_a_warm_table_from_the_first_decode_miss_and_serve_later_hits() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(1);
    // Production ceilings are set by the residency plan refresh; the tests
    // set a generous ceiling except the budget-refusal case.
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    let weights = streamed_weights(&runtime);

    cache
        .queue_pending_routed_expert_insert(0, &[7, 9], &weights, WARM_SLOT_COUNT)
        .expect("the warm insert should queue");
    let written_count = cache
        .flush_pending_inserts(&runtime)
        .expect("the first warm flush should succeed");
    assert_eq!(written_count, 2);

    // The warm table starts at the routed set size, holding only real experts.
    let residencies = cache.topology_snapshot(EXPERT_CAPACITY);
    assert_eq!(residencies.len(), 1);
    assert_eq!(
        residencies[0].class,
        RetainedExpertPageClass::ElasticRoutedExperts
    );
    assert_eq!(residencies[0].retained_expert_ids, vec![7, 9]);
    assert!(!cache.has_complete_layer(0, EXPERT_CAPACITY));

    // A warm table that covers the routed set serves a partial hit.
    let full_coverage_page = cache
        .full_coverage_page(0, &[7, 9], EXPERT_CAPACITY)
        .expect("a covering warm table should serve the routed set");
    assert_eq!(full_coverage_page.1.expert_ids, vec![7, 9]);
    assert_eq!(full_coverage_page.1.page_slot_by_global_expert_id[7], 0);
    assert_eq!(full_coverage_page.1.page_slot_by_global_expert_id[9], 1);

    // A routed set the table does not fully cover is a miss.
    assert!(
        cache
            .full_coverage_page(0, &[7, 11], EXPERT_CAPACITY)
            .is_none()
    );

    // The table grows to fit demonstrated demand before any eviction.
    cache
        .queue_pending_routed_expert_insert(0, &[11, 13], &weights, WARM_SLOT_COUNT)
        .expect("the second warm insert should queue");
    let second_written = cache
        .flush_pending_inserts(&runtime)
        .expect("the second warm flush should succeed");
    assert_eq!(second_written, 2);
    assert_eq!(
        cache.topology_snapshot(EXPERT_CAPACITY)[0].retained_expert_ids,
        vec![7, 9, 11, 13]
    );
}

#[test]
#[serial]
fn should_evict_the_least_read_expert_when_a_warm_table_overflows() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(1);
    // Production ceilings are set by the residency plan refresh; the tests
    // set a generous ceiling except the budget-refusal case.
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    let weights = streamed_weights(&runtime);

    for routed_expert_ids in [vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8]] {
        cache
            .queue_pending_routed_expert_insert(0, &routed_expert_ids, &weights, WARM_SLOT_COUNT)
            .expect("the warm insert should queue");
        cache
            .flush_pending_inserts(&runtime)
            .expect("the warm flush should succeed");
    }
    // The table is full: every slot is occupied.
    assert_eq!(
        cache.topology_snapshot(EXPERT_CAPACITY)[0].retained_expert_ids,
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );

    // Reads make experts 1 and 2 hot; the next insert must evict cold ones.
    for _read in 0..3 {
        cache.record_routed_reads(0, &[1, 2]);
    }
    cache
        .queue_pending_routed_expert_insert(0, &[9, 10], &weights, WARM_SLOT_COUNT)
        .expect("the overflow warm insert should queue");
    cache
        .flush_pending_inserts(&runtime)
        .expect("the overflow warm flush should succeed");
    let retained_ids = cache.topology_snapshot(EXPERT_CAPACITY)[0]
        .retained_expert_ids
        .clone();
    assert_eq!(retained_ids.len(), 8);
    // The hot experts stayed; two cold experts yielded their slots.
    assert!(retained_ids.contains(&1));
    assert!(retained_ids.contains(&2));
    assert!(retained_ids.contains(&9));
    assert!(retained_ids.contains(&10));
    assert!(!retained_ids.contains(&3));
    assert!(!retained_ids.contains(&4));
}

#[test]
#[serial]
fn should_keep_the_stream_operation_local_when_the_budget_refuses_the_warm_table() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(1);
    // Production ceilings are set by the residency plan refresh; the tests
    // set a generous ceiling except the budget-refusal case.
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    let weights = streamed_weights(&runtime);
    // A ceiling below the padded warm-table cost refuses creation.
    cache.update_maximum_resident_payload_bytes(1);

    cache
        .queue_pending_routed_expert_insert(0, &[7, 9], &weights, WARM_SLOT_COUNT)
        .expect("the warm insert should queue even under budget pressure");
    let written_count = cache
        .flush_pending_inserts(&runtime)
        .expect("a budget-refused flush should stay graceful");
    assert_eq!(written_count, 0);
    assert!(cache.topology_snapshot(EXPERT_CAPACITY).is_empty());
}

#[test]
#[serial]
fn should_scale_final_prefill_chunk_demand_above_earlier_prompt_frequency() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(2);
    // Production ceilings are set by the residency plan refresh; the tests
    // set a generous ceiling except the budget-refusal case.
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    let weights = streamed_weights(&runtime);
    for (layer_index, expert_ids) in [(0, &[7, 9][..]), (1, &[3, 5][..])] {
        cache
            .queue_pending_routed_expert_insert(layer_index, expert_ids, &weights, WARM_SLOT_COUNT)
            .expect("the warm insert should queue");
    }
    cache
        .flush_pending_inserts(&runtime)
        .expect("the warm flush should succeed");

    // Five earlier-prompt routes at plain weight; three final-chunk routes
    // recorded under the token-density boost.
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[7, 7, 7, 7, 7]);
    cache.set_demand_assignment_weight(2);
    cache.record_expert_demand(1, EXPERT_CAPACITY, &[3, 3, 3]);
    let snapshot = cache.topology_snapshot(EXPERT_CAPACITY);
    assert_eq!(snapshot[0].covered_weighted_demand, 5);
    assert_eq!(snapshot[1].covered_weighted_demand, 6);

    // A zero weight must not discard assignments; it floors at one.
    cache.set_demand_assignment_weight(0);
    cache.record_expert_demand(1, EXPERT_CAPACITY, &[3]);
    let snapshot = cache.topology_snapshot(EXPERT_CAPACITY);
    assert_eq!(snapshot[1].covered_weighted_demand, 7);

    // A cleared window restores the plain assignment weight.
    cache.clear_expert_demand();
    cache.record_expert_demand(0, EXPERT_CAPACITY, &[7]);
    let snapshot = cache.topology_snapshot(EXPERT_CAPACITY);
    assert_eq!(snapshot[0].covered_weighted_demand, 1);
}

#[test]
#[serial]
fn should_count_warm_inserts_only_for_hot_expert_warming() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(1);
    // Production ceilings are set by the residency plan refresh; the tests
    // set a generous ceiling except the budget-refusal case.
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    let weights = streamed_weights(&runtime);

    // A complete adoption (warm capacity 0) is whole-layer caching and
    // must not count toward hot-expert warming evidence.
    cache
        .insert_streamed_experts(&runtime, 0, &[0, 1, 2, 3, 4, 5, 6, 7], &weights, &[], 0)
        .expect("the complete adoption should succeed");
    assert_eq!(cache.warm_expert_insert_count, 0);

    // Hot-expert warming counts its experts.
    cache
        .queue_pending_routed_expert_insert(0, &[9, 11], &weights, WARM_SLOT_COUNT)
        .expect("the warm insert should queue");
    cache
        .flush_pending_inserts(&runtime)
        .expect("the warm flush should succeed");
    assert_eq!(cache.warm_expert_insert_count, 2);
}

const HOT_SET_EXPERT_COUNT: usize = 2;
const SIBLING_LAYER_COUNT: usize = 3;

/// Two sequential routed sets on one layer drive the table from its routed
/// size to its first growth, mirroring a layer's first two decode tokens.
fn warm_two_tokens(
    cache: &mut WarmExpertSlotTableCache,
    runtime: &MlxRuntime,
    layer_index: usize,
    first_routed_expert_ids: &[usize],
    second_routed_expert_ids: &[usize],
) {
    let weights = streamed_weights(runtime);
    cache
        .queue_pending_routed_expert_insert(
            layer_index,
            first_routed_expert_ids,
            &weights,
            WARM_SLOT_COUNT,
        )
        .expect("the first-token warm insert should queue");
    let first_written_count = cache
        .flush_pending_inserts(runtime)
        .expect("the first-token warm flush should succeed");
    assert_eq!(first_written_count, HOT_SET_EXPERT_COUNT as u64);
    cache
        .queue_pending_routed_expert_insert(
            layer_index,
            second_routed_expert_ids,
            &weights,
            WARM_SLOT_COUNT,
        )
        .expect("the second-token warm insert should queue");
    let second_written_count = cache
        .flush_pending_inserts(runtime)
        .expect("the second-token warm flush should succeed");
    assert_eq!(second_written_count, HOT_SET_EXPERT_COUNT as u64);
}

/// Issue #955, mechanism pin (mid-pyramid): warming must scale with
/// demonstrated demand. A table that starts at the routed set and grows only
/// when full leaves budget headroom for sibling layers after the first
/// tokens; eager full-capacity padding consumed the entire budget with
/// mostly-zero tables, the per-token slot resize then reported zero
/// affordable slots, and every later token re-streamed its experts from disk.
#[test]
#[serial]
fn should_keep_budget_headroom_for_sibling_layers_when_the_first_warm_tables_are_barely_filled() {
    let runtime = test_runtime();
    let mut probe_cache = WarmExpertSlotTableCache::new(1);
    probe_cache.update_maximum_resident_payload_bytes(u64::MAX >> 1);
    warm_two_tokens(&mut probe_cache, &runtime, 0, &[7, 9], &[11, 13]);
    let grown_table_payload_bytes = probe_cache.statistics().resident_payload_byte_count;
    drop(probe_cache);
    // The ceiling affords two grown tables plus one fresh routed-size table.
    let routed_size_table_payload_bytes = grown_table_payload_bytes / 2;
    let ceiling_bytes = 2 * grown_table_payload_bytes + routed_size_table_payload_bytes;
    let mut cache = WarmExpertSlotTableCache::new(SIBLING_LAYER_COUNT);
    cache.update_maximum_resident_payload_bytes(ceiling_bytes);

    warm_two_tokens(&mut cache, &runtime, 0, &[7, 9], &[11, 13]);
    warm_two_tokens(&mut cache, &runtime, 1, &[7, 9], &[11, 13]);

    // The first two layers warmed through their growth; the third layer's
    // routed set is exactly as hot and must still fit the budget.
    cache
        .queue_pending_routed_expert_insert(
            2,
            &[7, 9],
            &streamed_weights(&runtime),
            WARM_SLOT_COUNT,
        )
        .expect("the sibling-layer warm insert should queue");
    let sibling_written_count = cache
        .flush_pending_inserts(&runtime)
        .expect("the sibling-layer warm flush should succeed");
    assert_eq!(
        sibling_written_count, HOT_SET_EXPERT_COUNT as u64,
        "the third layer's routed set must still warm after the first two \
         layers warmed two tokens each"
    );
    let residencies = cache.topology_snapshot(EXPERT_CAPACITY);
    assert_eq!(residencies.len(), SIBLING_LAYER_COUNT);
    assert!(cache.statistics().resident_payload_byte_count <= ceiling_bytes);
}

/// Issue #955 fallback: when the budget refuses a table's growth, the insert
/// must still land through least-read eviction inside the existing capacity
/// instead of dropping the routed experts entirely.
#[test]
#[serial]
fn should_fall_back_to_least_read_eviction_when_the_budget_refuses_growth() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(1);
    let weights = streamed_weights(&runtime);
    // Two routed-size inserts grow the table 2 -> 4; the ceiling then refuses
    // any further growth, so the third new routed set must evict.
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    cache
        .queue_pending_routed_expert_insert(0, &[1, 2], &weights, WARM_SLOT_COUNT)
        .expect("the first warm insert should queue");
    cache
        .flush_pending_inserts(&runtime)
        .expect("the first warm flush should succeed");
    cache
        .queue_pending_routed_expert_insert(0, &[3, 4], &weights, WARM_SLOT_COUNT)
        .expect("the second warm insert should queue");
    cache
        .flush_pending_inserts(&runtime)
        .expect("the second warm flush should succeed");
    let grown_table_payload_bytes = cache.statistics().resident_payload_byte_count;
    // Budget admits the current table but not its doubling.
    cache.update_maximum_resident_payload_bytes(grown_table_payload_bytes);

    for _read in 0..3 {
        cache.record_routed_reads(0, &[1, 2]);
    }
    cache
        .queue_pending_routed_expert_insert(0, &[5, 6], &weights, WARM_SLOT_COUNT)
        .expect("the growth-refused warm insert should queue");
    let written_count = cache
        .flush_pending_inserts(&runtime)
        .expect("the growth-refused warm flush should succeed");
    assert_eq!(
        written_count, 2,
        "both routed experts must warm via eviction"
    );
    let retained_expert_ids = cache.topology_snapshot(EXPERT_CAPACITY)[0]
        .retained_expert_ids
        .clone();
    assert!(retained_expert_ids.contains(&1) && retained_expert_ids.contains(&2));
    assert!(retained_expert_ids.contains(&5) && retained_expert_ids.contains(&6));
    assert_eq!(
        u64::try_from(retained_expert_ids.len()).unwrap_or(0) * STREAMED_PER_EXPERT_PAYLOAD_BYTES,
        cache.topology_snapshot(EXPERT_CAPACITY)[0].payload_bytes,
        "eviction inside a growth-refused table must stay payload-net-constant"
    );
}

/// Incident reproduction (2026-10-15 paged-prefill stall), cache level: with
/// the ceiling set to the composed budget, complete-layer adoption offers
/// stop exactly at the budget — the cache's own admission arithmetic is sound.
/// Oversubscription therefore cannot originate in `try_adopt_complete_layer`;
/// it can only arrive through a ceiling raised above the composed budget.
#[test]
#[serial]
fn should_seat_at_most_the_composed_ceiling_and_refuse_every_further_complete_layer() {
    let runtime = test_runtime();
    let sparse_layer_count = 8_usize;
    let mut cache = WarmExpertSlotTableCache::new(sparse_layer_count);
    // Five tables of 768 bytes each: the composed-budget analogue of the
    // incident's 33 affordable layers out of 38.
    let affordable_complete_layer_count = 5_u64;
    let complete_layer_payload_bytes =
        u64::try_from(EXPERT_CAPACITY).unwrap_or(0) * STREAMED_PER_EXPERT_PAYLOAD_BYTES;
    let composed_ceiling_bytes = affordable_complete_layer_count * complete_layer_payload_bytes;
    cache.update_maximum_resident_payload_bytes(composed_ceiling_bytes);
    let complete_expert_ids: Vec<usize> = (0..EXPERT_CAPACITY).collect();

    let mut seated_layer_indexes = Vec::new();
    for layer_index in 0..sparse_layer_count {
        let complete_weights = streamed_weights_for_expert_count(&runtime, EXPERT_CAPACITY);
        let seated = cache
            .try_adopt_complete_layer(
                &runtime,
                layer_index,
                &complete_expert_ids,
                &complete_weights,
            )
            .expect("a complete-layer adoption offer must not fail");
        if seated {
            seated_layer_indexes.push(layer_index);
        }
    }

    assert_eq!(seated_layer_indexes, (0..5).collect::<Vec<_>>());
    assert_eq!(
        cache.statistics().resident_payload_byte_count,
        composed_ceiling_bytes,
        "resident payload must stop exactly at the composed ceiling"
    );
}

/// Incident reproduction, cache level: a sub-table pressure target evicts
/// exactly one whole table, and the evicted layer is then unseatable under
/// both documented ceilings — the pressure override and the budget-refresh
/// floor — so every later forward re-pays its complete-page SSD read. This is
/// the steady state the incident's halving loop executed for 69 seconds.
#[test]
#[serial]
fn should_evict_one_whole_table_for_a_sub_table_target_and_never_reseat_it() {
    let runtime = test_runtime();
    let sparse_layer_count = 8_usize;
    let mut cache = WarmExpertSlotTableCache::new(sparse_layer_count);
    let complete_layer_payload_bytes =
        u64::try_from(EXPERT_CAPACITY).unwrap_or(0) * STREAMED_PER_EXPERT_PAYLOAD_BYTES;
    let composed_ceiling_bytes = 5 * complete_layer_payload_bytes;
    cache.update_maximum_resident_payload_bytes(composed_ceiling_bytes);
    let complete_expert_ids: Vec<usize> = (0..EXPERT_CAPACITY).collect();
    for layer_index in 0..5_usize {
        let complete_weights = streamed_weights_for_expert_count(&runtime, EXPERT_CAPACITY);
        assert!(
            cache
                .try_adopt_complete_layer(
                    &runtime,
                    layer_index,
                    &complete_expert_ids,
                    &complete_weights,
                )
                .expect("the first five adoptions must seat")
        );
    }

    // The incident's recovery demanded 64,597,002 bytes while the smallest
    // eviction granularity was an 855,638,016-byte table; the scaled analogue
    // is a one-expert target against a 768-byte table.
    let sub_table_target_bytes = STREAMED_PER_EXPERT_PAYLOAD_BYTES;
    let released = cache.limit_for_request_pressure(sub_table_target_bytes);
    assert!(
        released,
        "the pressure pass must release at least one table"
    );
    assert_eq!(
        cache.statistics().resident_payload_byte_count,
        4 * complete_layer_payload_bytes,
        "exactly one whole table left despite a one-expert target"
    );

    // The evicted layer re-offers on every later forward; both ceilings
    // refuse it, so its complete page is re-read from SSD each attempt.
    let evicted_layer_index = 0_usize;
    let reoffered_weights = streamed_weights_for_expert_count(&runtime, EXPERT_CAPACITY);
    assert!(
        !cache
            .try_adopt_complete_layer(
                &runtime,
                evicted_layer_index,
                &complete_expert_ids,
                &reoffered_weights,
            )
            .expect("re-adoption under the pressure override must not fail")
    );

    // The next plan publication floors the ceiling at the post-eviction
    // payload (the incident's 31,658,606,592-byte floor against a
    // 28,974,982,400-byte composed budget).
    let floored_ceiling_bytes = crate::retained_resident_ceiling_after_budget_refresh(
        2 * complete_layer_payload_bytes,
        cache.statistics().resident_payload_byte_count,
    );
    cache.update_maximum_resident_payload_bytes(floored_ceiling_bytes);
    assert!(
        !cache
            .try_adopt_complete_layer(
                &runtime,
                evicted_layer_index,
                &complete_expert_ids,
                &reoffered_weights,
            )
            .expect("re-adoption under the floored ceiling must not fail")
    );
    assert_eq!(
        cache.statistics().resident_payload_byte_count,
        4 * complete_layer_payload_bytes,
        "the evicted layer never returns to the cache"
    );
}

/// Issue #955 classification boundary: the complete-layer floors and the
/// pressure reclamation paths treat a warm table that grew to hold every
/// expert of its layer as a pinned complete layer, while strictly partial
/// tables stay elastic and yield under request pressure. Cross-request
/// retention (issue #955) relies on this boundary deciding which tables the
/// pressure paths may reclaim, never on finalization releasing tables
/// outright.
#[test]
#[serial]
fn should_classify_a_filled_warm_table_as_pinned_and_a_partial_one_as_elastic() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(2);
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    let weights = streamed_weights(&runtime);

    // Layer 0 stays partial (elastic); layer 1 receives every expert
    // (pinned complete).
    cache
        .queue_pending_routed_expert_insert(0, &[7, 9], &weights, WARM_SLOT_COUNT)
        .expect("the partial warm insert should queue");
    let every_expert_of_layer_one: Vec<usize> = (0..EXPERT_CAPACITY).collect();
    let complete_layer_weights = streamed_weights_for_expert_count(&runtime, EXPERT_CAPACITY);
    cache
        .queue_pending_routed_expert_insert(
            1,
            &every_expert_of_layer_one,
            &complete_layer_weights,
            EXPERT_CAPACITY,
        )
        .expect("the complete warm insert should queue");
    cache
        .flush_pending_inserts(&runtime)
        .expect("the warm flush should succeed");

    let residencies = cache.topology_snapshot(EXPERT_CAPACITY);
    let residency_class_by_layer: Vec<(usize, RetainedExpertPageClass)> = residencies
        .iter()
        .map(|residency| (residency.layer_index, residency.class))
        .collect();
    assert_eq!(
        residency_class_by_layer,
        vec![
            (0, RetainedExpertPageClass::ElasticRoutedExperts),
            (1, RetainedExpertPageClass::StableCompleteLayer),
        ],
        "a filled warm table must classify as pinned complete so cross-request \
         retention can protect it"
    );
}

#[test]
#[serial]
fn should_raise_demand_coverage_ranking_from_seeded_usage_history() {
    let runtime = test_runtime();
    let mut seeded_cache = WarmExpertSlotTableCache::new(1);
    seeded_cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    // Persisted history says experts 4 and 5 of this layer are hot.
    seeded_cache.seed_expert_demand_history(vec![vec![0, 0, 0, 0, 9, 7]]);
    let weights = streamed_weights(&runtime);
    seeded_cache
        .queue_pending_routed_expert_insert(0, &[4, 5], &weights, WARM_SLOT_COUNT)
        .expect("the warm insert should queue");
    seeded_cache
        .flush_pending_inserts(&runtime)
        .expect("the warm flush should succeed");
    // Live demand adds one more sighting of expert 4 on top of the history.
    seeded_cache.record_expert_demand(0, EXPERT_CAPACITY, &[4]);

    let covered_weighted_demand =
        seeded_cache.topology_snapshot(EXPERT_CAPACITY)[0].covered_weighted_demand;
    assert_eq!(
        covered_weighted_demand, 17,
        "the planner's coverage score must rank seated layers by history plus \
         live demand: 9 + 7 seeded plus 1 live sighting"
    );

    // The control: an identical session without history ranks the same
    // seating strictly lower, so seeded heat genuinely raises priority.
    let mut cold_cache = WarmExpertSlotTableCache::new(1);
    cold_cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    cold_cache
        .queue_pending_routed_expert_insert(0, &[4, 5], &weights, WARM_SLOT_COUNT)
        .expect("the cold warm insert should queue");
    cold_cache
        .flush_pending_inserts(&runtime)
        .expect("the cold warm flush should succeed");
    cold_cache.record_expert_demand(0, EXPERT_CAPACITY, &[4]);
    assert_eq!(
        cold_cache.topology_snapshot(EXPERT_CAPACITY)[0].covered_weighted_demand,
        1
    );
}

#[test]
#[serial]
fn should_serve_a_second_chunk_route_from_a_layer_seated_by_the_first_chunk() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    // Chunk 1 routes the layer's whole expert set; the streamed page covers
    // every expert, so adoption seats the complete layer.
    let complete_weights = streamed_weights_for_expert_count(&runtime, EXPERT_CAPACITY);
    let every_expert_id: Vec<usize> = (0..EXPERT_CAPACITY).collect();
    cache
        .queue_pending_routed_expert_insert(0, &every_expert_id, &complete_weights, EXPERT_CAPACITY)
        .expect("the chunk-1 complete insert should queue");
    cache
        .flush_pending_inserts(&runtime)
        .expect("the chunk-1 flush should succeed");
    assert!(cache.has_complete_layer(0, EXPERT_CAPACITY));

    // Chunk 2 routes an arbitrary subset: coverage is a full hit against the
    // seated layer, so the dispatch serves it from RAM and streams nothing.
    assert_eq!(
        cache
            .missing_route_expert_ids(0, &[3, 7, 12])
            .expect("a seated layer must answer coverage probes"),
        Vec::<usize>::new(),
        "a layer seated by chunk 1 must leave nothing to re-stream in chunk 2"
    );
    let chunk_two_page = cache
        .full_coverage_page(0, &[3, 7, 12], EXPERT_CAPACITY)
        .expect("a seated complete layer must serve the chunk-2 route");
    assert_eq!(
        chunk_two_page.1.expert_ids,
        (0..EXPERT_CAPACITY).collect::<Vec<usize>>()
    );
}

#[test]
#[serial]
fn should_adopt_a_streamed_complete_layer_when_the_budget_admits_it() {
    let runtime = test_runtime();
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    let complete_weights = streamed_weights_for_expert_count(&runtime, EXPERT_CAPACITY);
    let every_expert_id: Vec<usize> = (0..EXPERT_CAPACITY).collect();

    // Positive path: a whole-layer read the budget admits must be adopted,
    // not dropped after serving one forward.
    let adopted = cache
        .try_adopt_complete_layer(&runtime, 0, &every_expert_id, &complete_weights)
        .expect("the adoption should not error");
    assert!(adopted, "a complete read within budget must be adopted");
    assert!(cache.has_complete_layer(0, EXPERT_CAPACITY));
    assert_eq!(
        cache.topology_snapshot(EXPERT_CAPACITY)[0].class,
        RetainedExpertPageClass::StableCompleteLayer
    );

    // Refusal path: a ceiling below the layer's payload keeps the page
    // operation-local and leaves the layer unseated.
    let mut refusing_cache = WarmExpertSlotTableCache::new(1);
    refusing_cache.update_maximum_resident_payload_bytes(8);
    let refused = refusing_cache
        .try_adopt_complete_layer(&runtime, 1, &every_expert_id, &complete_weights)
        .expect("the refused adoption should not error either");
    assert!(!refused, "a complete read above the ceiling must not adopt");
    assert!(!refusing_cache.has_complete_layer(1, EXPERT_CAPACITY));
}
