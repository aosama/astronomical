//! Direct-MLX unit tests for the retained expert slot-table cache: padded
//! warm-table creation, partial-hit serving, least-frequently-used eviction,
//! demand-coverage reporting, budget refusal, and warm-insert evidence scoping.

use super::*;
use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

const WARM_SLOT_COUNT: usize = 8;
const EXPERT_CAPACITY: usize = 16;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the warm-cache test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Two-row native projections: the streamed routed-set shape.
fn streamed_weights(runtime: &MlxRuntime) -> Qwen3_5PagedExpertWeights {
    let projection = |fill: f32| {
        runtime
            .array_from_f32(&vec![fill; 8], &[2, 4])
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
    }
}

#[test]
fn should_create_a_padded_warm_table_from_the_first_decode_miss_and_serve_later_hits() {
    let runtime = test_runtime();
    let mut cache = RetainedExpertCache::new(1);
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

    // The warm table holds the policy capacity, not just the routed rows.
    let residencies = cache.topology_snapshot(EXPERT_CAPACITY);
    assert_eq!(residencies.len(), 1);
    assert_eq!(
        residencies[0].class,
        RetainedExpertPageClass::ElasticRoutedExperts
    );
    assert_eq!(residencies[0].retained_expert_ids, vec![7, 9]);
    assert!(!cache.has_complete_layer(0, EXPERT_CAPACITY));

    // A warm table that covers the routed set serves a partial hit.
    let packed_page = cache
        .packed_page(0, &[7, 9], EXPERT_CAPACITY)
        .expect("a covering warm table should serve the routed set");
    assert_eq!(packed_page.1.expert_ids, vec![7, 9]);
    assert_eq!(packed_page.1.page_slot_by_global_expert_id[7], 0);
    assert_eq!(packed_page.1.page_slot_by_global_expert_id[9], 1);

    // A routed set the table does not fully cover is a miss.
    assert!(cache.packed_page(0, &[7, 11], EXPERT_CAPACITY).is_none());

    // Free slots fill before any eviction, growing the hot set.
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
fn should_evict_the_least_read_expert_when_a_warm_table_overflows() {
    let runtime = test_runtime();
    let mut cache = RetainedExpertCache::new(1);
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
fn should_report_real_demand_coverage_from_the_recorded_routing_evidence() {
    let runtime = test_runtime();
    let mut cache = RetainedExpertCache::new(1);
    // Production ceilings are set by the residency plan refresh; the tests
    // set a generous ceiling except the budget-refusal case.
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    let weights = streamed_weights(&runtime);

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

#[test]
fn should_keep_the_stream_operation_local_when_the_budget_refuses_the_warm_table() {
    let runtime = test_runtime();
    let mut cache = RetainedExpertCache::new(1);
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
fn should_count_warm_inserts_only_for_hot_expert_warming() {
    let runtime = test_runtime();
    let mut cache = RetainedExpertCache::new(1);
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
