//! Memory-behavior tests for the retained expert cache: the accounting
//! invariants that survive growth and eviction cycles, and the MLX graph-pin
//! guarantees of the warm-table write paths.

use super::tests::{
    EXPERT_CAPACITY, STREAMED_PER_EXPERT_PAYLOAD_BYTES, WARM_SLOT_COUNT,
    streamed_weights_for_expert_count, test_runtime,
};
use super::*;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;

/// Issue #955 invariant: after every flush, a table's real payload must equal
/// the per-expert geometry bytes times its slot-map occupancy. The second
/// identical ask validates this at plan time, so any accounting drift in
/// warming surfaces as a fatal paging error on the next request.
#[test]
fn should_keep_payload_accounting_exact_across_growth_and_eviction() {
    let runtime = test_runtime();
    let mut cache = RetainedExpertCache::new(1);
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    let routed_expert_set_by_token: [Vec<usize>; 8] = [
        vec![1, 2],
        vec![2, 3],
        vec![4, 5, 6],
        vec![1, 6],
        vec![7, 8, 9],
        vec![3, 9],
        vec![10, 11, 12, 13],
        vec![5, 13],
    ];
    for (token_index, routed_expert_ids) in routed_expert_set_by_token.iter().enumerate() {
        let weights = streamed_weights_for_expert_count(&runtime, routed_expert_ids.len());
        cache
            .queue_pending_routed_expert_insert(0, routed_expert_ids, &weights, WARM_SLOT_COUNT)
            .expect("the warm insert should queue");
        cache
            .flush_pending_inserts(&runtime)
            .unwrap_or_else(|error| panic!("token {token_index} warm flush failed: {error}"));
        let residencies = cache.topology_snapshot(EXPERT_CAPACITY);
        assert_eq!(residencies.len(), 1, "token {token_index}");
        assert_eq!(
            residencies[0].retained_expert_ids.len(),
            cache.statistics().entry_count,
            "token {token_index}"
        );
        let slot_map_expert_count = residencies[0].retained_expert_ids.len();
        assert_eq!(
            u64::try_from(slot_map_expert_count).unwrap_or(0) * STREAMED_PER_EXPERT_PAYLOAD_BYTES,
            residencies[0].payload_bytes,
            "token {token_index} drifted: {} map experts vs {} payload bytes",
            slot_map_expert_count,
            residencies[0].payload_bytes
        );
        // The cache-wide resident accounting must track every table's real
        // tensor bytes. A growth whose write-back is skipped on a later
        // early-return undercounts here and lets later admission overfill
        // the ceiling.
        let cache_wide_resident_payload_bytes = cache.statistics().resident_payload_byte_count;
        let sum_of_table_tensor_bytes = cache
            .tables_by_layer
            .iter()
            .flatten()
            .map(|table| table.full_padded_payload_bytes)
            .fold(0_u64, u64::saturating_add);
        assert_eq!(
            cache_wide_resident_payload_bytes, sum_of_table_tensor_bytes,
            "token {token_index} lost resident accounting: cache-wide {} vs tables {}",
            cache_wide_resident_payload_bytes, sum_of_table_tensor_bytes
        );
    }
}

/// Issue #955, growth transient: growing a table must not leave the retired
/// tensor or any fill source pinned in active memory. MLX retains an array's
/// graph inputs for its lifetime, so the growth copies and fills evaluate and
/// detach their results; the status-published memory decomposition closes
/// only when every active expert byte is claimed by the cache.
#[test]
fn should_not_leave_retired_or_source_tensors_active_after_growth() {
    let runtime = test_runtime();
    // Wide rows make each projection array tens of kilobytes, so a real pin
    // (a retired table or a fill source) towers over the CPU allocator's
    // per-array size-class rounding this suite-mode assertion must tolerate.
    let wide_streamed_weights = |expert_count: usize| {
        let projection = |fill: f32| {
            runtime
                .array_from_f32(
                    &vec![fill; expert_count * 2048],
                    &[
                        i32::try_from(expert_count).expect("the expert count fits i32"),
                        2048,
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
        }
    };
    let mut cache = RetainedExpertCache::new(1);
    cache.update_maximum_resident_payload_bytes(1024 * 1024 * 1024);
    // Other suite tests share the process-global runtime; sample this test's
    // own baseline after settling the allocator so only its allocations count.
    // The stream sync is mandatory: a committed Metal command buffer captures
    // its source buffers until completion, so active memory only becomes
    // deterministic once the stream has drained.
    runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .unwrap();
    let active_bytes_before_any_table = u64::try_from(
        runtime
            .memory_snapshot()
            .expect("the memory snapshot should be available")
            .active_memory_bytes(),
    )
    .unwrap_or(0);
    {
        let weights = wide_streamed_weights(2);
        cache
            .queue_pending_routed_expert_insert(0, &[1, 2], &weights, WARM_SLOT_COUNT)
            .expect("the first warm insert should queue");
        cache
            .flush_pending_inserts(&runtime)
            .expect("the first warm flush should succeed");
        // This insert forces the table to grow 2 -> 4 slots.
        cache
            .queue_pending_routed_expert_insert(
                0,
                &[3, 4],
                &wide_streamed_weights(2),
                WARM_SLOT_COUNT,
            )
            .expect("the growth warm insert should queue");
        cache
            .flush_pending_inserts(&runtime)
            .expect("the growth warm flush should succeed");
        // Drop the test-owned page reference so the only live expert arrays
        // are the ones the cache claims.
    }
    // Drain the fill/growth command buffers so their captured source buffers
    // are released before the active-memory sample.
    runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .unwrap();

    let claimed_table_payload_bytes = cache.statistics().resident_payload_byte_count;
    let active_bytes_after_growth = u64::try_from(
        runtime
            .memory_snapshot()
            .expect("the memory snapshot should be available")
            .active_memory_bytes(),
    )
    .unwrap_or(0);
    // One wide page is ~48 KB and the grown table ~96 KB, so a single pinned
    // source or stranded retired tensor dwarfs the 16 KB slack granted for
    // the CPU allocator's per-array size-class rounding.
    assert!(
        active_bytes_after_growth
            <= active_bytes_before_any_table + claimed_table_payload_bytes + 16 * 1024,
        "active memory kept unclaimed expert bytes after growth: before={} after={} claimed={}",
        active_bytes_before_any_table,
        active_bytes_after_growth,
        claimed_table_payload_bytes
    );
}
