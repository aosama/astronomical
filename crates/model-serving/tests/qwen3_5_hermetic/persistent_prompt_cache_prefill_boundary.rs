use astronomical_model_serving::{
    persistent_prompt_cache_boundary_clamped_prefill_chunk_end,
    persistent_prompt_cache_boundary_completed_prefill_chunk_tokens,
    sparse_anchored_dense_boundary_clamped_prefill_chunk_end,
    sparse_anchored_dense_boundary_completed_prefill_chunk_tokens,
};

#[test]
fn should_report_every_persistent_prompt_cache_boundary_crossed_by_one_prefill_chunk() {
    for (
        prefill_chunk_start,
        prefill_chunk_end,
        persistent_prompt_cache_block_token_count,
        expected_completed_prefill_chunk_tokens,
    ) in [
        (0, 128, 2_048, Vec::new()),
        (0, 2_048, 2_048, vec![2_048]),
        (0, 4_096, 2_048, vec![2_048, 4_096]),
        (128, 4_096, 2_048, vec![1_920, 3_968]),
        (22_528, 26_624, 2_048, vec![2_048, 4_096]),
        (22_528, 25_000, 2_048, vec![2_048]),
        (0, 1_536, 512, vec![512, 1_024, 1_536]),
        (2_048, 2_048, 2_048, Vec::new()),
        (4_096, 2_048, 2_048, Vec::new()),
        (0, 4_096, 0, Vec::new()),
        (usize::MAX - 1_024, usize::MAX, 2_048, Vec::new()),
    ] {
        assert_eq!(
            expected_completed_prefill_chunk_tokens,
            persistent_prompt_cache_boundary_completed_prefill_chunk_tokens(
                prefill_chunk_start,
                prefill_chunk_end,
                persistent_prompt_cache_block_token_count,
            ),
            "unexpected local boundary counts for [{prefill_chunk_start}, {prefill_chunk_end})"
        );
    }
}

#[test]
fn should_clamp_cache_enabled_prefill_to_the_next_persistent_boundary() {
    for (
        prefill_chunk_start,
        requested_prefill_chunk_end,
        persistent_prompt_cache_block_token_count,
        expected_prefill_chunk_end,
    ) in [
        (0, 128, 2_048, 128),
        (0, 4_096, 2_048, 2_048),
        (128, 4_096, 2_048, 2_048),
        (2_048, 8_192, 2_048, 4_096),
        (2_048, 2_048, 2_048, 2_048),
        (4_096, 2_048, 2_048, 2_048),
        (0, 4_096, 0, 4_096),
        (usize::MAX - 1_024, usize::MAX, 2_048, usize::MAX),
    ] {
        assert_eq!(
            expected_prefill_chunk_end,
            persistent_prompt_cache_boundary_clamped_prefill_chunk_end(
                prefill_chunk_start,
                requested_prefill_chunk_end,
                persistent_prompt_cache_block_token_count,
            ),
            "unexpected boundary clamp for [{prefill_chunk_start}, {requested_prefill_chunk_end})"
        );
    }
}

#[test]
fn should_report_sparse_anchored_boundaries_measured_from_the_restored_prefix() {
    // Anchor 2,599 with 512-token blocks: boundaries sit at 3,111 and 3,623, deliberately
    // not on absolute block multiples, because the restored sparse slab is compact.
    for (prefill_chunk_start, prefill_chunk_end, expected_completed_prefill_chunk_tokens) in [
        (2_599, 2_631, Vec::new()),
        (2_599, 3_111, vec![512]),
        (2_599, 3_623, vec![512, 1_024]),
        (3_087, 3_111, vec![24]),
        (3_111, 3_143, Vec::new()),
        (3_111, 3_623, vec![512]),
        (0, 2_599, Vec::new()),
        (2_599, 2_599, Vec::new()),
        (3_623, 3_655, Vec::new()),
    ] {
        assert_eq!(
            expected_completed_prefill_chunk_tokens,
            sparse_anchored_dense_boundary_completed_prefill_chunk_tokens(
                prefill_chunk_start,
                prefill_chunk_end,
                2_599,
                512,
            ),
            "unexpected anchored boundary report for [{prefill_chunk_start}, {prefill_chunk_end})"
        );
    }
}

#[test]
fn should_clamp_sparse_anchored_chunks_to_one_boundary_per_forward() {
    for (prefill_chunk_start, requested_prefill_chunk_end, expected_prefill_chunk_end) in [
        (2_599, 2_631, 2_631),
        (2_599, 3_111, 3_111),
        (2_599, 4_096, 3_111),
        (3_087, 3_143, 3_111),
        (3_111, 3_143, 3_143),
        (3_111, 4_096, 3_623),
        (0, 2_599, 2_599),
        (2_599, 2_599, 2_599),
    ] {
        assert_eq!(
            expected_prefill_chunk_end,
            sparse_anchored_dense_boundary_clamped_prefill_chunk_end(
                prefill_chunk_start,
                requested_prefill_chunk_end,
                2_599,
                512,
            ),
            "unexpected anchored clamp for [{prefill_chunk_start}, {requested_prefill_chunk_end})"
        );
    }
}
