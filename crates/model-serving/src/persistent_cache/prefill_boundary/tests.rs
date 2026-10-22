use super::{
    persistent_prompt_cache_boundary_clamped_prefill_chunk_end,
    persistent_prompt_cache_boundary_completed_prefill_chunk_tokens,
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
fn should_stop_tail_capture_one_token_short_of_a_partial_final_block() {
    for (
        prompt_token_count,
        persistent_prompt_cache_block_token_count,
        expected_tail_capture_end,
    ) in [
        (7_722, 3_328, 7_721),
        (1, 3_328, 0),
        (3_328, 3_328, 3_328),
        (6_656, 3_328, 6_656),
        (0, 3_328, 0),
    ] {
        assert_eq!(
            expected_tail_capture_end,
            super::persistent_prompt_cache_partial_tail_capture_end(
                prompt_token_count,
                persistent_prompt_cache_block_token_count,
            ),
            "tail capture end for a {prompt_token_count}-token prompt with \
             {persistent_prompt_cache_block_token_count}-token blocks"
        );
    }
}
