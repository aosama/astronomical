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

#[test]
fn should_compose_boundary_and_tail_clamps_for_a_nonaligned_terminal_chunk() {
    // Block 3328, chunk 2048, 7,722-token prompt: the terminal chunk must end
    // one token short of the prompt so the captured tail stays probeable.
    assert_eq!(
        7_721,
        super::persistent_prompt_cache_clamped_prefill_chunk_end(
            6_144, 7_722, 7_722, 3_328,
        ),
    );
}

#[test]
fn should_clamp_multi_boundary_small_block_chunks_to_one_boundary() {
    // Block 512, chunk 2048: a forward that would cross four durable
    // boundaries is clamped to publish only the first one. Crossing exactly
    // one boundary (or none) keeps the configured chunk, and the tail clamp
    // never fires mid-prompt.
    assert_eq!(
        512,
        super::persistent_prompt_cache_clamped_prefill_chunk_end(0, 2_048, 7_722, 512,),
    );
    // The next chunk starts exactly on a boundary and still crosses three
    // more, so it too is clamped to a single publication.
    assert_eq!(
        2_560,
        super::persistent_prompt_cache_clamped_prefill_chunk_end(2_048, 4_096, 7_722, 512,),
    );
}

#[test]
fn should_stop_the_small_block_terminal_chunk_one_token_short() {
    // Block 512, 7,722-token prompt: the final full chunk stops at 7,721 and
    // the remaining single token belongs to the terminal forward.
    assert_eq!(
        7_721,
        super::persistent_prompt_cache_clamped_prefill_chunk_end(
            7_168, 7_722, 7_722, 512,
        ),
    );
}

#[test]
fn should_keep_block_aligned_terminal_chunks_unchanged() {
    // An exactly block-aligned prompt has no partial tail, so the terminal
    // chunk keeps its complete-block publication boundary.
    assert_eq!(
        6_656,
        super::persistent_prompt_cache_clamped_prefill_chunk_end(
            3_328, 6_656, 6_656, 3_328,
        ),
    );
}

#[test]
fn should_not_produce_a_zero_token_terminal_chunk_for_a_one_token_remainder() {
    // A prompt ending one token past a block boundary leaves a single-token
    // remainder; the last full chunk must still be able to publish that block.
    assert_eq!(
        3_329,
        super::persistent_prompt_cache_clamped_prefill_chunk_end(
            3_328, 3_329, 3_329, 3_328,
        ),
    );
}
