use astronomical_model_serving::{
    PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT, persistent_prompt_cache_aligned_prefill_end,
};

#[test]
fn should_stop_cold_prefill_at_first_persistent_prompt_cache_block_boundary() {
    let prefill_chunck_start = 0;
    let candidate_prefill_chunck_end = PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT * 8;
    let final_prompt_index = PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT * 20;

    let aligned_prefill_chunck_end = persistent_prompt_cache_aligned_prefill_end(
        prefill_chunck_start,
        candidate_prefill_chunck_end,
        final_prompt_index,
    );

    assert_eq!(
        PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT,
        aligned_prefill_chunck_end
    );
}

#[test]
fn should_stop_restored_prefill_at_next_persistent_prompt_cache_block_boundary() {
    let prefill_chunck_start = PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT * 8;
    let candidate_prefill_chunck_end =
        prefill_chunck_start + PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT * 8;
    let final_prompt_index = PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT * 20;

    let aligned_prefill_chunck_end = persistent_prompt_cache_aligned_prefill_end(
        prefill_chunck_start,
        candidate_prefill_chunck_end,
        final_prompt_index,
    );

    assert_eq!(
        PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT * 9,
        aligned_prefill_chunck_end
    );
}

#[test]
fn should_keep_short_prefill_chunck_that_ends_before_the_next_cache_boundary() {
    let prefill_chunck_start = PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT * 8;
    let candidate_prefill_chunck_end = prefill_chunck_start + 128;
    let final_prompt_index = PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT * 20;

    let aligned_prefill_chunck_end = persistent_prompt_cache_aligned_prefill_end(
        prefill_chunck_start,
        candidate_prefill_chunck_end,
        final_prompt_index,
    );

    assert_eq!(candidate_prefill_chunck_end, aligned_prefill_chunck_end);
}

#[test]
fn should_not_cross_the_final_prompt_index_to_reach_a_cache_boundary() {
    let prefill_chunck_start = PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT * 8;
    let candidate_prefill_chunck_end =
        prefill_chunck_start + PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT * 8;
    let final_prompt_index = prefill_chunck_start + 127;

    let aligned_prefill_chunck_end = persistent_prompt_cache_aligned_prefill_end(
        prefill_chunck_start,
        candidate_prefill_chunck_end,
        final_prompt_index,
    );

    assert_eq!(final_prompt_index, aligned_prefill_chunck_end);
}
