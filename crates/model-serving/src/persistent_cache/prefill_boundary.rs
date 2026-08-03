use super::block_key::PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT;

/// Clips prompt-processing work so SSD prompt-cache capture observes every
/// complete persistent prompt-cache block boundary.
///
/// `candidate_prefill_chunck_end` remains the adaptive chunker's requested end;
/// this helper only prevents one model forward pass from crossing a cache
/// boundary that cannot be reconstructed later for recurrent layers.
#[must_use]
pub fn persistent_prompt_cache_aligned_prefill_end(
    prefill_chunck_start: usize,
    candidate_prefill_chunck_end: usize,
    final_prompt_index: usize,
) -> usize {
    let bounded_candidate_prefill_chunck_end = candidate_prefill_chunck_end.min(final_prompt_index);
    let next_persistent_prompt_cache_block_boundary =
        next_persistent_prompt_cache_block_boundary(prefill_chunck_start);
    bounded_candidate_prefill_chunck_end.min(next_persistent_prompt_cache_block_boundary)
}

fn next_persistent_prompt_cache_block_boundary(prefill_chunck_start: usize) -> usize {
    let completed_persistent_prompt_cache_block_count =
        prefill_chunck_start / PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT;
    completed_persistent_prompt_cache_block_count
        .saturating_add(1)
        .saturating_mul(PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT)
}
