use super::block_key::PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT;

/// Returns local completed-token counts for every persistent prompt-cache
/// boundary crossed by one attempted prefill forward.
#[must_use]
pub fn persistent_prompt_cache_boundary_completed_prefill_chunck_tokens(
    prefill_chunck_start: usize,
    prefill_chunck_end: usize,
) -> Vec<usize> {
    if prefill_chunck_end <= prefill_chunck_start {
        return Vec::new();
    }

    let completed_persistent_prompt_cache_block_count =
        prefill_chunck_start / PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT;
    let Some(mut absolute_persistent_prompt_cache_boundary) =
        completed_persistent_prompt_cache_block_count
            .checked_add(1)
            .and_then(|next_persistent_prompt_cache_block_count| {
                next_persistent_prompt_cache_block_count
                    .checked_mul(PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT)
            })
    else {
        return Vec::new();
    };
    let mut completed_prefill_chunck_tokens = Vec::new();
    while absolute_persistent_prompt_cache_boundary <= prefill_chunck_end {
        completed_prefill_chunck_tokens
            .push(absolute_persistent_prompt_cache_boundary - prefill_chunck_start);
        let Some(next_absolute_persistent_prompt_cache_boundary) =
            absolute_persistent_prompt_cache_boundary
                .checked_add(PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT)
        else {
            break;
        };
        absolute_persistent_prompt_cache_boundary = next_absolute_persistent_prompt_cache_boundary;
    }
    completed_prefill_chunck_tokens
}
