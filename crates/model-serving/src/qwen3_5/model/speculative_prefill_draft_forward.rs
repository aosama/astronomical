/// Returns the exclusive end position for one bounded drafter prompt forward.
///
/// Persistent capture requires each complete cache boundary to become visible
/// in parent order. A normal drafter forward may therefore be shortened when
/// it would otherwise cross the next boundary.
pub(crate) fn qwen3_5_speculative_prefill_draft_forward_end(
    draft_forward_start_token_count: usize,
    remaining_prompt_token_count: usize,
    maximum_draft_forward_token_count: usize,
    persistent_cache_block_token_count: Option<usize>,
) -> Option<usize> {
    if remaining_prompt_token_count == 0 || maximum_draft_forward_token_count == 0 {
        return None;
    }

    let mut next_draft_forward_token_count =
        remaining_prompt_token_count.min(maximum_draft_forward_token_count);
    if let Some(persistent_cache_block_token_count) = persistent_cache_block_token_count {
        if persistent_cache_block_token_count == 0 {
            return None;
        }
        let completed_tokens_in_current_cache_block =
            draft_forward_start_token_count % persistent_cache_block_token_count;
        let tokens_until_next_cache_boundary = if completed_tokens_in_current_cache_block == 0 {
            persistent_cache_block_token_count
        } else {
            persistent_cache_block_token_count - completed_tokens_in_current_cache_block
        };
        next_draft_forward_token_count =
            next_draft_forward_token_count.min(tokens_until_next_cache_boundary);
    }

    draft_forward_start_token_count.checked_add(next_draft_forward_token_count)
}
