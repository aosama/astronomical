//! Arithmetic for aligning prompt processing with durable cache boundaries.
//!
//! Cache-enabled prefill is clamped to one boundary per forward. That keeps the
//! captured decoder state and token slice at the same exact point, and ensures a
//! required synchronous publication succeeds before processing can advance.

/// Returns local completed-token counts for every persistent prompt-cache
/// boundary crossed by one attempted prefill forward.
#[must_use]
pub fn persistent_prompt_cache_boundary_completed_prefill_chunck_tokens(
    prefill_chunck_start: usize,
    prefill_chunck_end: usize,
    persistent_prompt_cache_block_token_count: usize,
) -> Vec<usize> {
    if prefill_chunck_end <= prefill_chunck_start || persistent_prompt_cache_block_token_count == 0
    {
        return Vec::new();
    }

    let completed_persistent_prompt_cache_block_count =
        prefill_chunck_start / persistent_prompt_cache_block_token_count;
    let Some(mut absolute_persistent_prompt_cache_boundary) =
        completed_persistent_prompt_cache_block_count
            .checked_add(1)
            .and_then(|next_persistent_prompt_cache_block_count| {
                next_persistent_prompt_cache_block_count
                    .checked_mul(persistent_prompt_cache_block_token_count)
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
                .checked_add(persistent_prompt_cache_block_token_count)
        else {
            break;
        };
        absolute_persistent_prompt_cache_boundary = next_absolute_persistent_prompt_cache_boundary;
    }
    completed_prefill_chunck_tokens
}

/// Clamps an attempted cache-enabled prefill chunk so it can publish at most one
/// mandatory persistent prompt-cache boundary.
#[must_use]
pub fn persistent_prompt_cache_boundary_clamped_prefill_chunck_end(
    prefill_chunck_start: usize,
    requested_prefill_chunck_end: usize,
    persistent_prompt_cache_block_token_count: usize,
) -> usize {
    if requested_prefill_chunck_end <= prefill_chunck_start
        || persistent_prompt_cache_block_token_count == 0
    {
        return requested_prefill_chunck_end;
    }
    // Integer division identifies the block containing the current cursor; the
    // next multiple is the earliest boundary this forward is allowed to cross.
    let Some(next_persistent_prompt_cache_boundary) = (prefill_chunck_start
        / persistent_prompt_cache_block_token_count)
        .checked_add(1)
        .and_then(|next_boundary_block_count| {
            next_boundary_block_count.checked_mul(persistent_prompt_cache_block_token_count)
        })
    else {
        return requested_prefill_chunck_end;
    };
    requested_prefill_chunck_end.min(next_persistent_prompt_cache_boundary)
}
