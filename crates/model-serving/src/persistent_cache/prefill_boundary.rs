//! Arithmetic for aligning prompt processing with durable cache boundaries.
//!
//! Cache-enabled prefill is clamped to one boundary per forward. That keeps the
//! captured decoder state and token slice at the same exact point, and ensures a
//! required synchronous publication succeeds before processing can advance.

#[cfg(test)]
mod tests;

/// Returns local completed-token counts for every persistent prompt-cache
/// boundary crossed by one attempted prefill forward.
#[must_use]
pub fn persistent_prompt_cache_boundary_completed_prefill_chunk_tokens(
    prefill_chunk_start: usize,
    prefill_chunk_end: usize,
    persistent_prompt_cache_block_token_count: usize,
) -> Vec<usize> {
    if prefill_chunk_end <= prefill_chunk_start || persistent_prompt_cache_block_token_count == 0 {
        return Vec::new();
    }

    let completed_persistent_prompt_cache_block_count =
        prefill_chunk_start / persistent_prompt_cache_block_token_count;
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
    let mut completed_prefill_chunk_tokens = Vec::new();
    while absolute_persistent_prompt_cache_boundary <= prefill_chunk_end {
        completed_prefill_chunk_tokens
            .push(absolute_persistent_prompt_cache_boundary - prefill_chunk_start);
        let Some(next_absolute_persistent_prompt_cache_boundary) =
            absolute_persistent_prompt_cache_boundary
                .checked_add(persistent_prompt_cache_block_token_count)
        else {
            break;
        };
        absolute_persistent_prompt_cache_boundary = next_absolute_persistent_prompt_cache_boundary;
    }
    completed_prefill_chunk_tokens
}

/// Returns the cache-enabled prefill chunk end after both durable-boundary clamps.
///
/// Composes the two rules that must never be applied independently: at most one
/// durable block publication per forward, and a terminal chunk that stops one
/// token short of a non-aligned prompt so the captured partial tail stays
/// probeable by the generation-start lookup. Both Qwen3.5 engines must call this
/// one function — the composition itself is the correctness contract.
#[must_use]
pub fn persistent_prompt_cache_clamped_prefill_chunk_end(
    prefill_chunk_start: usize,
    requested_prefill_chunk_end: usize,
    prompt_token_count: usize,
    persistent_prompt_cache_block_token_count: usize,
) -> usize {
    // The boundary clamp only intervenes when one forward would publish more
    // than one durable block; crossing exactly one boundary is the normal
    // cadence and must not split the configured chunk.
    let mut clamped_prefill_chunk_end =
        if persistent_prompt_cache_boundary_completed_prefill_chunk_tokens(
            prefill_chunk_start,
            requested_prefill_chunk_end,
            persistent_prompt_cache_block_token_count,
        )
        .len() <= 1
        {
            requested_prefill_chunk_end
        } else {
            persistent_prompt_cache_boundary_clamped_prefill_chunk_end(
                prefill_chunk_start,
                requested_prefill_chunk_end,
                persistent_prompt_cache_block_token_count,
            )
        };
    let partial_tail_capture_end = persistent_prompt_cache_partial_tail_capture_end(
        prompt_token_count,
        persistent_prompt_cache_block_token_count,
    );
    if clamped_prefill_chunk_end == prompt_token_count
        && partial_tail_capture_end < prompt_token_count
        && prefill_chunk_start < partial_tail_capture_end
    {
        clamped_prefill_chunk_end = partial_tail_capture_end;
    }
    clamped_prefill_chunk_end
}

/// Returns the prefill end position a partial tail can be durably captured from.
///
/// The generation-start lookup never consumes the final prompt token: the
/// terminal forward must process it to produce the first logits. A tail saved
/// through that token could therefore never be probed again, so capture stops
/// one token short — exactly the partial-tail probe's retained-suffix rule —
/// and the single remaining token belongs to the normal prefill path.
#[must_use]
pub fn persistent_prompt_cache_partial_tail_capture_end(
    prompt_token_count: usize,
    persistent_prompt_cache_block_token_count: usize,
) -> usize {
    if prompt_token_count % persistent_prompt_cache_block_token_count == 0 {
        prompt_token_count
    } else {
        prompt_token_count.saturating_sub(1)
    }
}

/// Clamps an attempted cache-enabled prefill chunk so it can publish at most one
/// mandatory persistent prompt-cache boundary.
#[must_use]
pub fn persistent_prompt_cache_boundary_clamped_prefill_chunk_end(
    prefill_chunk_start: usize,
    requested_prefill_chunk_end: usize,
    persistent_prompt_cache_block_token_count: usize,
) -> usize {
    if requested_prefill_chunk_end <= prefill_chunk_start
        || persistent_prompt_cache_block_token_count == 0
    {
        return requested_prefill_chunk_end;
    }
    // Integer division identifies the block containing the current cursor; the
    // next multiple is the earliest boundary this forward is allowed to cross.
    let Some(next_persistent_prompt_cache_boundary) = (prefill_chunk_start
        / persistent_prompt_cache_block_token_count)
        .checked_add(1)
        .and_then(|next_boundary_block_count| {
            next_boundary_block_count.checked_mul(persistent_prompt_cache_block_token_count)
        })
    else {
        return requested_prefill_chunk_end;
    };
    requested_prefill_chunk_end.min(next_persistent_prompt_cache_boundary)
}
