//! Arithmetic for aligning prompt processing with durable cache boundaries.
//!
//! Cache-enabled prefill is clamped to one boundary per forward. That keeps the
//! captured decoder state and token slice at the same exact point, and ensures a
//! required synchronous publication succeeds before processing can advance.

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

/// Returns local completed-token counts for every sparse-anchored dense boundary
/// crossed by one attempted prefill forward.
///
/// A request that restored SpecPrefill sparse target state processes its conversation
/// tail densely, but that tail continues from a compact selection-bound prefix whose
/// length is not a cache block multiple. Anchored boundaries therefore start one block
/// after the restored prefix instead of on an absolute block multiple (issue #659).
#[must_use]
pub fn sparse_anchored_dense_boundary_completed_prefill_chunk_tokens(
    prefill_chunk_start: usize,
    prefill_chunk_end: usize,
    sparse_anchor_token_count: usize,
    persistent_prompt_cache_block_token_count: usize,
) -> Vec<usize> {
    if prefill_chunk_end <= prefill_chunk_start || persistent_prompt_cache_block_token_count == 0 {
        return Vec::new();
    }
    // A boundary is durable only after the anchor's first complete block. Tokens
    // between the anchor and that boundary belong to no cacheable block.
    let Some(mut absolute_sparse_anchored_boundary) =
        sparse_anchor_token_count.checked_add(persistent_prompt_cache_block_token_count)
    else {
        return Vec::new();
    };
    let mut completed_prefill_chunk_tokens = Vec::new();
    // Skip whole anchored blocks already behind the cursor so a retried or
    // resumed chunk never reports a boundary it did not cross.
    while absolute_sparse_anchored_boundary <= prefill_chunk_start {
        let Some(next_absolute_sparse_anchored_boundary) = absolute_sparse_anchored_boundary
            .checked_add(persistent_prompt_cache_block_token_count)
        else {
            return completed_prefill_chunk_tokens;
        };
        absolute_sparse_anchored_boundary = next_absolute_sparse_anchored_boundary;
    }
    while absolute_sparse_anchored_boundary <= prefill_chunk_end {
        completed_prefill_chunk_tokens
            .push(absolute_sparse_anchored_boundary - prefill_chunk_start);
        let Some(next_absolute_sparse_anchored_boundary) = absolute_sparse_anchored_boundary
            .checked_add(persistent_prompt_cache_block_token_count)
        else {
            break;
        };
        absolute_sparse_anchored_boundary = next_absolute_sparse_anchored_boundary;
    }
    completed_prefill_chunk_tokens
}

/// Clamps an attempted sparse-anchored dense prefill chunk so it publishes at most
/// one anchored boundary, mirroring the ordinary one-boundary-per-forward rule.
#[must_use]
pub fn sparse_anchored_dense_boundary_clamped_prefill_chunk_end(
    prefill_chunk_start: usize,
    requested_prefill_chunk_end: usize,
    sparse_anchor_token_count: usize,
    persistent_prompt_cache_block_token_count: usize,
) -> usize {
    if requested_prefill_chunk_end <= prefill_chunk_start
        || persistent_prompt_cache_block_token_count == 0
    {
        return requested_prefill_chunk_end;
    }
    let Some(first_absolute_sparse_anchored_boundary) =
        sparse_anchor_token_count.checked_add(persistent_prompt_cache_block_token_count)
    else {
        return requested_prefill_chunk_end;
    };
    if prefill_chunk_start < first_absolute_sparse_anchored_boundary {
        return requested_prefill_chunk_end.min(first_absolute_sparse_anchored_boundary);
    }
    let completed_anchored_block_count = (prefill_chunk_start - sparse_anchor_token_count)
        / persistent_prompt_cache_block_token_count;
    let Some(next_absolute_sparse_anchored_boundary) = completed_anchored_block_count
        .checked_add(1)
        .and_then(|next_anchored_block_count| {
            next_anchored_block_count.checked_mul(persistent_prompt_cache_block_token_count)
        })
        .and_then(|anchored_block_offset| {
            sparse_anchor_token_count.checked_add(anchored_block_offset)
        })
    else {
        return requested_prefill_chunk_end;
    };
    requested_prefill_chunk_end.min(next_absolute_sparse_anchored_boundary)
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
