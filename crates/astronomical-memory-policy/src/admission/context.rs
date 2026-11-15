//! Sizes a request's context needs, known before the request runs.
//!
//! How many output tokens to hold state for, the workspace a restore from
//! the astro SSD astro SSD prompt cache needs, and the smallest limit a loaded model can be
//! lowered to. Whether a request fits is the memory governor's question
//! (`crate::ledger`).

/// Visible-answer KV reserve beyond the thinking allowance, in tokens.
///
/// One prefill chunk (the repo's canonical token granularity): long-form
/// answers beyond it still generate, with per-step KV growth admitted by the
/// adaptive growth guard instead of pre-reserved.
pub const VISIBLE_ANSWER_RESERVE_TOKENS: usize = 2_048;

/// Output tokens the generation-context memory reservation must cover.
///
/// Reserving the caller's full `maximum_output_token_count` charges
/// (max_output - thinking_budget - answer reserve) x per-token KV bytes as a
/// permanent phantom hold at admission. An agent client asking for 65,535
/// output tokens with a 16,384-token thinking budget pinned ~1.3 GB of KV
/// that never materialized, which pushed every turn of a long conversation
/// over the active-memory ceiling and forced a reclaim of warm expert layers
/// before each prefill (2026-10-16 incident). Generation beyond the reserved
/// count remains legal: that KV grows lazily and is admitted per step by the
/// growth guard, exactly like context growth beyond any other reserve.
#[must_use]
pub const fn bounded_output_reservation_token_count(
    maximum_output_token_count: usize,
    thinking_budget_token_count: Option<usize>,
) -> usize {
    let bounded_output_token_count = match thinking_budget_token_count {
        Some(thinking_budget_token_count) => {
            thinking_budget_token_count.saturating_add(VISIBLE_ANSWER_RESERVE_TOKENS)
        }
        None => VISIBLE_ANSWER_RESERVE_TOKENS,
    };
    if maximum_output_token_count < bounded_output_token_count {
        maximum_output_token_count
    } else {
        bounded_output_token_count
    }
}

/// Context-cache reconstruction temporarily owns both loaded and concatenated state.
///
/// `restored_context_token_count` is tokens that may already exist as astro-ssd-prompt-cache
/// blocks. Future output-budget tokens have no blocks and must not be multiplied in.
#[must_use]
pub fn persistent_context_restore_workspace_bytes(
    context_memory_reservation_bytes_per_token: usize,
    restored_context_token_count: usize,
) -> Option<usize> {
    context_memory_reservation_bytes_per_token.checked_mul(restored_context_token_count)
}

/// Smallest safe idle ceiling after reclaiming elastic expert payload.
#[must_use]
pub const fn safe_minimum_active_memory_ceiling_bytes(
    current_idle_active_memory_bytes: u64,
    evictable_retained_expert_payload_bytes: u64,
    maximum_expert_page_reserve_bytes: u64,
) -> u64 {
    current_idle_active_memory_bytes
        .saturating_sub(evictable_retained_expert_payload_bytes)
        .saturating_add(maximum_expert_page_reserve_bytes)
}
