/// Ends an ordinary target-prefill chunk at the exact control-span boundary so
/// one forward never mixes mandatory dense control context with sparse
/// conversation positions.
#[must_use]
pub const fn qwen3_5_prefill_chunck_end_at_ordinary_target_control_span_boundary(
    prefill_start_position: usize,
    candidate_prefill_end_position: usize,
    ordinary_target_prefill_control_span_end_position: usize,
) -> Option<usize> {
    if candidate_prefill_end_position <= prefill_start_position {
        return None;
    }
    if prefill_start_position < ordinary_target_prefill_control_span_end_position {
        return Some(
            if candidate_prefill_end_position
                < ordinary_target_prefill_control_span_end_position
            {
                candidate_prefill_end_position
            } else {
                ordinary_target_prefill_control_span_end_position
            },
        );
    }
    Some(candidate_prefill_end_position)
}

/// Returns whether the current target chunk belongs to sparse conversation
/// processing rather than ordinary system-and-tool prefill.
#[must_use]
pub const fn qwen3_5_speculative_prefill_sparse_target_is_active(
    should_use_speculative_prefill: bool,
    prefill_start_position: usize,
    ordinary_target_prefill_control_span_end_position: usize,
) -> bool {
    should_use_speculative_prefill
        && prefill_start_position >= ordinary_target_prefill_control_span_end_position
}
