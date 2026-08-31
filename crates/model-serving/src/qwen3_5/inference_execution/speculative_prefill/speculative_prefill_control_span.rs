//! Defines the dense-control to sparse-conversation boundary.
//!
//! System instructions and tool schemas form mandatory control context. They
//! must be processed densely by the target before any conversation-token
//! selection is allowed. These pure helpers make that transition explicit to
//! both the chunk planner and execution path.

/// Ends an ordinary target-prefill chunk at the exact control-span boundary so
/// one forward never mixes mandatory dense control context with sparse
/// conversation positions.
#[must_use]
pub const fn qwen3_5_prefill_chunk_end_at_ordinary_target_control_span_boundary(
    prefill_start_position: usize,
    candidate_prefill_end_position: usize,
    ordinary_target_prefill_control_span_end_position: usize,
) -> Option<usize> {
    // Refuse a non-advancing candidate instead of letting subtraction or an
    // outer loop disguise a chunk-planner defect.
    if candidate_prefill_end_position <= prefill_start_position {
        return None;
    }
    // While the cursor is inside dense control context, cap the chunk exactly at
    // the boundary. This prevents one target forward from mixing two execution
    // contracts (dense rows before the boundary and selected rows after it).
    if prefill_start_position < ordinary_target_prefill_control_span_end_position {
        return Some(
            if candidate_prefill_end_position < ordinary_target_prefill_control_span_end_position {
                candidate_prefill_end_position
            } else {
                ordinary_target_prefill_control_span_end_position
            },
        );
    }
    // Once the control span is complete, ordinary chunk sizing is unchanged.
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
    // Both conditions matter: a disabled/ineligible request is always ordinary,
    // and an eligible request remains ordinary until every control token is done.
    should_use_speculative_prefill
        && prefill_start_position >= ordinary_target_prefill_control_span_end_position
}
