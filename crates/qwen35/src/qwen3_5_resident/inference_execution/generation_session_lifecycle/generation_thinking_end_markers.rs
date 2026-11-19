//! Thinking-end marker assembly for one Qwen3.5 generation session.

/// Appends the model's think-end marker whenever the caller's natural
/// reasoning boundaries omit it, so a thinking block can always be closed —
/// even for direct engine callers that pass no tokenizer-derived boundary
/// such as a tool-call start.
pub(super) fn natural_reasoning_end_token_ids_with_model_safety_net(
    caller_supplied_marker_token_ids: &[u32],
    model_think_end_token_id: u32,
) -> Vec<u32> {
    let mut natural_reasoning_end_token_ids = caller_supplied_marker_token_ids.to_vec();
    if !natural_reasoning_end_token_ids.contains(&model_think_end_token_id) {
        natural_reasoning_end_token_ids.push(model_think_end_token_id);
    }
    natural_reasoning_end_token_ids
}

/// Returns whether a forced thinking transition ends exactly at the model's
/// think-end marker — the rule a thinking budget demands so reasoning can
/// always be closed by a model-owned token sequence.
pub(super) fn forced_thinking_transition_ends_with_model_marker(
    forced_transition_token_ids: &[u32],
    model_think_end_token_id: u32,
) -> bool {
    forced_transition_token_ids.last().copied() == Some(model_think_end_token_id)
}

#[cfg(test)]
#[path = "generation_thinking_end_markers_tests.rs"]
mod generation_thinking_end_markers_tests;
