use super::{check_configured_context, checked_full_attention_growth_tokens};
use crate::InferenceEngineError;

const CONTEXT_REASON: &str =
    "effective Qwen3.5 context must be positive and not exceed the artifact context";

fn is_fatal_with(check_outcome: Result<(), InferenceEngineError>, expected_reason: &str) -> bool {
    matches!(check_outcome, Err(InferenceEngineError::Fatal { reason }) if reason == expected_reason)
}

// Positive: a context inside the artifact's range is accepted.
#[test]
fn should_accept_a_context_inside_the_artifact_range() {
    assert!(check_configured_context(32_768, 262_144).is_ok());
}

// Edge: a context equal to the artifact's range is accepted.
#[test]
fn should_accept_a_context_equal_to_the_artifact_range() {
    assert!(check_configured_context(262_144, 262_144).is_ok());
}

// Negative: one token past the artifact's range is refused.
#[test]
fn should_refuse_a_context_past_the_artifact_range() {
    assert!(is_fatal_with(
        check_configured_context(262_145, 262_144),
        CONTEXT_REASON
    ));
}

// Negative: a zero context is refused.
#[test]
fn should_refuse_a_zero_context() {
    assert!(is_fatal_with(
        check_configured_context(0, 262_144),
        CONTEXT_REASON
    ));
}

// Positive and edge: growth steps up to the i32 range pass through.
#[test]
fn should_pass_growth_steps_up_to_the_i32_range() {
    assert_eq!(checked_full_attention_growth_tokens(256).ok(), Some(256));
    assert_eq!(
        checked_full_attention_growth_tokens(i32::MAX as u32).ok(),
        Some(i32::MAX)
    );
}

// Negative: a growth step past the i32 range is an engine failure.
#[test]
fn should_refuse_a_growth_step_past_the_i32_range() {
    let growth_outcome = checked_full_attention_growth_tokens(i32::MAX as u32 + 1);

    assert!(matches!(
        growth_outcome,
        Err(InferenceEngineError::Fatal { reason }) if reason == "full-attention growth tokens exceed Int32 range"
    ));
}
