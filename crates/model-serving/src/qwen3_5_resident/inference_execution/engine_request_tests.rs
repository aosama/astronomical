use std::collections::VecDeque;

use super::ResidentRequestExtension;
use crate::qwen3_5_core::engine_request_test_support::test_engine_request;

fn speculative_extension(queued_token_ids: &[u32]) -> ResidentRequestExtension {
    ResidentRequestExtension {
        speculative_draft_key_value_states: None,
        speculative_emission_queue: queued_token_ids.iter().copied().collect::<VecDeque<u32>>(),
        speculative_session_admitted: true,
    }
}

#[test]
fn should_have_speculative_tokens_to_advance_while_accepted_tokens_are_queued() {
    let engine_request = test_engine_request(speculative_extension(&[1_601, 42]));
    assert!(engine_request.has_speculative_tokens_to_advance());
}

#[test]
fn should_have_nothing_to_advance_with_an_empty_queue_and_no_pending_token() {
    let engine_request = test_engine_request(speculative_extension(&[]));
    assert!(!engine_request.has_speculative_tokens_to_advance());
}

#[test]
fn should_drop_queued_tokens_and_leave_nothing_to_advance_after_suspension() {
    let mut engine_request = test_engine_request(speculative_extension(&[1_601]));
    engine_request.suspend_speculative_session("forced_thinking_transition");
    assert!(!engine_request.engine_extension.speculative_session_admitted);
    assert!(!engine_request.has_speculative_tokens_to_advance());
}
