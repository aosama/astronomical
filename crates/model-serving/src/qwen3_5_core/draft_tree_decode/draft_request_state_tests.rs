//! Unit tests for `draft_request_state`: a request's drafting state.

use super::DraftRequestState;
use crate::Qwen3_5SamplingStrategy;

#[test]
fn should_draft_from_the_start_when_the_session_is_admitted() {
    let state = DraftRequestState::for_new_request(true, Qwen3_5SamplingStrategy::HighestLogit);

    assert!(state.is_drafting());
    assert!(state.draft_session.is_some());
    assert!(state.emission_queue.is_empty());
}

#[test]
fn should_hold_no_draft_session_when_the_session_is_not_admitted() {
    let state = DraftRequestState::for_new_request(false, Qwen3_5SamplingStrategy::HighestLogit);

    assert!(!state.is_drafting());
    assert!(state.draft_session.is_none());
}

#[test]
fn should_drop_queued_tokens_and_the_session_when_drafting_stops() {
    let mut state = DraftRequestState::for_new_request(true, Qwen3_5SamplingStrategy::HighestLogit);
    state.emission_queue.push_back(7);
    state.emission_queue.push_back(11);

    let was_drafting = state.stop_drafting();

    assert!(was_drafting);
    assert!(!state.is_drafting());
    assert!(state.emission_queue.is_empty());
    assert!(state.draft_session.is_none());
}

#[test]
fn should_report_no_change_when_a_session_that_never_drafted_is_stopped() {
    let mut state =
        DraftRequestState::for_new_request(false, Qwen3_5SamplingStrategy::HighestLogit);

    assert!(!state.stop_drafting());
}
