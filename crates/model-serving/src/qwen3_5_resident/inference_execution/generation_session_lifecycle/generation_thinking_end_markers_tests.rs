use super::forced_thinking_transition_ends_with_model_marker;
use super::natural_reasoning_end_token_ids_with_model_safety_net;

const MODEL_THINK_END_TOKEN_ID: u32 = 90;
const TOOL_CALL_START_TOKEN_ID: u32 = 91;

#[test]
fn should_return_the_caller_markers_unchanged_when_the_model_marker_is_present() {
    let caller_supplied_marker_token_ids = [MODEL_THINK_END_TOKEN_ID, TOOL_CALL_START_TOKEN_ID];

    let assembled_marker_token_ids = natural_reasoning_end_token_ids_with_model_safety_net(
        &caller_supplied_marker_token_ids,
        MODEL_THINK_END_TOKEN_ID,
    );

    assert_eq!(
        assembled_marker_token_ids,
        vec![MODEL_THINK_END_TOKEN_ID, TOOL_CALL_START_TOKEN_ID]
    );
}

#[test]
fn should_append_the_model_marker_to_empty_caller_markers() {
    let assembled_marker_token_ids =
        natural_reasoning_end_token_ids_with_model_safety_net(&[], MODEL_THINK_END_TOKEN_ID);

    assert_eq!(assembled_marker_token_ids, vec![MODEL_THINK_END_TOKEN_ID]);
}

#[test]
fn should_not_duplicate_the_model_marker_when_it_already_sits_midway() {
    let caller_supplied_marker_token_ids =
        [TOOL_CALL_START_TOKEN_ID, MODEL_THINK_END_TOKEN_ID, 777];

    let assembled_marker_token_ids = natural_reasoning_end_token_ids_with_model_safety_net(
        &caller_supplied_marker_token_ids,
        MODEL_THINK_END_TOKEN_ID,
    );

    assert_eq!(
        assembled_marker_token_ids,
        vec![TOOL_CALL_START_TOKEN_ID, MODEL_THINK_END_TOKEN_ID, 777]
    );
}

#[test]
fn should_accept_a_forced_transition_that_ends_with_the_model_marker() {
    assert!(forced_thinking_transition_ends_with_model_marker(
        &[70, 71, MODEL_THINK_END_TOKEN_ID],
        MODEL_THINK_END_TOKEN_ID
    ));
}

#[test]
fn should_reject_a_forced_transition_whose_marker_sits_before_the_final_token() {
    assert!(!forced_thinking_transition_ends_with_model_marker(
        &[70, MODEL_THINK_END_TOKEN_ID, 71],
        MODEL_THINK_END_TOKEN_ID
    ));
}

#[test]
fn should_reject_an_empty_forced_transition() {
    assert!(!forced_thinking_transition_ends_with_model_marker(
        &[],
        MODEL_THINK_END_TOKEN_ID
    ));
}
