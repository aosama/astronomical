use super::check_serving_request;
use crate::InferenceEngineError;

const VOCABULARY_SIZE: u32 = 256;
const MAXIMUM_CONTEXT_TOKENS: u32 = 16;

fn is_invalid_request(check_outcome: Result<(), InferenceEngineError>) -> bool {
    matches!(
        check_outcome,
        Err(InferenceEngineError::InvalidRequest { .. })
    )
}

// Positive: a prompt inside the vocabulary that fits with its output budget starts.
#[test]
fn should_accept_a_prompt_that_fits_the_context_window() {
    let check_outcome =
        check_serving_request(&[1, 2, 3], 8, VOCABULARY_SIZE, MAXIMUM_CONTEXT_TOKENS);

    assert!(check_outcome.is_ok());
}

// Edge: a prompt and budget that fill the window exactly still fit.
#[test]
fn should_accept_a_request_that_fills_the_context_window_exactly() {
    let check_outcome =
        check_serving_request(&[1, 2, 3, 4], 12, VOCABULARY_SIZE, MAXIMUM_CONTEXT_TOKENS);

    assert!(check_outcome.is_ok());
}

// Negative: one token past the window is refused as an invalid request.
#[test]
fn should_refuse_a_request_one_token_past_the_context_window() {
    let check_outcome =
        check_serving_request(&[1, 2, 3, 4], 13, VOCABULARY_SIZE, MAXIMUM_CONTEXT_TOKENS);

    assert!(is_invalid_request(check_outcome));
}

// Negative: the first id past the vocabulary is refused.
#[test]
fn should_refuse_a_prompt_token_outside_the_vocabulary() {
    let check_outcome = check_serving_request(
        &[1, VOCABULARY_SIZE],
        1,
        VOCABULARY_SIZE,
        MAXIMUM_CONTEXT_TOKENS,
    );

    assert!(is_invalid_request(check_outcome));
}

// Edge: the last id inside the vocabulary is accepted.
#[test]
fn should_accept_the_last_token_inside_the_vocabulary() {
    let check_outcome = check_serving_request(
        &[VOCABULARY_SIZE - 1],
        1,
        VOCABULARY_SIZE,
        MAXIMUM_CONTEXT_TOKENS,
    );

    assert!(check_outcome.is_ok());
}

// Negative: an empty prompt is refused.
#[test]
fn should_refuse_an_empty_prompt() {
    let check_outcome = check_serving_request(&[], 1, VOCABULARY_SIZE, MAXIMUM_CONTEXT_TOKENS);

    assert!(is_invalid_request(check_outcome));
}
