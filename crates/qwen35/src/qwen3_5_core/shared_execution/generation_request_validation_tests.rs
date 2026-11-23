use ipc_protocol::RequestId;

use super::{
    Qwen3_5GenerationRequestLimits, validate_generation_request_and_resolve_reserved_context,
};
use crate::{InferenceEngineError, Qwen3_5InferenceRequest};

const THINK_END_TOKEN_ID: u32 = 90;
const VOCABULARY_SIZE: u32 = 100;
const HARD_MAXIMUM_POSITION_COUNT: usize = 64;

fn request_limits() -> Qwen3_5GenerationRequestLimits {
    Qwen3_5GenerationRequestLimits {
        is_model_loaded: true,
        think_end_token_id: THINK_END_TOKEN_ID,
        vocabulary_size: VOCABULARY_SIZE,
        maximum_position_count: 32,
        hard_maximum_position_count: HARD_MAXIMUM_POSITION_COUNT,
    }
}

fn request_of(prompt_token_ids: Vec<u32>, max_output_tokens: u16) -> Qwen3_5InferenceRequest {
    Qwen3_5InferenceRequest::new(RequestId::new(7), prompt_token_ids, max_output_tokens)
}

fn is_invalid_request(validation_outcome: Result<usize, InferenceEngineError>) -> bool {
    matches!(
        validation_outcome,
        Err(InferenceEngineError::InvalidRequest { .. })
    )
}

// Positive: a plain request reserves its prompt plus its output reserve.
#[test]
fn should_reserve_the_prompt_and_output_for_a_plain_request() {
    let reserved_context_tokens = validate_generation_request_and_resolve_reserved_context(
        &request_limits(),
        &request_of(vec![1, 2, 3], 8),
    )
    .expect("a plain request should be accepted");

    assert_eq!(reserved_context_tokens, 3 + 8);
}

// Edge: a request that fills the hard position range exactly is served, even
// past the advisory configured limit.
#[test]
fn should_serve_a_request_that_fills_the_hard_position_range_exactly() {
    let validation_outcome = validate_generation_request_and_resolve_reserved_context(
        &request_limits(),
        &request_of(vec![1; 60], 4),
    );

    assert!(validation_outcome.is_ok());
}

// Negative: one position past the hard range is an invalid request.
#[test]
fn should_refuse_a_request_one_position_past_the_hard_range() {
    let validation_outcome = validate_generation_request_and_resolve_reserved_context(
        &request_limits(),
        &request_of(vec![1; 60], 5),
    );

    assert!(is_invalid_request(validation_outcome));
}

// Negative: a prompt token at the vocabulary size is refused.
#[test]
fn should_refuse_a_prompt_token_outside_the_vocabulary() {
    let validation_outcome = validate_generation_request_and_resolve_reserved_context(
        &request_limits(),
        &request_of(vec![1, VOCABULARY_SIZE], 4),
    );

    assert!(matches!(
        validation_outcome,
        Err(InferenceEngineError::Fatal { .. })
    ));
}

// Negative: a thinking budget whose transition does not end at the model's
// own marker is an invalid request.
#[test]
fn should_refuse_a_thinking_transition_without_the_model_marker() {
    let inference_request =
        request_of(vec![1, 2], 40).with_thinking_configuration(true, Some(8), vec![5, 6], vec![]);

    let validation_outcome = validate_generation_request_and_resolve_reserved_context(
        &request_limits(),
        &inference_request,
    );

    assert!(is_invalid_request(validation_outcome));
}

// Negative: a request before the model is loaded is refused.
#[test]
fn should_refuse_a_request_before_the_model_is_loaded() {
    let unloaded_limits = Qwen3_5GenerationRequestLimits {
        is_model_loaded: false,
        ..request_limits()
    };

    let validation_outcome = validate_generation_request_and_resolve_reserved_context(
        &unloaded_limits,
        &request_of(vec![1], 4),
    );

    assert!(matches!(
        validation_outcome,
        Err(InferenceEngineError::Fatal { .. })
    ));
}
