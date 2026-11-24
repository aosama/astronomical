use ipc_protocol::RequestId;
use serial_test::serial;

use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::FIXTURE_VOCAB_SIZE;
use crate::prism_hadamard_qwen35::engine::fixture_engine_test_support::{
    LoadedFixtureEngine, loaded_fixture_engine,
};
use crate::{GeneratedToken, InferenceEngineError, MlxInferenceExecution, Qwen3_5InferenceRequest};

const PROMPT_TOKEN_IDS: [u32; 3] = [11, 12, 13];
const FEEDBACK_TOKEN_IDS: [u32; 3] = [20, 21, 22];

/// Starts a greedy request and decodes until its first reply token, so the
/// prompt has been read and feedback lands after it.
fn engine_with_first_reply_token(
    request_id: u64,
    maximum_output_tokens: u16,
) -> LoadedFixtureEngine {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    fixture_engine
        .execution
        .start_generation(Qwen3_5InferenceRequest::new_sampling(
            RequestId::new(request_id),
            PROMPT_TOKEN_IDS.to_vec(),
            maximum_output_tokens,
            0,
            1_000,
            None,
        ))
        .expect("the request should start");
    for _ in 0..8 {
        if let GeneratedToken::TokenId { .. } = fixture_engine
            .execution
            .decode_next_token(RequestId::new(request_id))
            .expect("the decode step should succeed")
        {
            return fixture_engine;
        }
    }
    panic!("the request never produced its first reply token");
}

fn active_next_position_tokens(fixture_engine: &LoadedFixtureEngine) -> u32 {
    fixture_engine
        .execution
        .execution_core
        .active_request
        .as_ref()
        .expect("the request is still active")
        .next_position_tokens
}

// Positive: a feedback prefix and its final token extend the context, a
// pending reply token is sampled from it, and decoding continues.
#[test]
#[serial]
fn should_read_injected_feedback_and_keep_decoding() {
    let mut fixture_engine = engine_with_first_reply_token(91, 8);
    let position_before_feedback = active_next_position_tokens(&fixture_engine);

    fixture_engine
        .execution
        .inject_input_tokens(RequestId::new(91), FEEDBACK_TOKEN_IDS.to_vec())
        .expect("feedback inside the vocabulary and the context should inject");

    assert_eq!(
        active_next_position_tokens(&fixture_engine),
        position_before_feedback + 3
    );
    let active_request = fixture_engine
        .execution
        .execution_core
        .active_request
        .as_ref()
        .expect("the request is still active");
    assert!(active_request.pending_generated_token.is_some());
    assert!(matches!(
        fixture_engine
            .execution
            .decode_next_token(RequestId::new(91)),
        Ok(GeneratedToken::TokenId { .. })
    ));
}

// Edge: one feedback token has no prefix to read, and empty feedback
// changes nothing.
#[test]
#[serial]
fn should_inject_a_single_token_and_ignore_empty_feedback() {
    let mut fixture_engine = engine_with_first_reply_token(92, 8);
    let position_before_feedback = active_next_position_tokens(&fixture_engine);

    fixture_engine
        .execution
        .inject_input_tokens(RequestId::new(92), Vec::new())
        .expect("empty feedback is accepted");
    assert_eq!(
        active_next_position_tokens(&fixture_engine),
        position_before_feedback
    );
    fixture_engine
        .execution
        .inject_input_tokens(RequestId::new(92), vec![20])
        .expect("a single feedback token should inject");
    assert_eq!(
        active_next_position_tokens(&fixture_engine),
        position_before_feedback + 1
    );
}

// Edge: feedback past the configured 2,048-token window but inside the
// pack's 4,096 positions is served, as the shared rule allows.
#[test]
#[serial]
fn should_inject_past_the_configured_window_inside_the_pack_range() {
    let mut fixture_engine = engine_with_first_reply_token(93, 3_000);

    fixture_engine
        .execution
        .inject_input_tokens(RequestId::new(93), FEEDBACK_TOKEN_IDS.to_vec())
        .expect("feedback inside the pack's range should inject");
    assert!(
        fixture_engine
            .execution
            .execution_core
            .active_request
            .is_some()
    );
}

// Negative: a token outside the vocabulary and feedback for another
// request are engine failures that leave the running request alone.
#[test]
#[serial]
fn should_refuse_foreign_or_out_of_vocabulary_feedback_and_keep_the_request() {
    let mut fixture_engine = engine_with_first_reply_token(94, 8);

    assert!(matches!(
        fixture_engine
            .execution
            .inject_input_tokens(RequestId::new(94), vec![20, FIXTURE_VOCAB_SIZE]),
        Err(InferenceEngineError::Fatal { .. })
    ));
    assert!(matches!(
        fixture_engine
            .execution
            .inject_input_tokens(RequestId::new(95), FEEDBACK_TOKEN_IDS.to_vec()),
        Err(InferenceEngineError::Fatal { .. })
    ));
    assert!(
        fixture_engine
            .execution
            .execution_core
            .active_request
            .is_some()
    );
}

// Negative: feedback that would carry the context past the pack's 4,096
// positions is refused as the caller's and ends the request.
#[test]
#[serial]
fn should_refuse_feedback_past_the_pack_range_and_end_the_request() {
    let mut fixture_engine = engine_with_first_reply_token(96, 4_000);

    assert!(matches!(
        fixture_engine
            .execution
            .inject_input_tokens(RequestId::new(96), vec![20; 100]),
        Err(InferenceEngineError::InvalidRequest { .. })
    ));
    assert!(
        fixture_engine
            .execution
            .execution_core
            .active_request
            .is_none()
    );
}
