use ipc_protocol::{RequestId, StructuredGenerationConstraint};
use serde_json::json;
use serial_test::file_serial;

use super::successor_may_be_submitted_before_read;
use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::FIXTURE_VOCAB_SIZE;
use crate::prism_hadamard_qwen35::engine::execution::PrismRequestExtension;
use crate::prism_hadamard_qwen35::engine::fixture_engine_test_support::{
    LoadedFixtureEngine, loaded_fixture_engine,
};
use crate::qwen3_5_core::engine_request_test_support::test_engine_request;
use crate::structured_generation::StructuredTokenConstraint;
use crate::{GeneratedToken, InferenceEngineError, MlxInferenceExecution, Qwen3_5InferenceRequest};

const PROMPT_TOKEN_IDS: [u32; 3] = [11, 12, 13];

/// A greedy request: temperature zero selects the highest logit.
fn greedy_request(request_id: u64, maximum_output_tokens: u16) -> Qwen3_5InferenceRequest {
    Qwen3_5InferenceRequest::new_sampling(
        RequestId::new(request_id),
        PROMPT_TOKEN_IDS.to_vec(),
        maximum_output_tokens,
        0,
        1_000,
        None,
    )
}

/// Runs decode steps until the request finalizes itself, returning every
/// reply token and whether each carried the finalization.
fn decode_until_finalized(
    fixture_engine: &mut LoadedFixtureEngine,
    request_id: u64,
) -> Vec<(u32, bool, bool)> {
    let mut reply_tokens = Vec::new();
    for _ in 0..16 {
        match fixture_engine
            .execution
            .decode_next_token(RequestId::new(request_id))
            .expect("the decode step should succeed")
        {
            GeneratedToken::TokenId {
                token_id,
                is_reasoning_token,
                generation_finalization,
                ..
            } => {
                let is_finalized = generation_finalization.is_some();
                reply_tokens.push((token_id, is_reasoning_token, is_finalized));
                if is_finalized {
                    return reply_tokens;
                }
            }
            GeneratedToken::PrefillProgress { .. } => {}
            other_generated_token => panic!("unexpected advance {other_generated_token:?}"),
        }
    }
    panic!("the request never finalized itself");
}

#[test]
#[file_serial]
fn should_emit_the_end_of_sequence_token_with_its_finalization_and_end_the_request() {
    // Every vocabulary id ends the reply, so the first reply token does.
    let mut fixture_engine = loaded_fixture_engine(|fixture_pack| {
        fixture_pack.config["text_config"]["eos_token_id"] =
            json!((0..FIXTURE_VOCAB_SIZE).collect::<Vec<u32>>());
    });
    fixture_engine
        .execution
        .start_generation(greedy_request(71, 8))
        .expect("the request should start");

    let reply_tokens = decode_until_finalized(&mut fixture_engine, 71);

    assert_eq!(
        reply_tokens.len(),
        1,
        "the end-of-sequence token is emitted"
    );
    assert!(
        fixture_engine
            .execution
            .execution_core
            .active_request
            .is_none()
    );
    fixture_engine
        .execution
        .cancel_generation(RequestId::new(71))
        .expect("a cancel after the request ended on its own is harmless");
}

#[test]
#[file_serial]
fn should_end_the_reply_at_the_output_budget_with_its_finalization() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    fixture_engine
        .execution
        .start_generation(greedy_request(72, 3))
        .expect("the request should start");

    let reply_tokens = decode_until_finalized(&mut fixture_engine, 72);

    let finalization_flags: Vec<bool> = reply_tokens
        .iter()
        .map(|(_, _, is_finalized)| *is_finalized)
        .collect();
    assert_eq!(finalization_flags, [false, false, true]);
    assert!(
        fixture_engine
            .execution
            .execution_core
            .active_request
            .is_none()
    );
}

#[test]
#[file_serial]
fn should_force_the_thinking_transition_once_the_budget_is_spent() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    let forced_transition_token_ids = vec![FIXTURE_VOCAB_SIZE - 2, FIXTURE_VOCAB_SIZE - 1];
    let thinking_request = greedy_request(73, 4).with_thinking_configuration(
        true,
        Some(1),
        forced_transition_token_ids.clone(),
        vec![FIXTURE_VOCAB_SIZE - 1],
    );
    fixture_engine
        .execution
        .start_generation(thinking_request)
        .expect("the request should start");

    let reply_tokens = decode_until_finalized(&mut fixture_engine, 73);

    let reply_token_ids: Vec<u32> = reply_tokens
        .iter()
        .map(|(token_id, _, _)| *token_id)
        .collect();
    let reasoning_flags: Vec<bool> = reply_tokens
        .iter()
        .map(|(_, is_reasoning_token, _)| *is_reasoning_token)
        .collect();
    // One reasoning token spends the budget; the transition is forced next,
    // ending the reasoning; the answer continues until the output budget.
    assert_eq!(reply_token_ids[1..3], forced_transition_token_ids);
    assert_eq!(reasoning_flags, [true, true, false, false]);
}

#[test]
#[file_serial]
fn should_finalize_the_request_at_once_when_a_step_fails() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    let sampled_request =
        Qwen3_5InferenceRequest::new(RequestId::new(74), PROMPT_TOKEN_IDS.to_vec(), 4);
    fixture_engine
        .execution
        .start_generation(sampled_request)
        .expect("the request should start");
    // A sampled request without its random state cannot sample its first
    // reply token, so the step that reads the prompt fails.
    fixture_engine
        .execution
        .execution_core
        .active_request
        .as_mut()
        .expect("the request is active")
        .random_state = None;

    let failed_step = fixture_engine
        .execution
        .decode_next_token(RequestId::new(74));

    assert!(matches!(
        failed_step,
        Err(InferenceEngineError::Fatal { .. })
    ));
    assert!(
        fixture_engine
            .execution
            .execution_core
            .active_request
            .is_none()
    );
}

#[test]
fn should_submit_the_successor_early_only_when_the_read_cannot_change_it() {
    let mut active_generation = test_engine_request(PrismRequestExtension);
    active_generation.maximum_output_tokens = 3;
    assert!(successor_may_be_submitted_before_read(&active_generation));

    // Edge: the last token the budget allows has no successor worth running.
    active_generation.generated_token_count = 2;
    assert!(!successor_may_be_submitted_before_read(&active_generation));
    active_generation.generated_token_count = 1;
    assert!(successor_may_be_submitted_before_read(&active_generation));

    // Negative: the read changes the repetition history the successor reads.
    active_generation.repetition_penalty_thousandths = 1_100;
    assert!(!successor_may_be_submitted_before_read(&active_generation));
    active_generation.repetition_penalty_thousandths = 1_000;

    // Negative: the read changes the structured mask the successor reads.
    active_generation.structured_generation = Some(
        StructuredTokenConstraint::compile(
            &StructuredGenerationConstraint::Choice {
                choices: vec!["Juliet".to_owned()],
            },
            vec!["Juliet".to_owned(), "Romeo".to_owned()],
            vec![99],
            vec![vec![0]],
        )
        .expect("the fixture choice constraint should compile"),
    );
    assert!(!successor_may_be_submitted_before_read(&active_generation));
}
