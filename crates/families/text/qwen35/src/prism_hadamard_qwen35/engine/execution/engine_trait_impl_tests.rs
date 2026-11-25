use ipc_protocol::{RequestId, WorkerPromptProcessingPhase};
use serial_test::file_serial;

use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::FIXTURE_VOCAB_SIZE;
use crate::prism_hadamard_qwen35::engine::fixture_engine_test_support::{
    FIXTURE_THINK_END_TOKEN_ID, LoadedFixtureEngine, loaded_fixture_engine,
};
use crate::{InferenceEngineError, MlxInferenceExecution, TextInferenceRequest};

const PROMPT_TOKEN_IDS: [u32; 3] = [11, 12, 13];

fn start_greedy(
    fixture_engine: &mut LoadedFixtureEngine,
    prompt_token_ids: Vec<u32>,
    maximum_output_tokens: u16,
) -> Result<crate::EngineGenerationStart, InferenceEngineError> {
    fixture_engine
        .execution
        .start_generation(TextInferenceRequest::new_sampling(
            RequestId::new(81),
            prompt_token_ids,
            maximum_output_tokens,
            0,
            1_000,
            None,
        ))
}

#[test]
#[file_serial]
fn should_refuse_an_empty_prompt_and_an_out_of_vocabulary_token_as_engine_failures() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    assert!(matches!(
        start_greedy(&mut fixture_engine, Vec::new(), 4),
        Err(InferenceEngineError::Fatal { .. })
    ));
    assert!(matches!(
        start_greedy(&mut fixture_engine, vec![11, FIXTURE_VOCAB_SIZE], 4),
        Err(InferenceEngineError::Fatal { .. })
    ));
    assert!(matches!(
        start_greedy(&mut fixture_engine, PROMPT_TOKEN_IDS.to_vec(), 0),
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
#[file_serial]
fn should_serve_beyond_the_configured_context_up_to_the_pack_range() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    // 3 prompt tokens and 3,000 output tokens pass the configured 2,048 but
    // fit the pack's 4,096 positions.
    let generation_start = start_greedy(&mut fixture_engine, PROMPT_TOKEN_IDS.to_vec(), 3_000)
        .expect("a request within the pack's range should start");
    assert_eq!(
        generation_start.prompt_processing_phase(),
        Some(WorkerPromptProcessingPhase::Target)
    );
    assert_eq!(generation_start.restored_prompt_prefix_token_count(), 0);
    fixture_engine
        .execution
        .cancel_generation(RequestId::new(81))
        .expect("the request should cancel");

    // Edge: one position past the pack's range is refused as the caller's.
    assert!(matches!(
        start_greedy(&mut fixture_engine, PROMPT_TOKEN_IDS.to_vec(), 4_094),
        Err(InferenceEngineError::InvalidRequest { .. })
    ));
}

#[test]
#[file_serial]
fn should_hold_no_random_state_for_greedy_and_refuse_an_impossible_sampling_strategy() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    start_greedy(&mut fixture_engine, PROMPT_TOKEN_IDS.to_vec(), 4)
        .expect("a greedy request should start");
    let greedy_request_record = fixture_engine
        .execution
        .execution_core
        .active_request
        .as_ref()
        .expect("the request is active");
    assert!(greedy_request_record.random_state.is_none());
    fixture_engine
        .execution
        .cancel_generation(RequestId::new(81))
        .expect("the request should cancel");

    let zero_top_k_request = TextInferenceRequest::new_sampling_with_top_k(
        RequestId::new(82),
        PROMPT_TOKEN_IDS.to_vec(),
        4,
        700,
        0,
        950,
        Some(7),
    );
    assert!(matches!(
        fixture_engine
            .execution
            .start_generation(zero_top_k_request),
        Err(InferenceEngineError::Fatal { .. })
    ));
}

#[test]
#[file_serial]
fn should_require_a_thinking_transition_that_ends_with_the_model_marker() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    let unmarked_transition_request = TextInferenceRequest::new_sampling(
        RequestId::new(83),
        PROMPT_TOKEN_IDS.to_vec(),
        8,
        0,
        1_000,
        None,
    )
    .with_thinking_configuration(true, Some(2), vec![20, 21], vec![21]);
    assert!(matches!(
        fixture_engine
            .execution
            .start_generation(unmarked_transition_request),
        Err(InferenceEngineError::InvalidRequest { .. })
    ));

    // Negative: a marked transition still needs room for itself and an
    // answer inside the output budget (2 thinking + 2 forced + 1 answer).
    let cramped_budget_request = TextInferenceRequest::new_sampling(
        RequestId::new(84),
        PROMPT_TOKEN_IDS.to_vec(),
        4,
        0,
        1_000,
        None,
    )
    .with_thinking_configuration(
        true,
        Some(2),
        vec![20, FIXTURE_THINK_END_TOKEN_ID],
        vec![FIXTURE_THINK_END_TOKEN_ID],
    );
    assert!(matches!(
        fixture_engine
            .execution
            .start_generation(cramped_budget_request),
        Err(InferenceEngineError::InvalidRequest { .. })
    ));
}

#[test]
#[file_serial]
fn should_refuse_a_second_request_while_one_is_active() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    start_greedy(&mut fixture_engine, PROMPT_TOKEN_IDS.to_vec(), 4)
        .expect("the first request should start");
    assert!(matches!(
        start_greedy(&mut fixture_engine, PROMPT_TOKEN_IDS.to_vec(), 4),
        Err(InferenceEngineError::EngineBusy)
    ));
}

// Edge: a stop with nothing running, or meant for another request, reports
// nothing and leaves the running request alone, because a late stop for a
// finished request must not end its successor.
#[test]
#[file_serial]
fn should_ignore_a_stop_with_nothing_running_or_meant_for_another_request() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    assert_eq!(
        fixture_engine
            .execution
            .cancel_generation(RequestId::new(81))
            .expect("a stray stop is not an engine failure"),
        crate::GenerationFinalization::default()
    );
    start_greedy(&mut fixture_engine, PROMPT_TOKEN_IDS.to_vec(), 4)
        .expect("the request should start");

    assert_eq!(
        fixture_engine
            .execution
            .cancel_generation(RequestId::new(99))
            .expect("a stray stop is not an engine failure"),
        crate::GenerationFinalization::default()
    );
    assert!(
        fixture_engine
            .execution
            .execution_core
            .active_request
            .is_some()
    );
}

// Positive: a stop for the running request ends it and reports the idle
// placement and memory.
#[test]
#[file_serial]
fn should_end_the_running_request_a_stop_names() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    start_greedy(&mut fixture_engine, PROMPT_TOKEN_IDS.to_vec(), 4)
        .expect("the request should start");

    let generation_finalization = fixture_engine
        .execution
        .cancel_generation(RequestId::new(81))
        .expect("the request should cancel");

    assert_eq!(
        generation_finalization.expert_memory_mode(),
        Some(ipc_protocol::ExpertMemoryMode::Resident)
    );
    assert!(generation_finalization.mlx_memory_telemetry().is_some());
    assert!(
        fixture_engine
            .execution
            .execution_core
            .active_request
            .is_none()
    );
}
