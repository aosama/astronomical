use ipc_protocol::{RequestId, WorkerPromptWorkReuse};
use serial_test::file_serial;

use super::{PrismRequestStart, prism_active_generation};
use crate::prism_hadamard_qwen35::model::fixture_model_test_support::load_fixture_text_model;
use crate::{PerformanceAttribution, Qwen3_5InferenceRequest, Qwen3_5ThinkingBudgetState};

/// A five-token prompt started with no `astro_ssd_prompt_cache` open.
const PROMPT_TOKEN_IDS: [u32; 5] = [11, 12, 13, 14, 15];
const MAXIMUM_OUTPUT_TOKENS: u16 = 7;

#[test]
#[file_serial]
fn should_start_the_shared_record_at_the_prompt_start_without_a_store() {
    let fixture = load_fixture_text_model();
    let text_model = &fixture.text_model;
    let mut inference_request = Qwen3_5InferenceRequest::new(
        RequestId::new(41),
        PROMPT_TOKEN_IDS.to_vec(),
        MAXIMUM_OUTPUT_TOKENS,
    );
    let request_start = PrismRequestStart {
        prompt_token_ids: PROMPT_TOKEN_IDS.to_vec(),
        request_decoder_state: text_model
            .empty_request_decoder_state()
            .expect("the fixture layout should create decoder state"),
        restore_outcome: None,
        random_state: Some(
            text_model
                .runtime
                .random_key(7)
                .expect("the random key should build"),
        ),
        thinking_budget_state: Qwen3_5ThinkingBudgetState::new(false, None, Vec::new(), Vec::new())
            .expect("a request outside thinking needs no forced transition"),
        performance_attribution: PerformanceAttribution::disabled(),
        can_use_astro_ssd_prompt_cache: false,
    };

    let active_generation = prism_active_generation(&mut inference_request, request_start);

    assert_eq!(active_generation.request_id, RequestId::new(41));
    assert_eq!(active_generation.input_token_ids, PROMPT_TOKEN_IDS);
    assert_eq!(active_generation.prefill_cursor, 0);
    assert_eq!(active_generation.next_position_tokens, 0);
    assert_eq!(
        active_generation.prompt_work_reuse,
        WorkerPromptWorkReuse {
            target_eligible_token_count: 5,
            target_restored_token_count: 0,
        }
    );
    assert!(
        active_generation
            .astro_ssd_prompt_cache_diagnostics
            .is_none()
    );
    assert!(
        active_generation
            .last_restored_astro_ssd_prompt_cache_block_key
            .is_none()
    );
    assert_eq!(
        active_generation.maximum_output_tokens,
        MAXIMUM_OUTPUT_TOKENS
    );
    // The shared decode step counts the output budget from zero emitted.
    assert_eq!(active_generation.generated_token_count, 0);
    assert!(active_generation.random_state.is_some());
    assert!(active_generation.pending_generated_token.is_none());
    assert!(!active_generation.can_use_astro_ssd_prompt_cache);
    assert_eq!(
        active_generation.sampling_strategy,
        inference_request.sampling_strategy()
    );
    assert_eq!(
        active_generation.repetition_penalty_thousandths,
        inference_request.repetition_penalty_thousandths()
    );
}
