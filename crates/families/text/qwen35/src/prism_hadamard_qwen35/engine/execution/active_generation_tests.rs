use ipc_protocol::{RequestId, WorkerPromptWorkReuse};
use serial_test::serial;

use super::{PrismRequestStart, prism_active_generation};
use crate::prism_hadamard_qwen35::model::fixture_model_test_support::load_fixture_text_model;
use crate::{PerformanceAttribution, Qwen3_5InferenceRequest, Qwen3_5ThinkingBudgetState};

/// A five-token prompt whose first two tokens were restored from the
/// `astro_ssd_prompt_cache`.
const PROMPT_TOKEN_IDS: [u32; 5] = [11, 12, 13, 14, 15];
const RESTORED_TOKEN_COUNT: u32 = 2;
const MAXIMUM_OUTPUT_TOKENS: u16 = 7;

#[test]
#[serial]
fn should_start_the_shared_record_right_after_the_restored_prefix() {
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
        restored_token_count: RESTORED_TOKEN_COUNT,
        last_restored_block_key: None,
        random_state: text_model
            .runtime
            .random_key(7)
            .expect("the random key should build"),
        thinking_budget_state: Qwen3_5ThinkingBudgetState::new(false, None, Vec::new(), Vec::new())
            .expect("a request outside thinking needs no forced transition"),
        performance_attribution: PerformanceAttribution::disabled(),
        can_use_astro_ssd_prompt_cache: true,
    };

    let active_generation = prism_active_generation(&mut inference_request, request_start);

    assert_eq!(active_generation.request_id, RequestId::new(41));
    assert_eq!(active_generation.input_token_ids, PROMPT_TOKEN_IDS);
    assert_eq!(active_generation.prefill_cursor, 2);
    assert_eq!(active_generation.next_position_tokens, 2);
    assert_eq!(
        active_generation
            .engine_extension
            .remaining_prompt_token_ids,
        [13, 14, 15]
    );
    assert_eq!(
        active_generation.prompt_work_reuse,
        WorkerPromptWorkReuse {
            target_eligible_token_count: 5,
            target_restored_token_count: 2,
        }
    );
    assert_eq!(
        active_generation.maximum_output_tokens,
        MAXIMUM_OUTPUT_TOKENS
    );
    assert_eq!(
        active_generation.engine_extension.remaining_output_tokens,
        u32::from(MAXIMUM_OUTPUT_TOKENS)
    );
    assert!(!active_generation.engine_extension.has_reached_reply_end);
    assert!(active_generation.random_state.is_some());
    assert!(active_generation.pending_generated_token.is_none());
    assert!(active_generation.can_use_astro_ssd_prompt_cache);
    assert_eq!(
        active_generation.sampling_strategy,
        inference_request.sampling_strategy()
    );
    assert_eq!(
        active_generation.repetition_penalty_thousandths,
        inference_request.repetition_penalty_thousandths()
    );
}
