//! Thinking-allowance resolution contracts against the caller's output budget.

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{Qwen3_5Tokenizer, Qwen3_5TokenizerError};

use crate::common::qwen3_5_moe::frozen_ornith_1_0_image_processor;

const ORNITH_VOCABULARY_SIZE: u32 = 248_320;
const ORNITH_MAXIMUM_POSITION_COUNT: u32 = 262_144;
const SYNTHETIC_MODEL_ID: &str = "synthetic-qwen3.5";
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[test]
fn should_reject_a_thinking_budget_that_cannot_fit_its_transition_and_visible_answer() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &super::tokenizer::ornith_tokenizer_json_bytes(248_056),
        SYNTHETIC_MODEL_ID,
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        frozen_ornith_1_0_image_processor(),
    )
    .expect("the synthetic tokenizer should load");
    let preparation_error = tokenizer
        .prepare_chat(
            &ChatGenerationCommand {
                request_id: RequestId::new(4_004),
                model: SYNTHETIC_MODEL_ID.to_owned(),
                messages: vec![ChatMessage::User {
                    content: ROMEO_AND_JULIET_SOURCE.chars().take(128).collect(),
                    images: Vec::new(),
                }],
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                settings: ChatGenerationSettings {
                    max_output_tokens: 2,
                    temperature_thousandths: None,
                    top_p_thousandths: None,
                    seed: None,
                    thinking_budget: Some(1),
                },
                qwen_thinking_channel_seed: None,
                structured_generation: None,
            },
            true,
        )
        .expect_err("the request cannot reserve its complete model-owned transition");

    assert!(matches!(
        preparation_error,
        Qwen3_5TokenizerError::ThinkingBudgetOutputReservation { .. }
    ));
}

#[test]
fn should_clamp_the_thinking_allowance_to_fit_the_requested_output_budget() {
    // Issue #652: a caller that asks for "at most N output tokens" states an
    // upper bound. Pi compaction sends max_output 16,000 with a 16,384
    // high-effort allowance; the request must be served with a shrunken
    // allowance, not rejected.
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &super::tokenizer::ornith_tokenizer_json_bytes(248_056),
        SYNTHETIC_MODEL_ID,
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        frozen_ornith_1_0_image_processor(),
    )
    .expect("the synthetic tokenizer should load");
    let transition_token_count = tokenizer.forced_thinking_transition_token_ids().len();
    let requested_thinking_budget = 16_384_u16;
    let max_output_tokens = 16_000_u16;

    let prepared_request = tokenizer
        .prepare_chat(
            &ChatGenerationCommand {
                request_id: RequestId::new(4_005),
                model: SYNTHETIC_MODEL_ID.to_owned(),
                messages: vec![ChatMessage::User {
                    content: ROMEO_AND_JULIET_SOURCE.chars().take(128).collect(),
                    images: Vec::new(),
                }],
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                settings: ChatGenerationSettings {
                    max_output_tokens,
                    temperature_thousandths: None,
                    top_p_thousandths: None,
                    seed: None,
                    thinking_budget: Some(requested_thinking_budget),
                },
                qwen_thinking_channel_seed: None,
                structured_generation: None,
            },
            true,
        )
        .expect("an output cap below the requested allowance must clamp the allowance, not reject");

    let expected_allowance =
        u16::try_from(usize::from(max_output_tokens) - transition_token_count - 1)
            .expect("the clamped allowance should fit u16");
    assert_ne!(expected_allowance, requested_thinking_budget);
    assert_eq!(
        prepared_request.thinking_budget(),
        Some(expected_allowance),
        "the effective allowance must leave room for the transition and one visible answer token"
    );
    assert_eq!(prepared_request.max_output_tokens(), max_output_tokens);
}

#[test]
fn should_keep_a_thinking_allowance_that_already_fits_its_output_budget() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &super::tokenizer::ornith_tokenizer_json_bytes(248_056),
        SYNTHETIC_MODEL_ID,
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        frozen_ornith_1_0_image_processor(),
    )
    .expect("the synthetic tokenizer should load");
    let transition_token_count = tokenizer.forced_thinking_transition_token_ids().len();
    let requested_thinking_budget = 1_000_u16;
    let max_output_tokens =
        u16::try_from(1_000 + transition_token_count + 1).expect("the budget should fit u16");

    let prepared_request = tokenizer
        .prepare_chat(
            &ChatGenerationCommand {
                request_id: RequestId::new(4_006),
                model: SYNTHETIC_MODEL_ID.to_owned(),
                messages: vec![ChatMessage::User {
                    content: ROMEO_AND_JULIET_SOURCE.chars().take(128).collect(),
                    images: Vec::new(),
                }],
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                settings: ChatGenerationSettings {
                    max_output_tokens,
                    temperature_thousandths: None,
                    top_p_thousandths: None,
                    seed: None,
                    thinking_budget: Some(requested_thinking_budget),
                },
                qwen_thinking_channel_seed: None,
                structured_generation: None,
            },
            true,
        )
        .expect("a request whose allowance fits its output budget must pass through unchanged");

    assert_eq!(
        prepared_request.thinking_budget(),
        Some(requested_thinking_budget),
        "a fitting allowance must never be altered"
    );
}
