use super::super::validate_context_token_count;
use crate::qwen3_5_core::frozen_config_test_support::*;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};

use super::*;

const ORNITH_VOCABULARY_SIZE: u32 = 248_320;
const ORNITH_MAXIMUM_POSITION_COUNT: u32 = 262_144;
const SYNTHETIC_MODEL_ID: &str = "synthetic-qwen3.5";
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../../verification/acceptance/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[test]
fn should_discover_special_token_ids_from_tokenizer_json() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes(248_056),
        SYNTHETIC_MODEL_ID,
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        frozen_ornith_1_0_image_processor(),
    )
    .expect("the synthetic tokenizer should discover every special token ID");

    assert_eq!(tokenizer.image_pad_token_id(), 248_056);
    assert_eq!(tokenizer.end_of_text_token_id(), 248_044);
    assert_eq!(tokenizer.im_start_token_id(), 248_045);
    assert_eq!(tokenizer.im_end_token_id(), 248_046);
    assert_eq!(tokenizer.think_start_token_id(), 248_068);
    assert_eq!(tokenizer.think_end_token_id(), 248_069);
}

#[test]
fn should_digest_token_identifier_mappings_independently_of_json_serialization() {
    let compact_tokenizer_bytes = ornith_tokenizer_json_bytes(248_056);
    let tokenizer_document = serde_json::from_slice::<serde_json::Value>(&compact_tokenizer_bytes)
        .expect("the synthetic tokenizer should parse as JSON");
    let pretty_tokenizer_bytes = serde_json::to_vec_pretty(&tokenizer_document)
        .expect("the synthetic tokenizer should serialize with different formatting");

    assert_ne!(compact_tokenizer_bytes, pretty_tokenizer_bytes);
    assert_eq!(
        Qwen3_5Tokenizer::token_identifier_mapping_digest(&compact_tokenizer_bytes)
            .expect("the compact tokenizer mapping should digest"),
        Qwen3_5Tokenizer::token_identifier_mapping_digest(&pretty_tokenizer_bytes)
            .expect("the pretty tokenizer mapping should digest"),
    );
    assert_ne!(
        Qwen3_5Tokenizer::token_identifier_mapping_digest(&compact_tokenizer_bytes)
            .expect("the expected tokenizer mapping should digest"),
        Qwen3_5Tokenizer::token_identifier_mapping_digest(&ornith_tokenizer_json_bytes(248_057))
            .expect("the changed tokenizer mapping should digest"),
    );
}

#[test]
fn should_reject_a_tokenizer_missing_a_required_special_token() {
    let tokenizer_error = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes_missing_image_pad(),
        SYNTHETIC_MODEL_ID,
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        frozen_ornith_1_0_image_processor(),
    )
    .expect_err("the tokenizer should reject a missing special token");

    assert!(matches!(
        tokenizer_error,
        Qwen3_5TokenizerError::DiscoverTokenIds { .. }
    ));
}

#[test]
fn should_accept_an_opencode_context_above_the_retired_32k_input_limit() {
    assert!(validate_context_token_count(40_665, 4_096, 262_144, 262_144).is_ok());
}

#[test]
fn should_reject_a_context_above_the_frozen_ornith_1_0_position_limit() {
    assert!(validate_context_token_count(258_049, 4_096, 262_144, 262_144).is_err());
}

#[test]
fn should_serve_a_context_above_the_configured_limit_when_the_artifact_window_fits() {
    assert!(validate_context_token_count(150_500, 4_096, 262_144, 150_000).is_ok());
}

#[test]
fn should_reject_a_context_above_the_artifact_window_even_below_no_configured_limit() {
    assert!(validate_context_token_count(258_049, 4_096, 262_144, 150_000).is_err());
}

#[test]
fn should_prepare_a_zero_budget_chat_to_generate_outside_the_thinking_block() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes(248_056),
        SYNTHETIC_MODEL_ID,
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        frozen_ornith_1_0_image_processor(),
    )
    .expect("the synthetic tokenizer should load");
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(701),
        model: SYNTHETIC_MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: ROMEO_AND_JULIET_SOURCE.to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 8,
            temperature_thousandths: Some(1_000),
            top_p_thousandths: Some(950),
            repetition_penalty_thousandths: None,
            seed: None,
            thinking_budget: Some(0),
        },
        structured_generation: None,
    };

    let inference_request = tokenizer
        .prepare_chat(&chat_generation_command, false)
        .expect("a thinking-disabled Romeo and Juliet request should prepare");

    assert!(!inference_request.generation_starts_inside_thinking_block());
    assert_eq!(inference_request.thinking_budget(), None);
}

#[test]
fn should_preserve_a_positive_budget_for_generation_that_starts_inside_thinking() {
    let forced_transition_token_ids = vec![50, 51, 52];
    let inference_request =
        super::super::Qwen3_5InferenceRequest::new(RequestId::new(702), vec![1, 2, 3], 100)
            .with_thinking_configuration(
                true,
                Some(64),
                forced_transition_token_ids.clone(),
                vec![52, 53],
            );

    assert!(inference_request.generation_starts_inside_thinking_block());
    assert_eq!(inference_request.thinking_budget(), Some(64));
    assert_eq!(
        inference_request.forced_thinking_transition_token_ids(),
        forced_transition_token_ids
    );
}

#[test]
fn should_prepare_a_model_owned_multitoken_transition_ending_at_the_thinking_marker() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes(248_056),
        SYNTHETIC_MODEL_ID,
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        frozen_ornith_1_0_image_processor(),
    )
    .expect("the synthetic tokenizer should load");

    let forced_transition_token_ids = tokenizer.forced_thinking_transition_token_ids();
    assert!(forced_transition_token_ids.len() > 1);
    assert_eq!(
        forced_transition_token_ids.last().copied(),
        Some(tokenizer.think_end_token_id())
    );
    assert!(
        tokenizer
            .natural_reasoning_end_token_ids()
            .contains(&tokenizer.tool_call_start_token_id())
    );
}

pub(crate) fn ornith_tokenizer_json_bytes(image_pad_token_id: u32) -> Vec<u8> {
    let vocab = serde_json::json!({
        "<unk>": 0,
        "<unk>": 0,
        "<|endoftext|>": 248_044,
        "<|im_start|>": 248_045,
        "<|im_end|>": 248_046,
        "<|image_pad|>": image_pad_token_id,
        "<tool_call>": 248_058,
        "</tool_call>": 248_059,
        "<tool_response>": 248_066,
        "</tool_response>": 248_067,
        "<think>": 248_068,
        "</think>": 248_069
    });
    serde_json::to_vec(&serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>"
        }
    }))
    .expect("the synthetic tokenizer JSON should serialize")
}

pub(super) fn ornith_tokenizer_json_bytes_missing_image_pad() -> Vec<u8> {
    let vocab = serde_json::json!({
        "<unk>": 0,
        "<|endoftext|>": 248_044,
        "<|im_start|>": 248_045,
        "<|im_end|>": 248_046,
        "<tool_call>": 248_058,
        "</tool_call>": 248_059,
        "<tool_response>": 248_066,
        "</tool_response>": 248_067,
        "<think>": 248_068,
        "</think>": 248_069,
    });
    serde_json::to_vec(&serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>"
        }
    }))
    .expect("the synthetic tokenizer JSON should serialize")
}

fn thinking_budget_probe_command(
    thinking_budget: Option<u16>,
    max_output_tokens: u16,
) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(703),
        model: SYNTHETIC_MODEL_ID.to_owned(),
        messages: Vec::new(),
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens,
            temperature_thousandths: None,
            top_p_thousandths: None,
            repetition_penalty_thousandths: None,
            seed: None,
            thinking_budget,
        },
        structured_generation: None,
    }
}

#[test]
fn should_accept_a_provider_prefixed_model_id_resolving_to_the_loaded_leaf() {
    assert!(
        reject_mismatched_model_id("astronomical/synthetic-qwen3.5", SYNTHETIC_MODEL_ID,).is_ok()
    );
}

#[test]
fn should_reject_a_model_id_resolving_to_a_leaf_other_than_the_loaded_one() {
    let tokenizer_error =
        reject_mismatched_model_id("astronomical/some-other-model", SYNTHETIC_MODEL_ID)
            .expect_err("an unresolvable model ID should be rejected");

    assert!(matches!(
        tokenizer_error,
        Qwen3_5TokenizerError::ModelIdMismatch { actual_model_id }
            if actual_model_id == "astronomical/some-other-model"
    ));
}

#[test]
fn should_accept_the_exact_loaded_leaf_model_id() {
    assert!(reject_mismatched_model_id(SYNTHETIC_MODEL_ID, SYNTHETIC_MODEL_ID).is_ok());
}

#[test]
fn should_pass_through_a_positive_configured_top_k() {
    assert_eq!(clamp_model_top_k(7), 7);
}

#[test]
fn should_fall_back_to_the_default_top_k_for_a_zero_or_negative_configured_value() {
    assert_eq!(clamp_model_top_k(0), 20);
    assert_eq!(clamp_model_top_k(-3), 20);
}

#[test]
fn should_fall_back_to_the_default_top_k_above_the_u16_range_but_keep_the_boundary_value() {
    assert_eq!(clamp_model_top_k(65_536), 20);
    assert_eq!(clamp_model_top_k(65_535), 65_535);
}

#[test]
fn should_keep_a_thinking_allowance_that_fits_the_output_budget() {
    let chat_generation_command = thinking_budget_probe_command(Some(8), 100);

    assert_eq!(
        validate_thinking_budget_against_output_reservation(&chat_generation_command, true, 3)
            .expect("a fitting allowance should pass"),
        Some(8)
    );
}

#[test]
fn should_accept_an_output_budget_exactly_at_the_thinking_reservation_boundary() {
    // Budget 8 + transition 3 + one visible answer token = 12 output tokens
    // exactly, the smallest accepted budget for the requested allowance.
    let chat_generation_command = thinking_budget_probe_command(Some(8), 12);

    assert_eq!(
        validate_thinking_budget_against_output_reservation(&chat_generation_command, true, 3)
            .expect("the exact reservation boundary should pass"),
        Some(8)
    );
}

#[test]
fn should_reject_an_output_budget_below_the_thinking_transition_reservation() {
    // The 10-token transition exceeds the 5-token output budget, so the
    // allowance clamps to zero — yet even a zero allowance still reserves
    // the complete transition plus one visible token: 11 > 5.
    let chat_generation_command = thinking_budget_probe_command(Some(8), 5);

    let tokenizer_error =
        validate_thinking_budget_against_output_reservation(&chat_generation_command, true, 10)
            .expect_err("an output budget below the reservation should be rejected");

    assert!(matches!(
        tokenizer_error,
        Qwen3_5TokenizerError::ThinkingBudgetOutputReservation {
            max_output_tokens: 5,
            thinking_budget: 0,
            transition_token_count: 10,
        }
    ));
}

#[test]
fn should_clamp_an_oversized_thinking_allowance_to_fit_the_output_budget() {
    // Budget 64 needs 68 output tokens, so the allowance shrinks to
    // 8 - 3 - 1 = 4 while the request stays accepted.
    let chat_generation_command = thinking_budget_probe_command(Some(64), 8);

    assert_eq!(
        validate_thinking_budget_against_output_reservation(&chat_generation_command, true, 3)
            .expect("the clamped allowance should pass"),
        Some(4)
    );
}

#[test]
fn should_resolve_no_thinking_allowance_when_thinking_is_disabled() {
    let chat_generation_command = thinking_budget_probe_command(Some(8), 8);

    assert_eq!(
        validate_thinking_budget_against_output_reservation(&chat_generation_command, false, 3)
            .expect("a disabled allowance should pass"),
        None
    );
}
