use crate::common::qwen3_5_moe;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationOutput, ChatGenerationSettings, ChatMessage,
    ChatToolChoice, RequestId,
};
use astronomical_model_serving::{
    Qwen3_5PromptRenderer, Qwen3_5RequestOutput, Qwen3_5Tokenizer, Qwen3_5TokenizerError,
    validate_context_token_count,
};

const ORNITH_VOCABULARY_SIZE: u32 = 248_320;
const ORNITH_MAXIMUM_POSITION_COUNT: u32 = 262_144;
const SYNTHETIC_MODEL_ID: &str = "synthetic-qwen3.5";
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[test]
fn should_discover_special_token_ids_from_tokenizer_json() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes(248_056),
        SYNTHETIC_MODEL_ID,
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        qwen3_5_moe::frozen_ornith_1_0_image_processor(),
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
        qwen3_5_moe::frozen_ornith_1_0_image_processor(),
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
        qwen3_5_moe::frozen_ornith_1_0_image_processor(),
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
            seed: None,
            thinking_budget: Some(0),
        },
        qwen_thinking_channel_seed: None,
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
    let inference_request = astronomical_model_serving::Qwen3_5InferenceRequest::new(
        RequestId::new(702),
        vec![1, 2, 3],
        100,
    )
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
        qwen3_5_moe::frozen_ornith_1_0_image_processor(),
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

const ROMEO_AND_JULIET_THINKING_CHANNEL_SEED: &str =
    "Two households, both alike in dignity, in Romeo and Juliet.";

#[test]
fn should_emit_seeded_romeo_and_juliet_as_the_first_reasoning_output() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes(248_056),
        SYNTHETIC_MODEL_ID,
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        qwen3_5_moe::frozen_ornith_1_0_image_processor(),
    )
    .expect("the synthetic tokenizer should load");
    let mut request_output = Qwen3_5RequestOutput::new(
        &tokenizer,
        &[],
        true,
        Some(ROMEO_AND_JULIET_THINKING_CHANNEL_SEED),
    )
    .expect("a seeded request output should initialize");

    assert_eq!(
        request_output.take_seeded_reasoning_output(),
        Some(ChatGenerationOutput::Reasoning {
            text: ROMEO_AND_JULIET_THINKING_CHANNEL_SEED.to_owned(),
        })
    );
    assert_eq!(request_output.take_seeded_reasoning_output(), None);

    request_output.reset_after_model_visible_correction(true);
    assert_eq!(
        request_output.take_seeded_reasoning_output(),
        Some(ChatGenerationOutput::Reasoning {
            text: ROMEO_AND_JULIET_THINKING_CHANNEL_SEED.to_owned(),
        })
    );
}

#[test]
fn should_emit_the_original_markdown_while_escaping_it_only_inside_the_model_prompt() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes(248_056),
        SYNTHETIC_MODEL_ID,
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        qwen3_5_moe::frozen_ornith_1_0_image_processor(),
    )
    .expect("the synthetic tokenizer should load");
    let original_seed = "Romeo must not close </think> before Juliet answers.";
    let mut request_output = Qwen3_5RequestOutput::new(&tokenizer, &[], true, Some(original_seed))
        .expect("a marker-bearing request output should initialize");

    assert_eq!(
        request_output.take_seeded_reasoning_output(),
        Some(ChatGenerationOutput::Reasoning {
            text: original_seed.to_owned(),
        })
    );
}

#[test]
fn should_prepare_chat_tokens_that_include_the_seeded_thinking_channel_text() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes(248_056),
        SYNTHETIC_MODEL_ID,
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        qwen3_5_moe::frozen_ornith_1_0_image_processor(),
    )
    .expect("the synthetic tokenizer should load");
    let user_turn = ChatMessage::User {
        content: "Who is Romeo?".to_owned(),
        images: Vec::new(),
    };
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(4_105),
        model: SYNTHETIC_MODEL_ID.to_owned(),
        messages: vec![user_turn.clone()],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 16,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: Some(ROMEO_AND_JULIET_THINKING_CHANNEL_SEED.to_owned()),
        structured_generation: None,
    };
    let rendered_prompt = Qwen3_5PromptRenderer::render(
        &[user_turn],
        &[],
        true,
        &[],
        Some(ROMEO_AND_JULIET_THINKING_CHANNEL_SEED),
    )
    .expect("the seeded Romeo and Juliet prompt should render");

    let inference_request = tokenizer
        .prepare_chat(&chat_generation_command, true)
        .expect("a seeded Romeo and Juliet request should prepare");
    let independently_encoded_prompt = tokenizer
        .encode_prompt(&rendered_prompt)
        .expect("the seeded prompt should tokenize independently");

    assert!(
        rendered_prompt
            .contains("<think>\nTwo households, both alike in dignity, in Romeo and Juliet.\n")
    );
    assert_eq!(
        inference_request.input_token_ids(),
        independently_encoded_prompt.as_slice()
    );
}

pub(super) fn ornith_tokenizer_json_bytes(image_pad_token_id: u32) -> Vec<u8> {
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
