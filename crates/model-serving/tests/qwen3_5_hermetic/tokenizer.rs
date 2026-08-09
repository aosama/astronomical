#[allow(dead_code)]
#[path = "../../src/qwen3_5/inference_execution/speculative_prefill_selection.rs"]
mod speculative_prefill_selection;

use astronomical_ipc_protocol::{ChatMessage, ChatToolDefinition};
use astronomical_model_serving::{
    Qwen3_5PromptRenderer, Qwen3_5Tokenizer, Qwen3_5TokenizerError, validate_context_token_count,
};
use speculative_prefill_selection::qwen3_5_select_speculative_prefill_token_positions;

use crate::common::qwen3_5_moe::certified_ornith_image_processor;

const ORNITH_VOCABULARY_SIZE: u32 = 248_320;
const ORNITH_MAXIMUM_POSITION_COUNT: u32 = 262_144;
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[test]
fn should_discover_special_token_ids_from_tokenizer_json() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes(248_056),
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        certified_ornith_image_processor(),
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
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        certified_ornith_image_processor(),
    )
    .expect_err("the tokenizer should reject a missing special token");

    assert!(matches!(
        tokenizer_error,
        Qwen3_5TokenizerError::DiscoverTokenIds { .. }
    ));
}

#[test]
fn should_accept_an_opencode_context_above_the_retired_32k_input_limit() {
    assert!(validate_context_token_count(40_665, 4_096, 262_144).is_ok());
}

#[test]
fn should_reject_a_context_above_the_certified_ornith_position_limit() {
    assert!(validate_context_token_count(258_049, 4_096, 262_144).is_err());
}

#[test]
fn should_convert_the_rendered_system_and_tool_boundary_to_an_exact_token_count() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes(248_056),
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        certified_ornith_image_processor(),
    )
    .expect("the synthetic tokenizer should load");
    let rendered_prompt = Qwen3_5PromptRenderer::render_with_control_span(
        &[
            ChatMessage::System {
                content: "Use the declared tool.".to_owned(),
            },
            ChatMessage::User {
                content: "Inspect Romeo and Juliet.".to_owned(),
                images: Vec::new(),
            },
        ],
        &[ChatToolDefinition {
            name: "inspect_play".to_owned(),
            description: None,
            parameters_json: r#"{"type":"object"}"#.to_owned(),
        }],
        false,
        &[],
    )
    .expect("the tool-bearing prompt should render");

    let (prompt_token_ids, ordinary_target_prefill_control_span_token_count) = tokenizer
        .encode_rendered_prompt_with_control_span(&rendered_prompt)
        .expect("the control-span byte boundary should remain a token boundary");

    assert!(ordinary_target_prefill_control_span_token_count > 0);
    assert!(ordinary_target_prefill_control_span_token_count < prompt_token_ids.len());
}

#[test]
fn should_keep_the_complete_tool_control_span_outside_romeo_and_juliet_sparse_selection() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes(248_056),
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        certified_ornith_image_processor(),
    )
    .expect("the synthetic tokenizer should load");
    let rendered_prompt = Qwen3_5PromptRenderer::render_with_control_span(
        &[
            ChatMessage::System {
                content: "Use only the declared literary-analysis tool.".to_owned(),
            },
            ChatMessage::User {
                content: format!(
                    "Identify the central conflict from this source.\n\n{ROMEO_AND_JULIET_SOURCE}"
                ),
                images: Vec::new(),
            },
        ],
        &[ChatToolDefinition {
            name: "record_literary_analysis".to_owned(),
            description: Some("Record a structured literary analysis.".to_owned()),
            parameters_json: r#"{"type":"object","properties":{"central_conflict":{"type":"string"},"outcome":{"type":"string","enum":["tragic","comic"]}},"required":["central_conflict","outcome"]}"#.to_owned(),
        }],
        false,
        &[],
    )
    .expect("the Romeo and Juliet tool prompt should render");
    let independently_encoded_control_span_token_count = tokenizer
        .encode_prompt(rendered_prompt.ordinary_target_prefill_control_span())
        .expect("the protected control span should tokenize independently")
        .len();
    let (complete_prompt_token_ids, ordinary_target_prefill_control_span_token_count) = tokenizer
        .encode_rendered_prompt_with_control_span(&rendered_prompt)
        .expect("the complete tool prompt should tokenize with an exact control boundary");
    let final_generation_kickoff_position = complete_prompt_token_ids
        .len()
        .checked_sub(1)
        .expect("the rendered prompt should contain a generation-kickoff token");
    let selectable_conversation_token_count = final_generation_kickoff_position
        .checked_sub(ordinary_target_prefill_control_span_token_count)
        .expect("conversation tokens should follow the protected control span");
    let selection_chunck_token_count = 64_usize;
    let keep_percentage = 20_u32;
    let mandatory_trailing_token_count = 128_usize;
    let selectable_conversation_chunck_count =
        selectable_conversation_token_count.div_ceil(selection_chunck_token_count);
    let percentage_derived_conversation_chunck_budget =
        (selectable_conversation_chunck_count * keep_percentage as usize).div_ceil(100);
    let importance_scores = (0..selectable_conversation_token_count)
        .map(|conversation_token_position| (conversation_token_position % 97) as f32)
        .collect::<Vec<_>>();
    let selected_relative_conversation_positions =
        qwen3_5_select_speculative_prefill_token_positions(
            &importance_scores,
            keep_percentage,
            selection_chunck_token_count,
            mandatory_trailing_token_count,
        )
        .expect("the selectable conversation should produce a sparse target selection");
    let mut selected_conversation_chunck_indices = selected_relative_conversation_positions
        .iter()
        .map(|selected_relative_position| selected_relative_position / selection_chunck_token_count)
        .collect::<Vec<_>>();
    selected_conversation_chunck_indices.dedup();
    let selected_absolute_conversation_positions = selected_relative_conversation_positions
        .iter()
        .map(|selected_relative_position| {
            ordinary_target_prefill_control_span_token_count + selected_relative_position
        })
        .collect::<Vec<_>>();
    let complete_ordered_target_positions = (0..ordinary_target_prefill_control_span_token_count)
        .chain(selected_absolute_conversation_positions.iter().copied())
        .chain(std::iter::once(final_generation_kickoff_position))
        .collect::<Vec<_>>();

    assert_eq!(
        ordinary_target_prefill_control_span_token_count,
        independently_encoded_control_span_token_count
    );
    assert_eq!(
        ordinary_target_prefill_control_span_token_count..final_generation_kickoff_position,
        ordinary_target_prefill_control_span_token_count
            ..ordinary_target_prefill_control_span_token_count
                + selectable_conversation_token_count
    );
    assert_eq!(
        selected_conversation_chunck_indices.len(),
        percentage_derived_conversation_chunck_budget
    );
    assert!(
        (selectable_conversation_token_count - mandatory_trailing_token_count
            ..selectable_conversation_token_count)
            .all(
                |mandatory_relative_position| selected_relative_conversation_positions
                    .binary_search(&mandatory_relative_position)
                    .is_ok()
            )
    );
    assert_eq!(
        &complete_ordered_target_positions[..ordinary_target_prefill_control_span_token_count],
        (0..ordinary_target_prefill_control_span_token_count).collect::<Vec<_>>()
    );
    assert_eq!(
        complete_ordered_target_positions.last().copied(),
        Some(final_generation_kickoff_position)
    );
    assert!(
        complete_ordered_target_positions
            .windows(2)
            .all(|positions| positions[0] < positions[1])
    );
    assert!(
        selected_absolute_conversation_positions
            .iter()
            .all(|selected_position| *selected_position
                >= ordinary_target_prefill_control_span_token_count
                && *selected_position < final_generation_kickoff_position)
    );
}

fn ornith_tokenizer_json_bytes(image_pad_token_id: u32) -> Vec<u8> {
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

fn ornith_tokenizer_json_bytes_missing_image_pad() -> Vec<u8> {
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
