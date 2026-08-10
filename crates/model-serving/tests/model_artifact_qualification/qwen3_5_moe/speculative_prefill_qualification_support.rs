use std::path::Path;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::speculative_prefill::{RepresentativeGenerationMeasurement, RepresentativePrompt};

const REPRESENTATIVE_INPUT_TOKEN_COUNT: usize = 8_192;
const REPRESENTATIVE_OUTPUT_TOKEN_COUNT: u16 = 1_024;
const REPRESENTATIVE_SOURCE_TEXT: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

pub(super) fn prepare_romeo_and_juliet_three_paragraph_summary_prompt(
    model_directory: &Path,
    target_model_id: &str,
    request_id: RequestId,
    required_prompt_token_count: usize,
    maximum_output_token_count: u16,
) -> RepresentativePrompt {
    prepare_romeo_and_juliet_summary_prompt(
        model_directory,
        target_model_id,
        request_id,
        required_prompt_token_count,
        maximum_output_token_count,
        "Summarize the supplied Romeo and Juliet source in exactly three concise prose paragraphs. Do not use a heading, bullets, or a numbered list. Preserve the central conflict, the major decisions, and the tragic outcome.",
        Some(256),
        true,
    )
}

fn prepare_romeo_and_juliet_summary_prompt(
    model_directory: &Path,
    target_model_id: &str,
    request_id: RequestId,
    required_prompt_token_count: usize,
    maximum_output_token_count: u16,
    summary_instruction: &str,
    thinking_budget: Option<u16>,
    enable_thinking: bool,
) -> RepresentativePrompt {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, u32::from(maximum_output_token_count))
        .expect("the configured summary target artifact should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the configured summary tokenizer should load");
    let model_sampler_configuration = tokenizer.model_sampler_config();
    assert!(
        required_prompt_token_count < validated_artifact.config().maximum_position_count() as usize,
        "the summary prompt must remain below the validated model context limit"
    );
    let mut repeated_source_material = String::new();
    let prepared_chat_request = loop {
        if !repeated_source_material.is_empty() {
            repeated_source_material.push_str("\n\n");
        }
        repeated_source_material.push_str(REPRESENTATIVE_SOURCE_TEXT);
        let prepared_chat_request = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id,
                    model: target_model_id.to_owned(),
                    messages: vec![ChatMessage::User {
                        content: format!(
                            "{summary_instruction}\n\nSource material:\n{repeated_source_material}"
                        ),
                        images: Vec::new(),
                    }],
                    tools: Vec::new(),
                    tool_choice: ChatToolChoice::None,
                    settings: ChatGenerationSettings {
                        max_output_tokens: maximum_output_token_count,
                        temperature_thousandths: Some(
                            model_sampler_configuration.temperature_thousandths,
                        ),
                        top_p_thousandths: Some(model_sampler_configuration.top_p_thousandths),
                        seed: None,
                        thinking_budget,
                    },
                },
                enable_thinking,
            )
            .expect("the configured summary prompt should prepare");
        if prepared_chat_request.input_token_ids().len() >= required_prompt_token_count {
            break prepared_chat_request;
        }
    };
    let complete_prompt_token_ids = prepared_chat_request.input_token_ids();
    let assistant_suffix_start_token_index = complete_prompt_token_ids
        .iter()
        .rposition(|token_id| *token_id == tokenizer.im_end_token_id())
        .expect("the summary prompt should contain the assistant suffix marker");
    let assistant_suffix_token_ids =
        &complete_prompt_token_ids[assistant_suffix_start_token_index..];
    let retained_source_prefix_token_count = required_prompt_token_count
        .checked_sub(assistant_suffix_token_ids.len())
        .expect("the assistant suffix must fit the requested summary prompt length");
    assert!(retained_source_prefix_token_count < assistant_suffix_start_token_index);
    let mut prompt_token_ids =
        complete_prompt_token_ids[..retained_source_prefix_token_count].to_vec();
    prompt_token_ids.extend_from_slice(assistant_suffix_token_ids);
    assert_eq!(prompt_token_ids.len(), required_prompt_token_count);
    RepresentativePrompt {
        prompt_token_ids,
        image_pad_token_id: tokenizer.image_pad_token_id(),
        processed_visual_images: Vec::new(),
        ordinary_target_prefill_control_span_token_count: prepared_chat_request
            .ordinary_target_prefill_control_span_token_count(),
        sampling_temperature_thousandths: model_sampler_configuration.temperature_thousandths,
        sampling_top_p_thousandths: model_sampler_configuration.top_p_thousandths,
        sampling_seed: None,
    }
}

pub(super) fn decode_generated_output_text(
    tokenizer: &Qwen3_5Tokenizer,
    generated_token_ids: &[u32],
) -> String {
    let mut token_decoder = tokenizer.incremental_decoder();
    let mut decoded_output_text = String::new();
    for generated_token_id in generated_token_ids {
        if let Some(decoded_fragment) = token_decoder
            .push_token(*generated_token_id)
            .expect("the generated Shakespeare output should decode")
        {
            decoded_output_text.push_str(&decoded_fragment);
        }
    }
    if let Some(decoded_fragment) = token_decoder
        .finish()
        .expect("the generated Shakespeare output should flush")
    {
        decoded_output_text.push_str(&decoded_fragment);
    }
    decoded_output_text
}

pub(super) fn prepare_representative_prompt(model_directory: &Path) -> RepresentativePrompt {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the representative SpecPrefill target artifact should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the representative SpecPrefill tokenizer should load");
    let model_sampler_configuration = tokenizer.model_sampler_config();
    let mut repeated_source_material = String::new();
    let prepared_chat_request = loop {
        if !repeated_source_material.is_empty() {
            repeated_source_material.push_str("\n\n");
        }
        repeated_source_material.push_str(REPRESENTATIVE_SOURCE_TEXT);
        let prepared_chat_request = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id: RequestId::new(95_000),
                    model: validated_artifact.model_id().to_owned(),
                    messages: vec![ChatMessage::User {
                        content: format!(
                            "Write a coherent literary synopsis of Romeo and Juliet from the source material. Preserve the characters, relationships, major events, and tragic ending. Do not discuss these instructions or repeat the source text. Write at least seven hundred words.\n\nSource material:\n{repeated_source_material}"
                        ),
                        images: Vec::new(),
                    }],
                    tools: Vec::new(),
                    tool_choice: ChatToolChoice::None,
                    settings: ChatGenerationSettings {
                        max_output_tokens: REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
                        temperature_thousandths: Some(
                            model_sampler_configuration.temperature_thousandths,
                        ),
                        top_p_thousandths: Some(model_sampler_configuration.top_p_thousandths),
                        seed: None,
                        thinking_budget: Some(256),
                    },
                },
                false,
            )
            .expect("the representative SpecPrefill prompt should prepare");
        if prepared_chat_request.input_token_ids().len() >= REPRESENTATIVE_INPUT_TOKEN_COUNT {
            break prepared_chat_request;
        }
    };
    let complete_prompt_token_ids = prepared_chat_request.input_token_ids().to_vec();
    let assistant_suffix_start = complete_prompt_token_ids
        .iter()
        .rposition(|token_id| *token_id == tokenizer.im_end_token_id())
        .expect("the representative prompt should contain the assistant suffix marker");
    let assistant_suffix_token_ids = &complete_prompt_token_ids[assistant_suffix_start..];
    let retained_prompt_prefix_token_count = REPRESENTATIVE_INPUT_TOKEN_COUNT
        .checked_sub(assistant_suffix_token_ids.len())
        .expect("the assistant suffix should fit the representative input budget");
    assert!(retained_prompt_prefix_token_count < assistant_suffix_start);
    let mut prompt_token_ids =
        complete_prompt_token_ids[..retained_prompt_prefix_token_count].to_vec();
    prompt_token_ids.extend_from_slice(assistant_suffix_token_ids);
    assert_eq!(prompt_token_ids.len(), REPRESENTATIVE_INPUT_TOKEN_COUNT);
    RepresentativePrompt {
        prompt_token_ids,
        image_pad_token_id: tokenizer.image_pad_token_id(),
        processed_visual_images: Vec::new(),
        ordinary_target_prefill_control_span_token_count: prepared_chat_request
            .ordinary_target_prefill_control_span_token_count(),
        sampling_temperature_thousandths: 0,
        sampling_top_p_thousandths: 1_000,
        sampling_seed: None,
    }
}

pub(super) fn clear_reclaimable_mlx_memory(mlx_memory_limits: MlxMemoryLimits) {
    MlxRuntime::initialize(mlx_memory_limits)
        .expect("the representative benchmark should re-enter the configured MLX runtime")
        .clear_allocator_cache()
        .expect("the representative benchmark should clear reclaimable MLX memory");
}

pub(super) fn assert_within_memory_limits(
    measurement_name: &str,
    measurement: &RepresentativeGenerationMeasurement,
    mlx_memory_limits: MlxMemoryLimits,
) {
    let configured_memory_limit_bytes = mlx_memory_limits.active_memory_limit_bytes() as u64;
    assert!(
        measurement.maximum_active_memory_bytes <= configured_memory_limit_bytes,
        "{measurement_name} active MLX memory must remain within the configured limit",
    );
    assert!(
        measurement.maximum_peak_memory_bytes
            <= configured_memory_limit_bytes + configured_memory_limit_bytes / 100,
        "{measurement_name} peak MLX memory must remain within the one-percent allowance",
    );
}
