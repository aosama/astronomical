use std::{fs, net::SocketAddr, path::Path};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};
use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};

use super::model_artifact_rest_qualification::get_endpoint;

pub(super) const CACHE_PRESSURE_MODEL_ID: &str =
    crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID;
pub(super) const MAXIMUM_OUTPUT_TOKEN_COUNT: u16 = 2_048;
pub(super) const THINKING_BUDGET_TOKEN_COUNT: u16 = 256;
pub(super) const PROMPT_CACHE_MAXIMUM_SIZE_GB: u64 = 50;
const QUALIFICATION_SAMPLING_SEED: u64 = 42;

const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

pub(super) struct PreparedRomeoAndJulietPrompt {
    pub(super) user_message: String,
    pub(super) prompt_token_count: usize,
    pub(super) temperature: f32,
    pub(super) top_p: f32,
}

pub(super) struct StreamedAssistantResponse {
    pub(super) assistant_content: String,
    pub(super) prompt_token_count: u64,
    pub(super) streamed_output_character_count: usize,
}

pub(super) fn prepare_cacheable_romeo_and_juliet_prompt(
    model_directory: &Path,
    maximum_prompt_token_count: usize,
) -> PreparedRomeoAndJulietPrompt {
    prepare_romeo_and_juliet_prompt_with_instruction(
        model_directory,
        maximum_prompt_token_count,
        "Return only a factual Romeo and Juliet summary in no more than four short sentences and ninety words. Use one paragraph with no line breaks. Include the central conflict, major decisions, and tragic outcome.",
    )
}

fn prepare_romeo_and_juliet_prompt_with_instruction(
    model_directory: &Path,
    maximum_prompt_token_count: usize,
    summary_instruction: &str,
) -> PreparedRomeoAndJulietPrompt {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, u32::from(MAXIMUM_OUTPUT_TOKEN_COUNT))
        .expect("the cache-pressure artifact should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the cache-pressure tokenizer should load");
    let model_sampler_configuration = tokenizer.model_sampler_config();
    let source_repeat_count = maximum_prompt_token_count
        .saturating_add(2_999)
        .saturating_div(3_000)
        .max(3);
    let repeated_source_material = ROMEO_AND_JULIET_SOURCE.repeat(source_repeat_count);
    let source_character_boundaries = repeated_source_material
        .char_indices()
        .map(|(byte_position, _source_character)| byte_position)
        .chain(std::iter::once(repeated_source_material.len()))
        .collect::<Vec<_>>();
    let mut lower_character_position = 0_usize;
    let mut upper_character_position = source_character_boundaries.len() - 1;
    let mut selected_user_message = None;
    let mut selected_prompt_token_count = 0_usize;

    while lower_character_position <= upper_character_position {
        let candidate_character_position =
            lower_character_position + (upper_character_position - lower_character_position) / 2;
        let candidate_source_end_byte_position =
            source_character_boundaries[candidate_character_position];
        let candidate_user_message = format!(
            "{summary_instruction}\n\nSource material:\n{}",
            &repeated_source_material[..candidate_source_end_byte_position],
        );
        let candidate_prompt_token_count = prepared_chat_token_count(
            &tokenizer,
            validated_artifact.model_id(),
            &candidate_user_message,
            model_sampler_configuration.temperature_thousandths,
            model_sampler_configuration.top_p_thousandths,
        );
        if candidate_prompt_token_count <= maximum_prompt_token_count {
            selected_user_message = Some(candidate_user_message);
            selected_prompt_token_count = candidate_prompt_token_count;
            lower_character_position = candidate_character_position.saturating_add(1);
        } else if candidate_character_position == 0 {
            break;
        } else {
            upper_character_position = candidate_character_position - 1;
        }
    }

    assert!(
        selected_prompt_token_count >= maximum_prompt_token_count.saturating_sub(1),
        "the REST cache-pressure prompt should reach its requested token boundary; requested_token_count={maximum_prompt_token_count} actual_token_count={selected_prompt_token_count}"
    );
    PreparedRomeoAndJulietPrompt {
        user_message: selected_user_message
            .expect("the Romeo and Juliet source should contain a cacheable prompt prefix"),
        prompt_token_count: selected_prompt_token_count,
        temperature: f32::from(model_sampler_configuration.temperature_thousandths) / 1_000.0,
        top_p: f32::from(model_sampler_configuration.top_p_thousandths) / 1_000.0,
    }
}

pub(super) async fn send_streaming_chat_request(
    openai_client: &Client<OpenAIConfig>,
    messages: Vec<Value>,
    prepared_romeo_and_juliet_prompt: &PreparedRomeoAndJulietPrompt,
    qualification_log_prefix: &str,
    request_phase_name: &str,
    maximum_output_token_count: u16,
    thinking_budget_token_count: u16,
) -> StreamedAssistantResponse {
    let request_document = json!({
        "model": CACHE_PRESSURE_MODEL_ID,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
        "max_tokens": maximum_output_token_count,
        "temperature": prepared_romeo_and_juliet_prompt.temperature,
        "top_p": prepared_romeo_and_juliet_prompt.top_p,
        "seed": QUALIFICATION_SAMPLING_SEED,
        "thinking_budget": thinking_budget_token_count,
    });
    let mut streamed_chat_completion: StreamResponse<Value> = openai_client
        .chat()
        .create_stream_byot(request_document)
        .await
        .unwrap_or_else(|stream_start_error| {
            panic!("the {request_phase_name} REST request should start: {stream_start_error}")
        });
    let mut received_chunk_count = 0_usize;
    let mut streamed_output_character_count = 0_usize;
    let mut assistant_content = String::new();
    let mut prompt_token_count = None;
    while let Some(stream_item) = streamed_chat_completion.next().await {
        let stream_chunk = stream_item.unwrap_or_else(|stream_error| {
            panic!("the {request_phase_name} REST stream should remain healthy: {stream_error}")
        });
        received_chunk_count = received_chunk_count.saturating_add(1);
        if let Some(stream_prompt_token_count) = stream_chunk["usage"]["prompt_tokens"].as_u64() {
            prompt_token_count = Some(stream_prompt_token_count);
        }
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                assistant_content.push_str(content_fragment);
                streamed_output_character_count =
                    streamed_output_character_count.saturating_add(content_fragment.len());
            }
            if let Some(reasoning_fragment) = choice["delta"]["reasoning_content"].as_str() {
                streamed_output_character_count =
                    streamed_output_character_count.saturating_add(reasoning_fragment.len());
            }
        }
        if received_chunk_count.is_multiple_of(128) {
            eprintln!(
                "{qualification_log_prefix} status=progress phase={request_phase_name}_stream received_chunks={received_chunk_count} output_characters={streamed_output_character_count}"
            );
        }
    }
    assert!(
        received_chunk_count > 0,
        "the {request_phase_name} request must receive at least one public SSE chunk"
    );
    eprintln!(
        "{qualification_log_prefix} status=progress phase={request_phase_name}_complete received_chunks={received_chunk_count} output_characters={streamed_output_character_count}"
    );
    StreamedAssistantResponse {
        assistant_content,
        prompt_token_count: prompt_token_count
            .expect("the requested streaming usage must report prompt tokens"),
        streamed_output_character_count,
    }
}

pub(super) fn user_message(content: &str) -> Value {
    json!({ "role": "user", "content": content })
}

pub(super) fn assistant_message(content: &str) -> Value {
    json!({ "role": "assistant", "content": content })
}

pub(super) async fn get_json_endpoint(server_address: SocketAddr, endpoint_path: &str) -> Value {
    let http_response = get_endpoint(server_address, endpoint_path).await;
    assert!(
        http_response.starts_with("HTTP/1.1 200 OK"),
        "the {endpoint_path} endpoint should return success: {http_response}"
    );
    let (_, response_body) = http_response
        .split_once("\r\n\r\n")
        .expect("the endpoint response should contain a header/body boundary");
    serde_json::from_str(response_body).unwrap_or_else(|json_error| {
        panic!("the {endpoint_path} response should contain JSON: {json_error}")
    })
}

pub(super) fn required_u64(json_document: &Value, field_name: &str) -> u64 {
    json_document[field_name].as_u64().unwrap_or_else(|| {
        panic!("the cache stats response must contain {field_name}: {json_document}")
    })
}

pub(super) fn read_performance_records(performance_log_directory: &Path) -> Vec<Value> {
    let performance_log_document =
        fs::read_to_string(performance_log_directory.join("performance.jsonl"))
            .expect("completed REST requests should produce a performance log");
    performance_log_document
        .lines()
        .map(|performance_record| {
            serde_json::from_str(performance_record)
                .expect("each performance log row should contain valid JSON")
        })
        .collect()
}

pub(super) fn write_cache_pressure_worker_config(
    isolated_worker_home: &Path,
    model_directory: &Path,
    maximum_mlx_memory_bytes: u64,
) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the cache-pressure REST configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": maximum_mlx_memory_bytes / 1_000_000_000,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": true,
        "prompt_cache_max_size_gb": PROMPT_CACHE_MAXIMUM_SIZE_GB,
        "performance_attribution_enabled": true,
        "mtp_enabled": false,
        "chunking": {

            "fixed_prompt_processing_chunk_size_tokens": 2_048,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the cache-pressure REST configuration should serialize"),
    )
    .expect("the cache-pressure REST configuration should write");
}

fn prepared_chat_token_count(
    tokenizer: &Qwen3_5Tokenizer,
    target_model_id: &str,
    user_message: &str,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
) -> usize {
    tokenizer
        .prepare_chat(
            &ChatGenerationCommand {
                request_id: RequestId::new(95_282),
                model: target_model_id.to_owned(),
                messages: vec![ChatMessage::User {
                    content: user_message.to_owned(),
                    images: Vec::new(),
                }],
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                settings: ChatGenerationSettings {
                    max_output_tokens: MAXIMUM_OUTPUT_TOKEN_COUNT,
                    temperature_thousandths: Some(temperature_thousandths),
                    top_p_thousandths: Some(top_p_thousandths),
                    seed: None,
                    thinking_budget: Some(THINKING_BUDGET_TOKEN_COUNT),
                },
            },
            true,
        )
        .expect("the Romeo and Juliet REST prompt should prepare")
        .input_token_ids()
        .len()
}
