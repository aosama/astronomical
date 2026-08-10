use std::{fs, path::Path, time::Duration};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};
use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Instant, sleep, timeout};

use super::model_artifact_rest_qualification::{
    E2E_TIMEOUT, get_endpoint, launch_model_artifact_rest_server_for_model_with_memory_limit,
    stop_model_artifact_rest_server,
};

// This suite is the user-facing acceptance boundary for cache-pressure behavior: a client sends
// a long OpenAI Chat request, the real worker publishes reusable state, and the next identical
// request restores it. Direct-MLX coverage below the worker is retained separately for allocator
// arithmetic; this test protects the complete server-to-client journey.
const PINNED_MODEL_ID: &str = "Ornith-1.0-35B-MLX-2bit";
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");
const CACHEABLE_PROMPT_TOKEN_COUNT: usize = 10_001;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u16 = 2_048;
const THINKING_BUDGET_TOKEN_COUNT: u16 = 256;
const PROMPT_CACHE_MAXIMUM_SIZE_GB: u64 = 50;
const CACHE_PUBLICATION_TIMEOUT: Duration = Duration::from_secs(15);
const CACHE_PUBLICATION_POLL_INTERVAL: Duration = Duration::from_millis(250);

macro_rules! persistent_prompt_cache_memory_rest_qualification {
    ($test_name:ident, $maximum_mlx_memory_bytes:expr, $expected_expert_memory_mode:expr) => {
        #[tokio::test(flavor = "multi_thread")]
        #[ignore = "launches the production REST server and pinned model for a cold-and-warm 10K Romeo and Juliet cache journey"]
        async fn $test_name() {
            // Every memory cell owns a real model process and GPU. Keep its deadline inside the
            // test so an interrupted external runner cannot leave unbounded qualification work.
            timeout(
                E2E_TIMEOUT,
                run_persistent_prompt_cache_memory_rest_journey(
                    $maximum_mlx_memory_bytes,
                    $expected_expert_memory_mode,
                ),
            )
            .await
            .expect("the persistent prompt-cache REST journey must finish within 115 seconds");
        }
    };
}

persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_eleven_gb_through_rest,
    11_000_000_000,
    Some("paged")
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_eleven_point_five_gb_through_rest,
    11_500_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_twelve_gb_through_rest,
    12_000_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_twelve_point_five_gb_through_rest,
    12_500_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_thirteen_gb_through_rest,
    13_000_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_thirteen_point_five_gb_through_rest,
    13_500_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_fourteen_gb_through_rest,
    14_000_000_000,
    Some("resident")
);

async fn run_persistent_prompt_cache_memory_rest_journey(
    maximum_mlx_memory_bytes: u64,
    expected_expert_memory_mode: Option<&str>,
) {
    let qualification_log_prefix = format!(
        "[persistent-prompt-cache-rest:{}gb]",
        maximum_mlx_memory_bytes as f64 / 1_000_000_000.0
    );
    let model_directory = crate::common::configured_model_artifact_directory_by_id(PINNED_MODEL_ID);
    let configured_worker_home = tempfile::tempdir()
        .expect("the cache-pressure REST journey should create an isolated home");
    let performance_log_directory = tempfile::tempdir()
        .expect("the cache-pressure REST journey should create a performance log directory");
    write_cache_pressure_worker_config(
        configured_worker_home.path(),
        &model_directory,
        maximum_mlx_memory_bytes,
    );
    // Build a textual request instead of injecting token IDs: only the public REST tokenizer
    // may define the actual user journey. The local tokenizer is used solely to size a stable
    // Romeo-and-Juliet fixture near the cache boundary before it is sent as ordinary text.
    let prepared_romeo_and_juliet_prompt =
        prepare_cacheable_romeo_and_juliet_prompt(&model_directory);
    let model_artifact_rest_server = launch_model_artifact_rest_server_for_model_with_memory_limit(
        PINNED_MODEL_ID,
        model_directory,
        Some(configured_worker_home.path()),
        Some(performance_log_directory.path()),
        Some(maximum_mlx_memory_bytes),
    )
    .await;
    let server_address = model_artifact_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-qualification-client"),
    );

    // Cold request: verify an API client receives streamed model output, then wait until the
    // asynchronous writer has atomically published the cache block before issuing the warm call.
    eprintln!(
        "{qualification_log_prefix} status=progress phase=cold_request prompt_tokens={} maximum_output_tokens={MAXIMUM_OUTPUT_TOKEN_COUNT}",
        prepared_romeo_and_juliet_prompt.prompt_token_count,
    );
    let cold_streamed_output_character_count = send_streaming_summary_request(
        &openai_client,
        &prepared_romeo_and_juliet_prompt,
        &qualification_log_prefix,
        "cold",
    )
    .await;
    assert!(
        cold_streamed_output_character_count > 0,
        "the cold request must stream model-generated text through the public OpenAI-compatible client"
    );
    let cold_status_document = get_json_endpoint(server_address, "/v1/status").await;
    assert_expert_memory_mode(
        &cold_status_document,
        expected_expert_memory_mode,
        &qualification_log_prefix,
    );
    wait_for_persistent_prompt_cache_publication(server_address, &qualification_log_prefix).await;

    // Warm request: the same messages and model identity must recover the cold request prefix.
    // Cache counters, rather than timing, are the durable proof because machines differ widely.
    eprintln!("{qualification_log_prefix} status=progress phase=warm_request");
    let warm_streamed_output_character_count = send_streaming_summary_request(
        &openai_client,
        &prepared_romeo_and_juliet_prompt,
        &qualification_log_prefix,
        "warm",
    )
    .await;
    assert!(
        warm_streamed_output_character_count > 0,
        "the warm request must stream model-generated text through the public OpenAI-compatible client"
    );
    let warm_cache_stats_document = get_json_endpoint(server_address, "/v1/cache/stats").await;
    let restored_prompt_token_count = required_u64(
        &warm_cache_stats_document,
        "persistent_prompt_cache_tokens_saved",
    );
    assert!(
        required_u64(&warm_cache_stats_document, "persistent_prompt_cache_hits") > 0,
        "the warm request must restore the cold request's persistent prompt-cache state: {warm_cache_stats_document}"
    );
    assert!(
        restored_prompt_token_count > 0,
        "the warm request must report restored prompt tokens: {warm_cache_stats_document}"
    );
    stop_model_artifact_rest_server(model_artifact_rest_server).await;
    eprintln!(
        "{qualification_log_prefix} status=success cold_output_characters={cold_streamed_output_character_count} warm_output_characters={warm_streamed_output_character_count} restored_prompt_tokens={restored_prompt_token_count}"
    );
}

struct PreparedRomeoAndJulietPrompt {
    user_message: String,
    prompt_token_count: usize,
    temperature: f32,
    top_p: f32,
}

fn prepare_cacheable_romeo_and_juliet_prompt(
    model_directory: &Path,
) -> PreparedRomeoAndJulietPrompt {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, u32::from(MAXIMUM_OUTPUT_TOKEN_COUNT))
        .expect("the pinned cache-pressure artifact should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the pinned cache-pressure tokenizer should load");
    let model_sampler_configuration = tokenizer.model_sampler_config();
    let repeated_source_material = ROMEO_AND_JULIET_SOURCE.repeat(3);
    let source_character_boundaries = repeated_source_material
        .char_indices()
        .map(|(byte_position, _source_character)| byte_position)
        .chain(std::iter::once(repeated_source_material.len()))
        .collect::<Vec<_>>();
    let mut lower_character_position = 0_usize;
    let mut upper_character_position = source_character_boundaries.len() - 1;
    let mut selected_user_message = None;
    let mut selected_prompt_token_count = 0_usize;

    // Source characters are the safe slicing unit for UTF-8. Binary search finds the largest
    // textual prefix that stays within the 10,001-token target after the real chat template adds
    // its control tokens. This avoids fragile hardcoded byte offsets and keeps the test useful if
    // the fixture's Unicode content changes.
    while lower_character_position <= upper_character_position {
        let candidate_character_position =
            lower_character_position + (upper_character_position - lower_character_position) / 2;
        let candidate_source_end_byte_position =
            source_character_boundaries[candidate_character_position];
        let candidate_user_message = format!(
            "Return only a factual Romeo and Juliet summary in no more than four short sentences and ninety words. Use one paragraph with no line breaks. Include the central conflict, major decisions, and tragic outcome.\n\nSource material:\n{}",
            &repeated_source_material[..candidate_source_end_byte_position],
        );
        let candidate_prompt_token_count = prepared_chat_token_count(
            &tokenizer,
            validated_artifact.model_id(),
            &candidate_user_message,
            model_sampler_configuration.temperature_thousandths,
            model_sampler_configuration.top_p_thousandths,
        );
        if candidate_prompt_token_count <= CACHEABLE_PROMPT_TOKEN_COUNT {
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
        selected_prompt_token_count >= 10_000,
        "the REST cache-pressure prompt must contain at least 10K tokens; actual_token_count={selected_prompt_token_count}"
    );
    // Sampling is not invented by this test. Read the model's generation configuration and pass
    // its temperature and nucleus probability through the public request just as a real client
    // would, while retaining the required bounded thinking budget.
    PreparedRomeoAndJulietPrompt {
        user_message: selected_user_message
            .expect("the Romeo and Juliet source should contain a cacheable prompt prefix"),
        prompt_token_count: selected_prompt_token_count,
        temperature: f32::from(model_sampler_configuration.temperature_thousandths) / 1_000.0,
        top_p: f32::from(model_sampler_configuration.top_p_thousandths) / 1_000.0,
    }
}

fn prepared_chat_token_count(
    tokenizer: &Qwen3_5Tokenizer,
    target_model_id: &str,
    user_message: &str,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
) -> usize {
    // Mirror the public request shape only for prompt sizing. This count never bypasses the REST
    // boundary: the actual request below remains text and is tokenized again by the worker.
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

async fn send_streaming_summary_request(
    openai_client: &Client<OpenAIConfig>,
    prepared_romeo_and_juliet_prompt: &PreparedRomeoAndJulietPrompt,
    qualification_log_prefix: &str,
    request_phase_name: &str,
) -> usize {
    // async-openai's typed Chat request does not carry Astronomical's thinking_budget extension.
    // Its BYOT stream method still owns HTTP/SSE parsing while letting this compatibility test send
    // the complete public request document unchanged.
    let request_document = json!({
        "model": PINNED_MODEL_ID,
        "messages": [{
            "role": "user",
            "content": prepared_romeo_and_juliet_prompt.user_message,
        }],
        "stream": true,
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "temperature": prepared_romeo_and_juliet_prompt.temperature,
        "top_p": prepared_romeo_and_juliet_prompt.top_p,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
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
    while let Some(stream_item) = streamed_chat_completion.next().await {
        let stream_chunk = stream_item.unwrap_or_else(|stream_error| {
            panic!("the {request_phase_name} REST stream should remain healthy: {stream_error}")
        });
        received_chunk_count = received_chunk_count.saturating_add(1);
        // Qwen emits its bounded thought before ordinary content. Astronomical exposes both
        // delta fields on the Chat stream, so count both as client-visible progress; counting only
        // content would incorrectly report an empty successful reasoning response as a failure.
        streamed_output_character_count = streamed_output_character_count.saturating_add(
            stream_chunk["choices"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|choice| {
                    choice["delta"]["content"]
                        .as_str()
                        .into_iter()
                        .chain(choice["delta"]["reasoning_content"].as_str())
                        .map(str::len)
                        .sum::<usize>()
                })
                .sum::<usize>(),
        );
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
    streamed_output_character_count
}

async fn wait_for_persistent_prompt_cache_publication(
    server_address: std::net::SocketAddr,
    qualification_log_prefix: &str,
) {
    let publication_started_at = Instant::now();
    loop {
        let cache_stats_document = get_json_endpoint(server_address, "/v1/cache/stats").await;
        if required_u64(
            &cache_stats_document,
            "persistent_prompt_cache_sequence_state_block_count",
        ) > 0
        {
            eprintln!(
                "{qualification_log_prefix} status=progress phase=cache_published elapsed_seconds={:.1}",
                publication_started_at.elapsed().as_secs_f64()
            );
            return;
        }
        // The generation stream can finish before the writer publishes its atomic files. Poll the
        // public cache-stat endpoint instead of sleeping for a guessed storage latency.
        assert!(
            publication_started_at.elapsed() < CACHE_PUBLICATION_TIMEOUT,
            "the cold request did not publish persistent cache state within {} seconds: {cache_stats_document}",
            CACHE_PUBLICATION_TIMEOUT.as_secs(),
        );
        sleep(CACHE_PUBLICATION_POLL_INTERVAL).await;
    }
}

fn assert_expert_memory_mode(
    status_document: &Value,
    expected_expert_memory_mode: Option<&str>,
    qualification_log_prefix: &str,
) {
    let observed_expert_memory_mode = status_document["expert_memory_mode"].as_str();
    eprintln!(
        "{qualification_log_prefix} status=progress phase=cold_status expert_memory_mode={observed_expert_memory_mode:?}"
    );
    if let Some(expected_expert_memory_mode) = expected_expert_memory_mode {
        assert_eq!(
            observed_expert_memory_mode,
            Some(expected_expert_memory_mode),
            "the configured memory cell must report its expected expert-memory mode: {status_document}"
        );
    }
}

async fn get_json_endpoint(server_address: std::net::SocketAddr, endpoint_path: &str) -> Value {
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

fn required_u64(json_document: &Value, field_name: &str) -> u64 {
    json_document[field_name].as_u64().unwrap_or_else(|| {
        panic!("the cache stats response must contain {field_name}: {json_document}")
    })
}

fn write_cache_pressure_worker_config(
    isolated_worker_home: &Path,
    model_directory: &Path,
    maximum_mlx_memory_bytes: u64,
) {
    let configuration_directory = isolated_worker_home.join(".astronomical");
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
        "prefill_chunck_size_optimizer_enabled": false,
        "fixed_prefill_chunck_tokens": 2_048,
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the cache-pressure REST configuration should serialize"),
    )
    .expect("the cache-pressure REST configuration should write");
}
