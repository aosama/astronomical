//! Real user journey proving automatic Resident -> Paged -> Resident recovery.
//!
//! # What the user does
//!
//! 1. Start a worker at 25 GB, just enough for complete Ornith 4-bit residency
//!    when idle, but not enough for a long multimodal prompt beside it.
//! 2. Send a 32,000-token Romeo and Juliet request plus a processed image.
//! 3. After that request finishes, raise the public ceiling to 32 GB.
//! 4. Send a shorter follow-up on the same model.
//!
//! # What must happen
//!
//! - Request one forces paging and must still complete. Decode must start
//!   without killing the worker. Expert RAM during generation may stay below
//!   complete residency at 25 GB; the sister journey at 26 GB proves reclaim.
//! - Raising the ceiling must restore `resident` through replacement-aware
//!   promotion.
//! - Request two must then read zero expert bytes from the solid-state drive
//!   because the complete owner is already in RAM.
//!
//! This is a real REST/supervisor/worker/MLX journey, not a mock.

use std::{fs, net::SocketAddr, path::Path};

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    time::{Duration, Instant, sleep, timeout},
};

use crate::common::real_model_rest_server::{
    JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};

const MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
// Qualification values describe this measured journey, not production policy.
const INITIAL_MAXIMUM_MLX_MEMORY_GB: u64 = 25;
const RAISED_MAXIMUM_MLX_MEMORY_GB: u64 = 32;
const PRESSURE_PROMPT_TOKEN_COUNT: usize = 32_000;
const RESIDENT_PROMPT_TOKEN_COUNT: usize = 7_000;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 16;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 8;
// Matches the failed 6-bit chat's visual-embedding geometry (~1,225 merged rows).
const PRESSURE_IMAGE_WIDTH: u32 = 980;
const PRESSURE_IMAGE_HEIGHT: u32 = 980;
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the real REST server, worker, Ornith model, and MLX runtime for a live residency round trip"]
async fn should_page_a_tight_resident_model_to_finish_a_request_then_restore_full_residency_after_the_user_raises_the_ceiling()
 {
    timeout(JOURNEY_TIMEOUT, run_automatic_residency_round_trip())
        .await
        .expect("the automatic expert-residency round trip must finish within 115 seconds");
}

async fn run_automatic_residency_round_trip() {
    let model_directory = crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
    let isolated_worker_home =
        tempfile::tempdir().expect("the residency round-trip worker home should be created");
    write_acceptance_config(isolated_worker_home.path(), &model_directory);
    let repeated_source = ROMEO_AND_JULIET_SOURCE.repeat(12);
    let pressure_prompt = crate::common::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        &repeated_source,
        "Summarize Romeo and Juliet, preserving the major decisions and their consequences.",
        PRESSURE_PROMPT_TOKEN_COUNT,
    );
    let resident_prompt = crate::common::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        &repeated_source,
        "Explain the central conflict in Romeo and Juliet in one concise paragraph.",
        RESIDENT_PROMPT_TOKEN_COUNT,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        MODEL_ID,
        model_directory,
        isolated_worker_home.path(),
        INITIAL_MAXIMUM_MLX_MEMORY_GB * 1_000_000_000,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-residency-round-trip-client"),
    );

    let pressure_image_data_url = crate::common::solid_png::solid_rgb_png_data_url(
        PRESSURE_IMAGE_WIDTH,
        PRESSURE_IMAGE_HEIGHT,
        180,
        32,
        42,
    );
    let first_stream = start_streaming_request(
        &openai_client,
        pressure_prompt,
        Some(pressure_image_data_url),
    )
    .await;
    let (first_completion, first_memory_evidence) = tokio::join!(
        consume_completed_stream(first_stream),
        observe_resident_to_paged_request(server_address),
    );
    let first_status_after_request = get_json_endpoint(server_address, "/v1/status").await;
    if first_completion.is_err()
        || first_status_after_request["status"] == "unavailable"
        || first_status_after_request["expert_memory_mode"].is_null()
    {
        dump_decode_handoff_logs(isolated_worker_home.path());
        panic!(
            "pressure request died after paging; status={first_status_after_request}; stream={first_completion:?}"
        );
    }
    let first_completion = first_completion.expect("pressure request stream should complete");
    assert_completed_stream(&first_completion, "pressure-inducing request");
    assert!(first_memory_evidence.observed_resident);
    assert!(first_memory_evidence.observed_paged);
    assert!(
        first_status_after_request["mlx_memory_snapshot"]["expert_payload_bytes"]
            .as_u64()
            .is_some_and(|expert_payload_bytes| expert_payload_bytes > 4_000_000_000),
        "the pressure request must retain substantially more than the routed working set after decode: {first_status_after_request}"
    );

    let ceiling_update = put_maximum_mlx_memory(server_address, RAISED_MAXIMUM_MLX_MEMORY_GB).await;
    assert_eq!(
        ceiling_update["effective_mlx_memory_ceiling_bytes"].as_u64(),
        Some(RAISED_MAXIMUM_MLX_MEMORY_GB * 1_000_000_000),
        "the public memory update should apply the requested decimal-SI ceiling"
    );
    let promoted_status = wait_for_resident_ceiling(server_address).await;
    assert_eq!(
        promoted_status["expert_memory_mode"].as_str(),
        Some("resident")
    );

    let second_stream = start_streaming_request(&openai_client, resident_prompt, None).await;
    let (second_completion, second_final_status) = tokio::join!(
        consume_completed_stream(second_stream),
        observe_resident_request_until_idle(server_address),
    );
    let second_completion =
        second_completion.expect("the post-raise resident request stream should complete");
    assert_completed_stream(&second_completion, "post-raise resident request");
    assert_eq!(
        second_final_status["expert_memory_mode"].as_str(),
        Some("resident"),
        "the next request must finish with complete expert residency"
    );

    stop_real_model_rest_server(real_model_rest_server).await;
    let generation_source_read_bytes =
        generation_source_read_bytes_by_request(isolated_worker_home.path());
    assert_eq!(generation_source_read_bytes.len(), 2);
    assert!(
        generation_source_read_bytes[0] > 0,
        "the pressure request must prove that it actually entered SSD paging"
    );
    assert_eq!(
        generation_source_read_bytes[1], 0,
        "the request after the ceiling raise must execute with resident experts"
    );
    let worker_logs = isolated_worker_log_lines(isolated_worker_home.path());
    assert!(worker_logs.iter().any(|line| {
        line.contains("demoted complete resident experts to Rust streaming")
            && line.contains("transition_reason=RequestPressure")
    }));
    assert!(worker_logs.iter().any(|line| {
        line.contains("completed complete-model expert residency admission")
            && line.contains("transition_reason=CeilingRaise")
            && line.contains("outcome=\"promoted\"")
    }));
    eprintln!(
        "[expert-residency-round-trip] status=success initial_ceiling_gb={INITIAL_MAXIMUM_MLX_MEMORY_GB} raised_ceiling_gb={RAISED_MAXIMUM_MLX_MEMORY_GB} pressure_prompt_tokens={PRESSURE_PROMPT_TOKEN_COUNT} resident_prompt_tokens={RESIDENT_PROMPT_TOKEN_COUNT} first_request_source_read_bytes={} second_request_source_read_bytes={} first_output_characters={} second_output_characters={}",
        generation_source_read_bytes[0],
        generation_source_read_bytes[1],
        first_completion.model_text.len(),
        second_completion.model_text.len(),
    );
}

async fn start_streaming_request(
    openai_client: &Client<OpenAIConfig>,
    user_message: String,
    image_data_url: Option<String>,
) -> StreamResponse<Value> {
    let user_content = match image_data_url {
        Some(image_data_url) => json!([
            {"type": "text", "text": user_message},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]),
        None => json!(user_message),
    };
    let completion_request = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": user_content}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    });
    openai_client
        .chat()
        .create_stream_byot(completion_request)
        .await
        .expect("the real residency round-trip request should start")
}

struct ResidencyTransitionEvidence {
    observed_resident: bool,
    observed_paged: bool,
}

async fn observe_resident_to_paged_request(
    server_address: SocketAddr,
) -> ResidencyTransitionEvidence {
    let deadline = Instant::now() + JOURNEY_TIMEOUT;
    let mut observed_active_request = false;
    let mut observed_resident = false;
    let mut observed_paged = false;
    let mut last_status_log_at = Instant::now() - STATUS_LOG_INTERVAL;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let activity = status_document["activity"].as_str().unwrap_or("unknown");
        let expert_memory_mode = status_document["expert_memory_mode"].as_str();
        if activity != "idle" {
            observed_active_request = true;
            observed_resident |= expert_memory_mode == Some("resident");
            observed_paged |= expert_memory_mode == Some("paged");
        }
        if last_status_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            eprintln!(
                "[expert-residency-round-trip] status=pressure_progress activity={activity} expert_memory_mode={} processed_tokens={} total_tokens={}",
                expert_memory_mode.unwrap_or("unavailable"),
                status_document["progress"]["processed_tokens"],
                status_document["progress"]["total_tokens"],
            );
            last_status_log_at = Instant::now();
        }
        if status_document["status"] == "unavailable" {
            return ResidencyTransitionEvidence {
                observed_resident,
                observed_paged,
            };
        }
        if observed_active_request && activity == "idle" {
            return ResidencyTransitionEvidence {
                observed_resident,
                observed_paged,
            };
        }
        assert!(
            Instant::now() < deadline,
            "the pressure request did not return to idle"
        );
        sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_resident_ceiling(server_address: SocketAddr) -> Value {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let is_raised_resident = status_document["expert_memory_mode"] == "resident"
            && status_document["mlx_memory_ceiling_bytes"]
                == RAISED_MAXIMUM_MLX_MEMORY_GB * 1_000_000_000
            && status_document["pending_mlx_memory_ceiling_bytes"].is_null();
        if is_raised_resident {
            return status_document;
        }
        assert!(
            Instant::now() < deadline,
            "the raised ceiling did not restore residency: {status_document}"
        );
        sleep(Duration::from_millis(100)).await;
    }
}

async fn observe_resident_request_until_idle(server_address: SocketAddr) -> Value {
    let deadline = Instant::now() + JOURNEY_TIMEOUT;
    let mut observed_active_request = false;
    let mut remained_resident = true;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let activity = status_document["activity"].as_str().unwrap_or("unknown");
        if activity != "idle" {
            observed_active_request = true;
            remained_resident &= status_document["expert_memory_mode"] == "resident";
        }
        if observed_active_request && activity == "idle" {
            assert!(
                remained_resident,
                "the post-raise request left resident mode"
            );
            return status_document;
        }
        assert!(
            Instant::now() < deadline,
            "the post-raise request did not return to idle"
        );
        sleep(Duration::from_millis(50)).await;
    }
}

async fn put_maximum_mlx_memory(server_address: SocketAddr, maximum_mlx_memory_gb: u64) -> Value {
    let request_body = json!({"maximum_mlx_memory_gb": maximum_mlx_memory_gb}).to_string();
    let request_text = format!(
        "PUT /v1/config/maximum-mlx-memory HTTP/1.1\r\nHost: {server_address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request_body}",
        request_body.len(),
    );
    let mut tcp_stream = TcpStream::connect(server_address)
        .await
        .expect("the memory-control client should connect");
    tcp_stream
        .write_all(request_text.as_bytes())
        .await
        .expect("the memory-control request should write");
    let mut response_bytes = Vec::new();
    tcp_stream
        .read_to_end(&mut response_bytes)
        .await
        .expect("the memory-control response should read");
    let response_text = String::from_utf8(response_bytes)
        .expect("the memory-control response should contain UTF-8");
    assert!(
        response_text.starts_with("HTTP/1.1 200 OK"),
        "the live ceiling raise should succeed: {response_text}"
    );
    let (_, response_body) = response_text
        .split_once("\r\n\r\n")
        .expect("the memory-control response should contain HTTP headers");
    serde_json::from_str(response_body).expect("the memory-control response should be JSON")
}

#[derive(Debug)]
struct CompletedStream {
    model_text: String,
    finish_reason: Option<String>,
}

async fn consume_completed_stream(
    mut streamed_completion: StreamResponse<Value>,
) -> Result<CompletedStream, String> {
    let mut model_text = String::new();
    let mut finish_reason = None;
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = stream_item.map_err(|stream_error| stream_error.to_string())?;
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                model_text.push_str(content_fragment);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
        }
    }
    Ok(CompletedStream {
        model_text: model_text.trim().to_owned(),
        finish_reason,
    })
}

fn dump_decode_handoff_logs(isolated_worker_home: &Path) {
    for log_line in isolated_worker_log_lines(isolated_worker_home) {
        if log_line.contains("demoted complete resident")
            || log_line.contains("capped decode-warm")
            || log_line.contains("decode-warm expert pages ready")
            || log_line.contains("starting first decode forward")
            || log_line.contains("MLX inference owner stopped")
            || log_line.contains("fatal model execution failed")
        {
            eprintln!("[expert-residency-round-trip] status=decode_handoff_log {log_line}");
        }
    }
}

fn assert_completed_stream(completed_stream: &CompletedStream, request_description: &str) {
    assert!(
        !completed_stream.model_text.is_empty(),
        "{request_description} should return text"
    );
    assert!(
        matches!(
            completed_stream.finish_reason.as_deref(),
            Some("stop" | "length")
        ),
        "{request_description} should finish cleanly: {:?}",
        completed_stream.finish_reason
    );
}

fn generation_source_read_bytes_by_request(isolated_worker_home: &Path) -> Vec<u64> {
    let attribution_log = fs::read_to_string(
        isolated_worker_home
            .join(".astronomical-dev")
            .join("logs")
            .join("performance-attribution.jsonl"),
    )
    .expect("the round-trip journey should write performance attribution");
    attribution_log
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("attribution rows should be JSON"))
        .filter(|report| report["report_kind"] == "generation")
        .map(|report| {
            report["counters"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|counter| counter["counter"] == "positional_file_read_byte_count")
                .filter_map(|counter| counter["amount"].as_u64())
                .sum()
        })
        .collect()
}

fn isolated_worker_log_lines(isolated_worker_home: &Path) -> Vec<String> {
    let logging_directory = isolated_worker_home.join(".astronomical-dev").join("logs");
    fs::read_dir(logging_directory)
        .expect("the round-trip log directory should exist")
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .flat_map(|entry| {
            fs::read_to_string(entry.path())
                .unwrap_or_default()
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn write_acceptance_config(isolated_worker_home: &Path, model_directory: &Path) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the residency round-trip configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": INITIAL_MAXIMUM_MLX_MEMORY_GB,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": false,
        "performance_attribution_enabled": true,
        "mtp_enabled": false,
        "logging": {"level": "debug", "retained_files": 2},
        "chunking": {
            "prefill_size_optimizer_enabled": false,
            "fixed_prefill_tokens": 2_048,
            "experimental_ssd_paging_prefill_graph_submission_layer_interval": 1,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": 1,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the residency round-trip configuration should serialize"),
    )
    .expect("the residency round-trip configuration should be written");
}
