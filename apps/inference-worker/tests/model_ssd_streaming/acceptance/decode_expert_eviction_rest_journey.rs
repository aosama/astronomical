//! User journeys proving decode eviction and retention correctness on the
//! resident sparse MoE e2e fixture under a memory squeeze.
//!
//! These journeys exist because expert retention used to collapse during live
//! decode: a model whose experts fit the ceiling kept generating with a small
//! expert payload while the graphics processor sat idle waiting on SSD reads.
//! Each test names one eviction contract and fails on the user-visible symptom.
//!
//! Acceptance ceilings are evidence cells, not production constants. Production
//! policy derives capacity from the configured ceiling and model geometry; the
//! byte floors here only have to sit far above the broken behavior they catch.

use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
};

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::support::serving_rest::{
    get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker to accept decode retention across a repeated cached request"]
async fn should_retain_seated_experts_during_a_repeated_cached_request_at_a_25gb_ceiling() {
    tokio::time::timeout(
        JOURNEY_DEADLINE,
        run_repeated_cached_decode_retention_journey(DECODE_RETENTION_MLX_MEMORY_BYTES),
    )
    .await
    .expect("the repeated cached decode-retention journey must finish within 115 seconds");
}

/// Runs the identical request twice. The second request restores the cached
/// prompt prefix, which is exactly the dev-app flow where retention collapsed:
/// restoration drops and re-admits expert owners, so eviction correctness must
/// survive the repeat, not only the cold request.
async fn run_repeated_cached_decode_retention_journey(maximum_mlx_memory_bytes: u64) {
    run_two_request_retention_journey(maximum_mlx_memory_bytes, PROMPT_TOKEN_COUNT).await;
}

/// Large-request repeat: closest faithful proxy for the dev-app session where
/// retention collapsed — big prompts (~16k tokens), heavy prompt-cache reuse,
/// and a ceiling that leaves little slack beyond model core plus context.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker to accept retention across large cached requests"]
async fn should_retain_seated_experts_across_large_cached_requests_at_a_25gb_ceiling() {
    tokio::time::timeout(
        JOURNEY_DEADLINE,
        run_large_cached_retention_journey(DECODE_RETENTION_MLX_MEMORY_BYTES),
    )
    .await
    .expect("the large cached retention journey must finish within 115 seconds");
}

async fn run_large_cached_retention_journey(maximum_mlx_memory_bytes: u64) {
    run_two_request_retention_journey(maximum_mlx_memory_bytes, LARGE_PROMPT_TOKEN_COUNT).await;
}

/// Shared two-request retention cell: the identical request runs twice on one
/// worker, and every generating sample must keep a strong expert floor.
async fn run_two_request_retention_journey(
    maximum_mlx_memory_bytes: u64,
    prompt_token_count: usize,
) {
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id());
    let isolated_worker_home = isolated_eviction_journey_worker_home();
    write_acceptance_config(
        &isolated_worker_home,
        &model_directory,
        maximum_mlx_memory_bytes,
    );
    let user_message = crate::support::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Summarize Romeo and Juliet in one concise paragraph. Include the central conflict, major decisions, and tragic outcome.",
        prompt_token_count,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        model_id(),
        model_directory,
        &isolated_worker_home,
        maximum_mlx_memory_bytes,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-acceptance-client"),
    );
    let completion_request = json!({
        "model": model_id(),
        "messages": [{"role": "user", "content": user_message}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "temperature": 1,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    });
    let mut smallest_generating_expert_payload_bytes = u64::MAX;
    let mut final_expert_memory_mode = String::new();
    let mut last_request_status = serde_json::json!({});
    for request_number in 1..=2 {
        eprintln!(
            "[decode-expert-eviction] status=progress phase=request_send request_number={request_number} model={} prompt_characters={} ceiling_bytes={maximum_mlx_memory_bytes}",
            model_id(),
            user_message.len()
        );
        let streamed_completion: StreamResponse<Value> = timeout(
            Duration::from_secs(60),
            openai_client
                .chat()
                .create_stream_byot(completion_request.clone()),
        )
        .await
        .expect("the repeated decode-retention REST request must be accepted within 60 seconds")
        .expect("the repeated decode-retention REST request should start");
        let (completed_stream, generation_memory) = tokio::join!(
            consume_completed_stream(streamed_completion),
            observe_generation_expert_payload(server_address),
        );
        assert!(!completed_stream.model_text.is_empty());
        assert!(matches!(
            completed_stream.finish_reason.as_deref(),
            Some("stop" | "length")
        ));
        assert!(
            generation_memory.saw_generating_activity,
            "request {request_number} must reach live decode: {generation_memory:?}"
        );
        smallest_generating_expert_payload_bytes = smallest_generating_expert_payload_bytes
            .min(generation_memory.smallest_generating_expert_payload_bytes);
        final_expert_memory_mode = generation_memory.final_expert_memory_mode.clone();
        last_request_status = generation_memory.final_status;
        eprintln!(
            "[decode-expert-eviction] status=progress phase=request_complete request_number={request_number} smallest_generating_expert_payload_gb={:.2} average_generation_tokens_per_second={:.2}",
            generation_memory.smallest_generating_expert_payload_bytes as f64 / 1e9,
            generation_memory.average_generation_tokens_per_second,
        );
    }
    crate::support::memory_utilization_parity::assert_and_preserve_status_memory_ceiling_utilization(
        "decode-expert-eviction",
        &isolated_worker_home,
        &last_request_status,
    );
    stop_real_model_rest_server(real_model_rest_server).await;
    preserve_journey_evidence(&isolated_worker_home, prompt_token_count);
    assert!(
        smallest_generating_expert_payload_bytes >= MINIMUM_GENERATION_EXPERT_PAYLOAD_BYTES,
        "live decode must retain seated experts across cold and cached-repeat requests; the expert payload collapsing between requests is the eviction defect this journey pins: smallest={smallest_generating_expert_payload_bytes}"
    );
    eprintln!(
        "[decode-expert-eviction] status=success phase=repeat_complete model={} ceiling_bytes={maximum_mlx_memory_bytes} smallest_generating_expert_payload_gb={:.2} final_expert_memory_mode={final_expert_memory_mode}",
        model_id(),
        smallest_generating_expert_payload_bytes as f64 / 1e9,
    );
}

fn model_id() -> &'static str {
    crate::support::resident_sparse_moe_model_id()
}

// Evidence cells, not production constants. The 25 GB cell leaves the
// complete Ornith 4-bit expert payload (about 20.5 GB) almost entirely
// seated; correct retention must keep most of it, not the roughly 2.7 GB
// collapse that live decode previously showed while streaming from SSD.
const DECODE_RETENTION_MLX_MEMORY_BYTES: u64 = 25_000_000_000;
// Mirrors the dev-app usage pattern: large requests (~16k tokens) that reuse a
// cached prefix across a conversation, at a ceiling that leaves little slack.
const LARGE_PROMPT_TOKEN_COUNT: usize = 16_000;
// Half of the roughly 20.5 GB Ornith 4-bit expert payload. Correct retention
// keeps far more; the collapse this suite exists to catch reports under 3 GB.
const MINIMUM_GENERATION_EXPERT_PAYLOAD_BYTES: u64 = 10_000_000_000;
const PROMPT_TOKEN_COUNT: usize = 4_096;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 128;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 64;
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const REQUEST_MUST_BECOME_ACTIVE_WITHIN: Duration = Duration::from_secs(20);
const JOURNEY_DEADLINE: Duration = Duration::from_secs(115);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches Ornith-1.5-35B-A3B-OptiQ-4bit under a 25 GB ceiling and proves live decode keeps its seated experts"]
async fn should_retain_seated_experts_during_live_decode_at_a_25gb_ceiling() {
    tokio::time::timeout(
        JOURNEY_DEADLINE,
        run_decode_retention_journey(DECODE_RETENTION_MLX_MEMORY_BYTES),
    )
    .await
    .expect("the 25 GB decode-retention journey must finish within 115 seconds");
}

async fn run_decode_retention_journey(maximum_mlx_memory_bytes: u64) {
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id());
    let isolated_worker_home = isolated_eviction_journey_worker_home();
    write_acceptance_config(
        &isolated_worker_home,
        &model_directory,
        maximum_mlx_memory_bytes,
    );
    let user_message = crate::support::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Summarize Romeo and Juliet in one concise paragraph. Include the central conflict, major decisions, and tragic outcome.",
        PROMPT_TOKEN_COUNT,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        model_id(),
        model_directory,
        &isolated_worker_home,
        maximum_mlx_memory_bytes,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-acceptance-client"),
    );
    let completion_request = json!({
        "model": model_id(),
        "messages": [{"role": "user", "content": user_message}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "temperature": 1,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    });
    eprintln!(
        "[decode-expert-eviction] status=progress phase=request_send model={} prompt_characters={} ceiling_bytes={maximum_mlx_memory_bytes}",
        model_id(),
        user_message.len()
    );
    let streamed_completion: StreamResponse<Value> = timeout(
        Duration::from_secs(60),
        openai_client.chat().create_stream_byot(completion_request),
    )
    .await
    .expect("the decode-retention REST request must be accepted within 60 seconds")
    .expect("the decode-retention REST request should start");
    let (completed_stream, generation_memory) = tokio::join!(
        consume_completed_stream(streamed_completion),
        observe_generation_expert_payload(server_address),
    );
    crate::support::memory_utilization_parity::assert_and_preserve_status_memory_ceiling_utilization(
        "decode-expert-eviction",
        &isolated_worker_home,
        &generation_memory.final_status,
    );
    stop_real_model_rest_server(real_model_rest_server).await;
    assert!(!completed_stream.model_text.is_empty());
    assert!(matches!(
        completed_stream.finish_reason.as_deref(),
        Some("stop" | "length")
    ));
    assert!(
        generation_memory.saw_generating_activity,
        "the request must reach live decode: {generation_memory:?}"
    );
    assert!(
        generation_memory.smallest_generating_expert_payload_bytes
            >= MINIMUM_GENERATION_EXPERT_PAYLOAD_BYTES,
        "live decode must retain seated experts under the configured ceiling; the expert payload collapsing to a few GB while decode streams from SSD is the eviction defect this journey pins: {generation_memory:?}"
    );
    assert!(
        generation_memory
            .average_generation_tokens_per_second
            .is_finite()
            && generation_memory.average_generation_tokens_per_second > 0.0,
        "decode throughput must be a positive finite measurement: {generation_memory:?}"
    );
    eprintln!(
        "[decode-expert-eviction] status=success model={} ceiling_bytes={maximum_mlx_memory_bytes} expert_memory_mode={} smallest_generating_expert_payload_gb={:.2} largest_generating_expert_payload_gb={:.2} average_prefill_tokens_per_second={:.2} average_generation_tokens_per_second={:.2} output_characters={}",
        model_id(),
        generation_memory.final_expert_memory_mode,
        generation_memory.smallest_generating_expert_payload_bytes as f64 / 1e9,
        generation_memory.largest_generating_expert_payload_bytes as f64 / 1e9,
        generation_memory.average_prefill_tokens_per_second,
        generation_memory.average_generation_tokens_per_second,
        completed_stream.model_text.len(),
    );
}

#[derive(Debug)]
struct GenerationMemoryEvidence {
    saw_generating_activity: bool,
    smallest_generating_expert_payload_bytes: u64,
    largest_generating_expert_payload_bytes: u64,
    final_expert_memory_mode: String,
    average_prefill_tokens_per_second: f64,
    average_generation_tokens_per_second: f64,
    final_status: serde_json::Value,
}

async fn observe_generation_expert_payload(server_address: SocketAddr) -> GenerationMemoryEvidence {
    let request_started_at = Instant::now();
    let deadline = request_started_at + JOURNEY_DEADLINE;
    let mut saw_active_request = false;
    let mut saw_generating_activity = false;
    let mut smallest_generating_expert_payload_bytes = u64::MAX;
    let mut largest_generating_expert_payload_bytes = 0_u64;
    let mut last_status_log_at = Instant::now() - STATUS_LOG_INTERVAL;
    let mut final_status = json!({});
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let activity = status_document["activity"].as_str().unwrap_or("unknown");
        let expert_memory_mode = status_document["expert_memory_mode"]
            .as_str()
            .unwrap_or("unavailable");
        let expert_payload_bytes = status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
            .as_u64()
            .unwrap_or(0);
        if last_status_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            eprintln!(
                "[decode-expert-eviction] status=progress activity={activity} expert_memory_mode={expert_memory_mode} expert_payload_bytes={expert_payload_bytes}"
            );
            last_status_log_at = Instant::now();
        }
        if activity != "idle" {
            saw_active_request = true;
        }
        if activity == "generating" {
            saw_generating_activity = true;
            smallest_generating_expert_payload_bytes =
                smallest_generating_expert_payload_bytes.min(expert_payload_bytes);
            largest_generating_expert_payload_bytes =
                largest_generating_expert_payload_bytes.max(expert_payload_bytes);
        }
        if !saw_active_request && request_started_at.elapsed() >= REQUEST_MUST_BECOME_ACTIVE_WITHIN
        {
            panic!(
                "the decode-retention request stayed idle for {} seconds: {status_document}",
                REQUEST_MUST_BECOME_ACTIVE_WITHIN.as_secs()
            );
        }
        let snapshot_source = status_document["mlx_memory_snapshot"]["source"].as_str();
        if saw_active_request
            && activity == "idle"
            && matches!(snapshot_source, Some("finalized" | "idle_poll"))
        {
            final_status = status_document;
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the decode-retention journey did not return to idle: {status_document}"
        );
        sleep(Duration::from_millis(100)).await;
    }
    GenerationMemoryEvidence {
        saw_generating_activity,
        smallest_generating_expert_payload_bytes: if smallest_generating_expert_payload_bytes
            == u64::MAX
        {
            0
        } else {
            smallest_generating_expert_payload_bytes
        },
        largest_generating_expert_payload_bytes,
        final_expert_memory_mode: final_status["expert_memory_mode"]
            .as_str()
            .unwrap_or("unavailable")
            .to_owned(),
        average_prefill_tokens_per_second:
            final_status["serving_session"]["average_prefill_tok_per_second"]
                .as_f64()
                .unwrap_or(0.0),
        average_generation_tokens_per_second:
            final_status["serving_session"]["average_generation_tok_per_second"]
                .as_f64()
                .unwrap_or(0.0),
        final_status,
    }
}

struct CompletedStream {
    model_text: String,
    finish_reason: Option<String>,
}

async fn consume_completed_stream(
    mut streamed_completion: StreamResponse<Value>,
) -> CompletedStream {
    let mut streamed_model_text = String::new();
    let mut finish_reason = None;
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = match stream_item {
            Ok(stream_chunk) => stream_chunk,
            Err(stream_error) => {
                eprintln!("[decode-expert-eviction] status=stream_error error={stream_error}");
                break;
            }
        };
        if !stream_chunk["error"].is_null() {
            panic!("the decode-retention REST stream returned an error: {stream_chunk}");
        }
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                streamed_model_text.push_str(content_fragment);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
        }
    }
    CompletedStream {
        model_text: streamed_model_text.trim().to_owned(),
        finish_reason,
    }
}

fn write_acceptance_config(
    isolated_worker_home: &Path,
    model_directory: &Path,
    maximum_mlx_memory_bytes: u64,
) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the decode-eviction configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": maximum_mlx_memory_bytes / 1_000_000_000,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": true,
        "prompt_cache_max_size_gb": 50,
        "performance_attribution_enabled": true,
        "logging": {
            "level": "debug",
            "retained_files": 2,
        },
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 2_048,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the decode-eviction configuration should serialize"),
    )
    .expect("the decode-eviction configuration should be written");
}

fn isolated_eviction_journey_worker_home() -> PathBuf {
    let worker_home = std::env::temp_dir().join("astronomical-decode-expert-eviction-e2e");
    let _ = fs::remove_dir_all(&worker_home);
    fs::create_dir_all(&worker_home).expect("the decode-eviction worker home should be created");
    worker_home
        .canonicalize()
        .expect("the decode-eviction worker home should canonicalize")
}

/// Copies worker tracing and attribution evidence out of the disposable home so
/// a failing retention cell can be diagnosed after the runner exits.
fn preserve_journey_evidence(isolated_worker_home: &Path, prompt_token_count: usize) {
    let unix_millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|unix_time| unix_time.as_millis())
        .unwrap_or_default();
    let evidence_directory = std::env::current_dir()
        .expect("the journey should resolve its working directory")
        .join("target/acceptance-evidence/decode-expert-eviction")
        .join(format!("{unix_millis}-{prompt_token_count}"));
    fs::create_dir_all(&evidence_directory)
        .expect("the acceptance evidence directory should be created");
    let logging_directory = isolated_worker_home.join(".astronomical-dev").join("logs");
    let Ok(logging_entries) = fs::read_dir(&logging_directory) else {
        return;
    };
    for logging_entry in logging_entries.flatten() {
        let source_path = logging_entry.path();
        if source_path.is_file()
            && let Some(source_name) = source_path.file_name()
        {
            let _ = fs::copy(&source_path, evidence_directory.join(source_name));
        }
    }
}
