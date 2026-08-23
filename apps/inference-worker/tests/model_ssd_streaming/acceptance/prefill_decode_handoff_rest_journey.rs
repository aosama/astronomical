//! Reproduces the interaction between cached append-only prefill, expert
//! streaming, retained RAM topology, and the prefill-to-decode handoff.
//!
//! This permanent journey is intentionally verbose. It is the focused command a
//! maintainer reruns while changing phase-aware residency so a stalled token
//! frontier, memory growth, topology churn, or hidden disk read remains visible.

mod reports;
mod support;

use std::net::SocketAddr;

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::common::real_model_rest_server::{
    JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};
use reports::{assert_reported_interaction, print_comparison_summary, read_interaction_reports};
use support::{
    InteractionLiveEvidence, MemorySample, ProgressSample, production_shaped_tools,
    write_interaction_config,
};

const LOG_MARKER: &str = "[prefill-decode-residency-interaction]";
const MODEL_ID: &str = crate::common::ORNITH_SSD_STREAMING_MODEL_ID;
// This qualification cell is deliberately below complete-resident prefill need
// for the configured artifact, while the launcher still clamps it to the host's
// machine-derived ceiling. Production policy contains no corresponding constant.
const MAXIMUM_MLX_MEMORY_BYTES: u64 = 32_000_000_000;
const INITIAL_PROMPT_TOKEN_COUNT: usize = 10_000;
const PREFILL_CHUNK_TOKEN_COUNT: u32 = 4_096;
const PAGING_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 3;
// A short generation budget keeps the residency journey inside the bounded
// acceptance window. The long prefill source above supplies the memory pressure
// this journey qualifies without making success depend on one machine's speed.
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 32;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 0;
const TOOL_COUNT: usize = 46;
const STATUS_SAMPLE_INTERVAL: Duration = Duration::from_millis(100);
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches one real worker and reproduces cached prefill/decode expert residency"]
async fn should_complete_cold_and_cached_append_requests_with_consistent_prefill_decode_residency()
{
    timeout(JOURNEY_TIMEOUT, run_interaction_journey())
        .await
        .expect("the prefill/decode residency interaction must finish within 115 seconds");
}

async fn run_interaction_journey() {
    let model_directory = crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
    let isolated_worker_home =
        tempfile::tempdir().expect("the interaction worker home should be created");
    write_interaction_config(isolated_worker_home.path(), &model_directory);
    let initial_user_message = crate::common::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        &ROMEO_AND_JULIET_SOURCE.repeat(4),
        "Summarize the play's central conflict and the decisions that produce its tragic ending.",
        INITIAL_PROMPT_TOKEN_COUNT,
    );
    eprintln!(
        "{LOG_MARKER} request=journey status=start timeout_seconds={} mlx_ceiling_bytes={MAXIMUM_MLX_MEMORY_BYTES} mlx_ceiling_gb={:.3} initial_prompt_tokens={INITIAL_PROMPT_TOKEN_COUNT} fixed_prefill_tokens={PREFILL_CHUNK_TOKEN_COUNT} paging_graph_submission_layer_interval={PAGING_GRAPH_SUBMISSION_LAYER_INTERVAL} persistent_prompt_cache_enabled=true",
        JOURNEY_TIMEOUT.as_secs(),
        MAXIMUM_MLX_MEMORY_BYTES as f64 / 1_000_000_000.0,
    );

    let real_model_rest_server = launch_real_model_rest_server(
        MODEL_ID,
        model_directory,
        isolated_worker_home.path(),
        MAXIMUM_MLX_MEMORY_BYTES,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-residency-interaction-client"),
    );

    let cold_request = completion_request(json!([
        {"role": "user", "content": initial_user_message}
    ]));
    let cold_outcome =
        execute_observed_request(&openai_client, server_address, "cold", cold_request).await;
    assert_completed_request(&cold_outcome, "cold");

    let follow_up_source_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(1_000)
        .collect::<String>();
    let append_request = completion_request(json!([
        {"role": "user", "content": initial_user_message},
        {"role": "assistant", "content": cold_outcome.model_text},
        {
            "role": "user",
            "content": format!(
                "Relate that summary to the consequences of haste. Use this additional source context without quoting it: {}",
                follow_up_source_excerpt
            )
        }
    ]));
    let append_outcome =
        execute_observed_request(&openai_client, server_address, "append", append_request).await;
    assert_completed_request(&append_outcome, "append");

    stop_real_model_rest_server(real_model_rest_server).await;
    let reports = read_interaction_reports(isolated_worker_home.path());
    assert_reported_interaction(&reports, &cold_outcome, &append_outcome);
    print_comparison_summary(&reports, &cold_outcome, &append_outcome);
}

fn completion_request(messages: Value) -> Value {
    json!({
        "model": MODEL_ID,
        "messages": messages,
        "tools": production_shaped_tools(TOOL_COUNT),
        "tool_choice": "auto",
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    })
}

async fn execute_observed_request(
    openai_client: &Client<OpenAIConfig>,
    server_address: SocketAddr,
    request_label: &'static str,
    completion_request: Value,
) -> ObservedRequestOutcome {
    eprintln!(
        "{LOG_MARKER} request={request_label} status=start max_output_tokens={MAXIMUM_OUTPUT_TOKEN_COUNT} ETA_seconds=55"
    );
    let request_started_at = Instant::now();
    let streamed_completion: StreamResponse<Value> = openai_client
        .chat()
        .create_stream_byot(completion_request)
        .await
        .expect("the public interaction request should start");
    let (completed_stream, live_evidence) = tokio::join!(
        consume_stream(request_label, request_started_at, streamed_completion),
        observe_request(server_address, request_label, request_started_at),
    );
    eprintln!(
        "{LOG_MARKER} request={request_label} status=summary total_elapsed_seconds={:.3} first_output_millis={} generated_tokens={} max_active_gb={:.3} max_peak_gb={:.3} max_expert_gb={:.3} observed_prefill={} observed_preparation={} observed_generation={} final_activity={}",
        request_started_at.elapsed().as_secs_f64(),
        completed_stream.first_output_elapsed_millis.unwrap_or(0),
        completed_stream.generated_token_count,
        live_evidence.maximum_active_memory_bytes as f64 / 1_000_000_000.0,
        live_evidence.maximum_peak_memory_bytes as f64 / 1_000_000_000.0,
        live_evidence.maximum_expert_payload_bytes as f64 / 1_000_000_000.0,
        live_evidence.observed_prompt_processing,
        live_evidence.observed_generation_preparation,
        live_evidence.observed_generation,
        live_evidence.final_status["activity"]
            .as_str()
            .unwrap_or("unknown"),
    );
    ObservedRequestOutcome {
        model_text: completed_stream.model_text,
        finish_reason: completed_stream.finish_reason,
        generated_token_count: completed_stream.generated_token_count,
        first_output_elapsed_millis: completed_stream.first_output_elapsed_millis,
        live_evidence,
    }
}

async fn consume_stream(
    request_label: &'static str,
    request_started_at: Instant,
    mut streamed_completion: StreamResponse<Value>,
) -> CompletedStream {
    let mut model_text = String::new();
    let mut finish_reason = None;
    let mut generated_token_count = 0_u64;
    let mut first_output_elapsed_millis = None;
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk =
            stream_item.expect("the public interaction stream should remain healthy");
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                if !content_fragment.is_empty() {
                    first_output_elapsed_millis.get_or_insert_with(|| {
                        let elapsed_millis = request_started_at.elapsed().as_millis() as u64;
                        eprintln!(
                            "{LOG_MARKER} request={request_label} status=progress phase=first_output elapsed_millis={elapsed_millis}"
                        );
                        elapsed_millis
                    });
                    model_text.push_str(content_fragment);
                }
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
        }
        if let Some(completion_tokens) = stream_chunk["usage"]["completion_tokens"].as_u64() {
            generated_token_count = completion_tokens;
        }
    }
    CompletedStream {
        model_text: model_text.trim().to_owned(),
        finish_reason,
        generated_token_count,
        first_output_elapsed_millis,
    }
}

async fn observe_request(
    server_address: SocketAddr,
    request_label: &'static str,
    request_started_at: Instant,
) -> InteractionLiveEvidence {
    let mut evidence = InteractionLiveEvidence::default();
    let mut previous_sample = ProgressSample::default();
    let mut last_log_at = request_started_at - STATUS_LOG_INTERVAL;
    let mut last_activity = String::new();
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let activity = status_document["activity"].as_str().unwrap_or("unknown");
        evidence.observe(&status_document);
        let activity_changed = activity != last_activity;
        if activity_changed || last_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            print_status_sample(
                request_label,
                request_started_at,
                &status_document,
                &mut previous_sample,
            );
            last_log_at = Instant::now();
            last_activity = activity.to_owned();
        }
        if evidence.observed_active && activity == "idle" {
            evidence.final_status = status_document;
            return evidence;
        }
        sleep(STATUS_SAMPLE_INTERVAL).await;
    }
}

fn print_status_sample(
    request_label: &str,
    request_started_at: Instant,
    status_document: &Value,
    previous_sample: &mut ProgressSample,
) {
    let activity = status_document["activity"].as_str().unwrap_or("unknown");
    let phase = status_document["progress"]["phase"]
        .as_str()
        .unwrap_or(activity);
    let processed_tokens = status_document["progress"]["processed_tokens"]
        .as_u64()
        .unwrap_or(0);
    let total_tokens = status_document["progress"]["total_tokens"]
        .as_u64()
        .unwrap_or(0);
    let elapsed_millis = status_document["progress"]["elapsed_ms"]
        .as_u64()
        .unwrap_or(0);
    let completed_chunk_tokens = status_document["progress"]["completed_prefill_chunk_tokens"]
        .as_u64()
        .unwrap_or(0);
    let interval_elapsed_millis = elapsed_millis.saturating_sub(previous_sample.elapsed_millis);
    let interval_tokens = processed_tokens.saturating_sub(previous_sample.processed_tokens);
    let interval_tokens_per_second = if interval_elapsed_millis > 0 {
        interval_tokens as f64 * 1_000.0 / interval_elapsed_millis as f64
    } else {
        0.0
    };
    let cumulative_tokens_per_second = if elapsed_millis > 0 {
        processed_tokens as f64 * 1_000.0 / elapsed_millis as f64
    } else {
        0.0
    };
    // The worker uses a phase-local progress clock. Retain the latest rate from
    // each phase so every periodic line exposes both prefill and generation
    // throughput instead of making a maintainer infer them from generic fields.
    if activity == "prompt_processing" {
        previous_sample.prefill_tokens_per_second = cumulative_tokens_per_second;
    } else if activity == "generating" {
        previous_sample.generation_tokens_per_second = cumulative_tokens_per_second;
    }
    let prefill_tokens_per_second = previous_sample.prefill_tokens_per_second;
    let generation_tokens_per_second = previous_sample.generation_tokens_per_second;
    let memory = MemorySample::from_status(status_document);
    let complete_layer_count = status_document["expert_residency"]["complete_layer_count"]
        .as_u64()
        .unwrap_or(0);
    let partial_layer_count = status_document["expert_residency"]["partial_layer_count"]
        .as_u64()
        .unwrap_or(0);
    eprintln!(
        "{LOG_MARKER} request={request_label} status=progress activity={activity} phase={phase} request_elapsed_seconds={:.3} processed_tokens={processed_tokens} total_tokens={total_tokens} completed_chunk_tokens={completed_chunk_tokens} interval_tokens_per_second={interval_tokens_per_second:.2} prefill_tokens_per_second={prefill_tokens_per_second:.2} generation_tokens_per_second={generation_tokens_per_second:.2} active_bytes={} active_gb={:.3} allocator_bytes={} allocator_gb={:.3} peak_bytes={} peak_gb={:.3} expert_bytes={} expert_gb={:.3} model_core_bytes={} context_bytes={} runtime_bytes={} complete_layers={complete_layer_count} partial_layers={partial_layer_count}",
        request_started_at.elapsed().as_secs_f64(),
        memory.active_memory_bytes,
        memory.active_memory_bytes as f64 / 1_000_000_000.0,
        memory.allocator_cache_memory_bytes,
        memory.allocator_cache_memory_bytes as f64 / 1_000_000_000.0,
        memory.peak_memory_bytes,
        memory.peak_memory_bytes as f64 / 1_000_000_000.0,
        memory.expert_payload_bytes,
        memory.expert_payload_bytes as f64 / 1_000_000_000.0,
        memory.model_core_payload_bytes,
        memory.context_state_payload_bytes,
        memory.runtime_work_payload_bytes,
    );
    previous_sample.processed_tokens = processed_tokens;
    previous_sample.elapsed_millis = elapsed_millis;
}

fn assert_completed_request(request_outcome: &ObservedRequestOutcome, request_label: &str) {
    assert!(
        !request_outcome.model_text.is_empty(),
        "{request_label} request should produce visible model text"
    );
    assert!(matches!(
        request_outcome.finish_reason.as_deref(),
        Some("stop" | "length")
    ));
    assert!(request_outcome.live_evidence.observed_prompt_processing);
    assert!(
        request_outcome
            .live_evidence
            .observed_generation_preparation
    );
    assert!(
        request_outcome
            .live_evidence
            .observed_generation_preparation_with_consistent_residency,
        "{request_label} generation preparation must publish layer counts with matching payload bytes"
    );
    assert!(request_outcome.live_evidence.observed_generation);
    assert_eq!(
        request_outcome.live_evidence.final_status["status"],
        "ready"
    );
}

struct CompletedStream {
    model_text: String,
    finish_reason: Option<String>,
    generated_token_count: u64,
    first_output_elapsed_millis: Option<u64>,
}

struct ObservedRequestOutcome {
    model_text: String,
    finish_reason: Option<String>,
    generated_token_count: u64,
    first_output_elapsed_millis: Option<u64>,
    live_evidence: InteractionLiveEvidence,
}
