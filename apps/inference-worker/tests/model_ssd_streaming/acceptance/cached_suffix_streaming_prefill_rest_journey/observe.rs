//! Live request observation for the cached-suffix streaming prefill journey.
//!
//! Separated so the journey file stays under the source-size gate while the
//! maintainer can still watch token frontier, memory, and stream completion.

use std::net::SocketAddr;
use std::path::Path;

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep};

use crate::support::serving_rest::get_json_endpoint;

use super::reports::InteractionReports;
use super::support::{
    InteractionLiveEvidence, MemorySample, ProgressSample, print_worker_diagnostic_logs,
    production_shaped_tools,
};
use super::{LOG_MARKER, MAXIMUM_OUTPUT_TOKEN_COUNT};

const THINKING_BUDGET_TOKEN_COUNT: u32 = 0;
const TOOL_COUNT: usize = 46;
const STATUS_SAMPLE_INTERVAL: Duration = Duration::from_millis(100);
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const STALL_DIAGNOSTIC_SECONDS: f64 = 10.0;

pub(super) fn completion_request(model_id: &str, messages: Value) -> Value {
    json!({
        "model": model_id,
        "messages": messages,
        "tools": production_shaped_tools(TOOL_COUNT),
        "tool_choice": "auto",
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    })
}

pub(super) async fn execute_observed_request(
    openai_client: &Client<OpenAIConfig>,
    server_address: SocketAddr,
    logging_directory: &Path,
    request_label: &'static str,
    completion_request: Value,
) -> ObservedRequestOutcome {
    eprintln!(
        "{LOG_MARKER} request={request_label} status=start max_output_tokens={MAXIMUM_OUTPUT_TOKEN_COUNT} eta_seconds=unknown"
    );
    let request_started_at = Instant::now();
    let streamed_completion: StreamResponse<Value> = openai_client
        .chat()
        .create_stream_byot(completion_request)
        .await
        .expect("the public interaction request should start");
    let (completed_stream, live_evidence) = tokio::join!(
        consume_stream(request_label, request_started_at, streamed_completion),
        observe_request(
            server_address,
            logging_directory,
            request_label,
            request_started_at
        ),
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
        let stream_chunk = match stream_item {
            Ok(stream_chunk) => stream_chunk,
            Err(stream_error) => {
                eprintln!(
                    "{LOG_MARKER} request={request_label} status=stream_error error={stream_error}"
                );
                break;
            }
        };
        if !stream_chunk["error"].is_null() || stream_chunk.get("choices").is_none() {
            eprintln!(
                "{LOG_MARKER} request={request_label} status=stream_chunk chunk={stream_chunk}"
            );
        }
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
    logging_directory: &Path,
    request_label: &'static str,
    request_started_at: Instant,
) -> InteractionLiveEvidence {
    let mut evidence = InteractionLiveEvidence::default();
    let mut previous_sample = ProgressSample::default();
    let mut last_log_at = request_started_at - STATUS_LOG_INTERVAL;
    let mut last_activity = String::new();
    let mut last_prefill_processed_tokens = 0_u64;
    let mut last_prefill_progress_at = request_started_at;
    let mut last_stall_dump_at = request_started_at;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let activity = status_document["activity"].as_str().unwrap_or("unknown");
        evidence.observe(&status_document);
        if activity == "prompt_processing" {
            let processed_tokens = status_document["progress"]["processed_tokens"]
                .as_u64()
                .unwrap_or(0);
            if processed_tokens > last_prefill_processed_tokens {
                last_prefill_processed_tokens = processed_tokens;
                last_prefill_progress_at = Instant::now();
            } else if processed_tokens > 0 {
                let unmoving_seconds = last_prefill_progress_at.elapsed().as_secs_f64();
                evidence.longest_unmoving_prefill_seconds = evidence
                    .longest_unmoving_prefill_seconds
                    .max(unmoving_seconds);
                if unmoving_seconds >= STALL_DIAGNOSTIC_SECONDS
                    && last_stall_dump_at.elapsed().as_secs_f64() >= STALL_DIAGNOSTIC_SECONDS
                {
                    eprintln!(
                        "{LOG_MARKER} request={request_label} status=stall processed_tokens={processed_tokens} unmoving_seconds={unmoving_seconds:.1}"
                    );
                    print_worker_diagnostic_logs(logging_directory);
                    last_stall_dump_at = Instant::now();
                }
            }
        }
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
    let request_elapsed_millis =
        u64::try_from(request_started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
    // Prefer the request wall clock. The worker phase clock stays at zero while
    // generation preparation holds the first decode, which hid every tok/s.
    let interval_elapsed_millis =
        request_elapsed_millis.saturating_sub(previous_sample.request_elapsed_millis);
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
    if activity == "prompt_processing" && cumulative_tokens_per_second > 0.0 {
        previous_sample.prefill_tokens_per_second = cumulative_tokens_per_second;
    } else if activity == "generating" && cumulative_tokens_per_second > 0.0 {
        previous_sample.generation_tokens_per_second = cumulative_tokens_per_second;
    } else if activity == "generating" && interval_tokens_per_second > 0.0 {
        previous_sample.generation_tokens_per_second = interval_tokens_per_second;
    }
    let prefill_tokens_per_second = previous_sample.prefill_tokens_per_second;
    let generation_tokens_per_second = previous_sample.generation_tokens_per_second;
    let remaining_tokens = remaining_tokens_for_activity(
        activity,
        processed_tokens,
        total_tokens,
        MAXIMUM_OUTPUT_TOKEN_COUNT.into(),
    );
    let eta_rate_tokens_per_second = match activity {
        "prompt_processing" if interval_tokens_per_second > 0.0 => interval_tokens_per_second,
        "prompt_processing" => prefill_tokens_per_second,
        "generating" if interval_tokens_per_second > 0.0 => interval_tokens_per_second,
        "generating" => generation_tokens_per_second,
        _ => 0.0,
    };
    let eta_display = if eta_rate_tokens_per_second > 0.0 {
        format!(
            "{:.1}",
            remaining_tokens as f64 / eta_rate_tokens_per_second
        )
    } else {
        "unknown".to_owned()
    };
    let memory = MemorySample::from_status(status_document);
    let resident_expert_group_count = status_document["expert_residency"]["resident_expert_count"]
        .as_u64()
        .unwrap_or(0);
    eprintln!(
        "{LOG_MARKER} request={request_label} status=progress elapsed_seconds={:.3} activity={activity} phase={phase} processed={processed_tokens}/{total_tokens} remaining={remaining_tokens} prefill_tok_s={prefill_tokens_per_second:.2} generation_tok_s={generation_tokens_per_second:.2} interval_tok_s={interval_tokens_per_second:.2} eta_seconds={eta_display} completed_chunk_tokens={completed_chunk_tokens} active_gb={:.3} peak_gb={:.3} expert_gb={:.3} resident_expert_groups={resident_expert_group_count} active_bytes={} allocator_bytes={} allocator_gb={:.3} peak_bytes={} expert_bytes={} model_core_bytes={} context_bytes={} runtime_bytes={}",
        request_started_at.elapsed().as_secs_f64(),
        memory.active_memory_bytes as f64 / 1_000_000_000.0,
        memory.peak_memory_bytes as f64 / 1_000_000_000.0,
        memory.expert_payload_bytes as f64 / 1_000_000_000.0,
        memory.active_memory_bytes,
        memory.allocator_cache_memory_bytes,
        memory.allocator_cache_memory_bytes as f64 / 1_000_000_000.0,
        memory.peak_memory_bytes,
        memory.expert_payload_bytes,
        memory.model_core_payload_bytes,
        memory.context_state_payload_bytes,
        memory.runtime_work_payload_bytes,
    );
    previous_sample.processed_tokens = processed_tokens;
    previous_sample.elapsed_millis = elapsed_millis;
    previous_sample.request_elapsed_millis = request_elapsed_millis;
}

fn remaining_tokens_for_activity(
    activity: &str,
    processed_tokens: u64,
    total_tokens: u64,
    maximum_output_token_count: u64,
) -> u64 {
    match activity {
        "prompt_processing" => total_tokens.saturating_sub(processed_tokens),
        "generating" | "generation_preparation" => {
            maximum_output_token_count.saturating_sub(processed_tokens)
        }
        _ => total_tokens.saturating_sub(processed_tokens),
    }
}

pub(super) fn print_request_records(reports: &InteractionReports) {
    for (request_index, performance_record) in reports.performance_records.iter().enumerate() {
        eprintln!(
            "{LOG_MARKER} request=record index={request_index} cached_token_count={} prefill_tok_per_second={} generation_tok_per_second={} outcome={}",
            performance_record["cached_token_count"],
            performance_record["prefill_tok_per_second"],
            performance_record["generation_tok_per_second"],
            performance_record["outcome"],
        );
    }
    for (request_index, attribution_report) in reports.attribution_reports.iter().enumerate() {
        let counters = &attribution_report["counters"];
        eprintln!(
            "{LOG_MARKER} request=attribution index={request_index} outcome={} rejection_reason={} restored_tokens={} prefill_chunks={} streamed_payload_bytes={} positional_read_calls={} positional_read_bytes={} positional_read_elapsed_ns={} complete_layers_planned={} counters={}",
            attribution_report["outcome"],
            attribution_report["rejection_reason"],
            counters["RestoredPersistentPromptCacheTokenCount"],
            counters["PrefillChunckCount"],
            counters["RustExpertStreamingPayloadByteCount"],
            counters["PositionalFileReadCallCount"],
            counters["PositionalFileReadByteCount"],
            counters["PositionalFileReadElapsedNanoseconds"],
            counters["ExpertResidencyPlanCompleteLayerCount"],
            counters,
        );
    }
}

pub(super) fn assert_completed_request(
    request_outcome: &ObservedRequestOutcome,
    request_label: &str,
) {
    let finish_reason = request_outcome.finish_reason.as_deref();
    assert!(
        matches!(finish_reason, Some("stop" | "length" | "tool_calls")),
        "{request_label} request should finish through the public stream; finish_reason={finish_reason:?} generated_tokens={} last_error={} last_activity={}",
        request_outcome.generated_token_count,
        request_outcome.live_evidence.final_status["last_error"],
        request_outcome.live_evidence.final_status["activity"],
    );
    if matches!(finish_reason, Some("tool_calls")) {
        assert!(
            request_outcome.generated_token_count > 0,
            "{request_label} tool-call finish must still prove the worker generated tokens; last_error={} last_activity={}",
            request_outcome.live_evidence.final_status["last_error"],
            request_outcome.live_evidence.final_status["activity"],
        );
    } else {
        assert!(
            !request_outcome.model_text.is_empty(),
            "{request_label} request should produce visible model text; finish_reason={finish_reason:?} generated_tokens={} last_error={} last_activity={}",
            request_outcome.generated_token_count,
            request_outcome.live_evidence.final_status["last_error"],
            request_outcome.live_evidence.final_status["activity"],
        );
    }
    assert!(request_outcome.live_evidence.observed_prompt_processing);
    if request_outcome
        .live_evidence
        .observed_generation_preparation
    {
        assert!(
            request_outcome
                .live_evidence
                .observed_generation_preparation_with_consistent_residency,
            "{request_label} generation preparation must publish layer counts with matching payload bytes"
        );
    }
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

pub(super) struct ObservedRequestOutcome {
    pub(super) model_text: String,
    pub(super) finish_reason: Option<String>,
    pub(super) generated_token_count: u64,
    pub(super) first_output_elapsed_millis: Option<u64>,
    pub(super) live_evidence: InteractionLiveEvidence,
}
