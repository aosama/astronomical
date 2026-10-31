//! Driving one completion to its terminal event and reading its numbers back.
//!
//! The worker's performance log is the authority for the measured rates: it is
//! written by the serving process itself, so a client wall clock can never
//! contaminate the record. Everything here exists to make that one round trip —
//! launch a completion, wait for its terminal event, parse the sample the worker
//! wrote — and to keep the diagnostic output that explains a failure.

use std::fs;
use std::path::Path;
use std::time::Duration;

use tokio::time::sleep;

use astronomical_daemon::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, WorkerHandle, WorkerHealthStatus,
};
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatImageInput, ChatMessage, ChatToolChoice,
    RequestId, WorkerDecodeExpertCachePolicy,
};
use serde::Deserialize;

const PERFORMANCE_THROUGHPUT_SAMPLING_SEED: u64 = 42;

/// Values from the single measured completion, after the warmup is discarded.
#[derive(Clone, Debug)]
pub(crate) struct ThroughputMeasurement {
    pub(crate) model_id: String,
    pub(crate) decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
    pub(crate) total_input_tokens: u32,
    pub(crate) total_output_tokens: u16,
    pub(crate) cached_tokens: u32,
    pub(crate) prefill_time_seconds: f64,
    pub(crate) decode_time_seconds: f64,
    pub(crate) prefill_tokens_per_second: f64,
    pub(crate) decode_tokens_per_second: f64,
    pub(crate) mlx_peak_memory_bytes: Option<u64>,
    pub(crate) measured_output_text: Option<String>,
    pub(crate) decode_cache_diagnostic_evidence: Option<DecodeCacheDiagnosticEvidence>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct DecodeCacheDiagnosticEvidence {
    pub(crate) decode_token_forward_latency_p95_nanoseconds: Option<u64>,
    pub(crate) decode_token_forward_latency_sample_count: Option<u64>,
    pub(crate) decode_advance_span: Option<PerformanceOperationEvidence>,
    pub(crate) decode_expert_route_host_materialization: Option<PerformanceOperationEvidence>,
    pub(crate) decode_expert_route_contiguous_array_construction:
        Option<PerformanceOperationEvidence>,
    pub(crate) decode_expert_route_array_evaluation: Option<PerformanceOperationEvidence>,
    pub(crate) decode_expert_route_host_copy: Option<PerformanceOperationEvidence>,
    pub(crate) rust_expert_streaming_layer_preparation: Option<PerformanceOperationEvidence>,
    pub(crate) paged_router_graph_construction: Option<PerformanceOperationEvidence>,
    pub(crate) paged_moe_graph_construction: Option<PerformanceOperationEvidence>,
    pub(crate) forward_completion_wait: Option<PerformanceOperationEvidence>,
    pub(crate) generated_token_item_synchronization_wait: Option<PerformanceOperationEvidence>,
    pub(crate) decode_expert_source_payload_bytes: u64,
    pub(crate) decode_expert_source_summary_payload_bytes: u64,
    pub(crate) prefill_expert_source_payload_bytes: u64,
    pub(crate) retained_route_assignment_hit_count: u64,
    pub(crate) retained_route_assignment_miss_count: u64,
    pub(crate) hot_expert_route_fully_covered_count: u64,
    pub(crate) hot_expert_route_partially_covered_count: u64,
    pub(crate) hot_expert_route_fully_missed_count: u64,
    pub(crate) decode_expert_cache_slot_eviction_count: u64,
    pub(crate) positional_file_read_call_count: u64,
    pub(crate) positional_file_read_byte_count: u64,
    pub(crate) positional_file_read_elapsed_nanoseconds: u64,
    pub(crate) positional_file_read_maximum_elapsed_nanoseconds: u64,
    pub(crate) positional_file_read_maximum_concurrent_count: u64,
    pub(crate) process_physical_disk_read_bytes: Option<u64>,
    pub(crate) mlx_peak_memory_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PerformanceOperationEvidence {
    pub(crate) occurrence_count: u64,
    pub(crate) total_elapsed_nanoseconds: u64,
    pub(crate) minimum_elapsed_nanoseconds: u64,
    pub(crate) maximum_elapsed_nanoseconds: u64,
}

/// One line of the worker performance log, re-parsed for the fields the report needs.
#[derive(Debug, Deserialize)]
struct ThroughputSample {
    prompt_token_count: u32,
    cached_token_count: u32,
    generated_token_count: u16,
    prefill_tok_per_second: Option<f64>,
    generation_tok_per_second: Option<f64>,
    prefill_elapsed_millis: u64,
    generation_elapsed_millis: u64,
    mlx_peak_memory_bytes: Option<u64>,
}

/// Drives one completion to its terminal event, returning when the worker reports
/// completion so its performance record is written.
pub(crate) async fn drive_completion(
    worker_handle: &WorkerHandle,
    request_id: RequestId,
    model_id: &str,
    prompt: &str,
    images: Vec<ChatImageInput>,
    output_tokens: u16,
    temperature_thousandths: u16,
) -> Result<String, String> {
    let command = ChatGenerationCommand {
        request_id,
        model: model_id.to_owned(),
        messages: vec![ChatMessage::User {
            content: prompt.to_owned(),
            images,
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: output_tokens,
            temperature_thousandths: Some(temperature_thousandths),
            top_p_thousandths: None,
            repetition_penalty_thousandths: None,
            // Repeatable token routes keep baseline and candidate cache policies comparable.
            seed: Some(PERFORMANCE_THROUGHPUT_SAMPLING_SEED),
            thinking_budget: None,
        },
        structured_generation: None,
    };
    let mut generated_text = String::new();
    let mut receiver = match worker_handle.start_chat_generation(command).await {
        Ok(receiver) => receiver,
        Err(error) => {
            return Err(format!(
                "the worker rejected the completion request: {error:?}"
            ));
        }
    };
    while let Some(event) = receiver.recv().await {
        match event {
            ChatGenerationStreamEvent::ReasoningFragment(output_fragment)
            | ChatGenerationStreamEvent::TextFragment(output_fragment) => {
                generated_text.push_str(&output_fragment);
            }
            ChatGenerationStreamEvent::Completed { .. } => return Ok(generated_text),
            ChatGenerationStreamEvent::Failed { reason } => {
                return Err(format!(
                    "the worker failed the completion request: {reason:?}"
                ));
            }
            ChatGenerationStreamEvent::Error(error_code) => {
                return Err(format!(
                    "the worker stream reported a completion error: {error_code:?}"
                ));
            }
            _ => {}
        }
    }
    Err("the worker stream closed without a terminal completion event".to_owned())
}

/// Prints the tail of every file in the worker's persistent logging directory
/// so the last lines a failed worker wrote stay in the journey output.
///
/// Only a driver's own failure path calls this: the tail can be thousands of
/// lines, so a passing journey never pays for it.
pub(crate) fn dump_worker_logging_directory(
    logging_directory: &Path,
    worker_log_dump_line_limit: usize,
) {
    let Ok(entries) = fs::read_dir(logging_directory) else {
        eprintln!(
            "[performance-throughput] no worker logging directory at {}",
            logging_directory.display()
        );
        return;
    };
    for entry in entries.flatten().filter(|entry| entry.path().is_file()) {
        let log_file_path = entry.path();
        let Ok(contents) = fs::read_to_string(&log_file_path) else {
            continue;
        };
        eprintln!(
            "[performance-throughput] worker log {} (tail):",
            log_file_path.display()
        );
        for log_line in contents.lines().rev().take(worker_log_dump_line_limit) {
            eprintln!("[performance-throughput]   {log_line}");
        }
    }
}

/// Waits for the worker to finish loading and release the model, so the warmup
/// completion never races model loading and the measured run never starts on a
/// cold worker.
pub(crate) async fn wait_until_idle(worker_handle: &WorkerHandle, ready_attempt_limit: u8) {
    for _ in 1..=ready_attempt_limit {
        let snapshot = worker_handle.worker_health_snapshot();
        if snapshot.status == WorkerHealthStatus::Ready && snapshot.ready_model_id.is_none() {
            return;
        }
        sleep(Duration::from_secs(1)).await;
    }
    panic!("the real worker did not become idle before the throughput deadline");
}

/// Reads the worker's performance log and returns the measured completion.
///
/// The parsed sample type never leaves this module: callers want the
/// summarized measurement, and exposing the raw sample would let a journey
/// read a field the driver never meant to publish.
pub(crate) fn read_measured_completion(
    log_path: &Path,
    model_id: &str,
    decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
) -> Option<ThroughputMeasurement> {
    read_completion_by_index(
        log_path,
        model_id,
        decode_expert_cache_policy,
        CompletionIndex::Last,
    )
}

/// Which completion of a multi-completion journey to read. The warmup is
/// sample zero; the measured completion is the last sample.
#[derive(Clone, Copy, Debug)]
pub(crate) enum CompletionIndex {
    First,
    Last,
}

/// Reads one journey completion by its performance-log position, so a
/// two-turn journey can compare the first turn's evidence with the second's.
pub(crate) fn read_completion_by_index(
    log_path: &Path,
    model_id: &str,
    decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
    completion_index: CompletionIndex,
) -> Option<ThroughputMeasurement> {
    let samples = read_throughput_samples(log_path)?;
    let sample = match completion_index {
        CompletionIndex::First => samples.first()?,
        CompletionIndex::Last => samples.last()?,
    };
    Some(sample_to_measurement(
        model_id,
        decode_expert_cache_policy,
        sample,
    ))
}

fn read_throughput_samples(log_path: &Path) -> Option<Vec<ThroughputSample>> {
    let text = fs::read_to_string(log_path).ok()?;
    let samples = text
        .lines()
        .filter_map(|line| serde_json::from_str::<ThroughputSample>(line).ok())
        .collect::<Vec<_>>();
    (!samples.is_empty()).then_some(samples)
}

fn sample_to_measurement(
    model_id: &str,
    decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
    measured: &ThroughputSample,
) -> ThroughputMeasurement {
    ThroughputMeasurement {
        model_id: model_id.to_owned(),
        decode_expert_cache_policy,
        total_input_tokens: measured.prompt_token_count,
        total_output_tokens: measured.generated_token_count,
        cached_tokens: measured.cached_token_count,
        prefill_time_seconds: measured.prefill_elapsed_millis as f64 / 1_000.0,
        decode_time_seconds: measured.generation_elapsed_millis as f64 / 1_000.0,
        prefill_tokens_per_second: measured.prefill_tok_per_second.unwrap_or(0.0),
        decode_tokens_per_second: measured.generation_tok_per_second.unwrap_or(0.0),
        mlx_peak_memory_bytes: measured.mlx_peak_memory_bytes,
        measured_output_text: None,
        decode_cache_diagnostic_evidence: None,
    }
}

#[expect(clippy::too_many_lines)]
pub(crate) fn read_decode_cache_diagnostic_evidence(
    logging_directory: &Path,
    request_id: u64,
) -> Option<DecodeCacheDiagnosticEvidence> {
    let attribution_log =
        fs::read_to_string(logging_directory.join("performance-attribution.jsonl")).ok()?;
    // A request can carry more than one generation report: the worker emits
    // an early report (for example while expert loading is still settling)
    // before the final one that carries the complete counters. Consuming the
    // FIRST match silently reads the empty early report — which once turned
    // a 86-GB-streaming turn into a reported zero — so the LAST generation
    // report for the request is the authoritative one.
    let mut generation_report: Option<serde_json::Value> = None;
    for json_line in attribution_log.lines() {
        let Ok(attribution_report) = serde_json::from_str::<serde_json::Value>(json_line) else {
            continue;
        };
        if attribution_report["report_kind"] == "generation"
            && attribution_report["request_id"].as_u64() == Some(request_id)
        {
            generation_report = Some(attribution_report);
        }
    }
    let generation_report = generation_report?;
    let counters = generation_report.get("counters")?.as_array()?;
    let expert_streaming_source_summaries = generation_report
        .get("expert_streaming_source_summaries")?
        .as_array()?;
    let decode_expert_source_summary_payload_bytes = expert_streaming_source_summaries
        .iter()
        .filter(|source_summary| source_summary["phase"] == "decode")
        .try_fold(0_u64, |total_payload_bytes, source_summary| {
            source_summary["payload_byte_count"]
                .as_u64()
                .map(|payload_byte_count| total_payload_bytes.saturating_add(payload_byte_count))
        })?;
    let decode_token_forward_latency_sample_count = generation_report
        .get("decode_token_forward_latency_histogram")
        .and_then(serde_json::Value::as_array)
        .map(|histogram_buckets| {
            histogram_buckets
                .iter()
                .fold(0_u64, |sample_count, histogram_bucket| {
                    sample_count
                        .saturating_add(histogram_bucket["sample_count"].as_u64().unwrap_or(0))
                })
        });

    Some(DecodeCacheDiagnosticEvidence {
        decode_token_forward_latency_p95_nanoseconds: generation_report
            .get("decode_token_forward_latency_p95_nanoseconds")
            .and_then(serde_json::Value::as_u64),
        decode_token_forward_latency_sample_count,
        decode_advance_span: operation_evidence(&generation_report, "decode_advance_span"),
        decode_expert_route_host_materialization: operation_evidence(
            &generation_report,
            "decode_expert_route_host_materialization",
        ),
        decode_expert_route_contiguous_array_construction: operation_evidence(
            &generation_report,
            "decode_expert_route_contiguous_array_construction",
        ),
        decode_expert_route_array_evaluation: operation_evidence(
            &generation_report,
            "decode_expert_route_array_evaluation",
        ),
        decode_expert_route_host_copy: operation_evidence(
            &generation_report,
            "decode_expert_route_host_copy",
        ),
        rust_expert_streaming_layer_preparation: operation_evidence(
            &generation_report,
            "rust_expert_streaming_layer_preparation",
        ),
        paged_router_graph_construction: operation_evidence(
            &generation_report,
            "paged_router_graph_construction",
        ),
        paged_moe_graph_construction: operation_evidence(
            &generation_report,
            "paged_moe_graph_construction",
        ),
        forward_completion_wait: operation_evidence(
            &generation_report,
            "prefill_state_graphics_processor_completion_wait",
        ),
        generated_token_item_synchronization_wait: operation_evidence(
            &generation_report,
            "generated_token_item_synchronization_wait",
        ),
        decode_expert_source_payload_bytes: counter_amount(
            counters,
            "mandatory_decode_expert_source_payload_bytes",
        )?,
        decode_expert_source_summary_payload_bytes,
        prefill_expert_source_payload_bytes: counter_amount(
            counters,
            "mandatory_prefill_expert_source_payload_bytes",
        )?,
        retained_route_assignment_hit_count: counter_amount(
            counters,
            "retained_route_assignment_hit_count",
        )?,
        retained_route_assignment_miss_count: counter_amount(
            counters,
            "retained_route_assignment_miss_count",
        )?,
        hot_expert_route_fully_covered_count: counter_amount(
            counters,
            "hot_expert_route_fully_covered_count",
        )?,
        hot_expert_route_partially_covered_count: counter_amount(
            counters,
            "hot_expert_route_partially_covered_count",
        )?,
        hot_expert_route_fully_missed_count: counter_amount(
            counters,
            "hot_expert_route_fully_missed_count",
        )?,
        decode_expert_cache_slot_eviction_count: counter_amount(
            counters,
            "decode_expert_cache_slot_eviction_count",
        )?,
        positional_file_read_call_count: counter_amount(
            counters,
            "positional_file_read_call_count",
        )?,
        positional_file_read_byte_count: counter_amount(
            counters,
            "positional_file_read_byte_count",
        )?,
        positional_file_read_elapsed_nanoseconds: counter_amount(
            counters,
            "positional_file_read_elapsed_nanoseconds",
        )?,
        positional_file_read_maximum_elapsed_nanoseconds: counter_amount(
            counters,
            "positional_file_read_maximum_elapsed_nanoseconds",
        )?,
        positional_file_read_maximum_concurrent_count: counter_amount(
            counters,
            "positional_file_read_maximum_concurrent_count",
        )?,
        process_physical_disk_read_bytes: generation_report
            .get("process_physical_disk_read_bytes")
            .and_then(serde_json::Value::as_u64),
        mlx_peak_memory_bytes: generation_report
            .get("mlx_peak_memory_bytes")
            .and_then(serde_json::Value::as_u64),
    })
}

fn operation_evidence(
    generation_report: &serde_json::Value,
    operation_name: &str,
) -> Option<PerformanceOperationEvidence> {
    let operation_reports = generation_report.get("operations")?.as_array()?;
    let operation_report = operation_reports
        .iter()
        .find(|operation_report| operation_report["operation"] == operation_name)?;
    Some(PerformanceOperationEvidence {
        occurrence_count: operation_report["occurrence_count"].as_u64()?,
        total_elapsed_nanoseconds: operation_report["total_elapsed_nanoseconds"].as_u64()?,
        minimum_elapsed_nanoseconds: operation_report["minimum_elapsed_nanoseconds"].as_u64()?,
        maximum_elapsed_nanoseconds: operation_report["maximum_elapsed_nanoseconds"].as_u64()?,
    })
}

fn counter_amount(counters: &[serde_json::Value], counter_name: &str) -> Option<u64> {
    let matching_counter = counters
        .iter()
        .find(|counter_report| counter_report["counter"] == counter_name);
    matching_counter.map_or(Some(0), |counter_report| counter_report["amount"].as_u64())
}

/// Summarizes the measured completion, which is the last performance sample the
/// worker wrote (the warmup sample precedes it), so JIT warmup never enters the
/// reported values. A budget-exhausted completion never wrote a sample, so its
/// rates are zero and the caller records the partial evidence instead.
#[cfg(test)]
#[path = "completion_tests.rs"]
mod completion_tests;
