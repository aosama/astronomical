//! Reads and validates the durable performance evidence emitted by the journey.

use std::{fs, path::Path};

use serde_json::{Value, json};

use super::{
    LOG_MARKER,
    observe::ObservedRequestOutcome,
    support::{interaction_instance_paths, qualification_evidence_root, write_json_document},
};

/// Reads only the isolated worker reports produced by this acceptance journey.
pub(super) fn read_interaction_reports(isolated_worker_home: &Path) -> InteractionReports {
    let logging_directory = interaction_instance_paths(isolated_worker_home).logging_directory();
    InteractionReports {
        performance_records: read_json_lines(&logging_directory.join("performance.jsonl")),
        attribution_reports: read_json_lines(
            &logging_directory.join("performance-attribution.jsonl"),
        )
        .into_iter()
        .filter(|report| report["report_kind"] == "generation")
        .collect(),
    }
}

fn read_json_lines(log_path: &Path) -> Vec<Value> {
    fs::read_to_string(log_path)
        .unwrap_or_else(|read_error| {
            panic!("isolated JSONL report should be readable: {read_error}")
        })
        .lines()
        .map(|json_line| {
            serde_json::from_str(json_line).expect("each isolated report line should be valid JSON")
        })
        .collect()
}

/// Verifies memory bounds, cache reuse, truthful topology, and attribution.
pub(super) fn assert_reported_interaction(
    reports: &InteractionReports,
    cold_outcome: &ObservedRequestOutcome,
    append_outcome: &ObservedRequestOutcome,
    allocated_mlx_memory_bytes: u64,
) {
    assert!(
        cold_outcome.live_evidence.maximum_active_memory_bytes <= allocated_mlx_memory_bytes,
        "cold request active MLX memory must remain within the allocated ceiling"
    );
    assert!(
        append_outcome.live_evidence.maximum_active_memory_bytes <= allocated_mlx_memory_bytes,
        "append request active MLX memory must remain within the allocated ceiling"
    );
    assert_eq!(reports.performance_records.len(), 2);
    assert_eq!(reports.attribution_reports.len(), 2);
    // Throughput is evidence rather than a hardware-specific pass threshold.
    // Require both measurements to exist and remain mathematically usable so
    // every successful journey can report them on any supported laptop.
    for (request_label, performance_record) in [
        ("cold", &reports.performance_records[0]),
        ("append", &reports.performance_records[1]),
    ] {
        for throughput_field_name in ["prefill_tok_per_second", "generation_tok_per_second"] {
            let observed_tokens_per_second = performance_record[throughput_field_name]
                .as_f64()
                .unwrap_or(0.0);
            assert!(
                observed_tokens_per_second.is_finite() && observed_tokens_per_second > 0.0,
                "{request_label} {throughput_field_name} must be a positive finite measurement; observed {observed_tokens_per_second}"
            );
        }
    }
    let append_performance = &reports.performance_records[1];
    assert!(
        append_performance["cached_token_count"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    assert!(
        append_performance["time_to_first_output_millis"]
            .as_u64()
            .is_some()
    );
    assert!(
        append_performance["first_decode_forward_elapsed_millis"]
            .as_u64()
            .is_some(),
        "the performance record must isolate the first decode forward from total time to output"
    );
    assert_eq!(
        append_performance["generation_preparation_expert_source_read_byte_count"], 0,
        "the raw supervisor record must explicitly prove preparation performed no eager reads"
    );
    let append_attribution = &reports.attribution_reports[1];
    assert_eq!(
        attribution_counter(
            append_attribution,
            "generation_preparation_expert_source_read_byte_count"
        ),
        0,
        "generation preparation must not reread expert weights solely to warm RAM"
    );
    let cold_prefill_chunk_count =
        attribution_counter(&reports.attribution_reports[0], "prefill_chunck_count");
    let append_prefill_chunk_count =
        attribution_counter(append_attribution, "prefill_chunck_count");
    assert!(
        append_prefill_chunk_count >= 2,
        "the cached suffix must span at least two prompt-processing chunks so a streamed tail can reread: chunks={append_prefill_chunk_count}"
    );
    assert_prefill_stream_counts_stay_seated_or_fully_streamed(
        &reports.attribution_reports[0],
        cold_prefill_chunk_count,
    );
    assert_prefill_stream_counts_stay_seated_or_fully_streamed(
        append_attribution,
        append_prefill_chunk_count,
    );
    let preserved_expert_bytes = attribution_counter(
        append_attribution,
        "expert_topology_preserved_payload_byte_count",
    );
    assert!(
        preserved_expert_bytes > 0,
        "the append-only request must preserve useful expert topology"
    );
    assert!(
        cold_outcome.first_output_elapsed_millis.is_some()
            || matches!(cold_outcome.finish_reason.as_deref(), Some("tool_calls")),
        "cold request must publish visible text or finish with tool calls"
    );
    assert!(
        append_outcome.first_output_elapsed_millis.is_some()
            || matches!(append_outcome.finish_reason.as_deref(), Some("tool_calls")),
        "append request must publish visible text or finish with tool calls"
    );
    assert!(
        reports.performance_records[0]["mlx_peak_memory_bytes"]
            .as_u64()
            .unwrap_or(0)
            >= cold_outcome.live_evidence.maximum_peak_memory_bytes,
        "the performance record must retain the request maximum rather than overwrite it with final memory"
    );
    assert!(
        append_performance["mlx_peak_memory_bytes"]
            .as_u64()
            .unwrap_or(0)
            >= append_outcome.live_evidence.maximum_peak_memory_bytes,
        "the append performance record must retain the request maximum peak"
    );
    let final_residency = &append_outcome.live_evidence.final_status["expert_residency"];
    let resident_expert_count = final_residency["resident_expert_count"]
        .as_u64()
        .unwrap_or(0);
    let resident_expert_payload_bytes = final_residency["resident_expert_payload_bytes"]
        .as_u64()
        .unwrap_or(0);
    assert!(
        resident_expert_count > 0 && resident_expert_payload_bytes > 0,
        "decode must retain routed experts after prefill"
    );
    assert_eq!(
        append_performance["final_resident_expert_payload_bytes"],
        final_residency["resident_expert_payload_bytes"]
    );
}

fn assert_prefill_stream_counts_stay_seated_or_fully_streamed(
    generation_report: &Value,
    prefill_chunk_count: u64,
) {
    for source_summary in generation_report["expert_streaming_source_summaries"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|source_summary| source_summary["phase"] == "prefill")
    {
        let layer_index = source_summary["layer_index"].as_u64().unwrap_or(u64::MAX);
        let source_plan_count = source_summary["source_plan_count"].as_u64().unwrap_or(0);
        assert!(
            source_plan_count == 1 || source_plan_count == prefill_chunk_count,
            "a seated complete layer must remain seated for the rest of prefill; mixed reread means leftover-budget refresh evicted it: layer={layer_index} source_plan_count={source_plan_count} prefill_chunks={prefill_chunk_count}"
        );
    }
}

fn prefill_same_request_reread_bytes(generation_report: &Value) -> u64 {
    generation_report["expert_streaming_source_summaries"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|source_summary| source_summary["phase"] == "prefill")
        .map(|source_summary| {
            let source_plan_count = source_summary["source_plan_count"].as_u64().unwrap_or(0);
            let payload_byte_count = source_summary["payload_byte_count"].as_u64().unwrap_or(0);
            if source_plan_count <= 1 {
                0
            } else {
                payload_byte_count - payload_byte_count / source_plan_count
            }
        })
        .sum()
}

fn attribution_counter(attribution_report: &Value, counter_name: &str) -> u64 {
    attribution_report["counters"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|counter| counter["counter"] == counter_name)
        .and_then(|counter| counter["amount"].as_u64())
        .unwrap_or(0)
}

pub(super) fn persist_prefill_throughput_summary(
    reports: &InteractionReports,
    artifact_payload_bytes: u64,
    allocated_mlx_memory_bytes: u64,
) {
    let current_summary = json!({
        "artifact_payload_bytes": artifact_payload_bytes,
        "allocated_mlx_memory_bytes": allocated_mlx_memory_bytes,
        "cold_prefill_tokens_per_second": reports.performance_records[0]["prefill_tok_per_second"],
        "cold_prefill_elapsed_millis": reports.performance_records[0]["prefill_elapsed_millis"],
        "append_prefill_tokens_per_second": reports.performance_records[1]["prefill_tok_per_second"],
        "append_prefill_elapsed_millis": reports.performance_records[1]["prefill_elapsed_millis"],
        "append_time_to_first_output_millis": reports.performance_records[1]["time_to_first_output_millis"],
        "append_prefill_reread_bytes": prefill_same_request_reread_bytes(
            &reports.attribution_reports[1]
        ),
    });
    let evidence_root = qualification_evidence_root();
    let previous_summary_path = evidence_root.join("latest-summary.json");
    if let Ok(previous_summary_bytes) = fs::read_to_string(&previous_summary_path) {
        if let Ok(previous_summary) = serde_json::from_str::<Value>(&previous_summary_bytes) {
            let previous_append = previous_summary["append_prefill_tokens_per_second"]
                .as_f64()
                .unwrap_or(0.0);
            let current_append = current_summary["append_prefill_tokens_per_second"]
                .as_f64()
                .unwrap_or(0.0);
            eprintln!(
                "{LOG_MARKER} request=throughput-compare status=previous-vs-current previous_append_prefill_tokens_per_second={previous_append:.2} current_append_prefill_tokens_per_second={current_append:.2} previous_append_prefill_millis={} current_append_prefill_millis={} previous_append_reread_bytes={} current_append_reread_bytes={}",
                previous_summary["append_prefill_elapsed_millis"]
                    .as_u64()
                    .unwrap_or(0),
                current_summary["append_prefill_elapsed_millis"]
                    .as_u64()
                    .unwrap_or(0),
                previous_summary["append_prefill_reread_bytes"]
                    .as_u64()
                    .unwrap_or(0),
                current_summary["append_prefill_reread_bytes"]
                    .as_u64()
                    .unwrap_or(0),
            );
        }
    } else {
        eprintln!(
            "{LOG_MARKER} request=throughput-compare status=first-measurement append_prefill_tokens_per_second={:.2}",
            current_summary["append_prefill_tokens_per_second"]
                .as_f64()
                .unwrap_or(0.0),
        );
    }
    write_json_document(&previous_summary_path, &current_summary);
}

/// Prints compact cold/append evidence while the test output is still live.
pub(super) fn print_comparison_summary(
    reports: &InteractionReports,
    cold_outcome: &ObservedRequestOutcome,
    append_outcome: &ObservedRequestOutcome,
) {
    for (request_label, performance_record, attribution_report) in [
        (
            "cold",
            &reports.performance_records[0],
            &reports.attribution_reports[0],
        ),
        (
            "append",
            &reports.performance_records[1],
            &reports.attribution_reports[1],
        ),
    ] {
        eprintln!(
            "{LOG_MARKER} request={request_label} status=summary prompt_tokens={} cached_tokens={} generated_tokens={} prefill_tokens_per_second={:.2} generation_tokens_per_second={:.2} total_millis={} preparation_millis={} first_decode_millis={} first_output_millis={} logical_expert_read_bytes={} prefill_reread_bytes={} preserved_expert_bytes={} promoted_complete_bytes={} promoted_partial_bytes={} retired_expert_bytes={} peak_gb={:.3}",
            performance_record["prompt_token_count"]
                .as_u64()
                .unwrap_or(0),
            performance_record["cached_token_count"]
                .as_u64()
                .unwrap_or(0),
            performance_record["generated_token_count"]
                .as_u64()
                .unwrap_or(0),
            performance_record["prefill_tok_per_second"]
                .as_f64()
                .unwrap_or(0.0),
            performance_record["generation_tok_per_second"]
                .as_f64()
                .unwrap_or(0.0),
            performance_record["total_elapsed_millis"]
                .as_u64()
                .unwrap_or(0),
            performance_record["generation_preparation_elapsed_millis"]
                .as_u64()
                .unwrap_or(0),
            performance_record["first_decode_forward_elapsed_millis"]
                .as_u64()
                .unwrap_or(0),
            performance_record["time_to_first_output_millis"]
                .as_u64()
                .unwrap_or(0),
            attribution_counter(
                attribution_report,
                "rust_expert_streaming_payload_byte_count"
            ),
            prefill_same_request_reread_bytes(attribution_report),
            attribution_counter(
                attribution_report,
                "expert_topology_preserved_payload_byte_count"
            ),
            attribution_counter(
                attribution_report,
                "mandatory_prefill_complete_layer_promoted_payload_byte_count"
            ),
            attribution_counter(
                attribution_report,
                "mandatory_decode_routed_page_promoted_payload_byte_count"
            ),
            attribution_counter(
                attribution_report,
                "expert_topology_retired_payload_byte_count"
            ),
            performance_record["mlx_peak_memory_bytes"]
                .as_u64()
                .unwrap_or(0) as f64
                / 1_000_000_000.0,
        );
    }
    eprintln!(
        "{LOG_MARKER} request=comparison status=success cold_first_output_millis={} append_first_output_millis={} cold_generated_tokens={} append_generated_tokens={} append_resident_experts={}",
        cold_outcome.first_output_elapsed_millis.unwrap_or(0),
        append_outcome.first_output_elapsed_millis.unwrap_or(0),
        cold_outcome.generated_token_count,
        append_outcome.generated_token_count,
        append_outcome.live_evidence.final_status["expert_residency"]["resident_expert_count"]
            .as_u64()
            .unwrap_or(0),
    );
}

pub(super) struct InteractionReports {
    pub(super) performance_records: Vec<Value>,
    pub(super) attribution_reports: Vec<Value>,
}
