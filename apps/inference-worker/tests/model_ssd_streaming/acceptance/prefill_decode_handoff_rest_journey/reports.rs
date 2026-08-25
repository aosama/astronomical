//! Reads and validates the durable performance evidence emitted by the journey.

use std::{fs, path::Path};

use serde_json::Value;

use super::{
    LOG_MARKER, MAXIMUM_MLX_MEMORY_BYTES, ObservedRequestOutcome,
    support::interaction_instance_paths,
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
) {
    assert!(
        cold_outcome.live_evidence.maximum_active_memory_bytes <= MAXIMUM_MLX_MEMORY_BYTES,
        "cold request active MLX memory must remain within the configured ceiling"
    );
    assert!(
        append_outcome.live_evidence.maximum_active_memory_bytes <= MAXIMUM_MLX_MEMORY_BYTES,
        "append request active MLX memory must remain within the configured ceiling"
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
    let preserved_expert_bytes = attribution_counter(
        append_attribution,
        "expert_topology_preserved_payload_byte_count",
    );
    assert!(
        preserved_expert_bytes > 0,
        "the append-only request must preserve useful expert topology"
    );
    assert!(cold_outcome.first_output_elapsed_millis.is_some());
    assert!(append_outcome.first_output_elapsed_millis.is_some());
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
    let total_layer_count = final_residency["total_layer_count"].as_u64().unwrap_or(0);
    let complete_layer_count = final_residency["complete_layer_count"]
        .as_u64()
        .unwrap_or(0);
    let partial_layer_count = final_residency["partial_layer_count"].as_u64().unwrap_or(0);
    assert!(
        complete_layer_count > 0,
        "prefill must retain a complete-layer foundation"
    );
    assert!(
        partial_layer_count > 0,
        "decode must retain a routed-page overlay"
    );
    assert!(
        complete_layer_count + partial_layer_count <= total_layer_count,
        "retained ownership classes must not exceed the model's sparse and MTP layer inventory"
    );
    assert_eq!(
        append_performance["final_complete_expert_payload_bytes"],
        final_residency["complete_layer_payload_bytes"]
    );
    assert_eq!(
        append_performance["final_partial_expert_payload_bytes"],
        final_residency["partial_layer_payload_bytes"]
    );
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
            "{LOG_MARKER} request={request_label} status=summary prompt_tokens={} cached_tokens={} generated_tokens={} prefill_tokens_per_second={:.2} generation_tokens_per_second={:.2} total_millis={} preparation_millis={} first_decode_millis={} first_output_millis={} logical_expert_read_bytes={} preserved_expert_bytes={} promoted_complete_bytes={} promoted_partial_bytes={} retired_expert_bytes={} peak_gb={:.3}",
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
        "{LOG_MARKER} request=comparison status=success cold_first_output_millis={} append_first_output_millis={} cold_generated_tokens={} append_generated_tokens={} append_complete_layers={} append_partial_layers={}",
        cold_outcome.first_output_elapsed_millis.unwrap_or(0),
        append_outcome.first_output_elapsed_millis.unwrap_or(0),
        cold_outcome.generated_token_count,
        append_outcome.generated_token_count,
        append_outcome.live_evidence.final_status["expert_residency"]["complete_layer_count"]
            .as_u64()
            .unwrap_or(0),
        append_outcome.live_evidence.final_status["expert_residency"]["partial_layer_count"]
            .as_u64()
            .unwrap_or(0),
    );
}

pub(super) struct InteractionReports {
    performance_records: Vec<Value>,
    attribution_reports: Vec<Value>,
}
