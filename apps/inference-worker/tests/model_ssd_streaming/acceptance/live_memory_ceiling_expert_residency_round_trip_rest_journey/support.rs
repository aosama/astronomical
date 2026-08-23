//! Structural status, configuration, and attribution evidence for the round trip.

use std::{fs, net::SocketAddr, path::Path};

use astronomical_config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep};

use crate::common::real_model_rest_server::get_json_endpoint;

use super::{LOG_MARKER, MAXIMUM_OUTPUT_TOKEN_COUNT};

const STATUS_POLL_INTERVAL: Duration = Duration::from_millis(100);
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);

pub(super) fn write_round_trip_config(
    isolated_worker_home: &Path,
    model_directory: &Path,
    initial_mlx_memory_ceiling_bytes: u64,
) {
    let round_trip_instance_paths = instance_paths(isolated_worker_home);
    let configuration_directory = round_trip_instance_paths.state_directory();
    fs::create_dir(&configuration_directory)
        .expect("the live-ceiling round-trip configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": initial_mlx_memory_ceiling_bytes / 1_000_000_000,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": false,
        "performance_attribution_enabled": true,
        "logging": {"level": "debug", "retained_files": 2},
        "chunking": {"fixed_prompt_processing_chunk_size_tokens": 1_024},
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the live-ceiling round-trip configuration should serialize"),
    )
    .expect("the live-ceiling round-trip configuration should be written");
}

pub(super) async fn wait_for_idle_status(server_address: SocketAddr, request_label: &str) -> Value {
    let wait_started_at = Instant::now();
    let mut last_log_at = wait_started_at - STATUS_LOG_INTERVAL;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        if last_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            eprintln!(
                "{LOG_MARKER} phase={request_label} status=progress elapsed_seconds={:.3} activity={} expert_memory_mode={} active_gb={:.3}",
                wait_started_at.elapsed().as_secs_f64(),
                status_document["activity"].as_str().unwrap_or("unknown"),
                status_document["expert_memory_mode"]
                    .as_str()
                    .unwrap_or("unavailable"),
                status_document["mlx_memory_snapshot"]["active_memory_bytes"]
                    .as_u64()
                    .unwrap_or(0) as f64
                    / 1_000_000_000.0,
            );
            last_log_at = Instant::now();
        }
        if status_document["activity"] == "idle" {
            return status_document;
        }
        sleep(STATUS_POLL_INTERVAL).await;
    }
}

pub(super) fn assert_machine_supports_round_trip(status_document: &Value, raised_ceiling: u64) {
    let machine_ceiling = required_u64(status_document, "machine_mlx_memory_ceiling_bytes");
    assert!(
        machine_ceiling >= raised_ceiling,
        "this qualification cell requires a machine MLX ceiling of at least {:.0} GB; worker reported {:.3} GB",
        raised_ceiling as f64 / 1_000_000_000.0,
        machine_ceiling as f64 / 1_000_000_000.0,
    );
}

pub(super) fn assert_resident_status(status_document: &Value, expected_ceiling_bytes: u64) {
    assert_common_idle_status(status_document, expected_ceiling_bytes);
    assert_eq!(status_document["expert_memory_mode"], "resident");
    let expert_residency = &status_document["expert_residency"];
    let total_layer_count = required_u64(expert_residency, "total_layer_count");
    let complete_layer_count = required_u64(expert_residency, "complete_layer_count");
    let complete_payload_bytes = required_u64(expert_residency, "complete_layer_payload_bytes");
    assert!(total_layer_count > 0);
    assert_eq!(complete_layer_count, total_layer_count);
    assert!(complete_payload_bytes > 0);
    assert_eq!(required_u64(expert_residency, "partial_layer_count"), 0);
    assert_eq!(
        required_u64(expert_residency, "partial_layer_payload_bytes"),
        0
    );
    assert_eq!(
        memory_bytes(status_document, "expert_payload_bytes"),
        complete_payload_bytes
    );
}

pub(super) fn assert_streaming_status(status_document: &Value, expected_ceiling_bytes: u64) {
    assert_common_idle_status(status_document, expected_ceiling_bytes);
    assert_nonresident_topology(status_document);
    let expert_residency = &status_document["expert_residency"];
    let complete_layer_count = required_u64(expert_residency, "complete_layer_count");
    let partial_layer_count = required_u64(expert_residency, "partial_layer_count");
    assert!(
        complete_layer_count > 0 || partial_layer_count > 0,
        "a completed streaming request should retain structurally visible expert topology: {expert_residency}"
    );
}

pub(super) fn assert_nonresident_status_before_request(
    status_document: &Value,
    expected_ceiling_bytes: u64,
) {
    assert_common_idle_status(status_document, expected_ceiling_bytes);
    assert_nonresident_topology(status_document);
}

fn assert_nonresident_topology(status_document: &Value) {
    assert!(matches!(
        status_document["expert_memory_mode"].as_str(),
        Some("paged" | "hybrid")
    ));
    let expert_residency = &status_document["expert_residency"];
    let total_layer_count = required_u64(expert_residency, "total_layer_count");
    let complete_layer_count = required_u64(expert_residency, "complete_layer_count");
    let partial_layer_count = required_u64(expert_residency, "partial_layer_count");
    assert!(total_layer_count > 0);
    assert!(
        complete_layer_count < total_layer_count || partial_layer_count > 0,
        "streaming mode must not publish complete contiguous ownership: {expert_residency}"
    );
}

fn assert_common_idle_status(status_document: &Value, expected_ceiling_bytes: u64) {
    assert_eq!(status_document["status"], "ready");
    assert_eq!(status_document["activity"], "idle");
    assert_eq!(
        required_u64(status_document, "mlx_memory_ceiling_bytes"),
        expected_ceiling_bytes
    );
    assert!(status_document["pending_mlx_memory_ceiling_bytes"].is_null());
    assert!(status_document["mlx_memory_limit_error"].is_null());
    assert!(
        memory_bytes(status_document, "active_memory_bytes") <= expected_ceiling_bytes,
        "idle active memory must remain within the applied MLX ceiling: {status_document}"
    );
}

pub(super) fn read_generation_evidence(
    isolated_worker_home: &Path,
    applied_ceilings_by_generation: &[u64],
) -> Vec<GenerationEvidence> {
    let logging_directory = instance_paths(isolated_worker_home).logging_directory();
    let attribution_reports =
        read_json_lines(&logging_directory.join("performance-attribution.jsonl"))
            .into_iter()
            .filter(|report| report["report_kind"] == "generation")
            .collect::<Vec<_>>();
    let performance_records = read_json_lines(&logging_directory.join("performance.jsonl"));
    assert_eq!(
        attribution_reports.len(),
        applied_ceilings_by_generation.len()
    );
    assert_eq!(
        performance_records.len(),
        applied_ceilings_by_generation.len()
    );
    attribution_reports
        .iter()
        .zip(performance_records.iter())
        .zip(applied_ceilings_by_generation.iter().copied())
        .map(|((attribution_report, performance_record), applied_ceiling_bytes)| {
            let active_memory_bytes = required_u64(attribution_report, "mlx_active_memory_bytes");
            let peak_memory_bytes = required_u64(attribution_report, "mlx_peak_memory_bytes");
            let ceiling_tolerance_bytes = applied_ceiling_bytes / 100;
            assert!(active_memory_bytes <= applied_ceiling_bytes);
            assert!(
                peak_memory_bytes
                    <= applied_ceiling_bytes.saturating_add(ceiling_tolerance_bytes),
                "request peak {peak_memory_bytes} must remain within the applied ceiling plus derived tolerance {}",
                applied_ceiling_bytes.saturating_add(ceiling_tolerance_bytes),
            );
            GenerationEvidence {
                model_id: attribution_report["model_id"]
                    .as_str()
                    .expect("generation attribution should report model identity")
                    .to_owned(),
                model_revision: attribution_report["model_revision"]
                    .as_str()
                    .expect("generation attribution should report model revision")
                    .to_owned(),
                prompt_token_count: required_u64(performance_record, "prompt_token_count"),
                expert_source_read_bytes: attribution_counter(
                    attribution_report,
                    "positional_file_read_byte_count",
                ),
            }
        })
        .collect()
}

fn attribution_counter(attribution_report: &Value, counter_name: &str) -> u64 {
    attribution_report["counters"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|counter| counter["counter"] == counter_name)
        .filter_map(|counter| counter["amount"].as_u64())
        .sum()
}

fn read_json_lines(log_path: &Path) -> Vec<Value> {
    fs::read_to_string(log_path)
        .unwrap_or_else(|read_error| {
            panic!("{} should be readable: {read_error}", log_path.display())
        })
        .lines()
        .map(|json_line| {
            serde_json::from_str(json_line).expect("each isolated report row should be valid JSON")
        })
        .collect()
}

fn instance_paths(isolated_worker_home: &Path) -> AstronomicalInstancePaths {
    AstronomicalInstancePaths::for_home_directory(
        isolated_worker_home,
        AstronomicalRuntimeInstance::Development,
    )
}

fn memory_bytes(status_document: &Value, field_name: &str) -> u64 {
    required_u64(&status_document["mlx_memory_snapshot"], field_name)
}

fn required_u64(document: &Value, field_name: &str) -> u64 {
    document[field_name]
        .as_u64()
        .unwrap_or_else(|| panic!("{field_name} should be a u64 in {document}"))
}

pub(super) struct GenerationEvidence {
    pub(super) model_id: String,
    pub(super) model_revision: String,
    pub(super) prompt_token_count: u64,
    pub(super) expert_source_read_bytes: u64,
}
