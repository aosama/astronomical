//! Evidence preservation and status observation for the decode expert reuse
//! journey (issue #510): durable utilization evidence plus the status polling
//! helpers the journey reads the user-visible memory numbers through.

use std::{fs, path::Path};

use serde_json::Value;

pub(super) fn preserve_memory_utilization_evidence(
    isolated_worker_home: &Path,
    final_status: &Value,
    counters: &[(&str, u64)],
) {
    let unix_millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|unix_time| unix_time.as_millis())
        .unwrap_or_default();
    let evidence_directory = std::env::current_dir()
        .expect("the journey should resolve its working directory")
        .join("target/acceptance-evidence/ssd-paging-decode-expert-reuse")
        .join(unix_millis.to_string());
    fs::create_dir_all(&evidence_directory)
        .expect("the acceptance evidence directory should be created");
    let mut counter_document = serde_json::Map::new();
    for (counter_identifier, counter_amount) in counters {
        counter_document.insert(
            (*counter_identifier).to_owned(),
            serde_json::json!(counter_amount),
        );
    }
    let evidence_document = serde_json::json!({
        "attribution_counters": counter_document,
        "status_memory_ceiling_utilization": final_status["mlx_memory_snapshot"]["memory_ceiling_utilization"],
        "status_mlx_memory_snapshot": final_status["mlx_memory_snapshot"],
    });
    fs::write(
        evidence_directory.join("memory-utilization.json"),
        serde_json::to_string_pretty(&evidence_document).expect("evidence should serialize"),
    )
    .expect("the utilization evidence should be written");
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

pub(super) fn generation_attribution_counter(
    isolated_worker_home: &Path,
    counter_identifier: &str,
) -> u64 {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    fs::read_to_string(attribution_log_path)
        .expect("the completed request should flush performance attribution")
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .flat_map(|attribution_report| {
            attribution_report["counters"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter(|counter_report| counter_report["counter"] == counter_identifier)
        .filter_map(|counter_report| counter_report["amount"].as_u64())
        .sum()
}

pub(super) fn generation_attribution_report_count(isolated_worker_home: &Path) -> usize {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    fs::read_to_string(attribution_log_path)
        .expect("the completed request should flush performance attribution")
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .count()
}

pub(super) fn decode_streamed_layer_indices(
    isolated_worker_home: &Path,
) -> std::collections::BTreeSet<usize> {
    isolated_worker_log_lines(isolated_worker_home)
        .into_iter()
        .filter(|log_line| {
            log_line.contains("Rust expert layer streaming completed")
                && !log_line.contains("streamed_expert_count=256")
        })
        .filter_map(|log_line| {
            log_line
                .split_whitespace()
                .find_map(|field| field.strip_prefix("layer_index="))
                .and_then(|layer_index| layer_index.parse::<usize>().ok())
        })
        .collect()
}

pub(super) fn generation_expert_source_read_bytes(isolated_worker_home: &Path) -> u64 {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let attribution_log = fs::read_to_string(attribution_log_path)
        .expect("the paging acceptance journey should write performance attribution");
    attribution_log
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .filter_map(|attribution_report| {
            attribution_report["counters"]
                .as_array()
                .map(|counters| counters.to_owned())
        })
        .flatten()
        .filter(|counter_report| counter_report["counter"] == "positional_file_read_byte_count")
        .filter_map(|counter_report| counter_report["amount"].as_u64())
        .sum()
}

pub(super) fn isolated_worker_log_lines(isolated_worker_home: &Path) -> Vec<String> {
    let logging_directory = isolated_worker_home.join(".astronomical-dev").join("logs");
    let logging_entries = fs::read_dir(logging_directory)
        .expect("the paging acceptance journey should create its logging directory");
    let mut log_lines = Vec::new();
    for logging_entry in logging_entries {
        let log_path = logging_entry
            .expect("the isolated log entry should be readable")
            .path();
        if log_path.is_file() {
            let log_content = fs::read_to_string(&log_path).unwrap_or_else(|log_read_error| {
                panic!(
                    "{} should be readable: {log_read_error}",
                    log_path.display()
                )
            });
            log_lines.extend(log_content.lines().map(str::to_owned));
        }
    }
    log_lines
}

pub(super) fn log_status_progress(status_document: &Value) {
    let phase = status_document["progress"]["phase"]
        .as_str()
        .unwrap_or("idle");
    let processed_tokens = status_document["progress"]["processed_tokens"]
        .as_u64()
        .unwrap_or(0);
    let total_tokens = status_document["progress"]["total_tokens"]
        .as_u64()
        .unwrap_or(0);
    let elapsed_millis = status_document["progress"]["elapsed_ms"]
        .as_u64()
        .unwrap_or(0);
    let observed_tokens_per_second = if elapsed_millis == 0 {
        0.0
    } else {
        processed_tokens as f64 * 1_000.0 / elapsed_millis as f64
    };
    let expert_payload_bytes = status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
        .as_u64()
        .unwrap_or(0);
    eprintln!(
        "[ssd-paging-decode-expert-reuse] status=progress phase={phase} processed_tokens={processed_tokens} total_tokens={total_tokens} elapsed_seconds={:.3} observed_tokens_per_second={observed_tokens_per_second:.2} expert_payload_bytes={expert_payload_bytes}",
        elapsed_millis as f64 / 1_000.0,
    );
}

pub(super) fn record_expert_payload_increase(
    status_document: &Value,
    retained_expert_payload_bytes: &mut Vec<u64>,
) {
    let expert_payload_bytes = status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
        .as_u64()
        .unwrap_or(0);
    let largest_recorded_expert_payload_bytes = retained_expert_payload_bytes
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if expert_payload_bytes > largest_recorded_expert_payload_bytes {
        retained_expert_payload_bytes.push(expert_payload_bytes);
        eprintln!(
            "[ssd-paging-decode-expert-reuse] status=progress processed_tokens={} expert_payload_bytes={expert_payload_bytes}",
            status_document["progress"]["processed_tokens"]
        );
    }
}
