//! Evidence preservation and attribution reads for the paging memory-shape
//! experiment journey, so a run's measurements survive as comparable artifacts
//! instead of living only in test stdout.

use std::{fs, path::Path};

use serde_json::Value;

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

pub(super) fn attribution_reports(isolated_worker_home: &Path) -> Vec<Value> {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    fs::read_to_string(attribution_log_path)
        .expect("the completed request should flush performance attribution")
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .collect()
}

/// Prints the complete segment table for one attribution report: every
/// measured operation with its total, per-occurrence, and maximum elapsed
/// time, plus the report's attributed share, so a paging run's cost is
/// attributed to code segments instead of guessed.
pub(super) fn print_attribution_report(report: &Value) {
    let report_kind = report["report_kind"].as_str().unwrap_or("unknown");
    let report_elapsed_seconds =
        report["report_elapsed_nanoseconds"].as_u64().unwrap_or(0) as f64 / 1_000_000_000.0;
    let attributed_percent = report["attributed_percent"].as_f64().unwrap_or(0.0);
    eprintln!(
        "[paging-memory-shape] attribution report={report_kind} wall_seconds={report_elapsed_seconds:.3} attributed_percent={attributed_percent:.1}"
    );
    let mut operations = report["operations"].as_array().cloned().unwrap_or_default();
    operations.sort_by(|left, right| {
        let left_elapsed = left["total_elapsed_nanoseconds"].as_u64().unwrap_or(0);
        let right_elapsed = right["total_elapsed_nanoseconds"].as_u64().unwrap_or(0);
        right_elapsed.cmp(&left_elapsed)
    });
    eprintln!(
        "[paging-memory-shape] attribution {report_kind} operations: total_s avg_ms max_ms occ first_offset_s last_offset_s name"
    );
    for operation in &operations {
        let total_elapsed_nanoseconds =
            operation["total_elapsed_nanoseconds"].as_u64().unwrap_or(0);
        let occurrence_count = operation["occurrence_count"].as_u64().unwrap_or(1);
        eprintln!(
            "[paging-memory-shape] attribution {report_kind} op total_s={:.3} avg_ms={:.3} max_ms={:.3} occ={occurrence_count} first_offset_s={:.3} last_offset_s={:.3} wall_percent={:.1} name={}",
            total_elapsed_nanoseconds as f64 / 1_000_000_000.0,
            total_elapsed_nanoseconds as f64 / occurrence_count.max(1) as f64 / 1_000_000.0,
            operation["maximum_elapsed_nanoseconds"]
                .as_u64()
                .unwrap_or(0) as f64
                / 1_000_000.0,
            operation["first_started_offset_nanoseconds"]
                .as_u64()
                .unwrap_or(0) as f64
                / 1_000_000_000.0,
            operation["last_ended_offset_nanoseconds"]
                .as_u64()
                .unwrap_or(0) as f64
                / 1_000_000_000.0,
            if report_elapsed_seconds > 0.0 {
                total_elapsed_nanoseconds as f64 / 1_000_000_000.0 / report_elapsed_seconds * 100.0
            } else {
                0.0
            },
            operation["operation"].as_str().unwrap_or("unknown"),
        );
    }
    for counter in report["counters"].as_array().unwrap_or(&Vec::new()) {
        eprintln!(
            "[paging-memory-shape] attribution {report_kind} counter {}={}",
            counter["counter"].as_str().unwrap_or("unknown"),
            counter["amount"].as_u64().unwrap_or(0)
        );
    }
}

pub(super) fn preserve_experiment_evidence(
    isolated_worker_home: &Path,
    evidence_document: &Value,
) -> std::path::PathBuf {
    let unix_millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|unix_time| unix_time.as_millis())
        .unwrap_or_default();
    let evidence_directory = std::env::current_dir()
        .expect("the journey should resolve its working directory")
        .join("target/acceptance-evidence/large-sparse-moe-paging-memory-shape")
        .join(unix_millis.to_string());
    fs::create_dir_all(&evidence_directory)
        .expect("the acceptance evidence directory should be created");
    fs::write(
        evidence_directory.join("paging-memory-shape.json"),
        serde_json::to_string_pretty(evidence_document).expect("evidence should serialize"),
    )
    .expect("the evidence document should be written");
    let logging_directory = isolated_worker_home.join(".astronomical-dev").join("logs");
    if let Ok(logging_entries) = fs::read_dir(&logging_directory) {
        for logging_entry in logging_entries.flatten() {
            let source_path = logging_entry.path();
            if source_path.is_file()
                && let Some(source_name) = source_path.file_name()
            {
                let _ = fs::copy(&source_path, evidence_directory.join(source_name));
            }
        }
    }
    evidence_directory
}
