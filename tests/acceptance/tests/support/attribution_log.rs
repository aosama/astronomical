//! Performance-attribution log readers shared by the expert-paging journeys:
//! one durable JSONL log per isolated worker home, one reader per shape.

use std::path::Path;

/// Sums one named counter across every generation attribution report in the
/// isolated worker home's performance-attribution log.
pub(crate) fn generation_attribution_counter(
    isolated_worker_home: &Path,
    counter_identifier: &str,
) -> u64 {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let attribution_log_text = std::fs::read_to_string(attribution_log_path)
        .expect("the completed request should flush performance attribution");
    attribution_log_text
        .lines()
        .filter_map(|json_line| serde_json::from_str::<serde_json::Value>(json_line).ok())
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
