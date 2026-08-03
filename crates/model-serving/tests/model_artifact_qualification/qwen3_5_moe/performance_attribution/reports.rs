use serde_json::Value;
use std::{fs, path::Path};

pub(crate) fn read_attribution_report_documents(
    performance_attribution_log_path: &Path,
) -> Vec<Value> {
    fs::read_to_string(performance_attribution_log_path)
        .expect("the benchmark should write JSON Lines reports")
        .lines()
        .map(|json_line| {
            serde_json::from_str(json_line)
                .expect("each benchmark JSON Lines record should be valid JSON")
        })
        .collect()
}
pub(crate) fn generation_report_for_request(
    attribution_report_documents: &[Value],
    request_id: u64,
) -> &Value {
    attribution_report_documents
        .iter()
        .find(|document| {
            document["report_kind"] == "generation" && document["request_id"] == request_id
        })
        .unwrap_or_else(|| {
            panic!("the benchmark should write a generation report for request {request_id}")
        })
}
pub(super) fn model_loading_report(attribution_report_documents: &[Value]) -> &Value {
    attribution_report_documents
        .iter()
        .find(|document| document["report_kind"] == "model_loading")
        .unwrap_or_else(|| panic!("the benchmark should write a model-loading report"))
}
pub(crate) fn counter_amount(generation_report: &Value, counter_identifier: &str) -> u64 {
    generation_report["counters"]
        .as_array()
        .and_then(|reports| {
            reports.iter().find_map(|report| {
                (report["counter"] == counter_identifier)
                    .then(|| report["amount"].as_u64())
                    .flatten()
            })
        })
        .unwrap_or(0)
}
pub(crate) fn operation_total_elapsed_nanoseconds(
    report: &Value,
    operation_identifier: &str,
) -> u64 {
    report["operations"]
        .as_array()
        .and_then(|reports| {
            reports.iter().find_map(|operation| {
                (operation["operation"] == operation_identifier)
                    .then(|| operation["total_elapsed_nanoseconds"].as_u64())
                    .flatten()
            })
        })
        .unwrap_or(0)
}
