use std::path::Path;
use std::time::Duration;

use serde_json::Value;

use super::*;

const TEST_TIMEOUT: Duration = Duration::from_secs(10);

#[tokio::test]
async fn should_select_measured_generation_report_after_warmup_report() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let logging_directory =
            tempfile::tempdir().expect("the attribution log directory should be created");
        write_generation_reports(
            logging_directory.path(),
            &[
                generation_report(1, 557),
                generation_report(throughput_support::MEASURED_REQUEST_ID, 10_544),
            ],
        );

        let measured_report = read_generation_attribution_report(
            logging_directory.path(),
            throughput_support::MEASURED_REQUEST_ID,
        )
        .expect("the measured generation report should be present");

        assert_eq!(
            measured_report["request_id"].as_u64(),
            Some(throughput_support::MEASURED_REQUEST_ID),
        );
        assert_eq!(
            attribution_counter(&measured_report, "prompt_token_count"),
            10_544,
        );
    })
    .await
    .expect("the measured-report selection test should finish within ten seconds");
}

#[tokio::test]
async fn should_not_use_warmup_report_when_measured_report_is_missing() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let logging_directory =
            tempfile::tempdir().expect("the attribution log directory should be created");
        write_generation_reports(logging_directory.path(), &[generation_report(1, 557)]);

        assert!(
            read_generation_attribution_report(
                logging_directory.path(),
                throughput_support::MEASURED_REQUEST_ID,
            )
            .is_none(),
            "warmup evidence must not stand in for a missing measured report",
        );
    })
    .await
    .expect("the missing-measured-report test should finish within ten seconds");
}

fn generation_report(request_id: u64, prompt_token_count: u64) -> Value {
    serde_json::json!({
        "report_kind": "generation",
        "request_id": request_id,
        "counters": [{
            "counter": "prompt_token_count",
            "amount": prompt_token_count
        }]
    })
}

fn write_generation_reports(logging_directory: &Path, reports: &[Value]) {
    let report_lines = reports
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(
        logging_directory.join("performance-attribution.jsonl"),
        report_lines,
    )
    .expect("the attribution reports should be written");
}
