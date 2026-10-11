use std::path::Path;
use std::time::Duration;

use astronomical_supervisor::ChatGenerationExecutor;
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

#[tokio::test]
async fn should_persist_the_stage_of_a_budget_exhausted_cell() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let logging_directory =
            tempfile::tempdir().expect("the attribution log directory should be created");
        let history_path = logging_directory.path().join("throughput-history.jsonl");
        let machine_specs = MachineSpecs::capture().await;
        let stage_expectations = [
            (
                Some(MemoryCellBudgetExhaustedStage::LaunchOrIdle),
                Some("launch_or_idle"),
            ),
            (Some(MemoryCellBudgetExhaustedStage::Warmup), Some("warmup")),
            (
                Some(MemoryCellBudgetExhaustedStage::Measured),
                Some("measured"),
            ),
            (None, None),
        ];

        for (budget_exhausted_stage, _) in stage_expectations {
            let budget_exhausted = budget_exhausted_stage.is_some();
            let run_outcome = journey_run_outcome_with_budget_stage(
                logging_directory.path(),
                budget_exhausted_stage,
            );
            let memory_cell = compose_memory_cell(
                &run_outcome,
                23_000_000_000,
                budget_exhausted,
                &Value::Object(serde_json::Map::new()),
            );
            let throughput_record = ThroughputRecord {
                timestamp: "test-timestamp".to_owned(),
                model_id: "test-model".to_owned(),
                journey: ThroughputJourneyKind::MemoryCeilingSweep,
                prefill_tokens_per_second: 0,
                decode_tokens_per_second: 0,
                git_commit: None,
                machine: machine_specs.clone(),
                total_input_tokens: 0,
                total_output_tokens: 0,
                cached_tokens: 0,
                prefill_time_seconds: 0.0,
                decode_time_seconds: 0.0,
                memory_cell: Some(memory_cell),
            };
            append_throughput_history(&throughput_record, &history_path)
                .expect("the memory-cell record should append to durable history");
        }

        let persisted_rows = fs::read_to_string(&history_path)
            .expect("the durable history should be readable")
            .lines()
            .map(|persisted_line| {
                serde_json::from_str::<Value>(persisted_line)
                    .expect("each durable history row should contain valid JSON")
            })
            .collect::<Vec<_>>();
        assert_eq!(persisted_rows.len(), stage_expectations.len());
        for (persisted_row, (expected_stage, stage_label)) in
            persisted_rows.iter().zip(stage_expectations)
        {
            let persisted_memory_cell = &persisted_row["memory_cell"];
            assert_eq!(
                persisted_memory_cell["budget_exhausted"].as_bool(),
                Some(expected_stage.is_some()),
                "the persisted record should preserve whether the serving budget expired",
            );
            assert_eq!(
                persisted_memory_cell
                    .get("budget_exhausted_stage")
                    .and_then(Value::as_str),
                stage_label,
                "the persisted record should include only the actual exhaustion stage",
            );
        }
    })
    .await
    .expect("the memory-cell stage persistence test should finish within ten seconds");
}

#[tokio::test]
async fn should_append_an_exhausted_cell_before_failing() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let logging_directory =
            tempfile::tempdir().expect("the attribution log directory should be created");
        let history_directory =
            tempfile::tempdir().expect("the durable history directory should be created");
        let history_path = history_directory.path().join("throughput-history.jsonl");
        let run_outcome = journey_run_outcome_with_budget_stage(
            logging_directory.path(),
            Some(MemoryCellBudgetExhaustedStage::LaunchOrIdle),
        );
        let append_history_path = history_path.clone();

        let append_result = tokio::spawn(async move {
            append_memory_cell_record(run_outcome, 23_000_000_000, 23, &append_history_path).await
        })
        .await;
        let append_failure =
            append_result.expect_err("an exhausted cell must fail after persisting its record");
        assert!(
            append_failure.is_panic(),
            "the exhausted-cell failure should remain visible to the test harness",
        );

        let persisted_row = fs::read_to_string(&history_path)
            .expect("the exhausted-cell history record should exist");
        let persisted_record = serde_json::from_str::<Value>(persisted_row.trim())
            .expect("the exhausted-cell history record should contain valid JSON");
        assert_eq!(
            persisted_record["memory_cell"]["budget_exhausted"].as_bool(),
            Some(true),
        );
        assert_eq!(
            persisted_record["memory_cell"]["budget_exhausted_stage"].as_str(),
            Some("launch_or_idle"),
        );
    })
    .await
    .expect("the exhausted-cell persistence test should finish within ten seconds");
}

fn journey_run_outcome_with_budget_stage(
    logging_directory: &Path,
    budget_exhausted_stage: Option<MemoryCellBudgetExhaustedStage>,
) -> JourneyRunOutcome {
    JourneyRunOutcome {
        measurement: throughput_support::ThroughputMeasurement {
            model_id: "test-model".to_owned(),
            total_input_tokens: 0,
            total_output_tokens: 0,
            cached_tokens: 0,
            prefill_time_seconds: 0.0,
            decode_time_seconds: 0.0,
            prefill_tokens_per_second: 0.0,
            decode_tokens_per_second: 0.0,
        },
        measured_budget_exhausted: budget_exhausted_stage.is_some(),
        budget_exhausted_stage,
        final_health_snapshot: astronomical_supervisor::WorkerHandle::unavailable()
            .worker_health_snapshot(),
        last_observed_active_request_progress: None,
        logging_directory: logging_directory.to_path_buf(),
    }
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
