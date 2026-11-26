use std::path::Path;
use std::time::Duration;

use astronomical_daemon::ChatGenerationExecutor;
use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;
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
async fn should_record_the_stage_of_a_budget_exhausted_cell() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let logging_directory =
            tempfile::tempdir().expect("the attribution log directory should be created");
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
            // Edge: a cell that finished names no stage at all.
            (None, None),
        ];

        for (budget_exhausted_stage, stage_label) in stage_expectations {
            let printed_record = throughput_record_json(
                &fixture_memory_cell_record(logging_directory.path(), budget_exhausted_stage).await,
            );
            let printed_memory_cell = &printed_record["memory_cell"];
            assert_eq!(
                printed_memory_cell["budget_exhausted"].as_bool(),
                Some(budget_exhausted_stage.is_some()),
                "the record should say whether the serving budget expired",
            );
            assert_eq!(
                printed_memory_cell
                    .get("budget_exhausted_stage")
                    .and_then(Value::as_str),
                stage_label,
                "the record should include only the actual exhaustion stage",
            );
            assert_eq!(
                printed_record["worker_build_identity"]["source_commit"].as_str(),
                Some("fixture-commit"),
            );
        }
    })
    .await
    .expect("the memory-cell stage test should finish within ten seconds");
}

#[tokio::test]
async fn should_fail_an_exhausted_cell_after_reporting_it() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let logging_directory =
            tempfile::tempdir().expect("the attribution log directory should be created");
        let exhausted_record = fixture_memory_cell_record(
            logging_directory.path(),
            Some(MemoryCellBudgetExhaustedStage::LaunchOrIdle),
        )
        .await;

        let report_failure =
            tokio::task::spawn_blocking(move || report_memory_cell_record(&exhausted_record, 23))
                .await
                .expect_err("an exhausted cell must fail after its record is printed");

        assert!(
            report_failure.is_panic(),
            "the exhausted-cell failure should remain visible to the test harness",
        );
    })
    .await
    .expect("the exhausted-cell test should finish within ten seconds");
}

#[tokio::test]
async fn should_pass_a_finished_cell_below_the_hybrid_ceilings() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let logging_directory =
            tempfile::tempdir().expect("the attribution log directory should be created");
        let finished_record = fixture_memory_cell_record(logging_directory.path(), None).await;

        report_memory_cell_record(&finished_record, 23);
    })
    .await
    .expect("the finished-cell test should finish within ten seconds");
}

#[tokio::test]
async fn should_fail_a_finished_cell_at_a_hybrid_ceiling_that_did_not_serve_hybrid() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let logging_directory =
            tempfile::tempdir().expect("the attribution log directory should be created");
        let finished_record = fixture_memory_cell_record(logging_directory.path(), None).await;

        let report_failure =
            tokio::task::spawn_blocking(move || report_memory_cell_record(&finished_record, 38))
                .await
                .expect_err("a 38 GB cell without hybrid expert residency must fail");

        assert!(report_failure.is_panic());
    })
    .await
    .expect("the hybrid-ceiling test should finish within ten seconds");
}

async fn fixture_memory_cell_record(
    logging_directory: &Path,
    budget_exhausted_stage: Option<MemoryCellBudgetExhaustedStage>,
) -> ThroughputRecord {
    let mut run_outcome =
        journey_run_outcome_with_budget_stage(logging_directory, budget_exhausted_stage);
    if budget_exhausted_stage.is_none() {
        // A finished cell is held to the measured shape and to having
        // written its generation report.
        run_outcome.measurement.total_input_tokens = 10_544;
        run_outcome.measurement.total_output_tokens = 500;
        write_generation_reports(
            logging_directory,
            &[generation_report(
                throughput_support::MEASURED_REQUEST_ID,
                10_544,
            )],
        );
    }
    memory_cell_throughput_record(
        run_outcome,
        23 * BYTES_PER_GIGABYTE,
        fixture_worker_build_identity(),
    )
    .await
}

fn journey_run_outcome_with_budget_stage(
    logging_directory: &Path,
    budget_exhausted_stage: Option<MemoryCellBudgetExhaustedStage>,
) -> JourneyRunOutcome {
    JourneyRunOutcome {
        measurement: throughput_support::ThroughputMeasurement {
            model_id: "test-model".to_owned(),
            decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy::Baseline,
            total_input_tokens: 0,
            total_output_tokens: 0,
            cached_tokens: 0,
            prefill_time_seconds: 0.0,
            decode_time_seconds: 0.0,
            prefill_tokens_per_second: 0.0,
            decode_tokens_per_second: 0.0,
            mlx_peak_memory_bytes: None,
            measured_output_text: None,
            decode_astro_expert_cache_diagnostic_evidence: None,
            completion_ordinal_in_worker: 2,
        },
        measured_budget_exhausted: budget_exhausted_stage.is_some(),
        budget_exhausted_stage,
        final_health_snapshot: astronomical_daemon::WorkerHandle::unavailable()
            .worker_health_snapshot(),
        health_snapshot_after_warmup: None,
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

fn fixture_worker_build_identity() -> WorkerBuildIdentity {
    WorkerBuildIdentity {
        source_commit: "fixture-commit".to_owned(),
        source_is_dirty: false,
        executable_sha256: "ab".repeat(32),
    }
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
