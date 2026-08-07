use std::time::{Duration, Instant};

use astronomical_model_serving::InferenceEngine;

use super::engine_support::{load_mtp_test_engine, performance_counter_amount};

#[tokio::test]
#[ignore = "loads the resident configured target model and attributes positional file reads"]
async fn should_measure_resident_model_loading_positional_file_read_concurrency() {
    tokio::time::timeout(
        Duration::from_secs(60),
        run_resident_model_loading_positional_file_read_concurrency(),
    )
    .await
    .expect("resident model-loading file-read attribution should finish within 60 seconds");
}

async fn run_resident_model_loading_positional_file_read_concurrency() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::super::configured_depth_one_mtp_model_artifact_directory();
    let (mut target_only_engine, _temporary_log_directory, performance_attribution_log_path) =
        load_mtp_test_engine(&model_directory, false, true).await;

    eprintln!("[model-loading-file-read] status=start ETA_seconds=60");
    let model_loading_started_at = Instant::now();
    target_only_engine
        .load()
        .await
        .expect("the attributed resident target model should load");
    let model_loading_elapsed = model_loading_started_at.elapsed();
    drop(target_only_engine);

    let model_loading_report = std::fs::read_to_string(&performance_attribution_log_path)
        .expect("the model-loading attribution log should be readable")
        .lines()
        .map(|performance_attribution_line| {
            serde_json::from_str::<serde_json::Value>(performance_attribution_line)
                .expect("each model-loading attribution record should be valid JSON")
        })
        .find(|performance_attribution_report| {
            performance_attribution_report["report_kind"] == "model_loading"
        })
        .expect("the attributed model load should write a model-loading report");
    let read_call_count =
        performance_counter_amount(&model_loading_report, "positional_file_read_call_count");
    let read_byte_count =
        performance_counter_amount(&model_loading_report, "positional_file_read_byte_count");
    let maximum_concurrent_read_count = performance_counter_amount(
        &model_loading_report,
        "positional_file_read_maximum_concurrent_count",
    );
    let total_read_elapsed_nanoseconds = performance_counter_amount(
        &model_loading_report,
        "positional_file_read_elapsed_nanoseconds",
    );

    eprintln!(
        "[model-loading-file-read] status=success read_calls={} read_bytes={} maximum_concurrent_reads={} summed_read_milliseconds={:.3} model_loading_milliseconds={:.3}",
        read_call_count,
        read_byte_count,
        maximum_concurrent_read_count,
        total_read_elapsed_nanoseconds as f64 / 1_000_000.0,
        model_loading_elapsed.as_secs_f64() * 1_000.0,
    );
    assert!(
        read_call_count > 1,
        "model loading should read multiple tensors"
    );
    assert!(
        read_byte_count > 0,
        "model loading should read tensor bytes"
    );
}
