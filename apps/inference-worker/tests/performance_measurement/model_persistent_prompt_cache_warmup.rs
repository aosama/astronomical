use std::{path::PathBuf, time::Duration};

use astronomical_inference_worker::worker_startup::sample_iogpu_wired_limit_bytes;
use serde_json::{Value, json};
use tokio::time::{Instant, timeout};

use super::model_persistent_prompt_cache_benchmark_case::{
    PersistentPromptCacheWarmupCase, fifty_thousand_word_case, five_thousand_word_case,
    hundred_thousand_word_case,
};
use super::model_persistent_prompt_cache_warmup_worker::{
    BENCHMARK_TIMEOUT, MODEL_ID, StreamingMeasurement, WorkerPassMeasurement,
    persistent_prompt_cache_disk_usage, persistent_prompt_cache_enabled_worker_configuration,
    run_worker_pass, static_source_document,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model twice and benchmarks SSD prompt-cache warmup across two worker processes"]
async fn should_measure_model_persistent_prompt_cache_warmup_acceleration() {
    timeout(
        BENCHMARK_TIMEOUT,
        run_persistent_prompt_cache_warmup_e2e(five_thousand_word_case()),
    )
    .await
    .expect("the persistent prompt-cache warmup E2E must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model twice and measures 50,000-word SSD prompt-cache restoration scaling"]
async fn should_measure_model_persistent_prompt_cache_warmup_scaling_at_fifty_thousand_words() {
    timeout(
        BENCHMARK_TIMEOUT,
        run_persistent_prompt_cache_warmup_e2e(fifty_thousand_word_case()),
    )
    .await
    .expect("the 50K persistent prompt-cache warmup E2E must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model twice and explicitly measures 100,000-word SSD prompt-cache restoration scaling"]
async fn should_measure_model_persistent_prompt_cache_warmup_scaling_at_hundred_thousand_words() {
    timeout(
        BENCHMARK_TIMEOUT,
        run_persistent_prompt_cache_warmup_e2e(hundred_thousand_word_case()),
    )
    .await
    .expect("the 100K persistent prompt-cache warmup E2E must finish within 115 seconds");
}

async fn run_persistent_prompt_cache_warmup_e2e(
    persistent_prompt_cache_warmup_case: PersistentPromptCacheWarmupCase,
) {
    let benchmark_started_at = Instant::now();
    let maximum_gpu_wired_memory_bytes = sample_iogpu_wired_limit_bytes()
        .await
        .expect("the machine GPU wired-memory limit should be available");
    let source_document =
        static_source_document(persistent_prompt_cache_warmup_case.source_document);
    let configured_model_directory = configured_model_directory_from_user_config();
    let configured_prompt_cache_maximum_size_gb =
        configured_prompt_cache_maximum_size_gb_from_user_config();
    let (
        _isolated_home,
        persistent_prompt_cache_directory,
        worker_executable_path,
        worker_startup_configuration,
    ) = persistent_prompt_cache_enabled_worker_configuration(
        &configured_model_directory,
        configured_prompt_cache_maximum_size_gb,
    );

    let empty_persistent_prompt_cache_disk_usage =
        persistent_prompt_cache_disk_usage(&persistent_prompt_cache_directory);
    assert_eq!(
        empty_persistent_prompt_cache_disk_usage.block_count, 0,
        "the persistent prompt cache must start empty"
    );
    assert_eq!(
        empty_persistent_prompt_cache_disk_usage.byte_count, 0,
        "the persistent prompt cache must start at zero bytes"
    );

    let cold_worker_pass = run_worker_pass(
        "cold",
        benchmark_started_at,
        worker_executable_path.clone(),
        worker_startup_configuration.clone(),
        &configured_model_directory,
        &source_document,
        &persistent_prompt_cache_warmup_case,
        &persistent_prompt_cache_directory,
    )
    .await;
    assert_eq!(
        cold_worker_pass.streaming_measurement.cached_token_count, 0,
        "the first request must not see cached tokens"
    );

    let saved_persistent_prompt_cache_disk_usage =
        persistent_prompt_cache_disk_usage(&persistent_prompt_cache_directory);
    assert!(
        saved_persistent_prompt_cache_disk_usage.block_count > 0,
        "the cold request must save persistent prompt-cache blocks"
    );
    assert!(
        saved_persistent_prompt_cache_disk_usage.byte_count > 0,
        "saved persistent prompt-cache blocks must occupy disk space"
    );
    eprintln!(
        "[persistent-prompt-cache-warmup] cold pass saved {} blocks ({} bytes) to SSD",
        saved_persistent_prompt_cache_disk_usage.block_count,
        saved_persistent_prompt_cache_disk_usage.byte_count
    );

    let warm_worker_pass = run_worker_pass(
        "warm",
        benchmark_started_at,
        worker_executable_path,
        worker_startup_configuration,
        &configured_model_directory,
        &source_document,
        &persistent_prompt_cache_warmup_case,
        &persistent_prompt_cache_directory,
    )
    .await;
    assert_persistent_prompt_cache_warmup(
        &cold_worker_pass.streaming_measurement,
        &warm_worker_pass.streaming_measurement,
        saved_persistent_prompt_cache_disk_usage.block_count,
    );

    let metrics_report = benchmark_report(
        &persistent_prompt_cache_warmup_case,
        maximum_gpu_wired_memory_bytes,
        saved_persistent_prompt_cache_disk_usage.block_count,
        saved_persistent_prompt_cache_disk_usage.byte_count,
        &cold_worker_pass,
        &warm_worker_pass,
        benchmark_started_at.elapsed(),
    );
    eprintln!("[persistent-prompt-cache-warmup] {metrics_report}");
}

fn assert_persistent_prompt_cache_warmup(
    cold_streaming_measurement: &StreamingMeasurement,
    warm_streaming_measurement: &StreamingMeasurement,
    saved_persistent_prompt_cache_block_count: usize,
) {
    assert!(
        warm_streaming_measurement.cached_token_count > 0,
        "the second request should restore cached blocks from SSD"
    );
    assert!(
        warm_streaming_measurement.cached_token_count as usize
            >= saved_persistent_prompt_cache_block_count,
        "cached tokens should cover every saved persistent model-state block"
    );
    assert_eq!(
        warm_streaming_measurement.cached_token_count as usize
            % saved_persistent_prompt_cache_block_count,
        0,
        "the warm request must restore every block saved by the cold request"
    );
    assert_eq!(
        cold_streaming_measurement.prompt_token_count,
        warm_streaming_measurement.prompt_token_count
    );
    assert_eq!(
        cold_streaming_measurement.completion_token_count,
        warm_streaming_measurement.completion_token_count
    );
    assert_eq!(
        cold_streaming_measurement.generated_text, warm_streaming_measurement.generated_text,
        "cache restoration must preserve deterministic generated output"
    );
}

fn benchmark_report(
    persistent_prompt_cache_warmup_case: &PersistentPromptCacheWarmupCase,
    maximum_gpu_wired_memory_bytes: usize,
    saved_persistent_prompt_cache_block_count: usize,
    saved_persistent_prompt_cache_byte_count: u64,
    cold_worker_pass: &WorkerPassMeasurement,
    warm_worker_pass: &WorkerPassMeasurement,
    benchmark_duration: Duration,
) -> Value {
    let cold_prompt_seconds = cold_worker_pass
        .streaming_measurement
        .prompt_duration()
        .as_secs_f64();
    let warm_prompt_seconds = warm_worker_pass
        .streaming_measurement
        .prompt_duration()
        .as_secs_f64();
    let cold_total_seconds = cold_worker_pass
        .streaming_measurement
        .total_duration()
        .as_secs_f64();
    let warm_total_seconds = warm_worker_pass
        .streaming_measurement
        .total_duration()
        .as_secs_f64();
    let peak_physical_footprint_reduction_bytes = cold_worker_pass
        .physical_footprint
        .peak_bytes
        .saturating_sub(warm_worker_pass.physical_footprint.peak_bytes);
    let observed_block_token_count = warm_worker_pass
        .streaming_measurement
        .cached_token_count
        .checked_div(u64::try_from(saved_persistent_prompt_cache_block_count).unwrap_or(1))
        .unwrap_or(0);

    json!({
        "model": MODEL_ID,
        "benchmark": persistent_prompt_cache_warmup_case.benchmark_name,
        "source_words": persistent_prompt_cache_warmup_case.source_word_count,
        "gpu_wired_limit_bytes": maximum_gpu_wired_memory_bytes,
        "persistent_prompt_cache": {
            "observed_block_size_tokens": observed_block_token_count,
            "saved_block_count": saved_persistent_prompt_cache_block_count,
            "saved_bytes": saved_persistent_prompt_cache_byte_count,
        },
        "cold": worker_pass_report(cold_worker_pass, persistent_prompt_cache_warmup_case.maximum_output_tokens),
        "warm": worker_pass_report(warm_worker_pass, persistent_prompt_cache_warmup_case.maximum_output_tokens),
        "comparison": {
            "cached_prompt_percent": percent(
                warm_worker_pass.streaming_measurement.cached_token_count as f64,
                warm_worker_pass.streaming_measurement.prompt_token_count as f64,
            ),
            "prompt_latency_reduction_percent": percent(
                cold_prompt_seconds - warm_prompt_seconds,
                cold_prompt_seconds,
            ),
            "prompt_processing_speedup": cold_prompt_seconds / warm_prompt_seconds,
            "total_request_latency_reduction_percent": percent(
                cold_total_seconds - warm_total_seconds,
                cold_total_seconds,
            ),
            "total_request_speedup": cold_total_seconds / warm_total_seconds,
            "peak_ram_reduction_bytes": peak_physical_footprint_reduction_bytes,
            "peak_ram_reduction_percent": percent(
                peak_physical_footprint_reduction_bytes as f64,
                cold_worker_pass.physical_footprint.peak_bytes as f64,
            ),
        },
        "benchmark_seconds": benchmark_duration.as_secs_f64(),
    })
}

fn worker_pass_report(worker_pass: &WorkerPassMeasurement, maximum_output_tokens: u16) -> Value {
    let streaming_measurement = &worker_pass.streaming_measurement;
    assert!(streaming_measurement.prompt_token_count > 0);
    assert!(streaming_measurement.completion_token_count > 0);
    assert!(streaming_measurement.cached_token_count <= streaming_measurement.prompt_token_count);
    if maximum_output_tokens > 1 {
        assert!(!streaming_measurement.generated_text.trim().is_empty());
    }
    let prompt_duration = streaming_measurement.prompt_duration();
    let generation_duration = streaming_measurement.generation_duration();
    let uncached_prompt_token_count =
        streaming_measurement.prompt_token_count - streaming_measurement.cached_token_count;

    json!({
        "prompt_tokens": streaming_measurement.prompt_token_count,
        "uncached_prompt_tokens": uncached_prompt_token_count,
        "cached_tokens": streaming_measurement.cached_token_count,
        "completion_tokens": streaming_measurement.completion_token_count,
        "prompt_processing_seconds": prompt_duration.as_secs_f64(),
        "prompt_processing_tokens_per_second": streaming_measurement.prompt_token_count as f64
            / prompt_duration.as_secs_f64(),
        "uncached_prompt_tokens_per_second": uncached_prompt_token_count as f64
            / prompt_duration.as_secs_f64(),
        "first_prefill_progress_tokens_per_second": streaming_measurement.prefill_progress_samples
            .first()
            .map(|prefill_progress_sample| prefill_progress_sample.tokens_per_second()),
        "prefill_progress_samples": streaming_measurement.prefill_progress_samples
            .iter()
            .take(16)
            .map(|prefill_progress_sample| json!({
                "processed_tokens": prefill_progress_sample.processed_tokens,
                "total_tokens": prefill_progress_sample.total_tokens,
                "elapsed_millis": prefill_progress_sample.elapsed_millis,
                "tokens_per_second": prefill_progress_sample.tokens_per_second(),
            }))
            .collect::<Vec<_>>(),
        "generation_seconds": generation_duration.as_secs_f64(),
        "generation_tokens_per_second": (streaming_measurement.completion_token_count > 1).then(|| {
            (streaming_measurement.completion_token_count - 1) as f64 / generation_duration.as_secs_f64()
        }),
        "total_request_seconds": streaming_measurement.total_duration().as_secs_f64(),
        "idle_worker_startup_seconds": worker_pass.idle_worker_startup_duration.as_secs_f64(),
        "idle_worker_startup_plus_request_seconds": worker_pass.idle_worker_startup_duration.as_secs_f64()
            + streaming_measurement.total_duration().as_secs_f64(),
        "current_worker_physical_footprint_bytes": worker_pass.physical_footprint.current_bytes,
        "peak_worker_physical_footprint_bytes": worker_pass.physical_footprint.peak_bytes,
        "generated_utf8_bytes": streaming_measurement.generated_text.len(),
    })
}

fn percent(numerator: f64, denominator: f64) -> f64 {
    numerator * 100.0 / denominator
}

fn configured_model_directory_from_user_config() -> PathBuf {
    crate::common::configured_model_artifact_directory_by_id(MODEL_ID)
}

fn configured_prompt_cache_maximum_size_gb_from_user_config() -> u64 {
    let home_directory = std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("HOME should be set for persistent prompt-cache warmup");
    let config_file_path = home_directory.join(".astronomical/config.json");
    let config_file_bytes = std::fs::read(&config_file_path).expect(
        "~/.astronomical/config.json should be readable for persistent prompt-cache warmup",
    );
    let config_document: Value = serde_json::from_slice(&config_file_bytes)
        .expect("~/.astronomical/config.json should be valid JSON");
    config_document
        .get("prompt_cache")
        .and_then(|prompt_cache_document| prompt_cache_document.get("max_size_gb"))
        .and_then(Value::as_u64)
        .expect("~/.astronomical/config.json should define prompt_cache.max_size_gb")
}
