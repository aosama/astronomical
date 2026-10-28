use std::{path::PathBuf, time::Duration};

use astronomical_inference_worker::worker_startup;
use serde_json::{Value, json};
use tokio::time::{Instant, timeout};

use super::model_persistent_prompt_cache_benchmark_case::{
    PersistentPromptCacheWarmupCase, fifty_thousand_word_case, five_thousand_word_case,
    hundred_thousand_word_case,
};
use super::model_persistent_prompt_cache_warmup_worker::{
    BENCHMARK_TIMEOUT, StreamingMeasurement, WorkerPassMeasurement, model_id,
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

#[expect(clippy::too_many_lines)]
async fn run_persistent_prompt_cache_warmup_e2e(
    persistent_prompt_cache_warmup_case: PersistentPromptCacheWarmupCase,
) {
    let benchmark_started_at = Instant::now();
    let machine_mlx_memory_ceiling_bytes = worker_startup::machine_mlx_memory_ceiling_bytes()
        .await
        .expect("the machine MLX memory ceiling should be available");
    let source_document =
        static_source_document(persistent_prompt_cache_warmup_case.source_document);
    let configured_model_directory = configured_model_directory_from_user_config();
    let configured_prompt_cache_maximum_size_gb =
        configured_prompt_cache_maximum_size_gb_from_user_config();
    let (
        _isolated_home,
        persistent_prompt_cache_directory,
        worker_executable_path,
        model_policy_catalog,
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
        model_policy_catalog.clone(),
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
    let mut saved_complete_block_token_counts = saved_persistent_prompt_cache_disk_usage
        .sequence_state_token_counts
        .clone();
    saved_complete_block_token_counts.sort_unstable();
    // The largest declared token count is the model's block geometry; any
    // file declaring fewer is the prompt's best-effort partial tail.
    let persistent_prompt_cache_block_token_count = *saved_complete_block_token_counts
        .last()
        .expect("saved sequence-state files must declare a block token count");
    let saved_complete_block_count = saved_complete_block_token_counts
        .iter()
        .filter(|declared_token_count| {
            **declared_token_count == persistent_prompt_cache_block_token_count
        })
        .count();
    eprintln!(
        "[persistent-prompt-cache-warmup] cold pass saved {} blocks ({} bytes) to SSD: {} complete blocks of {} tokens plus {} partial-tail artifacts",
        saved_persistent_prompt_cache_disk_usage.block_count,
        saved_persistent_prompt_cache_disk_usage.byte_count,
        saved_complete_block_count,
        persistent_prompt_cache_block_token_count,
        saved_persistent_prompt_cache_disk_usage.block_count - saved_complete_block_count,
    );

    let warm_worker_pass = run_worker_pass(
        "warm",
        benchmark_started_at,
        worker_executable_path,
        worker_startup_configuration,
        model_policy_catalog,
        &source_document,
        &persistent_prompt_cache_warmup_case,
        &persistent_prompt_cache_directory,
    )
    .await;
    let saved_partial_tail_token_count = saved_persistent_prompt_cache_disk_usage
        .sequence_state_token_counts
        .iter()
        .filter(|declared_token_count| {
            **declared_token_count != persistent_prompt_cache_block_token_count
        })
        .sum();
    assert_persistent_prompt_cache_warmup(
        &cold_worker_pass.streaming_measurement,
        &warm_worker_pass.streaming_measurement,
        saved_complete_block_count,
        persistent_prompt_cache_block_token_count,
        saved_partial_tail_token_count,
    );

    let metrics_report = benchmark_report(
        &persistent_prompt_cache_warmup_case,
        machine_mlx_memory_ceiling_bytes,
        saved_complete_block_count,
        persistent_prompt_cache_block_token_count,
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
    saved_complete_block_count: usize,
    persistent_prompt_cache_block_token_count: usize,
    saved_partial_tail_token_count: usize,
) {
    let warm_cached_token_count = warm_streaming_measurement.cached_token_count as usize;
    let warm_uncached_token_count =
        warm_streaming_measurement.prompt_token_count as usize - warm_cached_token_count;
    eprintln!(
        "[persistent-prompt-cache-warmup] warm restore reported {warm_cached_token_count} cached tokens of {} prompt tokens",
        warm_streaming_measurement.prompt_token_count
    );
    assert!(
        warm_cached_token_count > 0,
        "the second request should restore cached blocks from SSD"
    );
    // The restored extent must cover every complete block the cold pass
    // persisted. It is not, in general, a multiple of the saved artifact
    // count: a partial tail contributes a non-block-sized amount, and the
    // final prompt token always stays on the prefill path.
    assert!(
        warm_cached_token_count
            >= saved_complete_block_count * persistent_prompt_cache_block_token_count,
        "the warm request must restore every complete block saved by the cold request: \
         restored {warm_cached_token_count} tokens, but {} saved complete blocks of \
         {persistent_prompt_cache_block_token_count} tokens require at least {}",
        saved_complete_block_count,
        saved_complete_block_count * persistent_prompt_cache_block_token_count,
    );
    // Only the tail remainder of the prompt may stay uncached; anything
    // larger means lookup stopped early and silently discarded blocks that
    // the configured persistent-cache size still admits.
    assert!(
        warm_uncached_token_count <= persistent_prompt_cache_block_token_count,
        "the warm request must leave at most one block of prompt work uncached: \
         {warm_uncached_token_count} tokens remain after restoring {warm_cached_token_count}"
    );
    // When the cold pass persisted a partial tail, the warm pass must restore
    // it too: the pre-fix regression (a tail saved through the final prompt
    // token could never be probed) restored only the complete blocks and left
    // exactly the whole tail uncached, which the bound above still admitted.
    assert!(
        saved_partial_tail_token_count == 0 || warm_uncached_token_count <= 1,
        "the warm request must restore the persisted partial tail: \
         {warm_uncached_token_count} tokens remain after restoring {warm_cached_token_count}, \
         but the cold pass saved a partial tail of {saved_partial_tail_token_count} tokens"
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
    machine_mlx_memory_ceiling_bytes: usize,
    saved_complete_block_count: usize,
    persistent_prompt_cache_block_token_count: usize,
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
    json!({
        "model": model_id(),
        "benchmark": persistent_prompt_cache_warmup_case.benchmark_name,
        "source_words": persistent_prompt_cache_warmup_case.source_word_count,
        "machine_mlx_memory_ceiling_bytes": machine_mlx_memory_ceiling_bytes,
        "persistent_prompt_cache": {
            "block_size_tokens": persistent_prompt_cache_block_token_count,
            "saved_complete_block_count": saved_complete_block_count,
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
    crate::support::configured_installed_model_directory_by_id(model_id())
}

fn configured_prompt_cache_maximum_size_gb_from_user_config() -> u64 {
    let home_directory = std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("HOME should be set for persistent prompt-cache warmup");
    let config_file_path = home_directory.join(".astronomical-dev/config.json");
    let config_file_bytes = std::fs::read(&config_file_path)
        .expect("Development config should be readable for persistent prompt-cache warmup");
    let config_document: Value = serde_json::from_slice(&config_file_bytes)
        .expect("Development config should be valid JSON");
    config_document
        .get("prompt_cache")
        .and_then(|prompt_cache_document| prompt_cache_document.get("maximum_size_gb"))
        .and_then(Value::as_u64)
        .expect("Development config should define prompt_cache.maximum_size_gb")
}
