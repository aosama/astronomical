//! Direct paged-decode tests for the real Ornith model.
//!
//! These tests intentionally bypass REST, the supervisor, the worker process,
//! the macOS app, and shell scripts. They call the loaded model directly with
//! token IDs so cache and decode bottlenecks are visible at the lowest useful
//! model boundary.

use std::future::Future;
use std::time::{Duration, Instant};

use astronomical_model_serving::{
    ExpertWeightMemoryCacheStatistics, Qwen3_5ArtifactValidator, Qwen3_5Config, Qwen3_5Model,
};
use astronomical_runtime_integration::MlxRuntime;
use tokio::time::{MissedTickBehavior, interval, sleep};

const EXPERT_PAGING_DECODE_TEST_TIMEOUT: Duration = Duration::from_secs(120);
const GENERATED_TOKEN_COUNT: u32 = 30;

use super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS;

#[tokio::test]
#[ignore = "loads the full Ornith model and generates 30 tokens through direct paged decode"]
async fn should_generate_30_tokens_with_expert_memory_cache_and_report_per_token_progress() {
    require_expert_paging_decode_completion(
        run_30_token_paged_decode_cache_probe(),
        "[paged-30]",
        "the direct 30-token paged decode diagnostic",
    )
    .await;
}

#[tokio::test]
#[ignore = "loads the full model, repeats an identical 30-token decode, and verifies stable expert residency"]
async fn should_repeat_identical_30_token_decode_with_stable_expert_residency() {
    require_expert_paging_decode_completion(
        run_30_token_warm_cache_decode_probe(),
        "[paged-30-warm]",
        "the direct warm-cache paged decode diagnostic",
    )
    .await;
}

async fn run_30_token_paged_decode_cache_probe() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_started_at = Instant::now();
    let (qwen3_5_model, config) = load_paged_qwen3_5_model_for_decode_probe("[paged-30]").await;

    let decode_pass_measurements = run_say_hi_30_token_paged_decode_pass(
        &qwen3_5_model,
        &config,
        "[paged-30]",
        "single",
        true,
    );

    eprintln!(
        "[paged-30] status=success elapsed_seconds={:.2} generated_tokens={} average_tok_per_second={:.2} cache_entries={} resident_payload_gib={:.2} disk_page_loads={} disk_batch_loads={} cache_hits={} cache_misses={}",
        test_started_at.elapsed().as_secs_f64(),
        GENERATED_TOKEN_COUNT,
        decode_pass_measurements.average_token_per_second(),
        decode_pass_measurements.final_cache_statistics.entry_count,
        bytes_to_gib(
            decode_pass_measurements
                .final_cache_statistics
                .resident_payload_byte_count
        ),
        decode_pass_measurements
            .final_cache_statistics
            .disk_page_load_count,
        decode_pass_measurements
            .final_cache_statistics
            .disk_batch_load_count,
        decode_pass_measurements
            .final_cache_statistics
            .cache_hit_count,
        decode_pass_measurements
            .final_cache_statistics
            .cache_miss_count
    );
    assert!(
        decode_pass_measurements.disk_page_load_count_delta > 0,
        "30-token paged decode should have loaded some experts from disk"
    );
    assert!(
        decode_pass_measurements.cache_hit_count_delta
            + decode_pass_measurements.complete_layer_hit_count_delta
            > 0,
        "30-token paged decode should reuse at least one partial expert or complete expert layer"
    );
}

async fn run_30_token_warm_cache_decode_probe() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_started_at = Instant::now();
    let (qwen3_5_model, config) =
        load_paged_qwen3_5_model_for_decode_probe("[paged-30-warm]").await;

    eprintln!("[paged-30-warm] status=progress phase=warmup_decode");
    let warmup_decode_measurements = run_say_hi_30_token_paged_decode_pass(
        &qwen3_5_model,
        &config,
        "[paged-30-warm]",
        "warmup",
        true,
    );
    eprintln!(
        "[paged-30-warm] status=progress phase=warmup_done average_tok_per_second={:.2} disk_page_loads={} disk_batch_loads={} cache_hits={} cache_misses={} cache_entries={} resident_payload_gib={:.2}",
        warmup_decode_measurements.average_token_per_second(),
        warmup_decode_measurements.disk_page_load_count_delta,
        warmup_decode_measurements.disk_batch_load_count_delta,
        warmup_decode_measurements.cache_hit_count_delta,
        warmup_decode_measurements.cache_miss_count_delta,
        warmup_decode_measurements
            .final_cache_statistics
            .entry_count,
        bytes_to_gib(
            warmup_decode_measurements
                .final_cache_statistics
                .resident_payload_byte_count
        )
    );

    eprintln!("[paged-30-warm] status=progress phase=measured_warm_decode");
    let measured_warm_decode_measurements = run_say_hi_30_token_paged_decode_pass(
        &qwen3_5_model,
        &config,
        "[paged-30-warm]",
        "measured_warm",
        true,
    );
    eprintln!(
        "[paged-30-warm] status=success elapsed_seconds={:.2} warmup_tok_per_second={:.2} measured_warm_tok_per_second={:.2} measured_disk_page_loads={} measured_disk_batch_loads={} measured_cache_hits={} measured_cache_misses={} cache_entries={} resident_payload_gib={:.2}",
        test_started_at.elapsed().as_secs_f64(),
        warmup_decode_measurements.average_token_per_second(),
        measured_warm_decode_measurements.average_token_per_second(),
        measured_warm_decode_measurements.disk_page_load_count_delta,
        measured_warm_decode_measurements.disk_batch_load_count_delta,
        measured_warm_decode_measurements.cache_hit_count_delta,
        measured_warm_decode_measurements.cache_miss_count_delta,
        measured_warm_decode_measurements
            .final_cache_statistics
            .entry_count,
        bytes_to_gib(
            measured_warm_decode_measurements
                .final_cache_statistics
                .resident_payload_byte_count
        )
    );
    assert_eq!(
        measured_warm_decode_measurements.generated_token_ids,
        warmup_decode_measurements.generated_token_ids,
        "warm measured pass must generate the same greedy token IDs as the warmup pass so cache attribution is comparable"
    );
    assert!(
        measured_warm_decode_measurements.cache_hit_count_delta
            + measured_warm_decode_measurements.complete_layer_hit_count_delta
            > 0,
        "the identical measured decode should reuse partial experts or complete expert layers"
    );
    assert_eq!(
        measured_warm_decode_measurements
            .final_cache_statistics
            .complete_layer_count,
        warmup_decode_measurements
            .final_cache_statistics
            .complete_layer_count,
        "the identical measured decode should preserve complete expert-layer residency"
    );
}

pub(crate) async fn load_paged_qwen3_5_model_for_decode_probe(
    log_prefix: &str,
) -> (Qwen3_5Model, Qwen3_5Config) {
    eprintln!("{log_prefix} status=start phase=artifact_validation");
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the pinned Ornith artifact should validate before 30-token paged decode");
    let config = validated_artifact.config().clone();
    eprintln!(
        "{log_prefix} status=progress phase=artifact_validated shards={} payload_bytes={}",
        validated_artifact.shard_count(),
        validated_artifact.total_payload_bytes()
    );

    eprintln!("{log_prefix} status=progress phase=runtime_init");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize for 30-token paged decode");

    eprintln!("{log_prefix} status=progress phase=model_load");
    let model_load_started_at = Instant::now();
    let qwen3_5_model = Qwen3_5Model::load(
        runtime,
        validated_artifact,
        &model_directory,
        false,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the Ornith model should load with automatic expert residency");
    eprintln!(
        "{log_prefix} status=progress phase=model_loaded elapsed_seconds={:.2}",
        model_load_started_at.elapsed().as_secs_f64()
    );
    (qwen3_5_model, config)
}

fn run_say_hi_30_token_paged_decode_pass(
    qwen3_5_model: &Qwen3_5Model,
    config: &Qwen3_5Config,
    log_prefix: &str,
    decode_pass_label: &str,
    should_print_per_token_progress: bool,
) -> PagedDecodePassMeasurements {
    eprintln!("{log_prefix} status=progress pass={decode_pass_label} phase=prefill token_count=14");
    let mut request_decoder_state = crate::common::standard_request_decoder_state(config);
    let prefill_started_at = Instant::now();
    qwen3_5_model
        .prefill_chunck(
            &SAY_HI_PROMPT_TOKEN_IDS[..SAY_HI_PROMPT_TOKEN_IDS.len() - 1],
            0,
            &mut request_decoder_state,
        )
        .expect("the prompt prefix should materialize decoder state before paged decode");
    eprintln!(
        "{log_prefix} status=progress pass={decode_pass_label} phase=prefill_done elapsed_ms={:.2}",
        prefill_started_at.elapsed().as_secs_f64() * 1000.0
    );

    let initial_cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
    let mut current_token_id = SAY_HI_PROMPT_TOKEN_IDS[SAY_HI_PROMPT_TOKEN_IDS.len() - 1];
    let first_decode_position_tokens = (SAY_HI_PROMPT_TOKEN_IDS.len() - 1) as u32;
    let generation_started_at = Instant::now();
    let mut generated_token_ids = Vec::with_capacity(GENERATED_TOKEN_COUNT as usize);

    for decode_step_index in 0..GENERATED_TOKEN_COUNT {
        let position_tokens = first_decode_position_tokens.saturating_add(decode_step_index);
        let before_step_cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
        let step_started_at = Instant::now();
        let logits = qwen3_5_model
            .forward_chunk(
                &[current_token_id],
                position_tokens,
                &mut request_decoder_state,
            )
            .unwrap_or_else(|error| {
                panic!("paged decode step {decode_step_index} failed: {error}")
            });
        current_token_id = qwen3_5_model
            .greedy_token_id(&logits)
            .unwrap_or_else(|error| {
                panic!("greedy token selection at paged decode step {decode_step_index} failed: {error}")
        });
        generated_token_ids.push(current_token_id);
        let step_elapsed = step_started_at.elapsed();
        let after_step_cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
        let mlx_memory_snapshot = qwen3_5_model
            .runtime()
            .memory_snapshot()
            .expect("the direct paged decode probe should sample MLX memory after each token");
        let disk_page_load_delta = after_step_cache_statistics.disk_page_load_count
            - before_step_cache_statistics.disk_page_load_count;
        let disk_batch_load_delta = after_step_cache_statistics.disk_batch_load_count
            - before_step_cache_statistics.disk_batch_load_count;
        let cache_hit_delta = after_step_cache_statistics.cache_hit_count
            - before_step_cache_statistics.cache_hit_count;
        let cache_miss_delta = after_step_cache_statistics.cache_miss_count
            - before_step_cache_statistics.cache_miss_count;
        let completed_token_count = decode_step_index + 1;
        let average_seconds_per_token =
            generation_started_at.elapsed().as_secs_f64() / f64::from(completed_token_count);
        let remaining_token_count = GENERATED_TOKEN_COUNT - completed_token_count;
        let estimated_remaining_seconds =
            average_seconds_per_token * f64::from(remaining_token_count);
        if should_print_per_token_progress {
            eprintln!(
                "{log_prefix} status=progress pass={decode_pass_label} step={completed_token_count:02}/{GENERATED_TOKEN_COUNT} token_id={current_token_id} elapsed_ms={:.2} tok_per_second={:.2} disk_load_delta={} disk_batch_delta={} cache_hit_delta={} cache_miss_delta={} cache_entries={} complete_layers={} resident_payload_gib={:.2} maximum_resident_payload_gib={:.2} mlx_active_gib={:.2} mlx_allocator_gib={:.2} mlx_peak_gib={:.2} cumulative_disk_loads={} cumulative_disk_batches={} cumulative_cache_hits={} ETA_seconds={:.1}",
                step_elapsed.as_secs_f64() * 1000.0,
                1.0 / step_elapsed.as_secs_f64(),
                disk_page_load_delta,
                disk_batch_load_delta,
                cache_hit_delta,
                cache_miss_delta,
                after_step_cache_statistics.entry_count,
                after_step_cache_statistics.complete_layer_count,
                bytes_to_gib(after_step_cache_statistics.resident_payload_byte_count),
                bytes_to_gib(after_step_cache_statistics.maximum_resident_payload_byte_count),
                bytes_to_gib(mlx_memory_snapshot.active_memory_bytes() as u64),
                bytes_to_gib(mlx_memory_snapshot.allocator_cache_memory_bytes() as u64),
                bytes_to_gib(mlx_memory_snapshot.peak_memory_bytes() as u64),
                after_step_cache_statistics.disk_page_load_count,
                after_step_cache_statistics.disk_batch_load_count,
                after_step_cache_statistics.cache_hit_count,
                estimated_remaining_seconds
            );
        }
    }

    let final_cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
    let generation_elapsed = generation_started_at.elapsed();
    let measurements = PagedDecodePassMeasurements {
        generation_elapsed,
        generated_token_ids,
        final_cache_statistics,
        disk_page_load_count_delta: final_cache_statistics.disk_page_load_count
            - initial_cache_statistics.disk_page_load_count,
        disk_batch_load_count_delta: final_cache_statistics.disk_batch_load_count
            - initial_cache_statistics.disk_batch_load_count,
        cache_hit_count_delta: final_cache_statistics.cache_hit_count
            - initial_cache_statistics.cache_hit_count,
        complete_layer_hit_count_delta: final_cache_statistics.complete_layer_hit_count
            - initial_cache_statistics.complete_layer_hit_count,
        cache_miss_count_delta: final_cache_statistics.cache_miss_count
            - initial_cache_statistics.cache_miss_count,
    };
    eprintln!(
        "{log_prefix} status=progress pass={decode_pass_label} phase=decode_done elapsed_ms={:.2} average_tok_per_second={:.2} disk_page_loads={} disk_batch_loads={} cache_hits={} cache_misses={} resident_payload_gib={:.2}",
        measurements.generation_elapsed.as_secs_f64() * 1000.0,
        measurements.average_token_per_second(),
        measurements.disk_page_load_count_delta,
        measurements.disk_batch_load_count_delta,
        measurements.cache_hit_count_delta,
        measurements.cache_miss_count_delta,
        bytes_to_gib(
            measurements
                .final_cache_statistics
                .resident_payload_byte_count
        )
    );
    measurements
}

struct PagedDecodePassMeasurements {
    generation_elapsed: Duration,
    generated_token_ids: Vec<u32>,
    final_cache_statistics: ExpertWeightMemoryCacheStatistics,
    disk_page_load_count_delta: u64,
    disk_batch_load_count_delta: u64,
    cache_hit_count_delta: u64,
    complete_layer_hit_count_delta: u64,
    cache_miss_count_delta: u64,
}

impl PagedDecodePassMeasurements {
    fn average_token_per_second(&self) -> f64 {
        f64::from(GENERATED_TOKEN_COUNT) / self.generation_elapsed.as_secs_f64()
    }
}

pub(crate) fn bytes_to_gib(byte_count: u64) -> f64 {
    byte_count as f64 / 1024.0 / 1024.0 / 1024.0
}

pub(crate) async fn require_expert_paging_decode_completion(
    test_future: impl Future<Output = ()>,
    progress_log_prefix: &str,
    timeout_description: &str,
) {
    let started_at = Instant::now();
    let timeout_deadline = sleep(EXPERT_PAGING_DECODE_TEST_TIMEOUT);
    let mut progress_interval = interval(Duration::from_secs(10));
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    tokio::pin!(test_future);
    tokio::pin!(timeout_deadline);
    progress_interval.tick().await;
    eprintln!(
        "{progress_log_prefix} status=timeout_guard_started timeout_seconds={}",
        EXPERT_PAGING_DECODE_TEST_TIMEOUT.as_secs()
    );

    loop {
        tokio::select! {
            () = &mut test_future => {
                eprintln!(
                    "{progress_log_prefix} status=completed elapsed_seconds={:.1}",
                    started_at.elapsed().as_secs_f64()
                );
                return;
            }
            () = &mut timeout_deadline => {
                panic!("{timeout_description} exceeded {} seconds", EXPERT_PAGING_DECODE_TEST_TIMEOUT.as_secs());
            }
            _ = progress_interval.tick() => {
                let elapsed = started_at.elapsed();
                let remaining = EXPERT_PAGING_DECODE_TEST_TIMEOUT.saturating_sub(elapsed);
                eprintln!(
                    "{progress_log_prefix} status=running elapsed_seconds={:.0} ETA<={:.0}",
                    elapsed.as_secs_f64(),
                    remaining.as_secs_f64()
                );
            }
        }
    }
}
