use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, GeneratedToken, InferenceEngine,
    InferenceEngineError, PerformanceAttribution, PerformanceAttributionLog,
    Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5PrefillChunckSizer,
};
use serde_json::Value;
use tokio::time::{Instant, MissedTickBehavior, interval, timeout};

const ENDURANCE_TEST_TIMEOUT: Duration = Duration::from_secs(120);
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const WARMUP_INPUT_TOKEN_COUNT: usize = 2_049;
const ENDURANCE_INPUT_TOKEN_COUNT: usize = 50_000;
const ENDURANCE_MAXIMUM_OUTPUT_TOKENS: u16 = 20_480;
const SUSTAINED_ENDURANCE_INPUT_TOKEN_COUNT: usize = 70_000;
const SUSTAINED_ENDURANCE_REQUESTED_OUTPUT_TOKEN_COUNT: u16 = 1_000;
const SUSTAINED_ENDURANCE_OBSERVED_OUTPUT_TOKEN_COUNT: u16 = 200;
const FIXED_PREFILL_CHUNCK_TOKENS: u32 = 2_048;
const DETERMINISTIC_PROMPT_TOKEN_ID: u32 = 198;
const IMAGE_PAD_TOKEN_ID: u32 = 248_069;

#[tokio::test]
#[ignore = "loads Qwen3.6-35B-A3B-8bit and proves long-context growth evicts retained experts"]
async fn should_serve_fifty_thousand_input_tokens_by_automatically_reclaiming_expert_residency() {
    timeout(
        ENDURANCE_TEST_TIMEOUT,
        run_automatic_residency_endurance_regression(),
    )
    .await
    .expect("the automatic-residency endurance regression must finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads Qwen3.6-35B-A3B-8bit and samples a realistic 70K-input/1K-output request"]
async fn should_preserve_automatic_expert_residency_during_seventy_thousand_input_token_generation()
{
    timeout(
        ENDURANCE_TEST_TIMEOUT,
        run_sustained_paged_mode_endurance_regression(),
    )
    .await
    .expect("the 70K-input/1K-output-budget sample must finish within 120 seconds");
}

async fn run_automatic_residency_endurance_regression() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::qwen3_6_35b_a3b_eight_bit_model_directory();
    assert!(
        model_directory.is_dir(),
        "the Qwen3.6-35B-A3B-8bit checkpoint must be available for endurance testing"
    );
    let (mut qwen3_5_engine, _temporary_log_directory, performance_attribution_log_path) =
        create_automatic_residency_endurance_engine(&model_directory).await;

    eprintln!(
        "[automatic-residency-endurance 0/3] status=progress phase=model_load input_tokens={ENDURANCE_INPUT_TOKEN_COUNT} ETA_seconds=120"
    );
    load_engine_with_progress(&mut qwen3_5_engine).await;

    eprintln!(
        "[paged-endurance 1/3] status=progress phase=warm_expert_retention input_tokens={WARMUP_INPUT_TOKEN_COUNT} ETA_seconds=100"
    );
    run_generation_until_first_token(
        &mut qwen3_5_engine,
        RequestId::new(10_000),
        WARMUP_INPUT_TOKEN_COUNT,
        1,
        "warmup",
    )
    .await
    .expect("the paged expert warmup should complete");

    eprintln!(
        "[paged-endurance 2/3] status=progress phase=long_context input_tokens={ENDURANCE_INPUT_TOKEN_COUNT} maximum_output_tokens={ENDURANCE_MAXIMUM_OUTPUT_TOKENS} ETA_seconds=90"
    );
    let endurance_outcome = run_generation_until_first_token(
        &mut qwen3_5_engine,
        RequestId::new(10_001),
        ENDURANCE_INPUT_TOKEN_COUNT,
        ENDURANCE_MAXIMUM_OUTPUT_TOKENS,
        "long_context",
    )
    .await;
    if endurance_outcome.is_ok() {
        qwen3_5_engine
            .cancel_generation(RequestId::new(10_001))
            .await
            .expect("the successful endurance control should finalize cleanly");
    }
    print_attribution_memory_timeline(&performance_attribution_log_path);
    assert!(
        endurance_outcome.is_ok(),
        "automatic expert residency must reclaim retained experts before rejecting context growth: {}",
        endurance_outcome.expect_err("the failed endurance outcome should contain the rejection")
    );
    eprintln!("[paged-endurance 3/3] status=success");
}

async fn run_sustained_paged_mode_endurance_regression() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::qwen3_6_35b_a3b_eight_bit_model_directory();
    assert!(
        model_directory.is_dir(),
        "the Qwen3.6-35B-A3B-8bit checkpoint must be available for endurance testing"
    );
    let sustained_endurance_prompt_token_ids = super::expert_paging_prefill_performance::
        prepare_reproduced_long_prompt_token_ids_for_model(
            &model_directory,
            "Qwen3.6-35B-A3B-8bit",
            SUSTAINED_ENDURANCE_INPUT_TOKEN_COUNT,
            SUSTAINED_ENDURANCE_REQUESTED_OUTPUT_TOKEN_COUNT,
        );
    let (mut qwen3_5_engine, _temporary_log_directory, performance_attribution_log_path) =
        create_automatic_residency_endurance_engine(&model_directory).await;

    eprintln!(
        "[paged-sustained-endurance 0/3] status=progress phase=model_load input_tokens={SUSTAINED_ENDURANCE_INPUT_TOKEN_COUNT} requested_output_tokens={SUSTAINED_ENDURANCE_REQUESTED_OUTPUT_TOKEN_COUNT} observed_output_tokens={SUSTAINED_ENDURANCE_OBSERVED_OUTPUT_TOKEN_COUNT} ETA_seconds=120"
    );
    load_engine_with_progress(&mut qwen3_5_engine).await;
    eprintln!(
        "[paged-sustained-endurance 1/3] status=progress phase=warm_expert_retention input_tokens={WARMUP_INPUT_TOKEN_COUNT} ETA_seconds=110"
    );
    run_generation_until_first_token(
        &mut qwen3_5_engine,
        RequestId::new(20_000),
        WARMUP_INPUT_TOKEN_COUNT,
        1,
        "warmup",
    )
    .await
    .expect("the paged expert warmup should complete");

    eprintln!(
        "[paged-sustained-endurance 2/3] status=progress phase=long_context input_tokens={SUSTAINED_ENDURANCE_INPUT_TOKEN_COUNT} requested_output_tokens={SUSTAINED_ENDURANCE_REQUESTED_OUTPUT_TOKEN_COUNT} observed_output_tokens={SUSTAINED_ENDURANCE_OBSERVED_OUTPUT_TOKEN_COUNT} ETA_seconds=100"
    );
    let sustained_endurance_outcome = run_generation_until_output_sample(
        &mut qwen3_5_engine,
        RequestId::new(20_001),
        sustained_endurance_prompt_token_ids,
        SUSTAINED_ENDURANCE_REQUESTED_OUTPUT_TOKEN_COUNT,
        SUSTAINED_ENDURANCE_OBSERVED_OUTPUT_TOKEN_COUNT,
        "long_context_sustained_decode",
    )
    .await;
    if sustained_endurance_outcome.is_ok() {
        qwen3_5_engine
            .cancel_generation(RequestId::new(20_001))
            .await
            .expect("the sustained endurance sample should finalize cleanly");
    }
    print_attribution_memory_timeline(&performance_attribution_log_path);
    sustained_endurance_outcome.expect(
        "automatic expert residency must preserve enough retained experts to sample sustained output after a 70K-token prompt",
    );
    eprintln!("[paged-sustained-endurance 3/3] status=success");
}

async fn create_automatic_residency_endurance_engine(
    model_directory: &Path,
) -> (Qwen3_5Engine, tempfile::TempDir, PathBuf) {
    let temporary_log_directory =
        tempfile::tempdir().expect("the endurance test should create a temporary log directory");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, u32::from(ENDURANCE_MAXIMUM_OUTPUT_TOKENS))
        .expect("the Qwen3.6 eight-bit artifact should validate before endurance testing");
    let performance_attribution_log =
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("the endurance test should open its attribution log");
    let qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer_and_performance_attribution(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(FIXED_PREFILL_CHUNCK_TOKENS)
            .expect("the production-sized prefill chunck should be valid"),
        IMAGE_PAD_TOKEN_ID,
        model_directory.to_path_buf(),
        DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
        true,
        false,
        PerformanceAttribution::enabled(),
        performance_attribution_log,
    )
    .expect("the paged endurance engine settings should be valid");
    (
        qwen3_5_engine,
        temporary_log_directory,
        performance_attribution_log_path,
    )
}

async fn load_engine_with_progress(qwen3_5_engine: &mut Qwen3_5Engine) {
    let model_load_started_at = Instant::now();
    let mut progress_interval = interval(PROGRESS_INTERVAL);
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    progress_interval.tick().await;
    let model_load_future = qwen3_5_engine.load();
    tokio::pin!(model_load_future);
    loop {
        tokio::select! {
            model_load_outcome = &mut model_load_future => {
                model_load_outcome.expect("the paged endurance model should load");
                return;
            }
            _ = progress_interval.tick() => eprintln!(
                "[paged-endurance] status=progress phase=model_load elapsed_seconds={:.1} ETA_seconds=unknown",
                model_load_started_at.elapsed().as_secs_f64(),
            ),
        }
    }
}

async fn run_generation_until_first_token(
    qwen3_5_engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    input_token_count: usize,
    maximum_output_tokens: u16,
    phase_name: &str,
) -> Result<(), InferenceEngineError> {
    let generation_started_at = Instant::now();
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                request_id,
                vec![DETERMINISTIC_PROMPT_TOKEN_ID; input_token_count],
                maximum_output_tokens,
            )
            .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID)
            .with_performance_attribution(PerformanceAttribution::enabled()),
        )
        .await?;
    let mut processed_input_token_count = 0usize;
    loop {
        let mut progress_interval = interval(PROGRESS_INTERVAL);
        progress_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        progress_interval.tick().await;
        let generation_advance_future = qwen3_5_engine.decode_next_token(request_id);
        tokio::pin!(generation_advance_future);
        let generation_event = loop {
            tokio::select! {
                generation_outcome = &mut generation_advance_future => break generation_outcome?,
                _ = progress_interval.tick() => eprintln!(
                    "[paged-endurance] status=progress phase={phase_name} processed_input_tokens={processed_input_token_count}/{input_token_count} elapsed_seconds={:.1} ETA_seconds=unknown",
                    generation_started_at.elapsed().as_secs_f64(),
                ),
            }
        };
        match generation_event {
            GeneratedToken::PrefillProgress {
                processed_token_count,
                mlx_memory_telemetry,
                ..
            } => {
                let mlx_memory_telemetry = mlx_memory_telemetry
                    .expect("the enabled memory guard should report prefill telemetry");
                processed_input_token_count =
                    processed_input_token_count.saturating_add(processed_token_count as usize);
                let elapsed_seconds = generation_started_at.elapsed().as_secs_f64();
                let estimated_total_seconds = elapsed_seconds * input_token_count as f64
                    / processed_input_token_count.max(1) as f64;
                eprintln!(
                    "[paged-endurance] status=progress phase={phase_name} processed_input_tokens={processed_input_token_count}/{input_token_count} mlx_active_bytes={} mlx_allocator_cache_bytes={} mlx_peak_bytes={} ETA_seconds={:.1}",
                    mlx_memory_telemetry.active_memory_bytes,
                    mlx_memory_telemetry.allocator_cache_memory_bytes,
                    mlx_memory_telemetry.peak_memory_bytes,
                    (estimated_total_seconds - elapsed_seconds).max(0.0),
                );
            }
            GeneratedToken::TokenId { .. } | GeneratedToken::EndOfSequence => return Ok(()),
        }
    }
}

async fn run_generation_until_output_sample(
    qwen3_5_engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    input_token_ids: Vec<u32>,
    maximum_output_tokens: u16,
    observed_output_token_count: u16,
    phase_name: &str,
) -> Result<(), InferenceEngineError> {
    let input_token_count = input_token_ids.len();
    let generation_started_at = Instant::now();
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new_sampling(
                request_id,
                input_token_ids,
                maximum_output_tokens,
                600,
                950,
                Some(1),
            )
            .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID)
            .with_performance_attribution(PerformanceAttribution::enabled()),
        )
        .await?;
    let mut processed_input_token_count = 0usize;
    let mut generated_token_count = 0u16;
    let mut decode_started_at: Option<Instant> = None;
    let mut progress_interval = interval(PROGRESS_INTERVAL);
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    progress_interval.tick().await;
    loop {
        if decode_started_at.is_none()
            && processed_input_token_count >= input_token_count.saturating_sub(1)
        {
            decode_started_at = Some(Instant::now());
        }
        let generation_advance_future = qwen3_5_engine.decode_next_token(request_id);
        tokio::pin!(generation_advance_future);
        let generation_event = loop {
            tokio::select! {
                generation_outcome = &mut generation_advance_future => break generation_outcome?,
                _ = progress_interval.tick() => {
                    let elapsed_seconds = generation_started_at.elapsed().as_secs_f64();
                    let prefill_elapsed_seconds = decode_started_at.map_or(
                        elapsed_seconds,
                        |decode_start| {
                            decode_start
                                .saturating_duration_since(generation_started_at)
                                .as_secs_f64()
                        },
                    );
                    let prefill_tok_per_second = processed_input_token_count as f64
                        / prefill_elapsed_seconds.max(f64::EPSILON);
                    let generation_elapsed_seconds = decode_started_at
                        .map_or(0.0, |decode_start| decode_start.elapsed().as_secs_f64());
                    let generation_tok_per_second = generated_token_count as f64
                        / generation_elapsed_seconds.max(f64::EPSILON);
                    let eta_seconds = if generated_token_count > 0 {
                        f64::from(maximum_output_tokens.saturating_sub(generated_token_count))
                            / generation_tok_per_second.max(f64::EPSILON)
                    } else {
                        input_token_count.saturating_sub(processed_input_token_count) as f64
                            / prefill_tok_per_second.max(f64::EPSILON)
                    };
                    eprintln!(
                        "[paged-sustained-endurance] status=progress phase={phase_name} processed_input_tokens={processed_input_token_count}/{input_token_count} generated_tokens={generated_token_count}/{maximum_output_tokens} prefill_tok_per_second={prefill_tok_per_second:.2} generation_tok_per_second={generation_tok_per_second:.2} elapsed_seconds={elapsed_seconds:.1} ETA_seconds={eta_seconds:.1}",
                    );
                },
            }
        };
        match generation_event {
            GeneratedToken::PrefillProgress {
                processed_token_count,
                mlx_memory_telemetry,
                ..
            } => {
                let mlx_memory_telemetry = mlx_memory_telemetry
                    .expect("the enabled memory guard should report prefill telemetry");
                processed_input_token_count =
                    processed_input_token_count.saturating_add(processed_token_count as usize);
                let elapsed_seconds = generation_started_at.elapsed().as_secs_f64();
                let prefill_tok_per_second =
                    processed_input_token_count as f64 / elapsed_seconds.max(f64::EPSILON);
                let eta_seconds = input_token_count.saturating_sub(processed_input_token_count)
                    as f64
                    / prefill_tok_per_second.max(f64::EPSILON);
                eprintln!(
                    "[paged-sustained-endurance] status=progress phase={phase_name} processed_input_tokens={processed_input_token_count}/{input_token_count} generated_tokens={generated_token_count}/{maximum_output_tokens} prefill_tok_per_second={prefill_tok_per_second:.2} generation_tok_per_second=0.00 mlx_active_bytes={} mlx_allocator_cache_bytes={} mlx_peak_bytes={} ETA_seconds={eta_seconds:.1}",
                    mlx_memory_telemetry.active_memory_bytes,
                    mlx_memory_telemetry.allocator_cache_memory_bytes,
                    mlx_memory_telemetry.peak_memory_bytes,
                );
            }
            GeneratedToken::TokenId { .. } => {
                generated_token_count = generated_token_count.saturating_add(1);
                if generated_token_count >= observed_output_token_count {
                    let generation_elapsed_seconds = decode_started_at
                        .map_or(0.0, |decode_start| decode_start.elapsed().as_secs_f64());
                    let generation_tok_per_second =
                        generated_token_count as f64 / generation_elapsed_seconds.max(f64::EPSILON);
                    eprintln!(
                        "[paged-sustained-endurance] status=sample_complete phase={phase_name} processed_input_tokens={processed_input_token_count}/{input_token_count} generated_tokens={generated_token_count}/{maximum_output_tokens} observed_output_tokens={observed_output_token_count} generation_tok_per_second={generation_tok_per_second:.2} projected_full_generation_ETA_seconds={:.1}",
                        f64::from(maximum_output_tokens.saturating_sub(generated_token_count))
                            / generation_tok_per_second.max(f64::EPSILON),
                    );
                    return Ok(());
                }
            }
            GeneratedToken::EndOfSequence => {
                return Err(InferenceEngineError::InvalidRequest {
                    reason: format!(
                        "model ended sustained generation after {generated_token_count} of {maximum_output_tokens} requested output tokens"
                    ),
                });
            }
        }
    }
}

fn print_attribution_memory_timeline(performance_attribution_log_path: &Path) {
    let Ok(performance_attribution_json_lines) =
        fs::read_to_string(performance_attribution_log_path)
    else {
        return;
    };
    let attribution_reports = performance_attribution_json_lines
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .collect::<Vec<_>>();
    if let Some(model_loading_report) = attribution_reports
        .iter()
        .find(|report| report["report_kind"] == "model_loading")
    {
        eprintln!(
            "[paged-endurance] status=attribution phase=model_loaded mlx_active_bytes={} mlx_allocator_cache_bytes={} mlx_peak_bytes={}",
            model_loading_report["mlx_active_memory_bytes"],
            model_loading_report["mlx_allocator_cache_memory_bytes"],
            model_loading_report["mlx_peak_memory_bytes"],
        );
    }
    for request_id in [10_000_u64, 10_001_u64, 20_000_u64, 20_001_u64] {
        let generation_report = attribution_reports.iter().find(|report| {
            report["report_kind"] == "generation" && report["request_id"] == request_id
        });
        let Some(generation_report) = generation_report else {
            continue;
        };
        eprintln!(
            "[paged-endurance] status=attribution request_id={request_id} outcome={} mlx_active_bytes={} mlx_allocator_cache_bytes={} mlx_peak_bytes={} expert_evictions={} complete_layer_hits={} expert_page_logical_payload_bytes={}",
            generation_report["outcome"],
            generation_report["mlx_active_memory_bytes"],
            generation_report["mlx_allocator_cache_memory_bytes"],
            generation_report["mlx_peak_memory_bytes"],
            counter_amount(
                generation_report,
                "expert_weight_memory_cache_eviction_count"
            ),
            counter_amount(
                generation_report,
                "expert_weight_memory_cache_complete_layer_hit_count"
            ),
            counter_amount(generation_report, "expert_page_logical_payload_bytes",),
        );
    }
}

fn counter_amount(generation_report: &Value, counter_identifier: &str) -> u64 {
    generation_report["counters"]
        .as_array()
        .and_then(|counter_reports| {
            counter_reports.iter().find_map(|counter_report| {
                (counter_report["counter"] == counter_identifier)
                    .then(|| counter_report["amount"].as_u64())
                    .flatten()
            })
        })
        .unwrap_or(0)
}
