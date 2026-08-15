use std::{fs, path::Path, time::Duration};

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, MlxMemoryTelemetry, PerformanceAttribution,
    PerformanceAttributionLog, Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5InferenceRequest,
    Qwen3_5Model, Qwen3_5PromptProcessingChunkSizer,
};
use astronomical_runtime_integration::MlxRuntime;
use serde_json::Value;
use tokio::time::{Instant, interval, timeout};

use super::expert_paging_prefill::maximum_absolute_difference;
use super::expert_paging_prefill_performance::prepare_reproduced_long_prompt_token_ids;

const QUALIFICATION_PROMPT_TOKEN_COUNT: usize = 4_097;
const QUALIFICATION_OUTPUT_TOKEN_COUNT: u16 = 512;
const QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);
const CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS: u32 = 8_192;
const CAPACITY_RETRY_PROMPT_TOKEN_COUNT: usize = CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS as usize + 1;
const CAPACITY_RETRY_REFERENCE_REQUEST_ID_VALUE: u64 = 72;
const CAPACITY_RETRY_CONSTRAINED_REQUEST_ID_VALUE: u64 = 73;

#[tokio::test]
#[ignore = "loads the configured model and requires exact final-prefill logits for 2048 and 4096 chunks"]
async fn should_preserve_exact_final_prefill_logits_between_fixed_prefill_sizes() {
    timeout(
        QUALIFICATION_TIMEOUT,
        assert_exact_final_prefill_logit_parity(),
    )
    .await
    .expect("the final-prefill logit parity contract must finish within 115 seconds");
}

async fn assert_exact_final_prefill_logit_parity() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let configured_model_directory = crate::common::configured_ornith_model_artifact_directory();
    let prompt_token_ids = prepare_reproduced_long_prompt_token_ids(
        QUALIFICATION_PROMPT_TOKEN_COUNT,
        QUALIFICATION_OUTPUT_TOKEN_COUNT,
    );
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(
            &configured_model_directory,
            u32::from(QUALIFICATION_OUTPUT_TOKEN_COUNT),
        )
        .expect("the Ornith artifact should validate before final-logit parity qualification");
    let qwen3_5_config = validated_artifact.config().clone();
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the machine-derived final-logit parity runtime should initialize");
    eprintln!("[prefill-logit-parity] status=progress phase=model_load");
    let qwen3_5_model = Qwen3_5Model::load(
        runtime,
        validated_artifact,
        &configured_model_directory,
        false,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the configured model should load for final-logit parity qualification");
    let (baseline_greedy_token_id, baseline_final_logits) = final_prefill_logits_for_chunk_size(
        &qwen3_5_model,
        &qwen3_5_config,
        &prompt_token_ids,
        2_048,
    );
    let (candidate_greedy_token_id, candidate_final_logits) = final_prefill_logits_for_chunk_size(
        &qwen3_5_model,
        &qwen3_5_config,
        &prompt_token_ids,
        4_096,
    );
    let maximum_absolute_logit_delta =
        maximum_absolute_difference(&baseline_final_logits, &candidate_final_logits);
    assert!(
        maximum_absolute_logit_delta.is_finite(),
        "full-chunk logit delta must remain finite"
    );
    assert_eq!(
        maximum_absolute_logit_delta, 0.0,
        "fixed 2048 and 4096 prefill sizes must produce exact final logits"
    );
    eprintln!(
        "[prefill-logit-parity] status=success baseline_greedy_token_id={baseline_greedy_token_id} candidate_greedy_token_id={candidate_greedy_token_id} maximum_absolute_logit_delta={maximum_absolute_logit_delta:.6}"
    );
}

fn final_prefill_logits_for_chunk_size(
    qwen3_5_model: &Qwen3_5Model,
    qwen3_5_config: &astronomical_model_serving::Qwen3_5Config,
    prompt_token_ids: &[u32],
    prefill_chunck_tokens: usize,
) -> (u32, Vec<f32>) {
    let mut request_decoder_state = crate::common::standard_request_decoder_state(qwen3_5_config);
    let final_prompt_token_index = prompt_token_ids
        .len()
        .checked_sub(1)
        .expect("the diagnostic prompt should contain a final decode-seeding token");
    for prefill_chunck_start in (0..final_prompt_token_index).step_by(prefill_chunck_tokens) {
        let prefill_chunck_end = prefill_chunck_start
            .saturating_add(prefill_chunck_tokens)
            .min(final_prompt_token_index);
        qwen3_5_model
            .prefill_chunck(
                &prompt_token_ids[prefill_chunck_start..prefill_chunck_end],
                u32::try_from(prefill_chunck_start)
                    .expect("the diagnostic prompt position should fit u32"),
                &mut request_decoder_state,
            )
            .expect("the diagnostic prefill chunk should complete");
    }
    let final_position_logits = qwen3_5_model
        .forward_chunk(
            &prompt_token_ids[final_prompt_token_index..],
            u32::try_from(final_prompt_token_index)
                .expect("the diagnostic final prompt position should fit u32"),
            &mut request_decoder_state,
        )
        .expect("the diagnostic final prompt token should produce logits");
    let greedy_token_id = qwen3_5_model
        .greedy_token_id(&final_position_logits)
        .expect("the diagnostic final logits should produce a greedy token");
    let final_logits = final_position_logits
        .to_vec_f32()
        .expect("the diagnostic final logits should materialize to CPU");
    assert!(
        final_logits.iter().all(|logit| logit.is_finite()),
        "the diagnostic final logits must remain finite"
    );
    (greedy_token_id, final_logits)
}

#[tokio::test]
#[ignore = "loads the configured model and forces one native prefill-capacity retry"]
async fn should_retry_native_prefill_capacity_rejection_after_automatic_expert_reclamation_without_changing_continuation()
 {
    timeout(
        QUALIFICATION_TIMEOUT,
        run_native_prefill_capacity_retry_qualification(),
    )
    .await
    .expect("the native prefill-capacity retry qualification must finish within 115 seconds");
}

struct CapacityRetryQualificationContinuation {
    generated_token_ids: Vec<u32>,
    completed_prefill_chunck_token_counts: Vec<u32>,
    final_mlx_memory_telemetry: MlxMemoryTelemetry,
    maximum_observed_peak_memory_bytes: u64,
}

async fn run_native_prefill_capacity_retry_qualification() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let configured_model_directory = crate::common::configured_ornith_model_artifact_directory();
    let prompt_token_ids = prepare_reproduced_long_prompt_token_ids(
        CAPACITY_RETRY_PROMPT_TOKEN_COUNT,
        QUALIFICATION_OUTPUT_TOKEN_COUNT,
    );
    assert_eq!(prompt_token_ids.len(), CAPACITY_RETRY_PROMPT_TOKEN_COUNT);
    let temporary_log_directory = tempfile::tempdir()
        .expect("the retry qualification should create a temporary log directory");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("prefill-capacity-retry-attribution.jsonl");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(
            &configured_model_directory,
            u32::from(QUALIFICATION_OUTPUT_TOKEN_COUNT),
        )
        .expect("the configured artifact should validate before retry qualification");
    let performance_attribution_log =
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("the retry qualification should open an attribution log");
    let mut qwen3_5_engine =
        Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
            validated_artifact,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            None,
            Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
                CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS,
            )
            .expect("the retry qualification prefill chunk size should be valid"),
            super::ORNITH_IMAGE_PAD_TOKEN_ID,
            configured_model_directory,
            crate::common::standard_worker_chunking_configuration(),
            true,
            false,
            crate::common::disabled_worker_speculative_prefill_configuration(),
            PerformanceAttribution::enabled(),
            performance_attribution_log,
        )
        .expect("the retry qualification engine settings should be valid");
    let engine_load_result =
        load_engine_with_progress(&mut qwen3_5_engine, "retry-reference").await;
    let reference_continuation = run_capacity_retry_continuation(
        &mut qwen3_5_engine,
        RequestId::new(CAPACITY_RETRY_REFERENCE_REQUEST_ID_VALUE),
        &prompt_token_ids,
        "retry-reference",
    )
    .await;
    let constrained_mlx_memory_ceiling_bytes = capacity_retry_memory_ceiling_bytes(
        &reference_continuation,
        engine_load_result.minimum_mlx_memory_ceiling_bytes(),
    );
    qwen3_5_engine
        .disable_adaptive_ram_growth_memory_guard_for_tests()
        .await
        .expect("the retry qualification should bypass predictive admission after calibration");
    let mlx_memory_limit_adjustment = qwen3_5_engine
        .update_mlx_memory_limit(constrained_mlx_memory_ceiling_bytes)
        .await
        .expect("the retry qualification should lower the idle MLX memory ceiling");
    assert_eq!(
        mlx_memory_limit_adjustment.effective_mlx_memory_ceiling_bytes(),
        constrained_mlx_memory_ceiling_bytes,
        "the retry qualification must apply its runtime-derived MLX memory ceiling"
    );
    qwen3_5_engine
        .reset_mlx_peak_memory_for_tests()
        .await
        .expect("the constrained retry qualification should reset its MLX peak sample");
    let constrained_continuation = run_capacity_retry_continuation(
        &mut qwen3_5_engine,
        RequestId::new(CAPACITY_RETRY_CONSTRAINED_REQUEST_ID_VALUE),
        &prompt_token_ids,
        "retry-constrained",
    )
    .await;
    let first_completed_prefill_chunck_tokens = constrained_continuation
        .completed_prefill_chunck_token_counts
        .first()
        .copied()
        .expect("the constrained request should complete a prefill chunk after retrying");
    assert!(
        is_geometric_prefill_chunck_reduction(
            CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS,
            first_completed_prefill_chunck_tokens,
        ),
        "the first successful prefill chunk must be a geometric reduction after native rejection: requested_tokens={CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS}, completed_tokens={first_completed_prefill_chunck_tokens}"
    );
    assert_eq!(
        constrained_continuation.generated_token_ids, reference_continuation.generated_token_ids,
        "native capacity retry changed the 512-token greedy continuation"
    );
    let allowed_peak_memory_bytes = allowed_peak_memory_bytes(constrained_mlx_memory_ceiling_bytes);
    assert!(
        constrained_continuation.maximum_observed_peak_memory_bytes <= allowed_peak_memory_bytes,
        "successful constrained forwards exceeded P: peak_bytes={}, allowed_peak_bytes={allowed_peak_memory_bytes}",
        constrained_continuation.maximum_observed_peak_memory_bytes,
    );
    assert!(
        constrained_continuation
            .final_mlx_memory_telemetry
            .active_memory_bytes
            <= constrained_mlx_memory_ceiling_bytes,
        "request-finalized stable memory exceeded C: active_bytes={}, ceiling_bytes={constrained_mlx_memory_ceiling_bytes}",
        constrained_continuation
            .final_mlx_memory_telemetry
            .active_memory_bytes,
    );
    drop(qwen3_5_engine);
    assert_capacity_retry_attribution(
        &performance_attribution_log_path,
        CAPACITY_RETRY_CONSTRAINED_REQUEST_ID_VALUE,
        constrained_continuation
            .completed_prefill_chunck_token_counts
            .len(),
    );
}

async fn run_capacity_retry_continuation(
    qwen3_5_engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    prompt_token_ids: &[u32],
    qualification_label: &str,
) -> CapacityRetryQualificationContinuation {
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                request_id,
                prompt_token_ids.to_vec(),
                QUALIFICATION_OUTPUT_TOKEN_COUNT,
            )
            .with_image_pad_token_id(super::ORNITH_IMAGE_PAD_TOKEN_ID)
            .with_performance_attribution(PerformanceAttribution::enabled()),
        )
        .await
        .expect("the retry qualification request should start");
    let mut generated_token_ids = Vec::with_capacity(usize::from(QUALIFICATION_OUTPUT_TOKEN_COUNT));
    let mut completed_prefill_chunck_token_counts = Vec::new();
    let mut final_mlx_memory_telemetry = None;
    let mut maximum_observed_peak_memory_bytes = 0;
    while generated_token_ids.len() < usize::from(QUALIFICATION_OUTPUT_TOKEN_COUNT) {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("the retry qualification request should advance")
        {
            GeneratedToken::PrefillProgress {
                completed_prefill_chunk_tokens,
                mlx_memory_telemetry,
                processed_token_count,
                ..
            } => {
                completed_prefill_chunck_token_counts.push(completed_prefill_chunk_tokens);
                record_peak_memory(
                    &mut maximum_observed_peak_memory_bytes,
                    mlx_memory_telemetry,
                );
                eprintln!(
                    "[prefill-capacity-retry:{qualification_label}] status=progress phase=prefill processed_tokens={processed_token_count}/{} completed_chunk_tokens={completed_prefill_chunk_tokens}",
                    prompt_token_ids.len() - 1
                );
            }
            GeneratedToken::TokenId {
                token_id,
                generation_finalization,
                mlx_memory_telemetry,
                ..
            } => {
                record_peak_memory(
                    &mut maximum_observed_peak_memory_bytes,
                    mlx_memory_telemetry,
                );
                if let Some(generation_finalization) = generation_finalization {
                    final_mlx_memory_telemetry = generation_finalization.mlx_memory_telemetry();
                    record_peak_memory(
                        &mut maximum_observed_peak_memory_bytes,
                        final_mlx_memory_telemetry,
                    );
                }
                generated_token_ids.push(token_id);
                if generated_token_ids.len().is_multiple_of(25) {
                    eprintln!(
                        "[prefill-capacity-retry:{qualification_label}] status=progress phase=decode generated_tokens={}/{}",
                        generated_token_ids.len(),
                        QUALIFICATION_OUTPUT_TOKEN_COUNT
                    );
                }
            }
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => {
                panic!("the retry qualification ended before 512 output tokens");
            }
        }
    }
    CapacityRetryQualificationContinuation {
        generated_token_ids,
        completed_prefill_chunck_token_counts,
        final_mlx_memory_telemetry: final_mlx_memory_telemetry
            .expect("the retry qualification should finalize MLX memory after 512 tokens"),
        maximum_observed_peak_memory_bytes,
    }
}

fn capacity_retry_memory_ceiling_bytes(
    reference_continuation: &CapacityRetryQualificationContinuation,
    minimum_mlx_memory_ceiling_bytes: u64,
) -> u64 {
    let reference_stable_memory_bytes = reference_continuation
        .final_mlx_memory_telemetry
        .active_memory_bytes;
    let reference_peak_memory_bytes = reference_continuation.maximum_observed_peak_memory_bytes;
    assert!(
        reference_peak_memory_bytes > reference_stable_memory_bytes,
        "the calibration request must expose a transient prefill window: stable_bytes={reference_stable_memory_bytes}, peak_bytes={reference_peak_memory_bytes}"
    );
    let reclaimable_idle_residency_bytes =
        reference_stable_memory_bytes.saturating_sub(minimum_mlx_memory_ceiling_bytes);
    assert!(
        reclaimable_idle_residency_bytes > 0,
        "the retry qualification needs some reclaimable expert residency"
    );
    let constrained_mlx_memory_ceiling_bytes =
        minimum_mlx_memory_ceiling_bytes.saturating_add(reclaimable_idle_residency_bytes / 3);
    eprintln!(
        "[prefill-capacity-retry] status=progress phase=calibrated stable_bytes={reference_stable_memory_bytes} peak_bytes={reference_peak_memory_bytes} minimum_ceiling_bytes={minimum_mlx_memory_ceiling_bytes} selected_ceiling_bytes={constrained_mlx_memory_ceiling_bytes} allowed_peak_bytes={}",
        allowed_peak_memory_bytes(constrained_mlx_memory_ceiling_bytes),
    );
    assert!(
        allowed_peak_memory_bytes(constrained_mlx_memory_ceiling_bytes)
            < reference_peak_memory_bytes,
        "the runtime-derived ceiling must reject the calibrated peak: ceiling_bytes={constrained_mlx_memory_ceiling_bytes}, allowed_peak_bytes={}, reference_peak_bytes={reference_peak_memory_bytes}",
        allowed_peak_memory_bytes(constrained_mlx_memory_ceiling_bytes),
    );
    constrained_mlx_memory_ceiling_bytes
}

const fn allowed_peak_memory_bytes(mlx_memory_ceiling_bytes: u64) -> u64 {
    mlx_memory_ceiling_bytes.saturating_add(mlx_memory_ceiling_bytes / 100)
}

fn is_geometric_prefill_chunck_reduction(
    requested_prefill_chunck_tokens: u32,
    completed_prefill_chunck_tokens: u32,
) -> bool {
    let mut reduced_prefill_chunck_tokens = requested_prefill_chunck_tokens;
    while reduced_prefill_chunck_tokens > 1 {
        reduced_prefill_chunck_tokens /= 2;
        if reduced_prefill_chunck_tokens == completed_prefill_chunck_tokens {
            return true;
        }
    }
    false
}

fn record_peak_memory(
    maximum_observed_peak_memory_bytes: &mut u64,
    mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
) {
    if let Some(mlx_memory_telemetry) = mlx_memory_telemetry {
        *maximum_observed_peak_memory_bytes =
            (*maximum_observed_peak_memory_bytes).max(mlx_memory_telemetry.peak_memory_bytes);
    }
}

fn assert_capacity_retry_attribution(
    performance_attribution_log_path: &Path,
    request_id: u64,
    completed_prefill_chunck_count: usize,
) {
    let generation_attribution_report = fs::read_to_string(performance_attribution_log_path)
        .expect("the retry qualification should write JSON Lines attribution")
        .lines()
        .map(|json_line| {
            serde_json::from_str::<Value>(json_line)
                .expect("every retry qualification attribution record should be valid JSON")
        })
        .find(|attribution_document| {
            attribution_document["report_kind"] == "generation"
                && attribution_document["request_id"] == request_id
        })
        .unwrap_or_else(|| {
            panic!(
                "the retry qualification should write a generation report for request {request_id}"
            )
        });
    let prefill_capacity_rejection_count = attribution_counter_amount(
        &generation_attribution_report,
        "prefill_capacity_rejection_count",
    );
    let prefill_capacity_retry_count = attribution_counter_amount(
        &generation_attribution_report,
        "prefill_capacity_retry_count",
    );
    let prefill_attempt_count =
        attribution_counter_amount(&generation_attribution_report, "prefill_chunck_count");
    assert!(
        prefill_capacity_rejection_count > 0,
        "the constrained request must observe a typed native capacity rejection"
    );
    assert!(
        prefill_capacity_retry_count > 0,
        "the constrained request must retry after a native capacity rejection"
    );
    assert!(
        prefill_attempt_count > u64::try_from(completed_prefill_chunck_count).unwrap_or(u64::MAX),
        "the failed prefill attempt must not advance observable progress"
    );
}

fn attribution_counter_amount(
    generation_attribution_report: &Value,
    counter_identifier: &str,
) -> u64 {
    generation_attribution_report["counters"]
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

async fn load_engine_with_progress(
    qwen3_5_engine: &mut Qwen3_5Engine,
    qualification_label: &str,
) -> astronomical_model_serving::EngineLoadResult {
    eprintln!("[prefill-capacity-retry:{qualification_label}] status=progress phase=model_load");
    let model_load_started_at = Instant::now();
    let mut model_load_progress_interval = interval(Duration::from_secs(5));
    model_load_progress_interval.tick().await;
    let model_load_future = qwen3_5_engine.load();
    tokio::pin!(model_load_future);
    loop {
        tokio::select! {
            model_load_outcome = &mut model_load_future => {
                return model_load_outcome.expect("the retry qualification engine should load");
            }
            _ = model_load_progress_interval.tick() => eprintln!(
                "[prefill-capacity-retry:{qualification_label}] status=progress phase=model_load elapsed_seconds={}",
                model_load_started_at.elapsed().as_secs()
            ),
        }
    }
}
