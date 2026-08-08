use std::path::Path;
use std::time::{Duration, Instant};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, ExpertMemoryMode,
    MtpRuntimeState, RequestId,
};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, Qwen3_5ArtifactValidator,
    Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5Tokenizer,
};
use astronomical_runtime_integration::MlxRuntime;

use super::benchmark_measurement::BenchmarkMeasurement;
use super::engine_support::{
    assert_terminal_only_speculative_prefill_attribution, generation_report_for_request,
    load_mtp_test_engine, performance_counter_amount,
};

const BENCHMARK_INPUT_TOKEN_COUNT: usize = 1_024;
const BENCHMARK_OUTPUT_TOKEN_COUNT: u16 = 1_024;
const BENCHMARK_WARMUP_OUTPUT_TOKEN_COUNT: u16 = 32;
const FOCUSED_PARITY_OUTPUT_TOKEN_COUNT: u16 = 64;
const BENCHMARK_SOURCE_TEXT: &str = include_str!(
    "../../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
#[ignore = "loads target-only and MTP production engines for the representative release gate"]
async fn should_keep_representative_mtp_generation_faster_and_target_verified() {
    tokio::time::timeout(
        Duration::from_secs(115),
        run_representative_mtp_release_gate(),
    )
    .await
    .expect("the representative MTP release gate should finish within 115 seconds");
}

#[tokio::test]
#[ignore = "loads target-only and MTP engines for focused representative greedy parity"]
async fn should_preserve_focused_representative_greedy_parity_with_mtp() {
    tokio::time::timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = super::super::configured_depth_one_mtp_model_artifact_directory();
        let benchmark_prompt_cases = prepare_benchmark_prompt_cases(&model_directory);
        let benchmark_prompt_case = &benchmark_prompt_cases[0];
        eprintln!("[mtp-parity] status=start output_tokens=64 ETA_seconds=60");
        let target_only_measurements = run_engine_benchmark_cases(
            &model_directory,
            false,
            FOCUSED_PARITY_OUTPUT_TOKEN_COUNT,
            &benchmark_prompt_cases,
            &[0],
            39_100,
        )
        .await;
        let mtp_measurements = run_engine_benchmark_cases(
            &model_directory,
            true,
            FOCUSED_PARITY_OUTPUT_TOKEN_COUNT,
            &benchmark_prompt_cases,
            &[0],
            39_200,
        )
        .await;
        let target_only_token_ids = &target_only_measurements[0].generated_token_ids;
        let mtp_token_ids = &mtp_measurements[0].generated_token_ids;
        assert_eq!(mtp_token_ids, target_only_token_ids);
        assert_eq!(
            mtp_token_ids.len(),
            usize::from(FOCUSED_PARITY_OUTPUT_TOKEN_COUNT)
        );
        eprintln!(
            "[mtp-parity] status=success phase={} output_tokens={}",
            benchmark_prompt_case.phase_name,
            mtp_token_ids.len()
        );
    })
    .await
    .expect("the focused representative MTP parity qualification should finish within 115 seconds");
}

async fn run_representative_mtp_release_gate() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::super::configured_depth_one_mtp_model_artifact_directory();
    let benchmark_prompt_cases = prepare_benchmark_prompt_cases(&model_directory);
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;

    eprintln!("[mtp-release] status=start phase=target_only_before_engine_load ETA_seconds=115");
    let target_only_before_measurements = run_engine_benchmark_cases(
        &model_directory,
        false,
        BENCHMARK_OUTPUT_TOKEN_COUNT,
        &benchmark_prompt_cases,
        &[0],
        37_000,
    )
    .await;

    MlxRuntime::initialize(mlx_memory_limits)
        .expect("the release gate should re-enter the configured MLX runtime")
        .clear_allocator_cache()
        .expect("the MTP run should not inherit reclaimable target-only allocations");

    eprintln!("[mtp-release] status=progress phase=mtp_engine_load ETA_seconds=90");
    let mtp_measurements = run_engine_benchmark_cases(
        &model_directory,
        true,
        BENCHMARK_OUTPUT_TOKEN_COUNT,
        &benchmark_prompt_cases,
        &[0],
        38_000,
    )
    .await;

    MlxRuntime::initialize(mlx_memory_limits)
        .expect("the release gate should re-enter the configured MLX runtime")
        .clear_allocator_cache()
        .expect("the final target-only run should not inherit reclaimable MTP allocations");

    eprintln!("[mtp-release] status=progress phase=target_only_after_engine_load ETA_seconds=55");
    let target_only_after_measurements = run_engine_benchmark_cases(
        &model_directory,
        false,
        BENCHMARK_OUTPUT_TOKEN_COUNT,
        &benchmark_prompt_cases,
        &[0],
        39_000,
    )
    .await;
    let configured_mlx_memory_ceiling_bytes = mlx_memory_limits.active_memory_limit_bytes() as u64;
    let allowed_peak_mlx_memory_bytes = configured_mlx_memory_ceiling_bytes
        .saturating_add(configured_mlx_memory_ceiling_bytes / 100);

    let mut paired_throughput_ratios = Vec::with_capacity(benchmark_prompt_cases.len());
    let mut paired_total_request_speedup_ratios = Vec::with_capacity(benchmark_prompt_cases.len());
    for prompt_case_index in 0..benchmark_prompt_cases.len() {
        let prompt_case = &benchmark_prompt_cases[prompt_case_index];
        let mtp_measurement = &mtp_measurements[prompt_case_index];
        let target_only_before_measurement = &target_only_before_measurements[prompt_case_index];
        let target_only_after_measurement = &target_only_after_measurements[prompt_case_index];
        assert_eq!(
            target_only_after_measurement.generated_token_ids,
            target_only_before_measurement.generated_token_ids,
            "bracketing target-only controls must preserve the same greedy sequence",
        );
        let target_only_measurement = if target_only_before_measurement
            .total_request_elapsed_seconds
            <= target_only_after_measurement.total_request_elapsed_seconds
        {
            target_only_before_measurement
        } else {
            target_only_after_measurement
        };
        for (measurement_mode, benchmark_measurement) in [
            ("mtp", mtp_measurement),
            ("target_only_before", target_only_before_measurement),
            ("target_only_after", target_only_after_measurement),
        ] {
            assert!(
                benchmark_measurement.maximum_active_mlx_memory_bytes
                    <= configured_mlx_memory_ceiling_bytes,
                "{measurement_mode} active MLX memory must remain within the configured ceiling",
            );
            assert!(
                benchmark_measurement.maximum_peak_mlx_memory_bytes
                    <= allowed_peak_mlx_memory_bytes,
                "{measurement_mode} peak MLX memory must remain within the one-percent allowance",
            );
        }
        let first_mismatched_token = mtp_measurement
            .generated_token_ids
            .iter()
            .zip(&target_only_measurement.generated_token_ids)
            .enumerate()
            .find(
                |(_generated_token_index, (mtp_token_id, target_only_token_id))| {
                    mtp_token_id != target_only_token_id
                },
            );
        assert_eq!(
            first_mismatched_token, None,
            "active MTP must preserve the target-only greedy sequence",
        );
        assert_eq!(
            mtp_measurement.generated_token_ids.len(),
            usize::from(BENCHMARK_OUTPUT_TOKEN_COUNT),
            "{} should exercise the complete representative output budget",
            prompt_case.phase_name,
        );
        assert_eq!(
            target_only_before_measurement.generated_token_ids.len(),
            usize::from(BENCHMARK_OUTPUT_TOKEN_COUNT),
            "{} leading target-only control should exercise the complete output budget",
            prompt_case.phase_name,
        );
        assert_eq!(
            target_only_after_measurement.generated_token_ids.len(),
            usize::from(BENCHMARK_OUTPUT_TOKEN_COUNT),
            "{} trailing target-only control should exercise the complete output budget",
            prompt_case.phase_name,
        );
        let target_only_throughput_baseline = target_only_before_measurement
            .tokens_per_second()
            .max(target_only_after_measurement.tokens_per_second());
        let paired_throughput_ratio =
            mtp_measurement.tokens_per_second() / target_only_throughput_baseline.max(f64::EPSILON);
        let paired_total_request_speedup_ratio = target_only_measurement
            .total_request_elapsed_seconds
            / mtp_measurement
                .total_request_elapsed_seconds
                .max(f64::EPSILON);
        paired_throughput_ratios.push(paired_throughput_ratio);
        paired_total_request_speedup_ratios.push(paired_total_request_speedup_ratio);
        eprintln!(
            "[mtp-release] status=sample phase={} prompt_tokens={} output_tokens={} target_only_before_total_request_seconds={:.3} target_only_after_total_request_seconds={:.3} target_only_before_tok_per_second={:.2} target_only_after_tok_per_second={:.2} target_only_baseline_prefill_millis={} mtp_prefill_millis={} target_only_baseline_time_to_first_token_seconds={:.3} mtp_time_to_first_token_seconds={:.3} target_only_baseline_total_request_seconds={:.3} mtp_total_request_seconds={:.3} target_only_baseline_tok_per_second={:.2} mtp_tok_per_second={:.2} throughput_ratio={:.3} total_request_speedup_ratio={:.3} target_only_active_mlx_bytes={} mtp_active_mlx_bytes={} target_only_peak_mlx_bytes={} mtp_peak_mlx_bytes={} exact_greedy_match={} first_greedy_mismatch={first_mismatched_token:?} target_only_fingerprint={:016x} mtp_fingerprint={:016x}",
            prompt_case.phase_name,
            prompt_case.prompt_token_ids.len(),
            mtp_measurement.generated_token_ids.len(),
            target_only_before_measurement.total_request_elapsed_seconds,
            target_only_after_measurement.total_request_elapsed_seconds,
            target_only_before_measurement.tokens_per_second(),
            target_only_after_measurement.tokens_per_second(),
            target_only_measurement.prefill_elapsed_millis,
            mtp_measurement.prefill_elapsed_millis,
            target_only_measurement.time_to_first_token_seconds,
            mtp_measurement.time_to_first_token_seconds,
            target_only_measurement.total_request_elapsed_seconds,
            mtp_measurement.total_request_elapsed_seconds,
            target_only_throughput_baseline,
            mtp_measurement.tokens_per_second(),
            paired_throughput_ratio,
            paired_total_request_speedup_ratio,
            target_only_measurement.maximum_active_mlx_memory_bytes,
            mtp_measurement.maximum_active_mlx_memory_bytes,
            target_only_measurement.maximum_peak_mlx_memory_bytes,
            mtp_measurement.maximum_peak_mlx_memory_bytes,
            first_mismatched_token.is_none(),
            target_only_measurement.generated_token_id_fingerprint(),
            mtp_measurement.generated_token_id_fingerprint(),
        );
    }
    let minimum_paired_throughput_ratio = paired_throughput_ratios
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let minimum_paired_total_request_speedup_ratio = paired_total_request_speedup_ratios
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    assert!(
        minimum_paired_throughput_ratio >= 1.05,
        "paired MTP throughput must exceed target-only by at least five percent"
    );
    assert!(
        minimum_paired_total_request_speedup_ratio >= 1.05,
        "paired MTP total request latency must improve by at least five percent"
    );
    eprintln!(
        "[mtp-release] status=success minimum_throughput_ratio={minimum_paired_throughput_ratio:.3} minimum_total_request_speedup_ratio={minimum_paired_total_request_speedup_ratio:.3}"
    );
}

fn prepare_benchmark_prompt_cases(model_directory: &Path) -> Vec<BenchmarkPromptCase> {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the representative MTP artifact should validate before tokenization");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the representative MTP tokenizer should load");
    let image_pad_token_id = tokenizer.image_pad_token_id();

    [
        (
            "representative_technical_briefing",
            "Explain the source material as a factual technical briefing. Write at least seven hundred words.",
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(prompt_case_index, (phase_name, prompt_text))| {
        let request_id = RequestId::new(37_900 + prompt_case_index as u64);
        let chat_generation_command = ChatGenerationCommand {
            request_id,
            model: model_directory
                .file_name()
                .expect("the configured MTP model directory should have a leaf name")
                .to_string_lossy()
                .into_owned(),
            messages: vec![ChatMessage::User {
                content: format!("{prompt_text}\n\nSource material:\n{BENCHMARK_SOURCE_TEXT}"),
                images: Vec::new(),
            }],
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            settings: ChatGenerationSettings {
                max_output_tokens: BENCHMARK_OUTPUT_TOKEN_COUNT,
                temperature_thousandths: Some(0),
                top_p_thousandths: Some(1_000),
                seed: None,
                thinking_budget: None,
            },
        };
        let inference_request = tokenizer
            .prepare_chat(&chat_generation_command, false)
            .expect("each representative prompt should prepare through the production tokenizer");
        let complete_prompt_token_ids = inference_request.input_token_ids();
        let assistant_suffix_start = complete_prompt_token_ids
            .iter()
            .rposition(|token_id| *token_id == tokenizer.im_end_token_id())
            .expect("the prepared benchmark prompt should contain the closing user marker");
        let assistant_suffix_token_ids = &complete_prompt_token_ids[assistant_suffix_start..];
        let retained_prompt_prefix_token_count = BENCHMARK_INPUT_TOKEN_COUNT
            .checked_sub(assistant_suffix_token_ids.len())
            .expect("the assistant suffix should fit the benchmark input budget");
        assert!(retained_prompt_prefix_token_count < assistant_suffix_start);
        let mut prompt_token_ids =
            complete_prompt_token_ids[..retained_prompt_prefix_token_count].to_vec();
        prompt_token_ids.extend_from_slice(assistant_suffix_token_ids);
        assert_eq!(prompt_token_ids.len(), BENCHMARK_INPUT_TOKEN_COUNT);
        BenchmarkPromptCase {
            phase_name,
            prompt_token_ids,
            image_pad_token_id,
        }
    })
    .collect::<Vec<_>>()
}

async fn run_engine_benchmark_cases(
    model_directory: &Path,
    mtp_enabled: bool,
    measured_output_token_count: u16,
    benchmark_prompt_cases: &[BenchmarkPromptCase],
    measured_prompt_order: &[usize],
    request_id_base: u64,
) -> Vec<BenchmarkMeasurement> {
    let (mut engine, _temporary_log_directory, performance_attribution_log_path) =
        load_mtp_test_engine(model_directory, mtp_enabled, false).await;
    let engine_load_result = engine
        .load()
        .await
        .expect("the representative benchmark engine should load");
    assert_eq!(
        engine_load_result.mtp_runtime_state(),
        if mtp_enabled {
            MtpRuntimeState::Active
        } else {
            MtpRuntimeState::Disabled
        }
    );
    engine
        .disable_adaptive_ram_growth_memory_guard_for_tests()
        .await
        .expect("the representative benchmark should disable adaptive memory guarding");
    engine
        .reset_mlx_peak_memory_for_tests()
        .await
        .expect("the representative warmup should start with a fresh MLX peak");
    let warmup_measurement = run_one_generation(
        &mut engine,
        RequestId::new(request_id_base),
        &benchmark_prompt_cases[measured_prompt_order[0]],
        BENCHMARK_WARMUP_OUTPUT_TOKEN_COUNT,
        PerformanceAttribution::enabled(),
    )
    .await;
    let warmup_request_id = RequestId::new(request_id_base);
    let warmup_generation_report =
        generation_report_for_request(&performance_attribution_log_path, warmup_request_id);
    assert_terminal_only_speculative_prefill_attribution(
        &warmup_generation_report,
        mtp_enabled,
        &warmup_measurement.completed_prefill_chunck_tokens,
    );
    if mtp_enabled {
        let admitted_attempt_count =
            performance_counter_amount(&warmup_generation_report, "mtp_admitted_attempt_count");
        let accepted_draft_count =
            performance_counter_amount(&warmup_generation_report, "mtp_accepted_draft_count");
        let rejected_draft_count =
            performance_counter_amount(&warmup_generation_report, "mtp_rejected_draft_count");
        let operational_fallback_count =
            performance_counter_amount(&warmup_generation_report, "mtp_operational_fallback_count");
        let memory_admission_fallback_count = performance_counter_amount(
            &warmup_generation_report,
            "mtp_memory_admission_fallback_count",
        );
        let prompt_history_initialization_fallback_count = performance_counter_amount(
            &warmup_generation_report,
            "mtp_prompt_history_initialization_fallback_count",
        );
        assert!(
            admitted_attempt_count > 0,
            "the MTP warmup must execute target-authoritative verification",
        );
        assert_eq!(
            accepted_draft_count + rejected_draft_count + operational_fallback_count,
            admitted_attempt_count,
            "every admitted MTP proposal must have one recorded verifier outcome",
        );
        assert_eq!(
            operational_fallback_count, 0,
            "the representative MTP warmup must not require operational fallback",
        );
        assert_eq!(
            memory_admission_fallback_count, 0,
            "the representative MTP warmup must fit its complete verification workspace",
        );
        assert_eq!(
            prompt_history_initialization_fallback_count, 0,
            "the representative MTP warmup must initialize prompt history",
        );
        eprintln!(
            "[mtp-release] status=diagnostic phase=warmup output_tokens={} admitted_attempts={} accepted_drafts={} rejected_drafts={} acceptance_rate={:.3}",
            warmup_measurement.generated_token_ids.len(),
            admitted_attempt_count,
            accepted_draft_count,
            rejected_draft_count,
            accepted_draft_count as f64 / admitted_attempt_count.max(1) as f64,
        );
    }

    let mut indexed_measurements = Vec::with_capacity(benchmark_prompt_cases.len());
    for (measured_position, prompt_case_index) in measured_prompt_order.iter().copied().enumerate()
    {
        let request_id = RequestId::new(request_id_base + measured_position as u64 + 1);
        engine
            .reset_mlx_peak_memory_for_tests()
            .await
            .expect("each representative measurement should start with a fresh MLX peak");
        eprintln!(
            "[mtp-release] status=progress runtime={} phase={} output_tokens={} ETA_seconds=40",
            if mtp_enabled { "mtp" } else { "target_only" },
            benchmark_prompt_cases[prompt_case_index].phase_name,
            measured_output_token_count,
        );
        let benchmark_measurement = run_one_generation(
            &mut engine,
            request_id,
            &benchmark_prompt_cases[prompt_case_index],
            measured_output_token_count,
            PerformanceAttribution::disabled(),
        )
        .await;
        indexed_measurements.push((prompt_case_index, benchmark_measurement));
    }
    drop(engine);
    indexed_measurements.sort_by_key(|(prompt_case_index, _measurement)| *prompt_case_index);
    indexed_measurements
        .into_iter()
        .map(|(_prompt_case_index, measurement)| measurement)
        .collect()
}

async fn run_one_generation(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    benchmark_prompt_case: &BenchmarkPromptCase,
    maximum_output_tokens: u16,
    performance_attribution: PerformanceAttribution,
) -> BenchmarkMeasurement {
    let request_started_at = Instant::now();
    let inference_request = Qwen3_5InferenceRequest::new(
        request_id,
        benchmark_prompt_case.prompt_token_ids.clone(),
        maximum_output_tokens,
    )
    .with_image_pad_token_id(benchmark_prompt_case.image_pad_token_id)
    .with_performance_attribution(performance_attribution);
    let generation_start = engine
        .start_generation(inference_request)
        .await
        .expect("the representative benchmark request should start");
    assert_eq!(
        generation_start.expert_memory_mode(),
        Some(ExpertMemoryMode::Resident),
        "the representative benchmark must isolate fully resident generation"
    );

    let mut measurement = BenchmarkMeasurement::default();
    let mut first_generated_token_at = None;
    loop {
        match engine
            .decode_next_token(request_id)
            .await
            .expect("the representative benchmark request should advance")
        {
            GeneratedToken::TokenId {
                token_id,
                mlx_memory_telemetry,
                generation_finalization,
                ..
            } => {
                let generated_token_at = Instant::now();
                if first_generated_token_at.is_none() {
                    measurement.time_to_first_token_seconds = generated_token_at
                        .saturating_duration_since(request_started_at)
                        .as_secs_f64();
                    first_generated_token_at = Some(generated_token_at);
                }
                measurement.record_mlx_memory_telemetry(mlx_memory_telemetry);
                measurement.generated_token_ids.push(token_id);
                if measurement.generated_token_ids.len().is_multiple_of(16) {
                    eprintln!(
                        "[mtp-generation-benchmark] status=progress phase={} generated_tokens={}/{}",
                        benchmark_prompt_case.phase_name,
                        measurement.generated_token_ids.len(),
                        maximum_output_tokens,
                    );
                }
                if let Some(generation_finalization) = generation_finalization {
                    measurement.generation_elapsed_seconds = generated_token_at
                        .saturating_duration_since(
                            first_generated_token_at.expect("the first token time should exist"),
                        )
                        .as_secs_f64();
                    measurement.total_request_elapsed_seconds = generated_token_at
                        .saturating_duration_since(request_started_at)
                        .as_secs_f64();
                    assert!(generation_finalization.has_reportable_state());
                    measurement.record_mlx_memory_telemetry(
                        generation_finalization.mlx_memory_telemetry(),
                    );
                    return measurement;
                }
            }
            GeneratedToken::PrefillProgress {
                completed_prefill_chunck_tokens,
                elapsed_millis,
                mlx_memory_telemetry,
                ..
            } => {
                measurement
                    .completed_prefill_chunck_tokens
                    .push(completed_prefill_chunck_tokens as usize);
                measurement.prefill_elapsed_millis = measurement
                    .prefill_elapsed_millis
                    .saturating_add(elapsed_millis);
                measurement.record_mlx_memory_telemetry(mlx_memory_telemetry);
            }
            GeneratedToken::EndOfSequence => {
                panic!("the Qwen benchmark should finalize on an emitted token")
            }
        }
    }
}

struct BenchmarkPromptCase {
    phase_name: &'static str,
    prompt_token_ids: Vec<u32>,
    image_pad_token_id: u32,
}
