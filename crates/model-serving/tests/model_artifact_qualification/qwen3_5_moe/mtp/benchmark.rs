use std::cmp::Reverse;
use std::hash::{DefaultHasher, Hash, Hasher};
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

use super::IMAGE_PAD_TOKEN_ID;
use super::engine_support::{generation_report_for_request, load_mtp_test_engine};

const BENCHMARK_INPUT_TOKEN_COUNT: usize = 1_024;
const BENCHMARK_OUTPUT_TOKEN_COUNT: u16 = 1_024;
const BENCHMARK_SOURCE_TEXT: &str = include_str!(
    "../../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
#[ignore = "loads target-only and MTP production engines for the representative release gate"]
async fn should_keep_representative_mtp_generation_faster_and_exact() {
    tokio::time::timeout(
        Duration::from_secs(115),
        run_representative_mtp_release_gate(),
    )
    .await
    .expect("the representative MTP release gate should finish within 115 seconds");
}

#[tokio::test]
#[ignore = "loads the fully resident target model for representative decode measurement"]
async fn should_measure_representative_fully_resident_target_only_decode() {
    tokio::time::timeout(
        Duration::from_secs(60),
        run_representative_fully_resident_target_only_decode(),
    )
    .await
    .expect("representative fully resident decode should finish within 60 seconds");
}

#[tokio::test]
#[ignore = "loads the fully resident target model for representative decode attribution"]
async fn should_attribute_representative_fully_resident_target_only_decode() {
    tokio::time::timeout(
        Duration::from_secs(60),
        run_representative_fully_resident_target_only_decode_attribution(),
    )
    .await
    .expect("representative fully resident decode attribution should finish within 60 seconds");
}

async fn run_representative_fully_resident_target_only_decode() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::super::qwen3_6_35b_a3b_oq4e_mtp_model_directory();
    let benchmark_prompt_cases = prepare_benchmark_prompt_cases(&model_directory);
    eprintln!(
        "[oq4e-resident-decode] status=start input_tokens={} output_tokens={} ETA_seconds=60",
        BENCHMARK_INPUT_TOKEN_COUNT, BENCHMARK_OUTPUT_TOKEN_COUNT,
    );
    let (mut target_only_engine, _temporary_log_directory, _performance_attribution_log_path) =
        load_mtp_test_engine(&model_directory, false).await;
    target_only_engine
        .load()
        .await
        .expect("the representative fully resident target-only engine should load");
    target_only_engine
        .disable_adaptive_ram_growth_memory_guard_for_tests()
        .await
        .expect("the representative resident benchmark should disable adaptive guarding");
    let target_only_measurement = run_one_generation(
        &mut target_only_engine,
        RequestId::new(40_100),
        &benchmark_prompt_cases[0],
        BENCHMARK_OUTPUT_TOKEN_COUNT,
        PerformanceAttribution::disabled(),
    )
    .await;
    assert_eq!(
        target_only_measurement.generated_token_ids.len(),
        usize::from(BENCHMARK_OUTPUT_TOKEN_COUNT),
        "the resident decode measurement must use the complete output budget"
    );
    eprintln!(
        "[oq4e-resident-decode] status=success input_tokens={} output_tokens={} output_fingerprint={:016x} target_only_tok_per_second={:.2}",
        BENCHMARK_INPUT_TOKEN_COUNT,
        BENCHMARK_OUTPUT_TOKEN_COUNT,
        target_only_measurement.generated_token_id_fingerprint(),
        target_only_measurement.tokens_per_second(),
    );
}

async fn run_representative_fully_resident_target_only_decode_attribution() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::super::qwen3_6_35b_a3b_oq4e_mtp_model_directory();
    let benchmark_prompt_cases = prepare_benchmark_prompt_cases(&model_directory);
    let (mut target_only_engine, _temporary_log_directory, performance_attribution_log_path) =
        load_mtp_test_engine(&model_directory, false).await;
    target_only_engine
        .load()
        .await
        .expect("the attributed fully resident target-only engine should load");
    target_only_engine
        .disable_adaptive_ram_growth_memory_guard_for_tests()
        .await
        .expect("the attributed resident benchmark should disable adaptive guarding");
    let request_id = RequestId::new(40_101);
    let target_only_measurement = run_one_generation(
        &mut target_only_engine,
        request_id,
        &benchmark_prompt_cases[0],
        BENCHMARK_OUTPUT_TOKEN_COUNT,
        PerformanceAttribution::enabled(),
    )
    .await;
    drop(target_only_engine);

    let generation_report =
        generation_report_for_request(&performance_attribution_log_path, request_id);
    assert_eq!(
        generation_report["retained_complete_expert_layer_count"], 40,
        "the attributed benchmark must remain fully resident"
    );
    let mut attributed_operations = generation_report["operations"]
        .as_array()
        .expect("the generation report should contain operations")
        .iter()
        .collect::<Vec<_>>();
    attributed_operations.sort_by_key(|attributed_operation| {
        Reverse(
            attributed_operation["total_elapsed_nanoseconds"]
                .as_u64()
                .unwrap_or(0),
        )
    });
    for attributed_operation in attributed_operations.into_iter().take(20) {
        eprintln!(
            "[oq4e-resident-attribution] operation={} occurrences={} total_milliseconds={:.3} average_microseconds={:.3}",
            attributed_operation["operation"]
                .as_str()
                .unwrap_or("unknown"),
            attributed_operation["occurrence_count"]
                .as_u64()
                .unwrap_or(0),
            attributed_operation["total_elapsed_nanoseconds"]
                .as_u64()
                .unwrap_or(0) as f64
                / 1_000_000.0,
            attributed_operation["total_elapsed_nanoseconds"]
                .as_u64()
                .unwrap_or(0) as f64
                / attributed_operation["occurrence_count"]
                    .as_u64()
                    .unwrap_or(1)
                    .max(1) as f64
                / 1_000.0,
        );
    }
    eprintln!(
        "[oq4e-resident-attribution] status=success input_tokens={} output_tokens={} target_only_tok_per_second={:.2} report_elapsed_milliseconds={:.3} attributed_percent={:.2}",
        BENCHMARK_INPUT_TOKEN_COUNT,
        BENCHMARK_OUTPUT_TOKEN_COUNT,
        target_only_measurement.tokens_per_second(),
        generation_report["report_elapsed_nanoseconds"]
            .as_u64()
            .unwrap_or(0) as f64
            / 1_000_000.0,
        generation_report["attributed_percent"]
            .as_f64()
            .unwrap_or(0.0),
    );
}

async fn run_representative_mtp_release_gate() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::super::qwen3_6_35b_a3b_oq4e_mtp_model_directory();
    let benchmark_prompt_cases = prepare_benchmark_prompt_cases(&model_directory);
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;

    eprintln!("[oq4e-mtp-release] status=start phase=mtp_engine_load ETA_seconds=115");
    let mtp_measurements = run_engine_benchmark_cases(
        &model_directory,
        true,
        &benchmark_prompt_cases,
        &[0, 1, 2],
        38_000,
    )
    .await;

    MlxRuntime::initialize(mlx_memory_limits)
        .expect("the release gate should re-enter the configured MLX runtime")
        .clear_allocator_cache()
        .expect("the target-only run should not inherit reclaimable MTP allocations");

    eprintln!("[oq4e-mtp-release] status=progress phase=target_only_engine_load ETA_seconds=70");
    let target_only_measurements = run_engine_benchmark_cases(
        &model_directory,
        false,
        &benchmark_prompt_cases,
        &[2, 1, 0],
        39_000,
    )
    .await;

    let mut paired_throughput_ratios = Vec::with_capacity(benchmark_prompt_cases.len());
    let mut individual_throughput_regressions = Vec::new();
    let mut output_mismatch_descriptions = Vec::new();
    for prompt_case_index in 0..benchmark_prompt_cases.len() {
        let prompt_case = &benchmark_prompt_cases[prompt_case_index];
        let mtp_measurement = &mtp_measurements[prompt_case_index];
        let target_only_measurement = &target_only_measurements[prompt_case_index];
        let first_mismatching_token_index = mtp_measurement
            .generated_token_ids
            .iter()
            .zip(&target_only_measurement.generated_token_ids)
            .position(|(mtp_token_id, target_only_token_id)| mtp_token_id != target_only_token_id);
        eprintln!(
            "[oq4e-mtp-release] status=diagnostic phase={} first_mismatch={first_mismatching_token_index:?}",
            prompt_case.phase_name,
        );
        if let Some(first_mismatching_token_index) = first_mismatching_token_index {
            output_mismatch_descriptions.push(format!(
                "{} first mismatched at output token {}",
                prompt_case.phase_name, first_mismatching_token_index,
            ));
        }
        assert_eq!(
            mtp_measurement.generated_token_ids.len(),
            usize::from(BENCHMARK_OUTPUT_TOKEN_COUNT),
            "{} should exercise the complete representative output budget",
            prompt_case.phase_name,
        );
        let paired_throughput_ratio = mtp_measurement.tokens_per_second()
            / target_only_measurement
                .tokens_per_second()
                .max(f64::EPSILON);
        paired_throughput_ratios.push(paired_throughput_ratio);
        if paired_throughput_ratio < 0.95 {
            individual_throughput_regressions.push(format!(
                "{} ratio {:.3}",
                prompt_case.phase_name, paired_throughput_ratio,
            ));
        }
        eprintln!(
            "[oq4e-mtp-release] status=sample phase={} prompt_tokens={} output_tokens={} target_only_tok_per_second={:.2} mtp_tok_per_second={:.2} throughput_ratio={:.3}",
            prompt_case.phase_name,
            prompt_case.prompt_token_ids.len(),
            mtp_measurement.generated_token_ids.len(),
            target_only_measurement.tokens_per_second(),
            mtp_measurement.tokens_per_second(),
            paired_throughput_ratio,
        );
    }
    paired_throughput_ratios.sort_by(f64::total_cmp);
    let paired_median_throughput_ratio = paired_throughput_ratios[1];
    assert!(
        output_mismatch_descriptions.is_empty(),
        "MTP output must exactly match target-only greedy output: {}",
        output_mismatch_descriptions.join(", "),
    );
    assert!(
        individual_throughput_regressions.is_empty(),
        "MTP throughput regressed by more than five percent: {}",
        individual_throughput_regressions.join(", "),
    );
    assert!(
        paired_median_throughput_ratio >= 1.05,
        "paired median MTP throughput must exceed target-only by at least five percent"
    );
    eprintln!(
        "[oq4e-mtp-release] status=success paired_median_throughput_ratio={paired_median_throughput_ratio:.3}"
    );
}

fn prepare_benchmark_prompt_cases(model_directory: &Path) -> Vec<BenchmarkPromptCase> {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the representative MTP artifact should validate before tokenization");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the representative MTP tokenizer should load");
    assert_eq!(tokenizer.image_pad_token_id(), IMAGE_PAD_TOKEN_ID);

    let benchmark_prompt_cases = [
        (
            "short_factual",
            "Explain the source material as a factual technical briefing. Write at least seven hundred words.",
        ),
        (
            "medium_code",
            "Derive a Rust data-processing design from the source material, include representative code, and explain complexity and edge cases. Write at least seven hundred words.",
        ),
        (
            "long_structured",
            "Transform the source material into a detailed numbered report with an executive summary, evidence, chronology, risks, recommendations, and follow-up actions. Write at least seven hundred words.",
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(prompt_case_index, (phase_name, prompt_text))| {
        let request_id = RequestId::new(37_900 + prompt_case_index as u64);
        let chat_generation_command = ChatGenerationCommand {
            request_id,
            model: "Jundot/Qwen3.6-35B-A3B-oQ4e-mtp".to_owned(),
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
        }
    })
    .collect::<Vec<_>>();
    benchmark_prompt_cases
}

async fn run_engine_benchmark_cases(
    model_directory: &Path,
    mtp_enabled: bool,
    benchmark_prompt_cases: &[BenchmarkPromptCase],
    measured_prompt_order: &[usize],
    request_id_base: u64,
) -> Vec<BenchmarkMeasurement> {
    let (mut engine, _temporary_log_directory, _performance_attribution_log_path) =
        load_mtp_test_engine(model_directory, mtp_enabled).await;
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
    run_one_generation(
        &mut engine,
        RequestId::new(request_id_base),
        &benchmark_prompt_cases[measured_prompt_order[0]],
        4,
        PerformanceAttribution::disabled(),
    )
    .await;

    let mut indexed_measurements = Vec::with_capacity(benchmark_prompt_cases.len());
    for (measured_position, prompt_case_index) in measured_prompt_order.iter().copied().enumerate()
    {
        let request_id = RequestId::new(request_id_base + measured_position as u64 + 1);
        eprintln!(
            "[oq4e-mtp-release] status=progress runtime={} phase={} output_tokens={} ETA_seconds=40",
            if mtp_enabled { "mtp" } else { "target_only" },
            benchmark_prompt_cases[prompt_case_index].phase_name,
            BENCHMARK_OUTPUT_TOKEN_COUNT,
        );
        let benchmark_measurement = run_one_generation(
            &mut engine,
            request_id,
            &benchmark_prompt_cases[prompt_case_index],
            BENCHMARK_OUTPUT_TOKEN_COUNT,
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
    let inference_request = Qwen3_5InferenceRequest::new(
        request_id,
        benchmark_prompt_case.prompt_token_ids.clone(),
        maximum_output_tokens,
    )
    .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID)
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
                generation_finalization,
                ..
            } => {
                let generated_token_at = Instant::now();
                first_generated_token_at.get_or_insert(generated_token_at);
                measurement.generated_token_ids.push(token_id);
                if measurement.generated_token_ids.len().is_multiple_of(16) {
                    eprintln!(
                        "[oq4e-generation-benchmark] status=progress phase={} generated_tokens={}/{}",
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
                    assert!(generation_finalization.has_reportable_state());
                    assert!(generation_finalization.mlx_memory_telemetry().is_some());
                    return measurement;
                }
            }
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::EndOfSequence => {
                panic!("the Qwen benchmark should finalize on an emitted token")
            }
        }
    }
}

struct BenchmarkPromptCase {
    phase_name: &'static str,
    prompt_token_ids: Vec<u32>,
}

#[derive(Default)]
struct BenchmarkMeasurement {
    generated_token_ids: Vec<u32>,
    generation_elapsed_seconds: f64,
}

impl BenchmarkMeasurement {
    fn generated_token_id_fingerprint(&self) -> u64 {
        let mut generated_token_id_hasher = DefaultHasher::new();
        self.generated_token_ids
            .hash(&mut generated_token_id_hasher);
        generated_token_id_hasher.finish()
    }

    fn tokens_per_second(&self) -> f64 {
        self.generated_token_ids.len().saturating_sub(1) as f64
            / self.generation_elapsed_seconds.max(f64::EPSILON)
    }
}
