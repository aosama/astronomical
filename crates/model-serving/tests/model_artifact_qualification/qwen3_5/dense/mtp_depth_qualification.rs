use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::common::mtp_depth_release_gate::{MtpDepthMeasurement, validate_mtp_depth_release_gate};
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, MtpRuntimeState,
    RequestId,
};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, PerformanceAttributionLog,
    Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5MtpArtifactCapability,
    Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
};
use astronomical_runtime_integration::MlxRuntime;

const MODEL_ID: &str = "Qwen3.8-27B-MTPLX-4bit";
const REPRESENTATIVE_INPUT_TOKEN_COUNT: usize = 1_024;
const REPRESENTATIVE_OUTPUT_TOKEN_COUNT: u16 = 1_024;
const WARMUP_OUTPUT_TOKEN_COUNT: u16 = 32;
const QUALIFICATION_DIRECTORY_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_MTP_DEPTH_QUALIFICATION_DIRECTORY";
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[test]
#[ignore = "validates the configured Qwen3.8 MTPLX sidecar and depth contract"]
fn should_recognize_qwen3_8_mtplx_as_a_depth_three_mtp_artifact() {
    eprintln!("[qwen3.8-mtplx-artifact] status=start ETA_seconds=30");
    let model_directory = super::qwen3_8_27b_mtplx_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the configured Qwen3.8 MTPLX artifact should validate");

    match validated_artifact.mtp_artifact_capability() {
        Qwen3_5MtpArtifactCapability::MtpCapable {
            stored_mtp_layer_count,
            artifact_maximum_draft_depth,
            artifact_default_draft_depth,
            mtp_tensor_count,
        } => {
            assert_eq!(*stored_mtp_layer_count, 1);
            assert_eq!(
                artifact_maximum_draft_depth.map(|depth| depth.get()),
                Some(3)
            );
            assert_eq!(*artifact_default_draft_depth, None);
            assert_eq!(*mtp_tensor_count, 15);
        }
        capability => panic!("Qwen3.8 MTPLX should be MTP-capable, got {capability:?}"),
    }
    assert_eq!(validated_artifact.model_id(), MODEL_ID);
    assert_eq!(validated_artifact.shard_index().mtp_tensor_count(), 0);
    assert_eq!(
        validated_artifact.mtp_sidecar_file_name(),
        Some("mtp.safetensors")
    );
    eprintln!("[qwen3.8-mtplx-artifact] status=success mtp_depth_max=3 mtp_tensors=15");
}

#[tokio::test]
#[ignore = "loads Qwen3.8 MTPLX and runs a depth-three target-verified smoke journey"]
async fn should_execute_qwen3_8_mtplx_depth_three_without_operational_fallback() {
    tokio::time::timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = super::qwen3_8_27b_mtplx_model_directory();
        let prompt = representative_prompt(&model_directory);
        let mut loaded_engine = load_engine(&model_directory, true, Some(3)).await;
        assert_eq!(loaded_engine.depth_status.effective_execution_draft_depth, Some(3));
        let diagnostic = run_generation(
            &mut loaded_engine.engine,
            RequestId::new(71_000),
            &prompt,
            64,
            PerformanceAttribution::enabled(),
        )
        .await;
        let report = generation_report(
            &loaded_engine.performance_attribution_log_path,
            RequestId::new(71_000),
        );
        assert_eq!(counter_amount(&report, "mtp_operational_fallback_count"), 0);
        assert!(counter_amount(&report, "mtp_proposed_draft_count") > 0);
        assert!(counter_amount(&report, "mtp_effective_depth_total") >= 3);
        assert_eq!(diagnostic.generated_token_ids.len(), 64);
        eprintln!(
            "[qwen3.8-mtplx-depth3] status=success output_tokens={} proposed_drafts={} accepted_drafts={}",
            diagnostic.generated_token_ids.len(),
            counter_amount(&report, "mtp_proposed_draft_count"),
            counter_amount(&report, "mtp_accepted_draft_count"),
        );
    })
    .await
    .expect("the Qwen3.8 depth-three smoke journey should finish within 115 seconds");
}

#[tokio::test]
#[ignore = "records the leading Qwen3.8 target-only representative control"]
async fn should_measure_qwen3_8_mtp_target_only_before() {
    run_representative_cell("target_only_before", false, None, 72_000).await;
}

#[tokio::test]
#[ignore = "records the representative Qwen3.8 fixed depth-one measurement"]
async fn should_measure_qwen3_8_mtp_depth_one() {
    run_representative_cell("depth_one", true, Some(1), 73_000).await;
}

#[tokio::test]
#[ignore = "records the representative Qwen3.8 fixed depth-two measurement"]
async fn should_measure_qwen3_8_mtp_depth_two() {
    run_representative_cell("depth_two", true, Some(2), 74_000).await;
}

#[tokio::test]
#[ignore = "records the representative Qwen3.8 fixed depth-three measurement"]
async fn should_measure_qwen3_8_mtp_depth_three() {
    run_representative_cell("depth_three", true, Some(3), 75_000).await;
}

#[tokio::test]
#[ignore = "records the trailing Qwen3.8 target-only representative control"]
async fn should_measure_qwen3_8_mtp_target_only_after() {
    run_representative_cell("target_only_after", false, None, 76_000).await;
}

#[test]
#[ignore = "compares independently bounded Qwen3.8 MTP depth measurement cells"]
fn should_accept_qwen3_8_mtp_depth_release_gate_reports() {
    let qualification_directory = qualification_directory();
    let target_only_before = read_measurement(&qualification_directory, "target_only_before");
    let target_only_after = read_measurement(&qualification_directory, "target_only_after");
    let depth_one = read_measurement(&qualification_directory, "depth_one");
    let depth_two = read_measurement(&qualification_directory, "depth_two");
    let depth_three = read_measurement(&qualification_directory, "depth_three");

    assert_eq!(
        target_only_before.generated_token_fingerprint,
        target_only_after.generated_token_fingerprint
    );
    let target_only_total_request_seconds = target_only_before
        .total_request_elapsed_seconds
        .min(target_only_after.total_request_elapsed_seconds);
    validate_mtp_depth_release_gate(
        &target_only_before,
        &target_only_after,
        &depth_one,
        &depth_two,
        &depth_three,
        usize::from(REPRESENTATIVE_OUTPUT_TOKEN_COUNT),
    )
    .expect("the isolated Qwen3.8 MTP measurements should satisfy the release gate");
    eprintln!(
        "[qwen3.8-mtplx-release] status=success target_only_total_seconds={target_only_total_request_seconds:.3} depth1_total_seconds={:.3} depth2_total_seconds={:.3} depth3_total_seconds={:.3}",
        depth_one.total_request_elapsed_seconds,
        depth_two.total_request_elapsed_seconds,
        depth_three.total_request_elapsed_seconds,
    );
}

async fn run_representative_cell(
    cell_name: &str,
    mtp_enabled: bool,
    draft_depth: Option<u8>,
    request_id_base: u64,
) {
    tokio::time::timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = super::qwen3_8_27b_mtplx_model_directory();
        let prompt = representative_prompt(&model_directory);
        eprintln!(
            "[qwen3.8-mtplx-measurement] status=start cell={cell_name} depth={draft_depth:?} ETA_seconds=115"
        );
        let mut loaded_engine = load_engine(&model_directory, mtp_enabled, draft_depth).await;
        loaded_engine
            .engine
            .reset_mlx_peak_memory_for_tests()
            .await
            .expect("the warmup should start with a fresh MLX peak");
        let warmup_request_id = RequestId::new(request_id_base);
        let _warmup = run_generation(
            &mut loaded_engine.engine,
            warmup_request_id,
            &prompt,
            WARMUP_OUTPUT_TOKEN_COUNT,
            PerformanceAttribution::enabled(),
        )
        .await;
        let warmup_report = generation_report(
            &loaded_engine.performance_attribution_log_path,
            warmup_request_id,
        );
        let warmup_operational_fallback_count =
            counter_amount(&warmup_report, "mtp_operational_fallback_count");
        assert_eq!(warmup_operational_fallback_count, 0);

        loaded_engine
            .engine
            .reset_mlx_peak_memory_for_tests()
            .await
            .expect("the measured request should start with a fresh MLX peak");
        let measured_request_id = RequestId::new(request_id_base + 1);
        let measured = run_generation(
            &mut loaded_engine.engine,
            measured_request_id,
            &prompt,
            REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
            PerformanceAttribution::enabled(),
        )
        .await;
        let measured_report = generation_report(
            &loaded_engine.performance_attribution_log_path,
            measured_request_id,
        );
        let measurement = measurement_from_run(
            cell_name,
            draft_depth,
            counter_amount(&measured_report, "mtp_operational_fallback_count"),
            counter_amount(&measured_report, "mtp_proposed_draft_count"),
            counter_amount(&measured_report, "mtp_effective_depth_total"),
            loaded_engine.mlx_memory_ceiling_bytes,
            measured,
        );
        write_measurement(&qualification_directory(), &measurement);
        eprintln!(
            "[qwen3.8-mtplx-measurement] status=success cell={cell_name} depth={draft_depth:?} tok_s={:.2} active_bytes={} peak_bytes={}",
            measurement.tokens_per_second,
            measurement.maximum_active_mlx_memory_bytes,
            measurement.maximum_peak_mlx_memory_bytes,
        );
        drop(loaded_engine);
        MlxRuntime::initialize(
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await,
        )
        .expect("the runtime should re-enter after the measurement")
        .clear_allocator_cache()
        .expect("the measurement should release reclaimable allocator storage");
    })
    .await
    .expect("the representative Qwen3.8 measurement should finish within 115 seconds");
}

struct LoadedEngine {
    engine: Qwen3_5Engine,
    performance_attribution_log_path: PathBuf,
    depth_status: astronomical_ipc_protocol::MtpDepthStatus,
    mlx_memory_ceiling_bytes: u64,
    _temporary_log_directory: tempfile::TempDir,
}

async fn load_engine(
    model_directory: &Path,
    mtp_enabled: bool,
    draft_depth: Option<u8>,
) -> LoadedEngine {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the Qwen3.8 artifact should validate before engine construction");
    let think_end_token_id = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the Qwen3.8 tokenizer should load")
        .think_end_token_id();
    let memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let temporary_log_directory = tempfile::tempdir()
        .expect("the Qwen3.8 qualification should create a temporary log directory");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");
    let performance_attribution_log =
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("the qualification should open its attribution log");
    let mut engine = Qwen3_5Engine::new_with_runtime_chunking_speculative_prefill_mtp_depth_and_performance_attribution(
        validated_artifact,
        memory_limits.active_memory_limit_bytes(),
        memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(1_024)
            .expect("the representative fixed chunk should be valid"),
        think_end_token_id,
        model_directory.to_path_buf(),
        crate::common::standard_worker_chunking_configuration(),
        true,
        mtp_enabled,
        draft_depth,
        crate::common::disabled_worker_speculative_prefill_configuration(),
        PerformanceAttribution::disabled(),
        performance_attribution_log,
    )
    .expect("the Qwen3.8 engine configuration should be valid");
    let load_result = engine.load().await.expect("the Qwen3.8 engine should load");
    assert_eq!(
        load_result.mtp_runtime_state(),
        if mtp_enabled {
            MtpRuntimeState::Active
        } else {
            MtpRuntimeState::Disabled
        },
    );
    if mtp_enabled {
        assert_eq!(
            load_result
                .mtp_depth_status()
                .effective_execution_draft_depth,
            draft_depth.or(Some(1))
        );
    }
    LoadedEngine {
        engine,
        performance_attribution_log_path,
        depth_status: load_result.mtp_depth_status(),
        mlx_memory_ceiling_bytes: memory_limits.active_memory_limit_bytes() as u64,
        _temporary_log_directory: temporary_log_directory,
    }
}

struct RepresentativePrompt {
    prompt_token_ids: Vec<u32>,
    image_pad_token_id: u32,
}

fn representative_prompt(model_directory: &Path) -> RepresentativePrompt {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the Qwen3.8 artifact should validate before tokenization");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the Qwen3.8 tokenizer should load");
    let command = ChatGenerationCommand {
        request_id: RequestId::new(70_000),
        model: MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: format!(
                "Explain this source as a factual technical briefing of at least seven hundred words.\n\nSource material:\n{ROMEO_AND_JULIET_SOURCE}"
            ),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: Some(256),
        },
        qwen_thinking_channel_seed: None,
    };
    let prepared = tokenizer
        .prepare_chat(&command, false)
        .expect("the representative Romeo and Juliet prompt should prepare");
    let complete_prompt = prepared.input_token_ids();
    let assistant_suffix_start = complete_prompt
        .iter()
        .rposition(|token_id| *token_id == tokenizer.im_end_token_id())
        .expect("the prepared prompt should contain the closing user marker");
    let assistant_suffix = &complete_prompt[assistant_suffix_start..];
    let prefix_length = REPRESENTATIVE_INPUT_TOKEN_COUNT
        .checked_sub(assistant_suffix.len())
        .expect("the assistant suffix should fit the representative prompt");
    assert!(prefix_length < assistant_suffix_start);
    let mut prompt_token_ids = complete_prompt[..prefix_length].to_vec();
    prompt_token_ids.extend_from_slice(assistant_suffix);
    assert_eq!(prompt_token_ids.len(), REPRESENTATIVE_INPUT_TOKEN_COUNT);
    RepresentativePrompt {
        prompt_token_ids,
        image_pad_token_id: tokenizer.image_pad_token_id(),
    }
}

struct GenerationMeasurement {
    generated_token_ids: Vec<u32>,
    generation_elapsed_seconds: f64,
    total_request_elapsed_seconds: f64,
    maximum_active_mlx_memory_bytes: u64,
    maximum_peak_mlx_memory_bytes: u64,
}

async fn run_generation(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    prompt: &RepresentativePrompt,
    output_token_count: u16,
    performance_attribution: PerformanceAttribution,
) -> GenerationMeasurement {
    // End-to-end timing starts before request admission and includes prefill plus first-token latency.
    let request_started_at = Instant::now();
    engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                request_id,
                prompt.prompt_token_ids.clone(),
                output_token_count,
            )
            .with_image_pad_token_id(prompt.image_pad_token_id)
            .with_performance_attribution(performance_attribution),
        )
        .await
        .expect("the representative request should start");
    let mut generated_token_ids = Vec::with_capacity(output_token_count as usize);
    let mut first_generated_token_at = None;
    let mut maximum_active_mlx_memory_bytes = 0_u64;
    let mut maximum_peak_mlx_memory_bytes = 0_u64;
    loop {
        match engine
            .decode_next_token(request_id)
            .await
            .expect("the representative request should advance")
        {
            GeneratedToken::TokenId {
                token_id,
                mlx_memory_telemetry,
                generation_finalization,
                ..
            } => {
                let generated_at = Instant::now();
                let first_generated_at = *first_generated_token_at.get_or_insert(generated_at);
                generated_token_ids.push(token_id);
                if let Some(telemetry) = mlx_memory_telemetry {
                    maximum_active_mlx_memory_bytes =
                        maximum_active_mlx_memory_bytes.max(telemetry.active_memory_bytes);
                    maximum_peak_mlx_memory_bytes =
                        maximum_peak_mlx_memory_bytes.max(telemetry.peak_memory_bytes);
                }
                if generated_token_ids.len().is_multiple_of(32) {
                    eprintln!(
                        "[qwen3.8-mtplx-generation] status=progress generated_tokens={}/{}",
                        generated_token_ids.len(),
                        output_token_count
                    );
                }
                if let Some(finalization) = generation_finalization {
                    if let Some(telemetry) = finalization.mlx_memory_telemetry() {
                        maximum_active_mlx_memory_bytes =
                            maximum_active_mlx_memory_bytes.max(telemetry.active_memory_bytes);
                        maximum_peak_mlx_memory_bytes =
                            maximum_peak_mlx_memory_bytes.max(telemetry.peak_memory_bytes);
                    }
                    return GenerationMeasurement {
                        generated_token_ids,
                        generation_elapsed_seconds: generated_at
                            .saturating_duration_since(first_generated_at)
                            .as_secs_f64(),
                        total_request_elapsed_seconds: generated_at
                            .saturating_duration_since(request_started_at)
                            .as_secs_f64(),
                        maximum_active_mlx_memory_bytes,
                        maximum_peak_mlx_memory_bytes,
                    };
                }
            }
            GeneratedToken::PrefillProgress {
                mlx_memory_telemetry,
                ..
            } => {
                if let Some(telemetry) = mlx_memory_telemetry {
                    maximum_active_mlx_memory_bytes =
                        maximum_active_mlx_memory_bytes.max(telemetry.active_memory_bytes);
                    maximum_peak_mlx_memory_bytes =
                        maximum_peak_mlx_memory_bytes.max(telemetry.peak_memory_bytes);
                }
            }
            GeneratedToken::PromptProcessingPhaseStarted { .. }
            | GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => {
                panic!("the request should finalize on an emitted token")
            }
        }
    }
}

fn measurement_from_run(
    cell_name: &str,
    draft_depth: Option<u8>,
    operational_fallback_count: u64,
    proposed_draft_count: u64,
    effective_depth_total: u64,
    mlx_memory_ceiling_bytes: u64,
    run: GenerationMeasurement,
) -> MtpDepthMeasurement {
    let mut token_hasher = DefaultHasher::new();
    run.generated_token_ids.hash(&mut token_hasher);
    let output_token_count = run.generated_token_ids.len();
    let tokens_per_second = output_token_count.saturating_sub(1) as f64
        / run.generation_elapsed_seconds.max(f64::EPSILON);
    MtpDepthMeasurement {
        cell_name: cell_name.to_owned(),
        draft_depth,
        output_token_count,
        generation_elapsed_seconds: run.generation_elapsed_seconds,
        total_request_elapsed_seconds: run.total_request_elapsed_seconds,
        tokens_per_second,
        maximum_active_mlx_memory_bytes: run.maximum_active_mlx_memory_bytes,
        maximum_peak_mlx_memory_bytes: run.maximum_peak_mlx_memory_bytes,
        mlx_memory_ceiling_bytes,
        operational_fallback_count,
        proposed_draft_count,
        effective_depth_total,
        generated_token_fingerprint: token_hasher.finish(),
    }
}

fn qualification_directory() -> PathBuf {
    std::env::var_os(QUALIFICATION_DIRECTORY_ENVIRONMENT_VARIABLE)
        .map(PathBuf::from)
        .expect("ASTRONOMICAL_MTP_DEPTH_QUALIFICATION_DIRECTORY must name the shared temporary qualification directory")
}

fn write_measurement(directory: &Path, measurement: &MtpDepthMeasurement) {
    std::fs::create_dir_all(directory).expect("the qualification directory should be created");
    let serialized =
        serde_json::to_vec_pretty(measurement).expect("the measurement should serialize");
    std::fs::write(
        directory.join(format!("{}.json", measurement.cell_name)),
        serialized,
    )
    .expect("the measurement should be written");
}

fn read_measurement(directory: &Path, cell_name: &str) -> MtpDepthMeasurement {
    let bytes = std::fs::read(directory.join(format!("{cell_name}.json")))
        .expect("the expected measurement cell should exist");
    serde_json::from_slice(&bytes).expect("the measurement cell should deserialize")
}

fn generation_report(path: &Path, request_id: RequestId) -> serde_json::Value {
    std::fs::read_to_string(path)
        .expect("the attribution log should be readable")
        .lines()
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .expect("the report should be valid JSON")
        })
        .find(|report| report["request_id"] == request_id.value())
        .expect("the requested attribution report should exist")
}

fn counter_amount(report: &serde_json::Value, identifier: &str) -> u64 {
    report["counters"]
        .as_array()
        .and_then(|counters| {
            counters
                .iter()
                .find(|counter| counter["counter"] == identifier)
        })
        .and_then(|counter| counter["amount"].as_u64())
        .unwrap_or(0)
}
