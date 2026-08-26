use std::{fs, path::Path, time::Instant};

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, PerformanceAttributionLog,
    Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5InferenceRequest,
    Qwen3_5PromptProcessingChunkSizer,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serde_json::Value;
use tokio::time::{Duration as TokioDuration, MissedTickBehavior, interval, timeout};

use crate::common::generation_progress::await_generation_advance_with_live_progress;

const FIRST_MODEL_ID: &str = crate::common::ORNITH_MODEL_SWAP_SOURCE_MODEL_ID;
const REPLACEMENT_MODEL_ID: &str = crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID;
const VALIDATION_MAXIMUM_OUTPUT_TOKENS: u32 = 20_480;
const FIXED_PREFILL_CHUNCK_TOKENS: u32 = 2_048;
const IMAGE_PAD_TOKEN_ID: u32 = 248_069;
const MODEL_SWAP_TEST_TIMEOUT: TokioDuration = TokioDuration::from_secs(105);
const LONG_OPERATION_PROGRESS_INTERVAL: TokioDuration = TokioDuration::from_secs(10);
const GENERATION_OUTPUT_TOKEN_COUNT: u16 = 2;
const FIRST_REQUEST_ID: u64 = 18_035;
const REPLACEMENT_REQUEST_ID: u64 = 18_122;
const SAY_HI_PROMPT_TOKEN_IDS: [u32; 15] = [
    248_045, 846, 198, 44_240, 15_131, 13, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
    248_069, 271,
];

#[tokio::test]
#[ignore = "loads the configured Ornith 1.5 oQ6e artifact, drops it, then reloads oQ6e"]
async fn should_clear_stale_mlx_allocator_memory_before_loading_the_replacement_model() {
    timeout(MODEL_SWAP_TEST_TIMEOUT, run_model_swap_allocator_contract())
        .await
        .expect("the 35B Ornith 1.5 MLX allocator contract must finish within 105 seconds");
}

async fn run_model_swap_allocator_contract() {
    let _direct_mlx_test_guard = crate::common::direct_mlx_test_guard().await;
    let first_model_directory = crate::common::configured_model_directory_by_id(FIRST_MODEL_ID)
        .unwrap_or_else(|| {
            panic!(
                "the memory contract requires the local {FIRST_MODEL_ID} artifact; configure it in ~/.astronomical-dev/config.json"
            )
        });
    let replacement_model_directory =
        crate::common::configured_model_directory_by_id(REPLACEMENT_MODEL_ID).unwrap_or_else(|| {
            panic!(
                "the memory contract requires the local {REPLACEMENT_MODEL_ID} artifact; configure it in ~/.astronomical-dev/config.json"
            )
        });
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    eprintln!("[memory-contract] status=progress phase=load_first_model model={FIRST_MODEL_ID}");
    let mut first_model_engine = create_engine(
        &first_model_directory,
        &mlx_memory_limits,
        PerformanceAttribution::disabled(),
        PerformanceAttributionLog::disabled(),
    );
    load_engine_with_progress(&mut first_model_engine, FIRST_MODEL_ID).await;
    run_bounded_generation(&mut first_model_engine, RequestId::new(FIRST_REQUEST_ID)).await;
    drop(first_model_engine);

    let post_drop_runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the Rust MLX runtime should reopen with the same machine-derived limits");
    let post_drop_snapshot = post_drop_runtime
        .memory_snapshot()
        .expect("the post-drop MLX allocator snapshot should be readable");
    eprintln!(
        "[memory-contract] status=progress phase=first_model_dropped active_bytes={} cache_bytes={} peak_bytes={}",
        post_drop_snapshot.active_memory_bytes(),
        post_drop_snapshot.allocator_cache_memory_bytes(),
        post_drop_snapshot.peak_memory_bytes(),
    );
    assert!(
        post_drop_snapshot.allocator_cache_memory_bytes() > 0,
        "dropping {FIRST_MODEL_ID} must leave reclaimable MLX allocator bytes before replacement loading"
    );
    assert_eq!(
        post_drop_snapshot.active_memory_bytes(),
        0,
        "dropping {FIRST_MODEL_ID} must release every active model allocation before replacement loading"
    );
    drop(post_drop_runtime);

    let temporary_log_directory =
        tempfile::tempdir().expect("the memory contract should create a temporary log directory");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");
    let performance_attribution_log =
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("the memory contract should open the replacement attribution log");
    eprintln!(
        "[memory-contract] status=progress phase=load_replacement_model model={REPLACEMENT_MODEL_ID}"
    );
    let mut replacement_model_engine = create_engine(
        &replacement_model_directory,
        &mlx_memory_limits,
        PerformanceAttribution::enabled(),
        performance_attribution_log,
    );
    load_engine_with_progress(&mut replacement_model_engine, REPLACEMENT_MODEL_ID).await;

    let attribution_report_documents =
        read_attribution_report_documents(&performance_attribution_log_path);
    let replacement_model_loading_report =
        model_loading_report(&attribution_report_documents, REPLACEMENT_MODEL_ID);
    assert_replacement_cleanup_order(replacement_model_loading_report);
    let replacement_model_loading_allocator_cache_memory_bytes =
        replacement_model_loading_report["mlx_allocator_cache_memory_bytes"]
            .as_u64()
            .expect("replacement model loading should report allocator-cache bytes");
    let replacement_model_loading_active_memory_bytes =
        replacement_model_loading_report["mlx_active_memory_bytes"]
            .as_u64()
            .expect("replacement model loading should report active memory bytes");
    eprintln!(
        "[memory-contract] status=progress phase=replacement_memory active_bytes={replacement_model_loading_active_memory_bytes} cache_bytes={replacement_model_loading_allocator_cache_memory_bytes} prior_stale_cache_bytes={}",
        post_drop_snapshot.allocator_cache_memory_bytes(),
    );
    assert!(
        replacement_model_loading_allocator_cache_memory_bytes
            < u64::try_from(post_drop_snapshot.allocator_cache_memory_bytes())
                .expect("the stale allocator-cache snapshot should fit in u64"),
        "replacement model loading must remove the prior model-scale allocator cache"
    );
    assert!(
        replacement_model_loading_allocator_cache_memory_bytes
            < replacement_model_loading_active_memory_bytes,
        "replacement model loading may retain tiny graph bookkeeping, but not model-scale cache bytes"
    );
    assert!(
        replacement_model_loading_active_memory_bytes > 0,
        "replacement model loading must retain active model residency"
    );

    run_bounded_generation(
        &mut replacement_model_engine,
        RequestId::new(REPLACEMENT_REQUEST_ID),
    )
    .await;
    drop(replacement_model_engine);

    let attribution_report_documents =
        read_attribution_report_documents(&performance_attribution_log_path);
    let replacement_generation_report =
        generation_report(&attribution_report_documents, REPLACEMENT_REQUEST_ID);
    assert_eq!(replacement_generation_report["outcome"], "success");
    let replacement_generation_allocator_cache_memory_bytes =
        replacement_generation_report["mlx_allocator_cache_memory_bytes"]
            .as_u64()
            .expect("replacement generation should report allocator-cache bytes");
    let replacement_generation_active_memory_bytes =
        replacement_generation_report["mlx_active_memory_bytes"]
            .as_u64()
            .expect("replacement generation should report active memory bytes");
    let replacement_generation_allocator_cleanup_operation =
        operation_report(replacement_generation_report, "mlx_allocator_cache_cleanup");
    assert_eq!(
        replacement_generation_allocator_cleanup_operation["occurrence_count"], 2,
        "replacement generation must clean up after prompt prefill and request finalization"
    );
    assert_eq!(
        replacement_generation_allocator_cache_memory_bytes, 0,
        "completed replacement generation must not retain reclaimable allocator-cache bytes"
    );
    assert!(
        replacement_generation_active_memory_bytes > 0,
        "completed replacement generation must preserve active model residency"
    );
    eprintln!("[memory-contract] status=success phase=model_swap");
}

fn create_engine(
    model_directory: &Path,
    mlx_memory_limits: &MlxMemoryLimits,
    model_loading_performance_attribution: PerformanceAttribution,
    performance_attribution_log: PerformanceAttributionLog,
) -> Qwen3_5Engine {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, VALIDATION_MAXIMUM_OUTPUT_TOKENS)
        .expect("the selected Qwen3.5-MoE artifact should validate before loading");
    Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
            FIXED_PREFILL_CHUNCK_TOKENS,
        )
        .expect("the fixed prefill chunck size should be valid"),
        IMAGE_PAD_TOKEN_ID,
        model_directory.to_path_buf(),
        crate::common::standard_worker_chunking_configuration(),
        true,
        false,
        crate::common::disabled_worker_speculative_prefill_configuration(),
        model_loading_performance_attribution,
        performance_attribution_log,
    )
    .expect("the selected Qwen3.5-MoE engine settings should be valid")
}

async fn load_engine_with_progress(qwen3_5_engine: &mut Qwen3_5Engine, model_id: &str) {
    let model_load_started_at = Instant::now();
    let mut progress_interval = interval(LONG_OPERATION_PROGRESS_INTERVAL);
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    progress_interval.tick().await;
    let model_load_future = qwen3_5_engine.load();
    tokio::pin!(model_load_future);
    loop {
        tokio::select! {
            model_load_outcome = &mut model_load_future => {
                model_load_outcome.expect("the selected model should load through the engine");
                eprintln!(
                    "[memory-contract] status=progress phase=model_loaded model={model_id} elapsed_seconds={:.1}",
                    model_load_started_at.elapsed().as_secs_f64(),
                );
                return;
            }
            _ = progress_interval.tick() => eprintln!(
                "[memory-contract] status=progress phase=model_load model={model_id} elapsed_seconds={:.1} ETA_seconds=unknown",
                model_load_started_at.elapsed().as_secs_f64(),
            ),
        }
    }
}

async fn run_bounded_generation(qwen3_5_engine: &mut Qwen3_5Engine, request_id: RequestId) {
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                request_id,
                SAY_HI_PROMPT_TOKEN_IDS.to_vec(),
                GENERATION_OUTPUT_TOKEN_COUNT,
            )
            .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID)
            .with_performance_attribution(PerformanceAttribution::enabled()),
        )
        .await
        .expect("the engine should accept the bounded memory-contract request");

    let mut generated_token_count = 0_usize;
    let generation_started_at = Instant::now();
    let mut generation_progress_interval = interval(LONG_OPERATION_PROGRESS_INTERVAL);
    generation_progress_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    generation_progress_interval.tick().await;
    loop {
        let generation_advance_outcome = await_generation_advance_with_live_progress(
            qwen3_5_engine.decode_next_token(request_id),
            &mut generation_progress_interval,
            || eprintln!(
                "[memory-contract] status=progress phase=generation_advance request_id={} generated_token_count={generated_token_count} elapsed_seconds={:.1} ETA_seconds=unknown",
                request_id.value(),
                generation_started_at.elapsed().as_secs_f64(),
            ),
        )
        .await;
        match generation_advance_outcome
            .expect("the bounded memory-contract request should advance successfully")
        {
            GeneratedToken::TokenId { .. } => {
                generated_token_count += 1;
                eprintln!(
                    "[memory-contract] status=progress phase=generated_token request_id={} generated_token_count={generated_token_count}",
                    request_id.value(),
                );
                assert!(
                    generated_token_count <= usize::from(GENERATION_OUTPUT_TOKEN_COUNT),
                    "the bounded memory-contract request generated more tokens than configured"
                );
                if generated_token_count == usize::from(GENERATION_OUTPUT_TOKEN_COUNT) {
                    return;
                }
            }
            GeneratedToken::PrefillProgress {
                mlx_memory_telemetry,
                ..
            } => {
                let mlx_memory_telemetry = mlx_memory_telemetry
                    .expect("the enabled memory guard should report prefill telemetry");
                eprintln!(
                    "[memory-contract] status=progress phase=prefill_progress request_id={} active_bytes={} peak_bytes={}",
                    request_id.value(),
                    mlx_memory_telemetry.active_memory_bytes,
                    mlx_memory_telemetry.peak_memory_bytes,
                );
            }
            GeneratedToken::PromptProcessingPhaseStarted {
                prompt_processing_phase,
                total_token_count,
            } => {
                eprintln!(
                    "[memory-contract] status=progress phase=prompt_processing_started request_id={} prompt_processing_phase={prompt_processing_phase:?} total_token_count={total_token_count}",
                    request_id.value(),
                );
            }
            GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => {
                assert!(
                    generated_token_count > 0,
                    "the bounded memory-contract request must generate at least one token"
                );
                return;
            }
        }
    }
}

fn read_attribution_report_documents(performance_attribution_log_path: &Path) -> Vec<Value> {
    fs::read_to_string(performance_attribution_log_path)
        .expect("the replacement attribution log should contain JSON Lines")
        .lines()
        .map(|json_line| {
            serde_json::from_str(json_line)
                .expect("every replacement attribution line should be valid JSON")
        })
        .collect()
}

fn model_loading_report<'a>(
    attribution_report_documents: &'a [Value],
    model_id: &str,
) -> &'a Value {
    let mut matching_model_loading_reports =
        attribution_report_documents
            .iter()
            .filter(|attribution_report_document| {
                attribution_report_document["report_kind"] == "model_loading"
                    && attribution_report_document["model_id"] == model_id
            });
    let model_loading_report = matching_model_loading_reports.next().unwrap_or_else(|| {
        panic!("the replacement model should have one model-loading report for {model_id}")
    });
    assert!(
        matching_model_loading_reports.next().is_none(),
        "the replacement model should have exactly one model-loading report for {model_id}"
    );
    model_loading_report
}

fn generation_report(attribution_report_documents: &[Value], request_id: u64) -> &Value {
    let mut matching_generation_reports =
        attribution_report_documents
            .iter()
            .filter(|attribution_report_document| {
                attribution_report_document["report_kind"] == "generation"
                    && attribution_report_document["request_id"] == request_id
            });
    let generation_report = matching_generation_reports.next().unwrap_or_else(|| {
        panic!("the replacement request should have a generation report for {request_id}")
    });
    assert!(
        matching_generation_reports.next().is_none(),
        "the replacement request should have exactly one generation report for {request_id}"
    );
    generation_report
}

fn operation_report<'a>(attribution_report: &'a Value, operation_identifier: &str) -> &'a Value {
    attribution_report["operations"]
        .as_array()
        .and_then(|operation_reports| {
            operation_reports
                .iter()
                .find(|operation_report| operation_report["operation"] == operation_identifier)
        })
        .unwrap_or_else(|| {
            panic!("the attribution report should contain operation {operation_identifier}")
        })
}

fn assert_replacement_cleanup_order(replacement_model_loading_report: &Value) {
    assert_eq!(replacement_model_loading_report["outcome"], "success");
    let runtime_initialization = operation_report(
        replacement_model_loading_report,
        "mlx_runtime_initialization",
    );
    let allocator_cleanup = operation_report(
        replacement_model_loading_report,
        "mlx_allocator_cache_cleanup",
    );
    let runtime_initialization_end = runtime_initialization["last_ended_offset_nanoseconds"]
        .as_u64()
        .expect("runtime initialization should report an end offset");
    let allocator_cleanup_start = allocator_cleanup["first_started_offset_nanoseconds"]
        .as_u64()
        .expect("allocator cleanup should report a start offset");
    assert_eq!(allocator_cleanup["occurrence_count"], 1);
    assert!(
        allocator_cleanup_start >= runtime_initialization_end,
        "allocator cleanup must start after runtime initialization"
    );

    let allocator_cleanup_end = allocator_cleanup["last_ended_offset_nanoseconds"]
        .as_u64()
        .expect("allocator cleanup should report an end offset");
    for operation_identifier in [
        "model_safetensors_mapping",
        "model_tensor_binding",
        "resident_weight_materialization_synchronization_wait",
    ] {
        let later_operation =
            operation_report(replacement_model_loading_report, operation_identifier);
        let later_operation_start = later_operation["first_started_offset_nanoseconds"]
            .as_u64()
            .expect("each replacement loading operation should report a start offset");
        assert!(
            allocator_cleanup_end <= later_operation_start,
            "allocator cleanup must finish before {operation_identifier}"
        );
    }
}
