//! Ignored real-artifact qualification for one configured standalone MTP pairing.

use std::time::{Duration, Instant};

use astronomical_config::{AstronomicalConfig, discover_models, discover_qwen3_5_mtp_drafters};
use astronomical_ipc_protocol::{MtpRuntimeState, RequestId};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, PerformanceAttributionLog,
    Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5Model,
    Qwen3_5MtpRequestState, Qwen3_5MtpSourceSelection, Qwen3_5PromptProcessingChunkSizer,
    Qwen3_5StandaloneMtpArtifactValidator, Qwen3_5Tokenizer, compare_qwen3_5_mtp_pairing_contracts,
};
use astronomical_runtime_integration::MlxRuntime;

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
#[ignore = "loads the first standalone MTP pairing from Development configuration"]
async fn should_attach_and_execute_the_configured_standalone_mtp_pairing() {
    tokio::time::timeout(Duration::from_secs(120), qualify_configured_pairing())
        .await
        .expect("standalone MTP qualification should finish within 120 seconds");
}

async fn qualify_configured_pairing() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let started_at = Instant::now();
    eprintln!("[standalone-mtp] status=start phase=configuration ETA_seconds=120");
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("Development configuration should load");
    let configured_pairings = astronomical_config
        .mtp_pairings()
        .expect("configured MTP pairings should validate");
    let Some(pairing) = configured_pairings.first() else {
        eprintln!(
            "[standalone-mtp] status=skipped reason=no_development_mtp_pairing elapsed_seconds={:.2}",
            started_at.elapsed().as_secs_f64()
        );
        return;
    };
    let discovered_target = discover_models(astronomical_config.model_directories(), 20_480)
        .expect("configured model roots should be discoverable")
        .into_iter()
        .flat_map(|directory_scan| directory_scan.discovered_models)
        .find(|model| model.model_id == pairing.target_model_id())
        .expect("the configured standalone MTP target should be discovered");
    let discovered_drafter = discover_qwen3_5_mtp_drafters(astronomical_config.model_directories())
        .expect("configured standalone MTP drafters should be discoverable")
        .into_iter()
        .find(|drafter| drafter.model_id == pairing.drafter_model_id())
        .expect("the configured standalone MTP drafter should be discovered");
    let target_model_directory = discovered_target.model_directory.clone();
    let drafter_model_directory = discovered_drafter.model_directory.clone();
    let drafter_model_id = discovered_drafter.model_id.clone();
    let drafter_model_revision = discovered_drafter.revision.clone();

    eprintln!("[standalone-mtp] status=progress phase=artifact_validation ETA_seconds=105");
    let validated_target = Qwen3_5ArtifactValidator::new()
        .validate(&discovered_target.model_directory, 20_480)
        .expect("the configured target should pass deep artifact validation");
    let target_config = validated_target.config().clone();
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target)
        .expect("the configured target tokenizer should load");
    let romeo_and_juliet_token_ids = tokenizer
        .encode_prompt(ROMEO_AND_JULIET_SOURCE)
        .expect("the Romeo and Juliet fixture should tokenize");
    let prompt_token_ids = romeo_and_juliet_token_ids
        .get(..2)
        .expect("the Romeo and Juliet fixture should contain at least two tokens");
    let validated_drafter = Qwen3_5StandaloneMtpArtifactValidator::new(
        &target_config,
        discovered_drafter.model_id,
        discovered_drafter.revision,
    )
    .validate(&discovered_drafter.model_directory)
    .expect("the configured standalone MTP drafter should validate");
    compare_qwen3_5_mtp_pairing_contracts(
        &target_config,
        validated_target
            .tokenizer_bytes()
            .expect("the target tokenizer descriptor should be retained"),
        validated_drafter.config(),
        validated_drafter.tokenizer_bytes(),
        Some(1),
    )
    .expect("the configured standalone MTP pairing should be compatible");

    eprintln!("[standalone-mtp] status=progress phase=model_loading ETA_seconds=90");
    let memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime =
        MlxRuntime::initialize(memory_limits).expect("the direct MLX runtime should initialize");
    let mut model = Qwen3_5Model::load(
        runtime,
        validated_target,
        &discovered_target.model_directory,
        false,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the target should load without artifact-local MTP binding");
    model
        .attach_and_materialize_standalone_mtp(validated_drafter)
        .expect("the standalone MTP drafter should attach and materialize");

    eprintln!("[standalone-mtp] status=progress phase=mtp_forward ETA_seconds=45");
    let mut target_state = crate::common::standard_request_decoder_state(&target_config);
    let first_target_output = model
        .forward_chunk_with_pre_final_normalization_hidden_states(
            &prompt_token_ids[..1],
            0,
            &mut target_state,
        )
        .expect("the first Romeo and Juliet token should produce target hidden state");
    let shifted_token = model
        .runtime()
        .array_from_u32(&prompt_token_ids[1..2], &[1, 1])
        .expect("the shifted Romeo and Juliet token should bind");
    let mut mtp_state = Qwen3_5MtpRequestState::empty_with_growth_tokens(256)
        .expect("the MTP request-state growth should validate");
    let mtp_output = model
        .forward_mtp_draft(
            first_target_output.pre_final_normalization_hidden_states(),
            &shifted_token,
            &mut mtp_state,
        )
        .expect("the standalone MTP drafter should execute one draft forward");
    assert_eq!(
        mtp_output.draft_logits().shape(),
        [1, 1, target_config.vocabulary_size() as i32]
    );
    drop(model);
    run_target_authoritative_pairing_parity(
        &target_model_directory,
        &drafter_model_directory,
        &drafter_model_id,
        &drafter_model_revision,
        prompt_token_ids,
        tokenizer.image_pad_token_id(),
    )
    .await;
    eprintln!(
        "[standalone-mtp] status=success elapsed_seconds={:.2}",
        started_at.elapsed().as_secs_f64()
    );
}

async fn run_target_authoritative_pairing_parity(
    target_model_directory: &std::path::Path,
    drafter_model_directory: &std::path::Path,
    drafter_model_id: &str,
    drafter_model_revision: &str,
    prompt_token_ids: &[u32],
    image_pad_token_id: u32,
) {
    eprintln!("[standalone-mtp] status=progress phase=target_only_parity ETA_seconds=35");
    let (mut target_only_engine, _target_log_directory, _target_log_path) =
        load_pairing_engine(target_model_directory, None, false).await;
    let target_only_token_ids = generate_with_injected_feedback(
        &mut target_only_engine,
        RequestId::new(81_001),
        prompt_token_ids,
        image_pad_token_id,
        false,
    )
    .await;
    drop(target_only_engine);
    let target_only_cleanup_runtime = MlxRuntime::initialize(
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await,
    )
    .expect("the runtime should re-enter after target-only parity");
    target_only_cleanup_runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .expect("target-only parity should release allocator cache");
    let target_only_post_drop_active_memory_bytes = target_only_cleanup_runtime
        .memory_snapshot()
        .expect("target-only teardown memory should be observable")
        .active_memory_bytes();

    eprintln!("[standalone-mtp] status=progress phase=paired_parity ETA_seconds=20");
    let pairing = Some((
        drafter_model_directory,
        drafter_model_id,
        drafter_model_revision,
    ));
    let (mut paired_engine, _paired_log_directory, paired_log_path) =
        load_pairing_engine(target_model_directory, pairing, true).await;
    let paired_token_ids = generate_with_injected_feedback(
        &mut paired_engine,
        RequestId::new(81_002),
        prompt_token_ids,
        image_pad_token_id,
        true,
    )
    .await;
    assert_eq!(
        paired_token_ids, target_only_token_ids,
        "verified standalone MTP output must equal target-only output after rejection and injection"
    );
    let generation_report = std::fs::read_to_string(paired_log_path)
        .expect("paired attribution should be readable")
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|report| report["report_kind"] == "generation")
        .expect("paired generation attribution should be present");
    assert_eq!(generation_report["drafter_model_id"], drafter_model_id);
    assert!(counter_amount(&generation_report, "mtp_proposed_draft_position_one_count") > 0);
    assert!(counter_amount(&generation_report, "mtp_rejected_draft_position_one_count") > 0);
    prove_paired_cancellation(&mut paired_engine, prompt_token_ids, image_pad_token_id).await;
    drop(paired_engine);
    let cleanup_runtime = MlxRuntime::initialize(
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await,
    )
    .expect("the runtime should re-enter after paired cancellation");
    cleanup_runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .expect("paired teardown should release reclaimable allocator storage");
    let cleanup_snapshot = cleanup_runtime
        .memory_snapshot()
        .expect("paired teardown memory should be observable");
    assert_eq!(
        cleanup_snapshot.active_memory_bytes(),
        target_only_post_drop_active_memory_bytes,
        "paired teardown must not retain active memory beyond the target-only runtime baseline"
    );
    assert_eq!(cleanup_snapshot.allocator_cache_memory_bytes(), 0);
}

async fn prove_paired_cancellation(
    engine: &mut Qwen3_5Engine,
    prompt_token_ids: &[u32],
    image_pad_token_id: u32,
) {
    let cancellation_request_id = RequestId::new(81_003);
    engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                cancellation_request_id,
                prompt_token_ids.iter().copied().take(64).collect(),
                8,
            )
            .with_image_pad_token_id(image_pad_token_id),
        )
        .await
        .expect("the paired cancellation request should start");
    loop {
        if matches!(
            engine
                .decode_next_token(cancellation_request_id)
                .await
                .expect("the paired cancellation request should advance"),
            GeneratedToken::TokenId { .. }
        ) {
            break;
        }
    }
    let cancellation_finalization = engine
        .cancel_generation(cancellation_request_id)
        .await
        .expect("paired cancellation should release request-local MTP state");
    assert!(cancellation_finalization.has_reportable_state());
}

async fn load_pairing_engine(
    target_model_directory: &std::path::Path,
    pairing: Option<(&std::path::Path, &str, &str)>,
    mtp_enabled: bool,
) -> (Qwen3_5Engine, tempfile::TempDir, std::path::PathBuf) {
    let validated_target = Qwen3_5ArtifactValidator::new()
        .validate(target_model_directory, 20_480)
        .expect("the parity target should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target)
        .expect("the parity target tokenizer should load");
    let mtp_source_selection =
        if let Some((drafter_directory, drafter_id, drafter_revision)) = pairing {
            let validated_drafter = Qwen3_5StandaloneMtpArtifactValidator::new(
                validated_target.config(),
                drafter_id,
                drafter_revision,
            )
            .validate(drafter_directory)
            .expect("the parity drafter should validate");
            let compatibility = compare_qwen3_5_mtp_pairing_contracts(
                validated_target.config(),
                validated_target
                    .tokenizer_bytes()
                    .expect("target tokenizer bytes should exist"),
                validated_drafter.config(),
                validated_drafter.tokenizer_bytes(),
                Some(1),
            )
            .expect("the parity pairing should remain compatible");
            Qwen3_5MtpSourceSelection::Standalone {
                artifact: validated_drafter,
                compatibility,
            }
        } else {
            Qwen3_5MtpSourceSelection::TargetLocal
        };
    let memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let temporary_log_directory = tempfile::tempdir().expect("parity should create a log root");
    let log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");
    let performance_log = PerformanceAttributionLog::open(&log_path, true)
        .expect("parity attribution log should open");
    let mut engine = Qwen3_5Engine::new_with_runtime_chunking_speculative_prefill_mtp_depth_and_performance_attribution(
        validated_target,
        memory_limits.active_memory_limit_bytes(),
        memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(64)
            .expect("the parity chunk size should validate"),
        tokenizer.think_end_token_id(),
        target_model_directory.to_path_buf(),
        crate::common::standard_worker_chunking_configuration(),
        true,
        mtp_enabled,
        Some(1),
        mtp_source_selection,
        crate::common::disabled_worker_speculative_prefill_configuration(),
        PerformanceAttribution::enabled(),
        performance_log,
    )
    .expect("the parity engine should construct");
    let load_result = engine.load().await.expect("the parity engine should load");
    assert_eq!(
        load_result.mtp_runtime_state(),
        if mtp_enabled {
            MtpRuntimeState::Active
        } else {
            MtpRuntimeState::Disabled
        }
    );
    (engine, temporary_log_directory, log_path)
}

async fn generate_with_injected_feedback(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    prompt_token_ids: &[u32],
    image_pad_token_id: u32,
    force_rejection: bool,
) -> Vec<u32> {
    let prompt_token_ids = prompt_token_ids
        .iter()
        .copied()
        .take(64)
        .collect::<Vec<_>>();
    engine
        .start_generation(
            Qwen3_5InferenceRequest::new(request_id, prompt_token_ids.clone(), 8)
                .with_image_pad_token_id(image_pad_token_id)
                .with_performance_attribution(PerformanceAttribution::enabled()),
        )
        .await
        .expect("the parity request should start");
    if force_rejection {
        engine
            .force_next_mtp_draft_rejection_for_tests(request_id)
            .await
            .expect("the paired request should arm one forced rejection");
    }
    let mut generated_token_ids = Vec::new();
    while generated_token_ids.len() < 8 {
        match engine
            .decode_next_token(request_id)
            .await
            .expect("parity decode should advance")
        {
            GeneratedToken::TokenId {
                token_id,
                generation_finalization,
                ..
            } => {
                generated_token_ids.push(token_id);
                if generated_token_ids.len() == 1 {
                    engine
                        .inject_input_tokens(request_id, prompt_token_ids[..2].to_vec())
                        .await
                        .expect(
                            "Romeo and Juliet feedback injection should reset private MTP state",
                        );
                }
                if generation_finalization.is_some() {
                    break;
                }
            }
            GeneratedToken::PrefillProgress { .. }
            | GeneratedToken::PromptProcessingPhaseStarted { .. }
            | GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => break,
        }
    }
    generated_token_ids
}

fn counter_amount(report: &serde_json::Value, counter_name: &str) -> u64 {
    report["counters"]
        .as_array()
        .and_then(|counters| {
            counters
                .iter()
                .find(|counter| counter["counter"] == counter_name)
        })
        .and_then(|counter| counter["amount"].as_u64())
        .unwrap_or(0)
}
