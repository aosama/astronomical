//! Drafter-engaged large-prompt activation-reserve acceptance journey.
//!
//! Encodes the target behavior of issue #644: on a production-shaped prompt
//! (50,400 tokens, above the configured 50,000 speculative-prefill floor) the
//! planned prefill activation reserve must stay within the MLX ceiling while
//! the request is served through expert SSD streaming. The journey fails with
//! the full per-chunk evidence table until the over-projection is fixed.

use std::time::Duration;

use astronomical_config::AstronomicalConfig;
use astronomical_ipc_protocol::{RequestId, WorkerSpeculativePrefillConfiguration};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, MemoryCeilingUtilization, PerformanceAttribution,
    PerformanceAttributionLog, PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator,
    Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
};

use super::support::{
    decode_generated_output_text, prepare_romeo_and_juliet_three_paragraph_summary_prompt,
};

/// Just above the configured 50,000 speculative-prefill floor so the drafter
/// engages, mirroring the production compaction prompts that surfaced #644.
const LARGE_PROMPT_TOKEN_COUNT: usize = 50_400;
// Completion-only output: the journey measures decode throughput and final
// memory state; prose quality is not a memory-policy outcome.
const LARGE_PROMPT_MAXIMUM_OUTPUT_TOKEN_COUNT: u16 = 48;
const LARGE_PROMPT_REQUEST_IDENTIFIER: u64 = 95_401;
const LARGE_PROMPT_JOURNEY_TIMEOUT: Duration = Duration::from_secs(118);

#[tokio::test]
#[ignore = "encodes the #644 target behavior: the planned prefill activation reserve must stay within the MLX ceiling on a drafter-engaged 50K-token prompt"]
async fn should_keep_the_planned_prefill_activation_reserve_within_the_mlx_ceiling_for_a_drafter_engaged_large_prompt()
 {
    tokio::time::timeout(
        LARGE_PROMPT_JOURNEY_TIMEOUT,
        run_drafter_engaged_large_prompt_journey(),
    )
    .await
    .expect("the drafter-engaged large-prompt journey must finish within 118 seconds");
}

struct ChunkMemoryEvidence {
    processed_token_count: u32,
    active_memory_bytes: u64,
    utilization: MemoryCeilingUtilization,
}

/// Asserts the engine-published utilization identity on one sample (issue #510).
///
/// The named owners' unoccupied reserve plus the unexplained residual must
/// account for every unused byte, and any excess promise must surface exactly
/// as the owner overrun. This is the contract the menu legend renders.
fn assert_utilization_identity(utilization: &MemoryCeilingUtilization) {
    let named_total_bytes = utilization
        .reserved_model_core_slack_bytes
        .saturating_add(utilization.reserved_context_growth_bytes)
        .saturating_add(utilization.reserved_activation_and_workspace_bytes)
        .saturating_add(utilization.unseated_expert_entitlement_bytes)
        .saturating_sub(utilization.speculative_draft_payload_bytes);
    assert_eq!(
        named_total_bytes.saturating_sub(utilization.unused_headroom_bytes),
        utilization.owner_overrun_bytes,
        "the owner overrun must equal the named-owner promise beyond the unused headroom"
    );
    assert_eq!(
        utilization
            .unused_headroom_bytes
            .saturating_sub(named_total_bytes),
        utilization.unexplained_headroom_bytes,
        "the unexplained residual must equal the unused headroom beyond the named-owner promise"
    );
}

async fn run_drafter_engaged_large_prompt_journey() {
    if let Err(test_tracing_initialization_error) = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init()
    {
        eprintln!(
            "[large-prompt-activation-overrun] status=progress tracing=already_initialized reason={test_tracing_initialization_error}"
        );
    }
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let sampled_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("the standard Astronomical configuration should load for the large-prompt journey");
    let target_model_id = crate::common::large_sparse_moe_model_id();
    let discovered_target_model =
        crate::common::configured_discovered_model_by_id(&astronomical_config, target_model_id);
    let target_model_directory = discovered_target_model.model_directory;
    let validated_target_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&target_model_directory, 20_480)
        .expect("the large-prompt target artifact should validate");
    let target_model_id = validated_target_artifact.model_id().to_owned();
    let target_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
        .expect("the large-prompt target tokenizer should load");
    let (draft_model_directory, draft_model_id) =
        crate::serving_acceptance::support::configured_speculative_prefill_draft_model(
            &target_model_directory,
        );
    let request_id = RequestId::new(LARGE_PROMPT_REQUEST_IDENTIFIER);
    let summary_prompt = prepare_romeo_and_juliet_three_paragraph_summary_prompt(
        &target_model_directory,
        &target_model_id,
        request_id,
        LARGE_PROMPT_TOKEN_COUNT,
        LARGE_PROMPT_MAXIMUM_OUTPUT_TOKEN_COUNT,
    );
    let temporary_prompt_cache_directory = tempfile::tempdir()
        .expect("the large-prompt journey should create an empty prompt-cache root");
    let temporary_attribution_directory = tempfile::tempdir()
        .expect("the large-prompt journey should create an attribution directory");
    let performance_attribution_log_path = temporary_attribution_directory
        .path()
        .join("performance-attribution.jsonl");
    let target_prompt_cache_directory = temporary_prompt_cache_directory.path().join("target");
    let persistent_prompt_cache_disk_store_config = PersistentPromptCacheDiskStoreConfig::new(
        target_prompt_cache_directory,
        temporary_prompt_cache_directory.path().to_path_buf(),
        astronomical_config
            .prompt_cache()
            .expect("the configured prompt-cache policy should resolve")
            .global_prompt_cache_maximum_size_bytes(),
    );
    // Production-shaped speculative prefill: the configured floor, keep
    // percentage, and mandatory trailing span from the user configuration that
    // surfaced #644, with the shared journey selection constants.
    let speculative_prefill_configuration = WorkerSpeculativePrefillConfiguration {
        enabled: true,
        target_model_id: Some(target_model_id.clone()),
        draft_model_id: Some(draft_model_id),
        draft_model_directory: Some(draft_model_directory),
        minimum_prompt_tokens: 50_000,
        keep_percentage: 50,
        selection_chunk_token_count: 32,
        mandatory_trailing_token_count: 10_240,
        lookahead_token_count: 8,
        importance_pooling_kernel_token_count: 13,
    };
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_speculative_prefill_mtp_depth_and_performance_attribution(
        validated_target_artifact,
        sampled_memory_limits.active_memory_limit_bytes(),
        sampled_memory_limits.allocator_cache_memory_limit_bytes(),
        Some(persistent_prompt_cache_disk_store_config),
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
            2_048,
            2_048,
        )
        .expect("the large-prompt prefill chunk size should be valid"),
        target_tokenizer.think_end_token_id(),
        target_model_directory,
        crate::common::standard_worker_chunking_configuration(),
        true,
        true,
        None,
        speculative_prefill_configuration,
        PerformanceAttribution::enabled(),
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("the large-prompt attribution log should open"),
    )
    .expect("the large-prompt SpecPrefill engine should construct");

    eprintln!(
        "[large-prompt-activation-overrun] status=progress phase=model_load prompt_tokens={} active_memory_limit_bytes={} timeout_seconds=118",
        summary_prompt.prompt_token_ids.len(),
        sampled_memory_limits.active_memory_limit_bytes(),
    );
    qwen3_5_engine
        .load()
        .await
        .expect("the large-prompt target and drafter should load");
    eprintln!("[large-prompt-activation-overrun] status=progress phase=model_loaded");
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new_sampling(
                request_id,
                summary_prompt.prompt_token_ids,
                LARGE_PROMPT_MAXIMUM_OUTPUT_TOKEN_COUNT,
                summary_prompt.sampling_temperature_thousandths,
                summary_prompt.sampling_top_p_thousandths,
                summary_prompt.sampling_seed,
            )
            .with_image_pad_token_id(summary_prompt.image_pad_token_id)
            .with_ordinary_target_prefill_control_span_token_count(
                summary_prompt.ordinary_target_prefill_control_span_token_count,
            )
            .with_thinking_configuration(false, None, Vec::new(), Vec::new())
            .with_performance_attribution(PerformanceAttribution::enabled()),
        )
        .await
        .expect("the large prompt should be admitted");

    let mut chunk_evidence: Vec<ChunkMemoryEvidence> = Vec::new();
    let mut observed_drafter_phase = false;
    let mut observed_live_drafter_memory = false;
    let mut generated_token_count = 0_usize;
    let mut generation_started_at: Option<std::time::Instant> = None;
    let mut prefill_elapsed_millis_total = 0_u64;
    loop {
        let generated_token = match qwen3_5_engine.decode_next_token(request_id).await {
            Ok(generated_token) => generated_token,
            Err(generation_error) => panic!(
                "the large prompt should complete without a memory failure; chunks={} last_chunk={chunk_evidence_last:?} error={generation_error:?}",
                chunk_evidence.len(),
                chunk_evidence_last = chunk_evidence.last().map(|evidence| (
                    evidence.processed_token_count,
                    evidence.utilization.reserved_activation_and_workspace_bytes,
                    evidence.utilization.owner_overrun_bytes,
                )),
            ),
        };
        match generated_token {
            GeneratedToken::PromptProcessingPhaseStarted {
                prompt_processing_phase,
                ..
            } => {
                observed_drafter_phase |= prompt_processing_phase
                    == astronomical_ipc_protocol::WorkerPromptProcessingPhase::Drafter;
                eprintln!(
                    "[large-prompt-activation-overrun] status=progress phase={prompt_processing_phase:?}"
                );
            }
            GeneratedToken::PrefillProgress {
                processed_token_count,
                elapsed_millis,
                mlx_memory_telemetry,
                speculative_prefill_draft_memory_telemetry,
                ..
            } => {
                prefill_elapsed_millis_total += elapsed_millis;
                observed_live_drafter_memory |= speculative_prefill_draft_memory_telemetry
                    .is_some_and(|draft_memory_telemetry| {
                        draft_memory_telemetry
                            .active_memory_breakdown
                            .speculative_prefill_draft_memory_bytes
                            > 0
                    });
                if let Some(mlx_memory_telemetry) = mlx_memory_telemetry {
                    if let Some(utilization) = mlx_memory_telemetry.memory_ceiling_utilization {
                        assert_utilization_identity(&utilization);
                        eprintln!(
                            "[large-prompt-activation-overrun] status=progress phase=prefill processed_tokens={processed_token_count} active_bytes={} activation_reserve_bytes={} owner_overrun_bytes={} context_reserve_bytes={} unused_bytes={}",
                            mlx_memory_telemetry.active_memory_bytes,
                            utilization.reserved_activation_and_workspace_bytes,
                            utilization.owner_overrun_bytes,
                            utilization.reserved_context_growth_bytes,
                            utilization.unused_headroom_bytes,
                        );
                        chunk_evidence.push(ChunkMemoryEvidence {
                            processed_token_count,
                            active_memory_bytes: mlx_memory_telemetry.active_memory_bytes,
                            utilization,
                        });
                    }
                }
            }
            GeneratedToken::GenerationPreparationStarted { .. } => {
                generation_started_at = Some(std::time::Instant::now());
            }
            GeneratedToken::TokenId {
                token_id,
                is_reasoning_token,
                generation_finalization,
                ..
            } => {
                if !is_reasoning_token {
                    generated_token_count += 1;
                }
                if let Some(generation_finalization) = generation_finalization {
                    eprintln!(
                        "[large-prompt-activation-overrun] status=progress phase=finalized active_bytes={} peak_bytes={}",
                        generation_finalization
                            .mlx_memory_telemetry()
                            .map(|telemetry| telemetry.active_memory_bytes)
                            .unwrap_or(0),
                        generation_finalization
                            .mlx_memory_telemetry()
                            .map(|telemetry| telemetry.peak_memory_bytes)
                            .unwrap_or(0),
                    );
                    break;
                }
                let _ = token_id;
            }
            GeneratedToken::EndOfSequence => break,
        }
    }
    let decode_elapsed_seconds = generation_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64())
        .unwrap_or_default();
    let prefill_tokens_per_second = f64::from(
        chunk_evidence
            .last()
            .map(|evidence| evidence.processed_token_count)
            .unwrap_or(0),
    ) / (prefill_elapsed_millis_total.max(1) as f64 / 1_000.0);
    let decode_tokens_per_second =
        generated_token_count as f64 / decode_elapsed_seconds.max(f64::EPSILON);
    eprintln!(
        "[large-prompt-activation-overrun] status=throughput prefill_tokens_per_second={prefill_tokens_per_second:.1} decode_tokens_per_second={decode_tokens_per_second:.1} prompt_tokens={} output_tokens={generated_token_count}",
        chunk_evidence
            .last()
            .map(|evidence| evidence.processed_token_count)
            .unwrap_or(0),
    );

    assert!(
        observed_drafter_phase,
        "the 50,400-token prompt must engage the drafter (above the 50,000 floor) for this journey to exercise the production shape"
    );
    assert!(
        observed_live_drafter_memory,
        "the engaged drafter must report live request-scoped draft memory during prefill"
    );
    assert!(
        generated_token_count > 0,
        "the large prompt must complete with generated output"
    );

    // The #644 target bound: the planned activation reserve is a promise the
    // engine makes about its own future workspace. Promising more than the
    // entire ceiling cannot be honored and inflates the expert retention
    // ceiling loss. This assertion is expected to fail until #644 is fixed;
    // the panic message carries the per-chunk evidence table.
    let ceiling_bytes = chunk_evidence
        .first()
        .map(|evidence| evidence.utilization.mlx_active_memory_ceiling_bytes)
        .unwrap_or(0);
    let worst_activation_reserve_bytes = chunk_evidence
        .iter()
        .map(|evidence| evidence.utilization.reserved_activation_and_workspace_bytes)
        .max()
        .unwrap_or(0);
    let evidence_table = chunk_evidence
        .iter()
        .map(|evidence| {
            format!(
                "processed={} active_gb={:.2} activation_reserve_gb={:.2} overrun_gb={:.2}",
                evidence.processed_token_count,
                evidence.active_memory_bytes as f64 / 1_000_000_000.0,
                evidence.utilization.reserved_activation_and_workspace_bytes as f64
                    / 1_000_000_000.0,
                evidence.utilization.owner_overrun_bytes as f64 / 1_000_000_000.0,
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    assert!(
        worst_activation_reserve_bytes <= ceiling_bytes,
        "the planned prefill activation reserve must stay within the MLX ceiling (issue #644); worst_reserve_bytes={worst_activation_reserve_bytes} ceiling_bytes={ceiling_bytes} chunks={} evidence=[{evidence_table}]",
        chunk_evidence.len(),
    );
    let _ = decode_generated_output_text;
}
