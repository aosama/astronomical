//! `astro_ssd_prompt_cache` acceptance journeys over the resident sparse MoE role.
//!
//! Every journey in this binary runs the resident sparse MoE role
//! (`resident_sparse_moe`) because it measures `astro_ssd_prompt_cache` behaviour, not expert
//! paging: the surfaces under test (`astro_ssd_prompt_cache` disabled, `astro_ssd_prompt_cache` miss memory, restore
//! peak, parity, block reporting, partial tail reuse, startup cleanup
//! attribution, and the vision journeys) are identical in resident mode. The
//! whole binary takes about three minutes. The journeys are `#[ignore]`d
//! real-model journeys: run the binary directly with
//! `-- --ignored --nocapture --test-threads=1`.
//!
//! The interaction-matrix tests name the `LiveReuse`, `WorkerRestart`, and
//! `DeletedWhileLive` cells explicitly. Run each full test name in a separate
//! process through `cargo xtask test-run --exact --ignored`; see that module's
//! documentation for the complete runner invocation.
//!
//! The wide-chunk parity journey selects its prefill span with
//! `prompt_processing_chunk_tokens` in the explicit test-run configuration;
//! the cell accepts only 2048 or 4096 and defaults to 2048.

use std::{future::Future, path::Path, time::Duration};

use ipc_protocol::{RequestId, WorkerEvent};
use model_serving::{
    AstroSsdPromptCacheDiskStoreConfig, AstroSsdPromptCacheModelContract, GeneratedToken,
    InferenceEngine, PromptProcessingChunkSizer, Qwen3_5ArtifactValidator, Qwen3_5TextEngine,
    Qwen3_5Tokenizer, TextInferenceRequest, TextMlxEngineConstructors,
    qwen3_5_astro_decoder_state_cache_layout,
};
use tokio::time::{Instant, MissedTickBehavior, interval, sleep};

use super::large_prefill_prompt::{
    LARGE_PREFILL_ACCEPTANCE_OUTPUT_TOKEN_COUNT,
    representative_long_generation_prompt_token_ids_capped_at,
};

const ASTRO_SSD_PROMPT_CACHE_ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);
const SAY_HI_PROMPT_TOKEN_IDS: [u32; 15] = [
    248_045, 846, 198, 44_240, 15_131, 13, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
    248_069, 271,
];
const MAXIMUM_OUTPUT_TOKEN_COUNT: usize = 10;

struct AstroSsdPromptCacheParityAcceptanceOutcome {
    cold_generated_token_ids: Vec<u32>,
    cold_completed_prefill_chunk_token_counts: Vec<u32>,
    astro_ssd_prompt_cache_restored_token_count: u32,
    restored_generated_token_ids: Vec<u32>,
}

#[tokio::test]
#[serial_test::serial]
#[ignore = "loads and generates with the complete Ornith artifact"]
async fn should_restore_astro_ssd_prompt_cache_blocks_and_report_astro_ssd_prompt_cache_restored_tokens_on_the_second_run()
 {
    require_astro_ssd_prompt_cache_acceptance_completion(
        run_astro_ssd_prompt_cache_restore_acceptance(),
    )
    .await;
}

async fn run_astro_ssd_prompt_cache_restore_acceptance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_resident_sparse_moe_model_directory();
    let acceptance_outcome = run_astro_ssd_prompt_cache_parity_acceptance(
        &model_directory,
        astro_ssd_prompt_cache_eligible_prompt_token_ids(
            astro_ssd_prompt_cache_eligible_prompt_token_count_for_block_multiplier(
                &model_directory,
                2,
            )
            .await,
        ),
        1,
        2_048,
    )
    .await;

    assert_eq!(acceptance_outcome.cold_generated_token_ids.len(), 1);
    assert_eq!(
        acceptance_outcome.cold_generated_token_ids,
        acceptance_outcome.restored_generated_token_ids
    );
}

#[tokio::test]
#[serial_test::serial]
#[ignore = "loads Ornith and compares cold and restored astro SSD prompt cache tokens"]
async fn should_preserve_tokens_after_astro_ssd_prompt_cache_restore() {
    require_astro_ssd_prompt_cache_acceptance_completion(async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = crate::common::configured_resident_sparse_moe_model_directory();
        let acceptance_outcome = run_astro_ssd_prompt_cache_parity_acceptance(
            &model_directory,
            astro_ssd_prompt_cache_eligible_prompt_token_ids(
                astro_ssd_prompt_cache_eligible_prompt_token_count_for_block_multiplier(
                    &model_directory,
                    4,
                )
                .await,
            ),
            MAXIMUM_OUTPUT_TOKEN_COUNT,
            2_048,
        )
        .await;

        assert!(
            !acceptance_outcome.cold_generated_token_ids.is_empty(),
            "the cold Ornith request should emit at least one token before terminal EOS"
        );
        assert_eq!(
            acceptance_outcome.cold_generated_token_ids,
            acceptance_outcome.restored_generated_token_ids
        );
    })
    .await;
}

#[tokio::test]
#[serial_test::serial]
#[ignore = "runs one fixed 2048 or 4096 astro SSD prompt cache safe prefill acceptance cell"]
async fn should_restore_exact_astro_ssd_prompt_cache_parity_for_one_selected_large_prefill_size() {
    require_astro_ssd_prompt_cache_acceptance_completion(async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let configured_prefill_chunk_tokens =
            test_harness::test_run::TestRunContext::current()
                .configuration
                .prompt_processing_chunk_tokens
                .unwrap_or(2_048);
        assert!(
            [2_048, 4_096].contains(&configured_prefill_chunk_tokens),
            "the selected astro-ssd-prompt-cache prefill acceptance size must be 2048 or 4096"
        );
        let model_directory = crate::common::configured_resident_sparse_moe_model_directory();
        let prompt_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 20_480)
            .expect("the selected acceptance artifact should validate for prompt preparation");
        let prompt_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&prompt_artifact)
            .expect("the selected acceptance tokenizer should load");
        let representative_prompt_token_ids =
            representative_long_generation_prompt_token_ids_capped_at(
                &prompt_tokenizer,
                prompt_artifact.model_id(),
                // The block size equals the selected chunk here, and a prompt
                // that is an exact multiple of the block size keeps its last
                // block for a forward pass, so the partial tail is what makes
                // all four complete blocks restorable
                // (see astro_ssd_prompt_cache::prefix_lookup). Four blocks plus a
                // half-block tail also keeps the parity cell inside its
                // 115-second budget at 2,048-token chunks.
                4 * configured_prefill_chunk_tokens as usize
                    + configured_prefill_chunk_tokens as usize / 2,
            );
        let acceptance_outcome = run_astro_ssd_prompt_cache_parity_acceptance(
            &model_directory,
            representative_prompt_token_ids,
            LARGE_PREFILL_ACCEPTANCE_OUTPUT_TOKEN_COUNT,
            configured_prefill_chunk_tokens,
        )
        .await;

        assert!(
            acceptance_outcome
                .cold_completed_prefill_chunk_token_counts
                .iter()
                .sum::<u32>()
                >= configured_prefill_chunk_tokens,
            "the cold request must read at least one prompt chunk of the selected size"
        );
        assert!(
            acceptance_outcome.astro_ssd_prompt_cache_restored_token_count
                >= configured_prefill_chunk_tokens,
            "the restored request recovered {} tokens but must recover every 2048-token block produced by one selected {configured_prefill_chunk_tokens}-token forward",
            acceptance_outcome.astro_ssd_prompt_cache_restored_token_count,
        );
        assert!(
            acceptance_outcome.cold_generated_token_ids.len() >= 32,
            // This floor guards against a degenerate answer (an immediate
            // end-of-sequence), not against a particular answer length: the
            // pinned resident artifact answers this four-block-plus-tail
            // fixture with an end-of-sequence at 432 of the 1,024 requested
            // tokens (measured 2026-10-10), and the currently installed
            // revision measures 52 (measured 2026-10-31) — so a percentage
            // near the requested length would assert the pinned model's
            // phrasing instead of `astro_ssd_prompt_cache` behaviour. The exact-parity assert
            // below is the contract; this floor only rejects degeneration.
            "the representative acceptance produced only {} of approximately {} requested output tokens",
            acceptance_outcome.cold_generated_token_ids.len(),
            LARGE_PREFILL_ACCEPTANCE_OUTPUT_TOKEN_COUNT,
        );
        assert_eq!(
            acceptance_outcome.cold_generated_token_ids,
            acceptance_outcome.restored_generated_token_ids,
            "cold and restored deterministic generation must remain identical"
        );
    })
    .await;
}

async fn run_astro_ssd_prompt_cache_parity_acceptance(
    model_directory: &Path,
    prompt_token_ids: Vec<u32>,
    generated_token_count: usize,
    fixed_prefill_chunk_tokens: u32,
) -> AstroSsdPromptCacheParityAcceptanceOutcome {
    let astro_ssd_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create an astro SSD prompt cache directory");

    // First run: cold prefill, should populate the `astro_ssd_prompt_cache`.
    // The exact storage contract now depends on bound affine dtypes. Read block
    // geometry from the loaded engine rather than reconstructing a config-only
    // contract that could disagree with production serialization.
    let (mut qwen3_5_engine, _model_id, _model_revision, astro_ssd_prompt_cache_block_token_count) =
        load_astro_ssd_prompt_cache_acceptance_engine(
            model_directory,
            astro_ssd_prompt_cache_directory.path(),
            fixed_prefill_chunk_tokens,
        )
        .await;
    let first_request_id = RequestId::new(2_001);
    let first_generation_start = qwen3_5_engine
        .start_generation(locked_astro_ssd_prompt_cache_comparison_request(
            first_request_id,
            prompt_token_ids.clone(),
            generated_token_count,
        ))
        .await
        .expect("the first engine should accept the request");
    assert_eq!(
        first_generation_start.astro_ssd_prompt_cache_restored_token_count(),
        0
    );
    let (cold_generated_token_ids, cold_completed_prefill_chunk_token_counts) =
        generate_token_ids(&mut qwen3_5_engine, first_request_id, generated_token_count).await;
    let minimum_expected_sequence_state_block_count = (usize::try_from(fixed_prefill_chunk_tokens)
        .unwrap_or(usize::MAX)
        / astro_ssd_prompt_cache_block_token_count)
        .max(1);
    wait_for_astro_ssd_prompt_cache_blocks(
        &qwen3_5_engine,
        minimum_expected_sequence_state_block_count,
    )
    .await;
    // Second run: same prompt, same loaded engine, same `astro_ssd_prompt_cache` directory. The
    // prompt is longer than one persistent `astro_ssd_prompt_cache` block, so the second start
    // must report a hit without paying another full model load in this proof.
    let second_request_id = RequestId::new(2_002);
    let second_generation_start = qwen3_5_engine
        .start_generation(locked_astro_ssd_prompt_cache_comparison_request(
            second_request_id,
            prompt_token_ids,
            generated_token_count,
        ))
        .await
        .expect("the second engine should accept the request");
    assert!(
        second_generation_start.astro_ssd_prompt_cache_restored_token_count()
            >= astro_ssd_prompt_cache_block_token_count as u32,
        "the second run should report at least one restored astro-ssd-prompt-cache block"
    );
    let astro_ssd_prompt_cache_restored_token_count =
        second_generation_start.astro_ssd_prompt_cache_restored_token_count();
    let (restored_generated_token_ids, _restored_completed_prefill_chunk_token_counts) =
        generate_token_ids(
            &mut qwen3_5_engine,
            second_request_id,
            generated_token_count,
        )
        .await;
    AstroSsdPromptCacheParityAcceptanceOutcome {
        cold_generated_token_ids,
        cold_completed_prefill_chunk_token_counts,
        astro_ssd_prompt_cache_restored_token_count,
        restored_generated_token_ids,
    }
}

pub(super) async fn load_astro_ssd_prompt_cache_acceptance_engine(
    model_directory: &Path,
    astro_ssd_prompt_cache_directory: &Path,
    fixed_prefill_chunk_tokens: u32,
) -> (Qwen3_5TextEngine, String, String, usize) {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the model-artifact checkpoint should validate before engine loading");
    let model_id = validated_artifact.model_id().to_owned();
    let model_revision = validated_artifact.revision().to_owned();
    let prefill_chunk_sizer =
        PromptProcessingChunkSizer::for_chunk_size_tokens(fixed_prefill_chunk_tokens)
            .expect("the selected resident prefill size should be valid");
    let mlx_memory_limits =
        crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    let mut worker_chunking_configuration = crate::common::standard_worker_chunking_configuration();
    worker_chunking_configuration.ssd_prompt_cache_block_tokens = Some(fixed_prefill_chunk_tokens);
    let mut qwen3_5_engine = Qwen3_5TextEngine::new_with_prompt_processing_chunk_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.mlx_buffer_cache_limit_bytes(),
        Some(AstroSsdPromptCacheDiskStoreConfig::new(
            astro_ssd_prompt_cache_directory.to_path_buf(),
            astro_ssd_prompt_cache_directory.to_path_buf(),
            crate::common::configured_model_artifact_astro_ssd_prompt_cache_maximum_size_bytes(),
        )),
        prefill_chunk_sizer,
        248_069,
        model_directory.to_path_buf(),
        worker_chunking_configuration,
    )
    .expect("the engine should accept the astro-ssd-prompt-cache directory");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should load the model");
    super::keep_the_hosted_drafter_silent(&qwen3_5_engine).await;
    let astro_ssd_prompt_cache_block_token_count =
        astro_ssd_prompt_cache_block_token_count(&qwen3_5_engine).await;
    (
        qwen3_5_engine,
        model_id,
        model_revision,
        astro_ssd_prompt_cache_block_token_count,
    )
}

async fn astro_ssd_prompt_cache_eligible_prompt_token_count_for_block_multiplier(
    model_directory: &Path,
    block_multiplier: usize,
) -> usize {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the model-artifact checkpoint should validate before sizing the prompt");
    let mlx_memory_limits =
        crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    // This pre-load helper needs only a prompt comfortably beyond a boundary.
    // Use the widest supported state dtype and a caller-supplied multiplier for
    // sizing; hit assertions later use the exact block count reported after load.
    let astro_ssd_prompt_cache_model_contract = AstroSsdPromptCacheModelContract::resolve(
        validated_artifact.model_id().to_owned(),
        validated_artifact.revision().to_owned(),
        qwen3_5_astro_decoder_state_cache_layout(
            validated_artifact.config(),
            crate::common::standard_worker_chunking_configuration()
                .full_attention_key_value_growth_tokens as usize,
            &crate::common::qwen3_5_moe::float32_astro_decoder_state_cache_layer_dtypes(
                validated_artifact.config(),
            ),
        )
        .expect("the validated artifact should provide an astro decoder state cache layout"),
        validated_artifact.config().maximum_position_count() as usize,
        mlx_memory_limits.active_memory_limit_bytes() as u64,
        crate::common::configured_model_artifact_astro_ssd_prompt_cache_maximum_size_bytes(),
        crate::common::standard_worker_chunking_configuration()
            .ssd_prompt_cache_block_tokens
            .map(|block_token_count| block_token_count as usize),
        crate::common::standard_worker_chunking_configuration()
            .ssd_prompt_cache_common_prefix_stride_blocks,
    )
    .expect("the model should resolve a persistent storage contract");
    astro_ssd_prompt_cache_eligible_prompt_token_ids(
        astro_ssd_prompt_cache_model_contract
            .block_token_count()
            .saturating_mul(block_multiplier)
            .saturating_add(16),
    )
    .len()
}

async fn astro_ssd_prompt_cache_block_token_count(qwen3_5_engine: &Qwen3_5TextEngine) -> usize {
    // Stats are emitted from the engine-owned, load-derived `astro_ssd_prompt_cache` contract and
    // therefore share the exact geometry used by lookup and publication.
    let astro_ssd_prompt_cache_stats = qwen3_5_engine
        .collect_astro_ssd_prompt_cache_stats()
        .await
        .expect("the engine should report persistent astro-ssd-prompt-cache stats")
        .expect("the acceptance engine should have the astro SSD prompt cache enabled");
    let WorkerEvent::AstroSsdPromptCacheStats {
        astro_ssd_prompt_cache_block_token_count,
        ..
    } = astro_ssd_prompt_cache_stats
    else {
        panic!("the engine returned an unexpected astro-ssd-prompt-cache stats event")
    };
    usize::try_from(astro_ssd_prompt_cache_block_token_count)
        .expect("the astro-ssd-prompt-cache block token count should fit usize")
}
pub(super) async fn wait_for_astro_ssd_prompt_cache_blocks(
    qwen3_5_engine: &Qwen3_5TextEngine,
    expected_sequence_state_block_count: usize,
) {
    let publication_started_at = Instant::now();
    loop {
        let astro_ssd_prompt_cache_stats = qwen3_5_engine
            .collect_astro_ssd_prompt_cache_stats()
            .await
            .expect("the engine should report persistent astro-ssd-prompt-cache stats")
            .expect("the acceptance engine should have the astro SSD prompt cache enabled");
        let WorkerEvent::AstroSsdPromptCacheStats {
            astro_ssd_prompt_cache_block_token_count,
            astro_ssd_prompt_cache_sequence_state_block_count,
            astro_ssd_prompt_cache_boundary_state_snapshot_count,
            ..
        } = astro_ssd_prompt_cache_stats
        else {
            panic!("the engine returned an unexpected astro-ssd-prompt-cache stats event")
        };
        if astro_ssd_prompt_cache_sequence_state_block_count
            >= u64::try_from(expected_sequence_state_block_count).unwrap_or(u64::MAX)
        {
            eprintln!(
                "[astro-ssd-prompt-cache-acceptance] status=astro-ssd-prompt-cache-ready sequence_blocks={astro_ssd_prompt_cache_sequence_state_block_count} boundary_snapshots={astro_ssd_prompt_cache_boundary_state_snapshot_count}"
            );
            return;
        }
        assert!(
            publication_started_at.elapsed() < Duration::from_secs(10),
            "only {astro_ssd_prompt_cache_sequence_state_block_count} of {expected_sequence_state_block_count} expected astro-ssd-prompt-cache blocks were published"
        );
        eprintln!(
            "[astro-ssd-prompt-cache-acceptance] status=waiting-for-astro-ssd-prompt-cache block_tokens={astro_ssd_prompt_cache_block_token_count} published_blocks={astro_ssd_prompt_cache_sequence_state_block_count} expected_blocks={expected_sequence_state_block_count}"
        );
        sleep(Duration::from_millis(250)).await;
    }
}

pub(super) async fn generate_token_ids(
    qwen3_5_engine: &mut Qwen3_5TextEngine,
    request_id: RequestId,
    generated_token_count: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut generated_token_ids = Vec::new();
    let mut completed_prefill_chunk_token_counts = Vec::new();
    loop {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("the engine should advance")
        {
            GeneratedToken::TokenId {
                token_id,
                generation_finalization,
                ..
            } => {
                generated_token_ids.push(token_id);
                if generation_finalization.is_some()
                    || generated_token_ids.len() == generated_token_count
                {
                    return (generated_token_ids, completed_prefill_chunk_token_counts);
                }
            }
            GeneratedToken::PrefillProgress {
                completed_prefill_chunk_tokens,
                ..
            } => completed_prefill_chunk_token_counts.push(completed_prefill_chunk_tokens),
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => {
                return (generated_token_ids, completed_prefill_chunk_token_counts);
            }
        }
    }
}

pub(super) fn astro_ssd_prompt_cache_eligible_prompt_token_ids(
    target_token_count: usize,
) -> Vec<u32> {
    let mut prompt_token_ids = Vec::with_capacity(target_token_count);
    while prompt_token_ids.len() < target_token_count {
        prompt_token_ids.extend_from_slice(&SAY_HI_PROMPT_TOKEN_IDS);
    }
    prompt_token_ids.truncate(target_token_count);
    prompt_token_ids
}

pub(super) async fn require_astro_ssd_prompt_cache_acceptance_completion(
    test_future: impl Future<Output = ()>,
) {
    // The pinned artifact is intentionally large, but a local acceptance
    // must never leave the laptop compiling or evaluating indefinitely.
    let started_at = Instant::now();
    let timeout_deadline = sleep(ASTRO_SSD_PROMPT_CACHE_ACCEPTANCE_TIMEOUT);
    let mut progress_interval = interval(Duration::from_secs(10));
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    tokio::pin!(test_future);
    tokio::pin!(timeout_deadline);
    progress_interval.tick().await;
    eprintln!(
        "[astro-ssd-prompt-cache-acceptance] status=start timeout_seconds={}",
        ASTRO_SSD_PROMPT_CACHE_ACCEPTANCE_TIMEOUT.as_secs()
    );

    loop {
        tokio::select! {
            () = &mut test_future => {
                eprintln!(
                    "[astro-ssd-prompt-cache-acceptance] status=success elapsed_seconds={:.1}",
                    started_at.elapsed().as_secs_f64()
                );
                return;
            }
            () = &mut timeout_deadline => {
                panic!(
                    "the persistent astro-ssd-prompt-cache acceptance exceeded {} seconds",
                    ASTRO_SSD_PROMPT_CACHE_ACCEPTANCE_TIMEOUT.as_secs()
                );
            }
            _ = progress_interval.tick() => {
                let elapsed = started_at.elapsed();
                let remaining = ASTRO_SSD_PROMPT_CACHE_ACCEPTANCE_TIMEOUT.saturating_sub(elapsed);
                eprintln!(
                    "[astro-ssd-prompt-cache-acceptance] status=running elapsed_seconds={:.0} ETA<={:.0}",
                    elapsed.as_secs_f64(),
                    remaining.as_secs_f64()
                );
            }
        }
    }
}

fn locked_astro_ssd_prompt_cache_comparison_request(
    request_id: RequestId,
    prompt_token_ids: Vec<u32>,
    generated_token_count: usize,
) -> TextInferenceRequest {
    TextInferenceRequest::new_sampling(
        request_id,
        prompt_token_ids,
        u16::try_from(generated_token_count)
            .expect("the generated token count should fit the engine request"),
        0,
        1_000,
        None,
    )
    .with_image_pad_token_id(248_069)
    .with_thinking_configuration(false, None, Vec::new(), Vec::new())
}
