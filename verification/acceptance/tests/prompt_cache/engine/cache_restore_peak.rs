use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PersistentPromptCacheDiskStoreConfig,
    Qwen3_5ArtifactValidator, Qwen3_5InferenceRequest, Qwen3_5ResidentEngine,
    Qwen3_5ResidentPromptProcessingChunkSizer, Qwen3_5Tokenizer,
};
use astronomical_runtime_integration::MlxRuntime;
use tokio::time::timeout;

use super::engine_prompt_cache::{
    require_persistent_prompt_cache_acceptance_completion, wait_for_persistent_prompt_cache_blocks,
};
use super::large_prefill_prompt;

const CACHE_RESTORE_PEAK_ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);
const CACHE_RESTORE_PEAK_PREFILL_CHUNK_TOKENS: u32 = 2_048;
const CACHE_RESTORE_PEAK_BLOCK_TOKENS: u32 = 2_048;
const CACHE_RESTORE_PEAK_COMPLETE_BLOCK_COUNT: usize = 4;
const CACHE_RESTORE_PEAK_PARTIAL_TAIL_TOKENS: usize = CACHE_RESTORE_PEAK_BLOCK_TOKENS as usize / 2;
const CACHE_RESTORE_PEAK_OUTPUT_TOKEN_COUNT: u16 = 1;
const ROMEO_AND_JULIET_REQUEST_ID_COLD: u64 = 81_001;
const ROMEO_AND_JULIET_REQUEST_ID_RESTORED: u64 = 81_002;

#[tokio::test]
#[ignore = "loads Ornith and measures the real long-prefix prompt-cache restore peak"]
async fn should_restore_a_long_ornith_prefix_with_one_block_of_workspace() {
    require_persistent_prompt_cache_acceptance_completion(async {
        timeout(
            CACHE_RESTORE_PEAK_ACCEPTANCE_TIMEOUT,
            // MlxRuntime holds a raw MLX device pointer and is therefore not
            // Send, so the in-phase memory ticker runs as a local task inside a
            // LocalSet rather than a spawned Send task.
            tokio::task::LocalSet::new().run_until(run_cache_restore_peak_acceptance()),
        )
        .await
        .expect("the real prompt-cache peak acceptance should finish within 115 seconds");
    })
    .await;
}

async fn run_cache_restore_peak_acceptance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let journey_started_at = tokio::time::Instant::now();
    // This journey measures the prompt-cache restore peak, not SSD streaming:
    // the resident sparse MoE role keeps the whole artifact wired so the peak
    // measurement is not confounded by expert paging.
    let model_directory = crate::common::configured_resident_sparse_moe_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the installed model should validate before prompt preparation");
    let prompt_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the installed tokenizer should load before prompt preparation");
    let prompt_token_ids =
        large_prefill_prompt::representative_long_generation_prompt_token_ids_capped_at(
            &prompt_tokenizer,
            validated_artifact.model_id(),
            // The partial tail is deliberate: a prompt whose length is an exact
            // multiple of the block size keeps its last block for a forward pass,
            // because the final prompt token must still produce the logits that
            // begin decode (see persistent_cache::prefix_lookup). A prompt of whole
            // blocks only would therefore restore one block fewer than published,
            // and the journey would measure a different restore shape than intended.
            CACHE_RESTORE_PEAK_COMPLETE_BLOCK_COUNT * CACHE_RESTORE_PEAK_BLOCK_TOKENS as usize
                + CACHE_RESTORE_PEAK_PARTIAL_TAIL_TOKENS,
        );
    let expected_complete_block_count =
        prompt_token_ids.len() / CACHE_RESTORE_PEAK_BLOCK_TOKENS as usize;
    assert!(
        expected_complete_block_count >= 4,
        "the source prompt should cover at least four complete cache blocks"
    );

    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the acceptance should create a temporary cache root");
    let memory_limits = crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    let mut worker_chunking_configuration = crate::common::standard_worker_chunking_configuration();
    worker_chunking_configuration.prompt_cache_block_tokens = Some(CACHE_RESTORE_PEAK_BLOCK_TOKENS);
    let prompt_processing_chunk_sizer =
        Qwen3_5ResidentPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
            CACHE_RESTORE_PEAK_PREFILL_CHUNK_TOKENS,
        )
        .expect("the selected prefill chunk size should be valid");
    let mut qwen3_5_engine = Qwen3_5ResidentEngine::new_with_prompt_processing_chunk_sizer(
        validated_artifact,
        memory_limits.active_memory_limit_bytes(),
        memory_limits.allocator_cache_memory_limit_bytes(),
        Some(PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_directory.path().to_path_buf(),
            persistent_prompt_cache_directory.path().to_path_buf(),
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        )),
        prompt_processing_chunk_sizer,
        248_069,
        model_directory,
        worker_chunking_configuration,
    )
    .expect("the engine should accept the temporary cache configuration");
    qwen3_5_engine
        .load()
        .await
        .expect("the installed model should load for the restore journey");
    eprintln!(
        "[prompt-cache-restore-peak] status=engine-loaded elapsed_seconds={}",
        journey_started_at.elapsed().as_secs()
    );
    let mlx_runtime = std::sync::Arc::new(
        MlxRuntime::initialize(memory_limits).expect("the journey should share engine limits"),
    );

    eprintln!(
        "[prompt-cache-restore-peak] status=cold-start blocks={expected_complete_block_count} prompt_tokens={} elapsed_seconds={}",
        prompt_token_ids.len(),
        journey_started_at.elapsed().as_secs()
    );
    let cold_start_requested_at = tokio::time::Instant::now();
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                RequestId::new(ROMEO_AND_JULIET_REQUEST_ID_COLD),
                prompt_token_ids.clone(),
                CACHE_RESTORE_PEAK_OUTPUT_TOKEN_COUNT,
            )
            .with_image_pad_token_id(248_069),
        )
        .await
        .expect("the cold Romeo and Juliet request should start");
    eprintln!(
        "[prompt-cache-restore-peak] status=cold-started start_milliseconds={} elapsed_seconds={}",
        cold_start_requested_at.elapsed().as_millis(),
        journey_started_at.elapsed().as_secs()
    );
    let (cold_generated_token_ids, cold_prefill_chunk_token_counts) =
        generate_token_ids_with_event_timing(
            &mut qwen3_5_engine,
            RequestId::new(ROMEO_AND_JULIET_REQUEST_ID_COLD),
            usize::from(CACHE_RESTORE_PEAK_OUTPUT_TOKEN_COUNT),
            "cold_prefill",
            journey_started_at,
            mlx_runtime.clone(),
        )
        .await;
    eprintln!(
        "[prompt-cache-restore-peak] status=cold-completed elapsed_seconds={} chunk_token_counts={cold_prefill_chunk_token_counts:?}",
        journey_started_at.elapsed().as_secs()
    );
    wait_for_persistent_prompt_cache_blocks(&qwen3_5_engine, expected_complete_block_count).await;
    let cache_file_sizes = safetensors_file_sizes(persistent_prompt_cache_directory.path())
        .expect("the temporary cache should expose the published block sizes");
    assert!(
        cache_file_sizes.len() >= expected_complete_block_count,
        "the cold prefill should publish multiple sequence blocks and snapshots"
    );
    let destination_byte_bound = cache_file_sizes.iter().sum::<u64>();
    let largest_source_block_bytes = cache_file_sizes
        .iter()
        .copied()
        .max()
        .expect("the cache should contain source blocks");

    let active_memory_bytes_before_restore = mlx_runtime
        .memory_snapshot()
        .expect("the journey should sample active memory before the cache hit")
        .active_memory_bytes() as u64;
    mlx_runtime
        .reset_peak_memory()
        .expect("the journey should reset the process peak before cache restore");
    let restored_generation_start = qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                RequestId::new(ROMEO_AND_JULIET_REQUEST_ID_RESTORED),
                prompt_token_ids,
                CACHE_RESTORE_PEAK_OUTPUT_TOKEN_COUNT,
            )
            .with_image_pad_token_id(248_069),
        )
        .await
        .expect("the restored request should start");
    assert!(
        restored_generation_start.cached_token_count()
            >= (expected_complete_block_count * CACHE_RESTORE_PEAK_BLOCK_TOKENS as usize) as u32,
        "the warm request should restore every complete prefix block: cached_tokens={} expected_at_least={} blocks={expected_complete_block_count} block_tokens={}",
        restored_generation_start.cached_token_count(),
        expected_complete_block_count * CACHE_RESTORE_PEAK_BLOCK_TOKENS as usize,
        CACHE_RESTORE_PEAK_BLOCK_TOKENS
    );
    let restore_memory_snapshot = mlx_runtime
        .memory_snapshot()
        .expect("the journey should sample the restore peak");
    let restore_peak_delta_bytes = (restore_memory_snapshot.peak_memory_bytes() as u64)
        .saturating_sub(active_memory_bytes_before_restore);
    let concat_restore_peak_bound_bytes = destination_byte_bound
        .saturating_mul(2)
        .saturating_add(largest_source_block_bytes.saturating_mul(3));
    eprintln!(
        "[prompt-cache-restore-peak] status=measured cached_tokens={} cache_files={} active_before_bytes={active_memory_bytes_before_restore} peak_bytes={} peak_delta_bytes={restore_peak_delta_bytes} destination_byte_bound={destination_byte_bound} largest_source_block_bytes={largest_source_block_bytes} concat_restore_peak_bound_bytes={concat_restore_peak_bound_bytes}",
        restored_generation_start.cached_token_count(),
        cache_file_sizes.len(),
        restore_memory_snapshot.peak_memory_bytes(),
    );
    assert!(
        restore_peak_delta_bytes <= concat_restore_peak_bound_bytes,
        "the real cache-hit peak must fit the destination plus the complete source block set \
         and MLX scratch: peak_delta={restore_peak_delta_bytes} \
         bound={concat_restore_peak_bound_bytes}"
    );

    let (restored_generated_token_ids, restored_prefill_chunk_token_counts) =
        generate_token_ids_with_event_timing(
            &mut qwen3_5_engine,
            RequestId::new(ROMEO_AND_JULIET_REQUEST_ID_RESTORED),
            usize::from(CACHE_RESTORE_PEAK_OUTPUT_TOKEN_COUNT),
            "restored_prefill",
            journey_started_at,
            mlx_runtime,
        )
        .await;
    assert_eq!(
        cold_generated_token_ids, restored_generated_token_ids,
        "the restored prompt should continue with the same next token as cold prefill"
    );
    eprintln!(
        "[prompt-cache-restore-peak] status=success elapsed_seconds={} restored_chunk_token_counts={restored_prefill_chunk_token_counts:?}",
        journey_started_at.elapsed().as_secs()
    );
}

/// Drives one generate-to-token phase while attributing wall time to every
/// engine event, because a stalled journey must show whether the cost sits
/// inside a single forward, between prefill chunks, or in cache publication.
/// The ten-second memory ticker answers the paired question of whether the GPU
/// is wired and computing while no event arrives.
async fn generate_token_ids_with_event_timing(
    qwen3_5_engine: &mut Qwen3_5ResidentEngine,
    request_id: RequestId,
    generated_token_count: usize,
    phase_label: &'static str,
    journey_started_at: tokio::time::Instant,
    mlx_runtime: std::sync::Arc<MlxRuntime>,
) -> (Vec<u32>, Vec<u32>) {
    let ticker_mlx_runtime = mlx_runtime.clone();
    let ticker_journey_started_at = journey_started_at;
    let memory_ticker = tokio::task::spawn_local(async move {
        let mut ticker_interval = tokio::time::interval(Duration::from_secs(10));
        ticker_interval.tick().await;
        loop {
            ticker_interval.tick().await;
            let memory_report = ticker_mlx_runtime
                .memory_snapshot()
                .map(|memory_snapshot| {
                    format!(
                        "active_gb={:.3} peak_gb={:.3}",
                        memory_snapshot.active_memory_bytes() as f64 / 1_000_000_000.0,
                        memory_snapshot.peak_memory_bytes() as f64 / 1_000_000_000.0,
                    )
                })
                .unwrap_or_else(|snapshot_error| {
                    format!("memory_snapshot_unavailable={snapshot_error}")
                });
            eprintln!(
                "[prompt-cache-restore-peak] status=waiting phase={phase_label} elapsed_seconds={} {memory_report}",
                ticker_journey_started_at.elapsed().as_secs()
            );
        }
    });

    let mut generated_token_ids = Vec::new();
    let mut completed_prefill_chunk_token_counts = Vec::new();
    let phase_started_at = tokio::time::Instant::now();
    let mut previous_event_at = phase_started_at;
    let outcome = loop {
        let decoded_event = qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("the engine should advance");
        let event_observed_at = tokio::time::Instant::now();
        let (event_label, should_finish) = match decoded_event {
            GeneratedToken::TokenId {
                token_id,
                generation_finalization,
                ..
            } => {
                generated_token_ids.push(token_id);
                (
                    "token_id".to_owned(),
                    generation_finalization.is_some()
                        || generated_token_ids.len() == generated_token_count,
                )
            }
            GeneratedToken::PrefillProgress {
                completed_prefill_chunk_tokens,
                ..
            } => {
                completed_prefill_chunk_token_counts.push(completed_prefill_chunk_tokens);
                (
                    format!("prefill_progress chunk_tokens={completed_prefill_chunk_tokens}"),
                    false,
                )
            }
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {
                ("prompt_processing_phase_started".to_owned(), false)
            }
            GeneratedToken::GenerationPreparationStarted { .. } => {
                ("generation_preparation_started".to_owned(), false)
            }
            GeneratedToken::EndOfSequence => ("end_of_sequence".to_owned(), true),
        };
        eprintln!(
            "[prompt-cache-restore-peak] status=event phase={phase_label} event={event_label} gap_ms={} phase_milliseconds={} elapsed_seconds={}",
            (event_observed_at - previous_event_at).as_millis(),
            (event_observed_at - phase_started_at).as_millis(),
            journey_started_at.elapsed().as_secs()
        );
        previous_event_at = event_observed_at;
        if should_finish {
            break (generated_token_ids, completed_prefill_chunk_token_counts);
        }
    };
    memory_ticker.abort();
    outcome
}

fn safetensors_file_sizes(directory: &Path) -> std::io::Result<Vec<u64>> {
    let mut pending_directories = vec![PathBuf::from(directory)];
    let mut file_sizes = Vec::new();
    while let Some(current_directory) = pending_directories.pop() {
        for directory_entry in fs::read_dir(current_directory)? {
            let directory_entry = directory_entry?;
            let entry_type = directory_entry.file_type()?;
            if entry_type.is_dir() {
                pending_directories.push(directory_entry.path());
            } else if entry_type.is_file()
                && directory_entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "safetensors")
            {
                file_sizes.push(directory_entry.metadata()?.len());
            }
        }
    }
    Ok(file_sizes)
}
