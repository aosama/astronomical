use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    InferenceEngine, PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5InferenceRequest, Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
};
use astronomical_runtime_integration::MlxRuntime;
use tokio::time::timeout;

use super::engine_prompt_cache::{
    generate_token_ids, require_persistent_prompt_cache_acceptance_completion,
    wait_for_persistent_prompt_cache_blocks,
};
use super::large_prefill_prompt;

const CACHE_RESTORE_PEAK_ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);
const CACHE_RESTORE_PEAK_PREFILL_CHUNK_TOKENS: u32 = 8_192;
const CACHE_RESTORE_PEAK_BLOCK_TOKENS: u32 = 2_048;
const CACHE_RESTORE_PEAK_OUTPUT_TOKEN_COUNT: u16 = 1;
const ROMEO_AND_JULIET_REQUEST_ID_COLD: u64 = 81_001;
const ROMEO_AND_JULIET_REQUEST_ID_RESTORED: u64 = 81_002;

#[tokio::test]
#[ignore = "loads Ornith and measures the real long-prefix prompt-cache restore peak"]
async fn should_restore_a_long_ornith_prefix_with_one_block_of_workspace() {
    require_persistent_prompt_cache_acceptance_completion(async {
        timeout(
            CACHE_RESTORE_PEAK_ACCEPTANCE_TIMEOUT,
            run_cache_restore_peak_acceptance(),
        )
        .await
        .expect("the real prompt-cache peak acceptance should finish within 115 seconds");
    })
    .await;
}

async fn run_cache_restore_peak_acceptance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the installed model should validate before prompt preparation");
    let prompt_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the installed tokenizer should load before prompt preparation");
    let prompt_token_ids = large_prefill_prompt::representative_long_generation_prompt_token_ids(
        &prompt_tokenizer,
        validated_artifact.model_id(),
    );
    let expected_complete_block_count =
        prompt_token_ids.len() / CACHE_RESTORE_PEAK_BLOCK_TOKENS as usize;
    assert!(
        expected_complete_block_count >= 8,
        "the source prompt should cover at least eight complete cache blocks"
    );

    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the acceptance should create a temporary cache root");
    let memory_limits = crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    let mut worker_chunking_configuration = crate::common::standard_worker_chunking_configuration();
    worker_chunking_configuration.prompt_cache_block_tokens = Some(CACHE_RESTORE_PEAK_BLOCK_TOKENS);
    let prompt_processing_chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
            CACHE_RESTORE_PEAK_PREFILL_CHUNK_TOKENS,
        )
        .expect("the selected prefill chunk size should be valid");
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prompt_processing_chunk_sizer(
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
    let mlx_runtime =
        MlxRuntime::initialize(memory_limits).expect("the journey should share engine limits");

    eprintln!(
        "[prompt-cache-restore-peak] status=cold-prefill blocks={expected_complete_block_count} prompt_tokens={}",
        prompt_token_ids.len()
    );
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
    let (cold_generated_token_ids, _) = generate_token_ids(
        &mut qwen3_5_engine,
        RequestId::new(ROMEO_AND_JULIET_REQUEST_ID_COLD),
        usize::from(CACHE_RESTORE_PEAK_OUTPUT_TOKEN_COUNT),
    )
    .await;
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
        "the warm request should restore every complete prefix block"
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

    let (restored_generated_token_ids, _) = generate_token_ids(
        &mut qwen3_5_engine,
        RequestId::new(ROMEO_AND_JULIET_REQUEST_ID_RESTORED),
        usize::from(CACHE_RESTORE_PEAK_OUTPUT_TOKEN_COUNT),
    )
    .await;
    assert_eq!(
        cold_generated_token_ids, restored_generated_token_ids,
        "the restored prompt should continue with the same next token as cold prefill"
    );
    eprintln!("[prompt-cache-restore-peak] status=success");
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
