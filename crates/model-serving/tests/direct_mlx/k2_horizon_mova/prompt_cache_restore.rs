use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{ChatMessage, ChatToolChoice, RequestId};
use astronomical_model_serving::{
    GeneratedToken, K2HorizonMoVAConfig, K2HorizonMoVAInferenceRequest,
    K2HorizonMoVAPromptRenderer, K2HorizonMoVAServingSettings, K2HorizonMoVATokenizer,
    MlxInferenceExecution, initialize_k2_horizon_mova_execution_with_serving_settings,
};
use astronomical_runtime_integration::MlxRuntime;
use tokio::time::timeout;

const ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);
const CACHE_BLOCK_TOKEN_COUNT: u32 = 2_048;
const RESTORED_BLOCK_COUNT: usize = 4;
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
#[ignore = "loads K2 and measures a real multi-block prompt-cache restore"]
async fn should_restore_k2_prompt_cache_with_one_concat_per_layer() {
    eprintln!(
        "[k2-prompt-cache-restore] status=start timeout_seconds={}",
        ACCEPTANCE_TIMEOUT.as_secs()
    );
    timeout(ACCEPTANCE_TIMEOUT, run_prompt_cache_restore_acceptance())
        .await
        .expect("the K2 prompt-cache restore acceptance should finish within 115 seconds");
}

async fn run_prompt_cache_restore_acceptance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_installed_model_directory_by_id(
        crate::common::k2_horizon_mova_model_id(),
    );
    let cache_directory =
        tempfile::tempdir().expect("the K2 acceptance should create a cache root");
    let memory_limits = crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    let mut worker_chunking_configuration = crate::common::standard_worker_chunking_configuration();
    worker_chunking_configuration.prompt_cache_block_tokens = Some(CACHE_BLOCK_TOKEN_COUNT);
    worker_chunking_configuration.prompt_cache_common_prefix_stride_blocks = 1;
    let mut serving_settings = K2HorizonMoVAServingSettings::default_fixed();
    serving_settings.prompt_processing_chunk_tokens = CACHE_BLOCK_TOKEN_COUNT;
    serving_settings.chunking = Some(worker_chunking_configuration);
    serving_settings.persistent_prompt_cache_enabled = true;
    serving_settings.prompt_cache_config = Some(PromptCacheConfig::new(
        cache_directory.path().to_path_buf(),
        50_000_000_000,
    ));

    eprintln!("[k2-prompt-cache-restore] status=loading-model");
    let (_processor, mut execution) = initialize_k2_horizon_mova_execution_with_serving_settings(
        &model_directory,
        memory_limits.active_memory_limit_bytes(),
        memory_limits.allocator_cache_memory_limit_bytes(),
        true,
        serving_settings,
    )
    .expect("the installed K2 artifact should start with persistent caching enabled");
    execution
        .load()
        .expect("K2 weights and the persistent cache should load");
    let mlx_runtime = MlxRuntime::initialize(memory_limits)
        .expect("the K2 acceptance should share the engine's MLX memory limits");
    let prompt_token_ids = romeo_and_juliet_prompt_token_ids(&model_directory);
    eprintln!(
        "[k2-prompt-cache-restore] status=model-ready prompt_tokens={}",
        prompt_token_ids.len()
    );

    let cold_generation_start = execution
        .start_generation(k2_request(prompt_token_ids.clone()))
        .expect("the cold prompt should start");
    assert_eq!(cold_generation_start.cached_token_count(), 0);
    let cold_token_id = generate_first_token_to_finalization(&mut execution, RequestId::new(1));
    let cache_block_file_sizes = safetensors_file_sizes(cache_directory.path())
        .expect("the cache root should expose published block sizes");
    assert!(
        cache_block_file_sizes.len() >= RESTORED_BLOCK_COUNT,
        "the cold request should publish multiple complete K2 blocks, found {}",
        cache_block_file_sizes.len()
    );
    let restored_destination_bytes = cache_block_file_sizes.iter().sum::<u64>();
    let largest_loaded_block_bytes = cache_block_file_sizes
        .iter()
        .copied()
        .max()
        .expect("the K2 cache should publish at least one block");

    let active_memory_bytes_before_restore = mlx_runtime
        .memory_snapshot()
        .expect("the K2 acceptance should sample active memory before restore")
        .active_memory_bytes() as u64;
    mlx_runtime
        .reset_peak_memory()
        .expect("the K2 restore journey should reset the MLX peak counter");
    let warm_generation_start = execution
        .start_generation(k2_request(prompt_token_ids))
        .expect("the warm prompt should start from its saved prefix");
    assert!(
        warm_generation_start.cached_token_count() >= CACHE_BLOCK_TOKEN_COUNT,
        "the warm request should restore at least one complete cache block"
    );
    let memory_snapshot_after_restore = mlx_runtime
        .memory_snapshot()
        .expect("the K2 acceptance should sample the restore peak");
    let restore_peak_delta_bytes = (memory_snapshot_after_restore.peak_memory_bytes() as u64)
        .saturating_sub(active_memory_bytes_before_restore);
    // The concat restore materializes the destination while the complete
    // source block set streams through in one pass, so the honest peak is
    // the destination plus the complete block set plus MLX scratch headroom.
    let concat_restore_peak_bound_bytes = restored_destination_bytes
        .saturating_mul(2)
        .saturating_add(largest_loaded_block_bytes.saturating_mul(3));
    eprintln!(
        "[k2-prompt-cache-restore] status=measured blocks={} cached_tokens={} active_before_bytes={active_memory_bytes_before_restore} peak_bytes={} restore_peak_delta_bytes={restore_peak_delta_bytes} destination_bytes={restored_destination_bytes} largest_block_bytes={largest_loaded_block_bytes} concat_restore_peak_bound_bytes={concat_restore_peak_bound_bytes}",
        cache_block_file_sizes.len(),
        warm_generation_start.cached_token_count(),
        memory_snapshot_after_restore.peak_memory_bytes(),
    );
    assert!(
        restore_peak_delta_bytes <= concat_restore_peak_bound_bytes,
        "the real K2 restore peak should remain within destination plus the \
         complete source block set: peak_delta={restore_peak_delta_bytes} \
         bound={concat_restore_peak_bound_bytes}"
    );

    let restored_token_id = generate_first_token_to_finalization(&mut execution, RequestId::new(2));
    assert_eq!(
        cold_token_id, restored_token_id,
        "cold and restored K2 requests should continue with the same token"
    );
    eprintln!("[k2-prompt-cache-restore] status=success");
}

fn k2_request(prompt_token_ids: Vec<u32>) -> K2HorizonMoVAInferenceRequest {
    K2HorizonMoVAInferenceRequest::new(prompt_token_ids, 8, 1_000, 950, Some(1))
}

fn romeo_and_juliet_prompt_token_ids(model_directory: &Path) -> Vec<u32> {
    let model_config = K2HorizonMoVAConfig::from_json_bytes(
        &fs::read(model_directory.join("config.json"))
            .expect("the installed K2 config should be readable"),
    )
    .expect("the installed K2 config should parse");
    let tokenizer = K2HorizonMoVATokenizer::from_json_bytes(
        &fs::read(model_directory.join("tokenizer.json"))
            .expect("the installed K2 tokenizer should be readable"),
        &model_config,
    )
    .expect("the installed K2 tokenizer should load");
    let prompt_renderer = K2HorizonMoVAPromptRenderer::new();
    let minimum_prompt_token_count = CACHE_BLOCK_TOKEN_COUNT as usize * RESTORED_BLOCK_COUNT + 1;
    let maximum_prompt_token_count = usize::try_from(model_config.max_position_embeddings())
        .expect("the K2 context limit should fit usize")
        .saturating_sub(1);
    assert!(
        maximum_prompt_token_count >= minimum_prompt_token_count,
        "the installed K2 artifact should allow a multi-block cache prefix"
    );

    for source_repetition_count in 1..=8 {
        let user_prompt = format!(
            "Use the supplied Romeo and Juliet source as the only source.\n\n{}",
            ROMEO_AND_JULIET_SOURCE.repeat(source_repetition_count),
        );
        let rendered_prompt = prompt_renderer.render(
            &[ChatMessage::User {
                content: user_prompt,
                images: Vec::new(),
            }],
            &[],
            &ChatToolChoice::None,
        );
        let prompt_token_ids = tokenizer
            .encode_prompt(&rendered_prompt)
            .expect("the Romeo and Juliet prompt should encode");
        if prompt_token_ids.len() >= minimum_prompt_token_count {
            return prompt_token_ids
                .into_iter()
                .take(maximum_prompt_token_count)
                .collect();
        }
    }
    panic!(
        "the Romeo and Juliet fixture should reach {minimum_prompt_token_count} K2 prompt tokens"
    )
}

fn generate_first_token_to_finalization(
    execution: &mut astronomical_model_serving::K2HorizonMoVAInferenceExecution,
    request_id: RequestId,
) -> Option<u32> {
    eprintln!(
        "[k2-prompt-cache-restore] status=generating request_id={}",
        request_id.value()
    );
    let mut first_generated_token_id = None;
    for _advance_attempt in 0..10_000 {
        match execution
            .decode_next_token(request_id)
            .expect("K2 should advance prefill or generation")
        {
            GeneratedToken::PrefillProgress {
                processed_token_count,
                ..
            } => eprintln!(
                "[k2-prompt-cache-restore] status=prefill processed_tokens={processed_token_count}"
            ),
            GeneratedToken::TokenId {
                token_id,
                generation_finalization,
                ..
            } => {
                first_generated_token_id.get_or_insert(token_id);
                if generation_finalization.is_some() {
                    return first_generated_token_id;
                }
            }
            GeneratedToken::EndOfSequence => return first_generated_token_id,
            other => panic!("K2 emitted an unexpected generation event: {other:?}"),
        }
    }
    panic!("the K2 request should reach terminal generation")
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
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
            {
                file_sizes.push(directory_entry.metadata()?.len());
            }
        }
    }
    Ok(file_sizes)
}
