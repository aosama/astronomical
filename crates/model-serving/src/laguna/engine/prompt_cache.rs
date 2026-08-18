//! Laguna-owned persistent prompt-cache open, restore, and capture.

use std::collections::HashMap;
use std::sync::Arc;

use astronomical_runtime_integration::MlxRuntime;

use crate::laguna::{
    LagunaDecoderState, LagunaExecutionError, LagunaTargetContract, laguna_decoder_cache_layout,
};
use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreConfig, PersistentPromptCacheDiskStoreError,
    PersistentPromptCacheModelContract, PersistentPromptCacheModelContractError,
    PersistentPromptCachePrefixLookup, PersistentPromptCachePrefixLookupResult,
    PersistentPromptCachePublicationOutcome,
};

/// Separates recoverable allocation pressure from durable cache/publication failures.
pub(super) enum LagunaPromptCacheCaptureError {
    Capacity(LagunaExecutionError),
    Engine(InferenceEngineError),
}

/// Opens the SSD store for one loaded Laguna revision.
pub(super) fn open_prompt_cache_store(
    runtime_active_memory_limit_bytes: usize,
    target_contract: &LagunaTargetContract,
    model_id: &str,
    model_revision: &str,
    configured_block_token_count: Option<usize>,
    common_prefix_checkpoint_stride_blocks: u32,
    disk_store_config: PersistentPromptCacheDiskStoreConfig,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Arc<PersistentPromptCacheDiskStore>, InferenceEngineError> {
    let decoder_cache_layout =
        laguna_decoder_cache_layout(target_contract).map_err(|layout_error| {
            InferenceEngineError::Fatal {
                reason: format!("Laguna prompt-cache layout is invalid: {layout_error}"),
            }
        })?;
    let model_contract = resolve_quota_bounded_prompt_cache_contract(
        model_id,
        model_revision,
        decoder_cache_layout,
        target_contract.model().maximum_position_count() as usize,
        runtime_active_memory_limit_bytes as u64,
        disk_store_config.global_prompt_cache_maximum_size_bytes(),
        configured_block_token_count,
        common_prefix_checkpoint_stride_blocks.max(1),
    )
    .map_err(|contract_error| InferenceEngineError::Fatal {
        reason: format!("Laguna prompt-cache contract is invalid: {contract_error}"),
    })?;
    let persistent_prompt_cache = performance_attribution
        .measure_operation(
            PerformanceOperation::PersistentPromptCacheOpenAndScan,
            |_performance_attribution| {
                PersistentPromptCacheDiskStore::open(disk_store_config, model_contract)
            },
        )
        .map_err(|store_error| InferenceEngineError::Fatal {
            reason: format!("Laguna prompt-cache store could not open: {store_error}"),
        })?;
    Ok(Arc::new(persistent_prompt_cache))
}

#[allow(clippy::too_many_arguments)]
fn resolve_quota_bounded_prompt_cache_contract(
    model_id: &str,
    model_revision: &str,
    decoder_cache_layout: crate::DecoderCacheLayout,
    model_maximum_context_token_count: usize,
    effective_mlx_memory_ceiling_bytes: u64,
    global_ssd_quota_bytes: u64,
    configured_block_token_count: Option<usize>,
    common_prefix_checkpoint_stride_blocks: u32,
) -> Result<PersistentPromptCacheModelContract, PersistentPromptCacheModelContractError> {
    let minimum_cacheable_context_token_count = configured_block_token_count.unwrap_or(1).max(1);
    let mut cacheable_context_token_count = model_maximum_context_token_count;
    loop {
        match PersistentPromptCacheModelContract::resolve(
            model_id.to_owned(),
            model_revision.to_owned(),
            decoder_cache_layout.clone(),
            cacheable_context_token_count,
            effective_mlx_memory_ceiling_bytes,
            global_ssd_quota_bytes,
            configured_block_token_count,
            common_prefix_checkpoint_stride_blocks,
        ) {
            Ok(model_contract) => {
                if cacheable_context_token_count < model_maximum_context_token_count {
                    tracing::info!(
                        model_maximum_context_token_count,
                        cacheable_context_token_count,
                        global_ssd_quota_bytes,
                        "Laguna bounded persistent prompt-cache context to the available SSD quota"
                    );
                }
                return Ok(model_contract);
            }
            Err(
                quota_error @ (PersistentPromptCacheModelContractError::ConfiguredBlockChainExceedsSsdQuota { .. }
                | PersistentPromptCacheModelContractError::BlockFilesExceedSsdQuota { .. }),
            ) => {
                let reduced_context_token_count = (cacheable_context_token_count / 2)
                    .max(minimum_cacheable_context_token_count);
                if reduced_context_token_count == cacheable_context_token_count {
                    return Err(quota_error);
                }
                cacheable_context_token_count = reduced_context_token_count;
            }
            Err(contract_error) => return Err(contract_error),
        }
    }
}

/// Finds the longest usable prefix without creating any MLX array owners.
pub(super) fn lookup_prompt_prefix(
    persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    prompt_token_ids: &[u32],
    performance_attribution: &mut PerformanceAttribution,
) -> PersistentPromptCachePrefixLookupResult {
    performance_attribution.measure_operation(
        PerformanceOperation::PersistentPromptCachePrefixLookup,
        |_performance_attribution| {
            PersistentPromptCachePrefixLookup::for_prompt(
                &persistent_prompt_cache.model_contract,
                prompt_token_ids,
                |block_hash| persistent_prompt_cache.has_kv_block(block_hash),
                |block_hash| persistent_prompt_cache.has_recurrent_snapshot(block_hash),
            )
        },
    )
}

/// Restores a previously admitted prefix and returns its last block key and token count.
pub(super) fn restore_prompt_prefix(
    runtime: &MlxRuntime,
    persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    prompt_token_ids: &[u32],
    lookup_result: &PersistentPromptCachePrefixLookupResult,
    decoder_state: &mut LagunaDecoderState,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(Option<PersistentPromptCacheBlockKey>, u32), InferenceEngineError> {
    let restored_token_count = lookup_result.restored_token_count();
    if restored_token_count == 0 {
        return Ok((None, 0));
    }
    let block_token_count = persistent_prompt_cache.model_contract.block_token_count();
    let complete_block_count = restored_token_count / block_token_count;
    let mut sequence_blocks = Vec::with_capacity(complete_block_count);
    let mut last_restored_block_key = None;
    for block_index in 0..complete_block_count {
        let block_start = block_index * block_token_count;
        let block_end = block_start + block_token_count;
        let block_key = cache_block_key(
            &persistent_prompt_cache.model_contract,
            &prompt_token_ids[block_start..block_end],
            last_restored_block_key.as_ref(),
        )?;
        let loaded_sequence_block = performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheKvBlockRead,
                |attribution| {
                    persistent_prompt_cache.load_kv_block(
                        runtime,
                        &block_key,
                        attribution.positional_file_read_metrics(),
                    )
                },
            )
            .map_err(|store_error| InferenceEngineError::Fatal {
                reason: format!("Laguna prompt-cache sequence block could not load: {store_error}"),
            })?
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "Laguna prompt-cache sequence block was reported present but missing"
                    .to_owned(),
            })?;
        sequence_blocks.push(loaded_sequence_block);
        last_restored_block_key = Some(block_key);
    }
    let snapshot_key =
        last_restored_block_key
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "Laguna prompt-cache restore lost the snapshot key".to_owned(),
            })?;
    let mut boundary_snapshot = if persistent_prompt_cache
        .model_contract
        .decoder_cache_layout()
        .has_boundary_state()
    {
        performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheRecurrentSnapshotRead,
                |attribution| {
                    persistent_prompt_cache.load_recurrent_snapshot(
                        runtime,
                        snapshot_key,
                        attribution.positional_file_read_metrics(),
                    )
                },
            )
            .map_err(|store_error| InferenceEngineError::Fatal {
                reason: format!(
                    "Laguna prompt-cache boundary snapshot could not load: {store_error}"
                ),
            })?
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "Laguna prompt-cache boundary snapshot was reported present but missing"
                    .to_owned(),
            })?
    } else {
        HashMap::new()
    };
    performance_attribution
        .measure_operation(
            PerformanceOperation::PersistentPromptCacheStateReconstruction,
            |_performance_attribution| {
                decoder_state.restore_from_cache_blocks(
                    runtime,
                    &mut sequence_blocks,
                    &mut boundary_snapshot,
                )
            },
        )
        .map_err(|restore_error| InferenceEngineError::Fatal {
            reason: format!("Laguna prompt-cache restore failed: {restore_error:?}"),
        })?;
    drop(sequence_blocks);
    Ok((
        last_restored_block_key,
        u32::try_from(restored_token_count).unwrap_or(u32::MAX),
    ))
}

/// Publishes every complete cache block crossed by a successful prefill forward.
pub(super) fn capture_completed_cache_blocks(
    runtime: &MlxRuntime,
    persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    prompt_token_ids: &[u32],
    decoder_state: &LagunaDecoderState,
    absolute_chunk_start: usize,
    absolute_chunk_end: usize,
    last_published_block_key: &mut Option<PersistentPromptCacheBlockKey>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), LagunaPromptCacheCaptureError> {
    let block_token_count = persistent_prompt_cache.model_contract.block_token_count();
    if block_token_count == 0 || !absolute_chunk_end.is_multiple_of(block_token_count) {
        return Ok(());
    }
    let block_start = absolute_chunk_end.saturating_sub(block_token_count);
    if block_start < absolute_chunk_start {
        return Ok(());
    }
    let block_tokens = &prompt_token_ids[block_start..absolute_chunk_end];
    let block_key = cache_block_key(
        &persistent_prompt_cache.model_contract,
        block_tokens,
        last_published_block_key.as_ref(),
    )
    .map_err(LagunaPromptCacheCaptureError::Engine)?;
    let (sequence_state_tensors, boundary_state_tensors) = performance_attribution
        .measure_operation(
            PerformanceOperation::PersistentPromptCacheStateExtraction,
            |_performance_attribution| {
                decoder_state.extract_cache_block_tensors(runtime, block_start, absolute_chunk_end)
            },
        )
        .map_err(|extract_error| {
            if extract_error.is_recoverable_memory_pressure() {
                LagunaPromptCacheCaptureError::Capacity(extract_error)
            } else {
                LagunaPromptCacheCaptureError::Engine(InferenceEngineError::InvalidRequest {
                    reason: format!(
                        "persistent prompt cache failed during required persistent prompt-state capture; the request was stopped: {extract_error:?}"
                    ),
                })
            }
        })?;
    let publication_outcome = persistent_prompt_cache
        .publish_block_with_performance_attribution(
            runtime,
            &block_key,
            last_published_block_key.as_ref(),
            &sequence_state_tensors,
            &boundary_state_tensors,
            performance_attribution,
        )
        .map_err(prompt_cache_publication_error)?;
    if matches!(
        publication_outcome,
        PersistentPromptCachePublicationOutcome::Published
            | PersistentPromptCachePublicationOutcome::AlreadyPublished
    ) {
        *last_published_block_key = Some(block_key);
    }
    Ok(())
}

fn prompt_cache_publication_error(
    publish_error: PersistentPromptCacheDiskStoreError,
) -> LagunaPromptCacheCaptureError {
    match publish_error {
        PersistentPromptCacheDiskStoreError::SaveSafetensors { source }
            if matches!(
                &source,
                astronomical_runtime_integration::MlxRuntimeError::ActiveMemoryLimitExceeded { .. }
            ) || source.is_recoverable_graphics_processor_out_of_memory() =>
        {
            LagunaPromptCacheCaptureError::Capacity(LagunaExecutionError::Runtime(source))
        }
        publish_error => {
            LagunaPromptCacheCaptureError::Engine(InferenceEngineError::InvalidRequest {
                reason: format!(
                    "persistent prompt cache failed during required persistent prompt-state capture; the request was stopped: {publish_error}"
                ),
            })
        }
    }
}

fn cache_block_key(
    model_contract: &PersistentPromptCacheModelContract,
    block_tokens: &[u32],
    parent_block_key: Option<&PersistentPromptCacheBlockKey>,
) -> Result<PersistentPromptCacheBlockKey, InferenceEngineError> {
    match parent_block_key {
        None => PersistentPromptCacheBlockKey::for_root_block(model_contract, block_tokens),
        Some(parent_block_key) => parent_block_key.for_child_block(block_tokens),
    }
    .map_err(|block_key_error| InferenceEngineError::InvalidRequest {
        reason: format!(
            "persistent prompt cache failed during required persistent prompt-state capture; the request was stopped: {block_key_error}"
        ),
    })
}
