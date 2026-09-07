//! K2-owned persistent prompt-cache open, restore, and capture.
//!
//! The disk store and prefix lookup are family-neutral. This module only maps
//! append-only key/value tensors onto that contract.

use std::collections::HashMap;
use std::sync::Arc;

use astronomical_runtime_integration::MlxRuntime;

use crate::k2_horizon_mova::cache_layout::k2_horizon_mova_decoder_cache_layout;
use crate::k2_horizon_mova::configuration::K2HorizonMoVAConfig;
use crate::k2_horizon_mova::model::K2HorizonMoVAKvState;
use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreConfig, PersistentPromptCacheModelContract,
    PersistentPromptCacheModelContractError, PersistentPromptCachePrefixLookup,
    PersistentPromptCachePrefixLookupResult, PersistentPromptCachePublicationOutcome,
};

/// Opens the SSD store for one loaded K2 Horizon MoVA revision.
pub(super) fn open_prompt_cache_store(
    runtime_active_memory_limit_bytes: usize,
    config: &K2HorizonMoVAConfig,
    model_id: &str,
    model_revision: &str,
    configured_block_token_count: Option<usize>,
    common_prefix_checkpoint_stride_blocks: u32,
    disk_store_config: PersistentPromptCacheDiskStoreConfig,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Arc<PersistentPromptCacheDiskStore>, InferenceEngineError> {
    let decoder_cache_layout =
        k2_horizon_mova_decoder_cache_layout(config).map_err(|layout_error| {
            InferenceEngineError::Fatal {
                reason: format!("K2 Horizon MoVA prompt-cache layout is invalid: {layout_error}"),
            }
        })?;
    let model_contract = resolve_quota_bounded_prompt_cache_contract(
        model_id,
        model_revision,
        decoder_cache_layout,
        config.max_position_embeddings() as usize,
        runtime_active_memory_limit_bytes as u64,
        disk_store_config.global_prompt_cache_maximum_size_bytes(),
        configured_block_token_count,
        common_prefix_checkpoint_stride_blocks.max(1),
    )
    .map_err(|contract_error| InferenceEngineError::Fatal {
        reason: format!("K2 Horizon MoVA prompt-cache contract is invalid: {contract_error}"),
    })?;
    let persistent_prompt_cache = performance_attribution
        .measure_operation(
            PerformanceOperation::PersistentPromptCacheOpenAndScan,
            |_performance_attribution| {
                PersistentPromptCacheDiskStore::open(disk_store_config, model_contract)
            },
        )
        .map_err(|store_error| InferenceEngineError::Fatal {
            reason: format!("K2 Horizon MoVA prompt-cache store could not open: {store_error}"),
        })?;
    Ok(Arc::new(persistent_prompt_cache))
}

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
            Ok(model_contract) => return Ok(model_contract),
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
                |_block_hash| false,
            )
        },
    )
}

pub(super) fn restore_prompt_prefix(
    runtime: &MlxRuntime,
    persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    prompt_token_ids: &[u32],
    restored_token_count: usize,
    caches: &mut [K2HorizonMoVAKvState],
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(Option<PersistentPromptCacheBlockKey>, u32), InferenceEngineError> {
    if restored_token_count == 0 {
        return Ok((None, 0));
    }
    let block_token_count = persistent_prompt_cache.model_contract.block_token_count();
    if block_token_count == 0 {
        return Ok((None, 0));
    }
    let complete_block_count = restored_token_count / block_token_count;
    if complete_block_count == 0 {
        return Ok((None, 0));
    }
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
                reason: format!(
                    "K2 Horizon MoVA prompt-cache sequence block could not load: {store_error}"
                ),
            })?
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason:
                    "K2 Horizon MoVA prompt-cache sequence block was reported present but missing"
                        .to_owned(),
            })?;
        sequence_blocks.push(loaded_sequence_block);
        last_restored_block_key = Some(block_key);
    }
    performance_attribution.measure_operation(
        PerformanceOperation::PersistentPromptCacheStateReconstruction,
        |_performance_attribution| {
            restore_caches_from_sequence_blocks(runtime, caches, &mut sequence_blocks)
        },
    )?;
    Ok((
        last_restored_block_key,
        u32::try_from(restored_token_count).unwrap_or(u32::MAX),
    ))
}

pub(super) fn capture_completed_cache_blocks(
    runtime: &MlxRuntime,
    persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    prompt_token_ids: &[u32],
    caches: &[K2HorizonMoVAKvState],
    absolute_chunk_start: usize,
    absolute_chunk_end: usize,
    last_published_block_key: &mut Option<PersistentPromptCacheBlockKey>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    let block_token_count = persistent_prompt_cache.model_contract.block_token_count();
    if block_token_count == 0 {
        return Ok(());
    }
    let mut block_end = ((absolute_chunk_start / block_token_count) + 1) * block_token_count;
    while block_end <= absolute_chunk_end {
        let block_start = block_end.saturating_sub(block_token_count);
        publish_one_cache_block(
            runtime,
            persistent_prompt_cache,
            prompt_token_ids,
            caches,
            block_start,
            block_end,
            last_published_block_key,
            performance_attribution,
        )?;
        block_end = block_end.saturating_add(block_token_count);
    }
    Ok(())
}

fn publish_one_cache_block(
    runtime: &MlxRuntime,
    persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    prompt_token_ids: &[u32],
    caches: &[K2HorizonMoVAKvState],
    block_start: usize,
    block_end: usize,
    last_published_block_key: &mut Option<PersistentPromptCacheBlockKey>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    let block_tokens = &prompt_token_ids[block_start..block_end];
    let block_key = cache_block_key(
        &persistent_prompt_cache.model_contract,
        block_tokens,
        last_published_block_key.as_ref(),
    )?;
    let sequence_state_tensors = performance_attribution.measure_operation(
        PerformanceOperation::PersistentPromptCacheStateExtraction,
        |_performance_attribution| {
            extract_sequence_block_tensors(runtime, caches, block_start, block_end)
        },
    )?;
    let publication_outcome = persistent_prompt_cache
        .publish_block_with_performance_attribution(
            runtime,
            &block_key,
            last_published_block_key.as_ref(),
            &sequence_state_tensors,
            &HashMap::new(),
            performance_attribution,
        )
        .map_err(|publish_error| InferenceEngineError::InvalidRequest {
            reason: format!(
                "persistent prompt cache failed during required persistent prompt-state capture; the request was stopped: {publish_error}"
            ),
        })?;
    if matches!(
        publication_outcome,
        PersistentPromptCachePublicationOutcome::Published
            | PersistentPromptCachePublicationOutcome::AlreadyPublished
    ) {
        *last_published_block_key = Some(block_key);
    }
    Ok(())
}

fn extract_sequence_block_tensors(
    runtime: &MlxRuntime,
    caches: &[K2HorizonMoVAKvState],
    block_start: usize,
    block_end: usize,
) -> Result<HashMap<String, astronomical_runtime_integration::MlxArray>, InferenceEngineError> {
    let mut sequence_state_tensors = HashMap::new();
    for (layer_index, layer_cache) in caches.iter().enumerate() {
        match layer_cache {
            K2HorizonMoVAKvState::FullPrecision(state) => {
                let keys_state = state.keys_state().ok_or_else(|| {
                    InferenceEngineError::Fatal {
                        reason: format!(
                            "K2 Horizon MoVA prompt-cache capture is missing layer {layer_index} keys"
                        ),
                    }
                })?;
                let values_state = state.values_state().ok_or_else(|| {
                    InferenceEngineError::Fatal {
                        reason: format!(
                            "K2 Horizon MoVA prompt-cache capture is missing layer {layer_index} values"
                        ),
                    }
                })?;
                sequence_state_tensors.insert(
                    format!("layer_{layer_index}_attention.keys"),
                    slice_token_range(runtime, keys_state, block_start, block_end)?,
                );
                sequence_state_tensors.insert(
                    format!("layer_{layer_index}_attention.values"),
                    slice_token_range(runtime, values_state, block_start, block_end)?,
                );
            }
            K2HorizonMoVAKvState::Quantized(state) => {
                // The SSD prompt-cache format stays bfloat16; a quantized
                // state dequantizes the captured token range on write, so the
                // result is already the block and must not be sliced again.
                let keys_state = state
                    .dequantized_token_range(runtime, true, block_start as i32, block_end as i32)
                    .map_err(|dequantize_error| InferenceEngineError::Fatal {
                        reason: format!(
                            "K2 Horizon MoVA prompt-cache capture failed to dequantize keys: {dequantize_error}"
                        ),
                    })?;
                let values_state = state
                    .dequantized_token_range(runtime, false, block_start as i32, block_end as i32)
                    .map_err(|dequantize_error| InferenceEngineError::Fatal {
                        reason: format!(
                            "K2 Horizon MoVA prompt-cache capture failed to dequantize values: {dequantize_error}"
                        ),
                    })?;
                sequence_state_tensors
                    .insert(format!("layer_{layer_index}_attention.keys"), keys_state);
                sequence_state_tensors.insert(
                    format!("layer_{layer_index}_attention.values"),
                    values_state,
                );
            }
        };
    }
    Ok(sequence_state_tensors)
}

fn restore_caches_from_sequence_blocks(
    runtime: &MlxRuntime,
    caches: &mut [K2HorizonMoVAKvState],
    sequence_blocks: &mut [HashMap<String, astronomical_runtime_integration::MlxArray>],
) -> Result<(), InferenceEngineError> {
    for (layer_index, layer_cache) in caches.iter_mut().enumerate() {
        let keys = concatenate_taken_blocks(
            runtime,
            sequence_blocks,
            &format!("layer_{layer_index}_attention.keys"),
        )?;
        let values = concatenate_taken_blocks(
            runtime,
            sequence_blocks,
            &format!("layer_{layer_index}_attention.values"),
        )?;
        match layer_cache {
            K2HorizonMoVAKvState::FullPrecision(state) => {
                state
                    .restore_from_blocks_with_growth_headroom(runtime, keys, values)
                    .map_err(|restore_error| InferenceEngineError::Fatal {
                        reason: format!(
                            "K2 Horizon MoVA restored cache state is invalid: {restore_error}"
                        ),
                    })?;
            }
            K2HorizonMoVAKvState::Quantized(state) => {
                state
                    .restore_from_bf16_blocks_with_growth_headroom(runtime, keys, values)
                    .map_err(|restore_error| InferenceEngineError::Fatal {
                        reason: format!(
                            "K2 Horizon MoVA restored cache state is invalid: {restore_error}"
                        ),
                    })?;
            }
        }
        let (restored_keys, restored_values) = match layer_cache {
            K2HorizonMoVAKvState::FullPrecision(state) => {
                (state.keys_state(), state.values_state())
            }
            K2HorizonMoVAKvState::Quantized(state) => {
                (state.quantized_keys_state(), state.quantized_values_state())
            }
        };
        if let (Some(keys), Some(values)) = (restored_keys, restored_values) {
            runtime.evaluate_arrays(&[keys, values]).map_err(|error| {
                InferenceEngineError::Fatal {
                    reason: format!("K2 Horizon MoVA restored cache evaluation failed: {error}"),
                }
            })?;
        }
    }
    Ok(())
}

fn concatenate_taken_blocks(
    runtime: &MlxRuntime,
    sequence_blocks: &mut [HashMap<String, astronomical_runtime_integration::MlxArray>],
    tensor_name: &str,
) -> Result<astronomical_runtime_integration::MlxArray, InferenceEngineError> {
    let mut owned_arrays = Vec::with_capacity(sequence_blocks.len());
    for sequence_block in sequence_blocks.iter_mut() {
        owned_arrays.push(sequence_block.remove(tensor_name).ok_or_else(|| {
            InferenceEngineError::Fatal {
                reason: format!("K2 Horizon MoVA restored cache is missing {tensor_name}"),
            }
        })?);
    }
    let array_refs = owned_arrays.iter().collect::<Vec<_>>();
    runtime
        .concatenate_axis(&array_refs, 2)
        .map_err(|error| InferenceEngineError::Fatal {
            reason: format!("K2 Horizon MoVA restored cache concatenate failed: {error}"),
        })
}

fn slice_token_range(
    runtime: &MlxRuntime,
    tensor: &astronomical_runtime_integration::MlxArray,
    start_tokens: usize,
    end_tokens: usize,
) -> Result<astronomical_runtime_integration::MlxArray, InferenceEngineError> {
    let shape = tensor.shape();
    if shape.len() != 4 {
        return Err(InferenceEngineError::Fatal {
            reason: "K2 Horizon MoVA cache tensors must have rank four".to_owned(),
        });
    }
    let start = i32::try_from(start_tokens).unwrap_or(i32::MAX);
    let end = i32::try_from(end_tokens).unwrap_or(i32::MAX);
    runtime
        .slice(
            tensor,
            &[0, 0, start, 0],
            &[shape[0], shape[1], end, shape[3]],
            &[1, 1, 1, 1],
        )
        .map_err(|error| InferenceEngineError::Fatal {
            reason: format!("K2 Horizon MoVA cache slice failed: {error}"),
        })
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
