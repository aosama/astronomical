//! Folded-ternary persistent astro SSD prompt cache open, restore, capture, and stats.
//!
//! The disk store, contract, and prefix lookup are family-neutral; this module
//! only maps the shared decoder-state stack onto that contract, addressing
//! every tensor by its decoder-layer index. Full-attention
//! KV spans persist per block, and each published block also carries the
//! convolution and gated-delta boundary snapshot taken at that block's forward
//! boundary, so a restored prefix resumes the linear-attention recurrence
//! exactly where capture stopped. Every chunk that ends on a block boundary is
//! clamped by the caller so one forward produces at most one durable boundary —
//! the snapshot is only valid at a forward edge.

use std::collections::HashMap;
use std::sync::Arc;

use ipc_protocol::WorkerEvent;
use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;

use crate::prism_hadamard_qwen35::engine::execution::{
    PrismHadamardQwen35ActiveGeneration, PrismHadamardQwen35InferenceExecution,
};
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::{
    AstroDecoderStateCacheState, ConvolutionState, FullAttentionKeyValueState,
    GatedDeltaRecurrentState, RequestDecoderStateStack,
};
use crate::{
    AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheDiskStoreConfig,
    AstroSsdPromptCacheModelContract, AstroSsdPromptCachePublicationOutcome, InferenceEngineError,
    PerformanceAttribution, PerformanceOperation, build_astro_ssd_prompt_cache_stats_event,
};
use engine_core::astro_ssd_prompt_cache::AstroSsdPromptCacheCounters;
use engine_core::astro_ssd_prompt_cache::prefix_lookup::AstroSsdPromptCachePrefixLookup;
use engine_core::astro_ssd_prompt_cache::quota_bounded_contract::resolve_quota_bounded_astro_ssd_prompt_cache_contract;

#[allow(clippy::too_many_arguments)]
/// Opens the SSD store for one loaded folded-ternary revision.
pub(super) fn open_astro_ssd_prompt_cache_store(
    effective_mlx_memory_ceiling_bytes: usize,
    model: &PrismHadamardTextModel,
    model_id: &str,
    model_revision: &str,
    configured_block_token_count: Option<usize>,
    common_prefix_checkpoint_stride_blocks: u32,
    disk_store_config: AstroSsdPromptCacheDiskStoreConfig,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Arc<AstroSsdPromptCacheDiskStore>, InferenceEngineError> {
    let model_contract = resolve_quota_bounded_astro_ssd_prompt_cache_contract(
        model_id,
        model_revision,
        model.astro_decoder_state_cache_layout.clone(),
        model.pack.config().text().max_position_embeddings as usize,
        effective_mlx_memory_ceiling_bytes as u64,
        disk_store_config.global_astro_ssd_prompt_cache_maximum_size_bytes(),
        configured_block_token_count,
        common_prefix_checkpoint_stride_blocks.max(1),
    )
    .map_err(|contract_error| InferenceEngineError::Fatal {
        reason: format!(
            "folded-ternary astro-ssd-prompt-cache contract is invalid: {contract_error}"
        ),
    })?;
    let astro_ssd_prompt_cache = performance_attribution
        .measure_operation(
            PerformanceOperation::AstroSsdPromptCacheOpenAndScan,
            |_performance_attribution| {
                AstroSsdPromptCacheDiskStore::open(disk_store_config, model_contract)
            },
        )
        .map_err(|store_error| InferenceEngineError::Fatal {
            reason: format!(
                "folded-ternary astro-ssd-prompt-cache store could not open: {store_error}"
            ),
        })?;
    Ok(Arc::new(astro_ssd_prompt_cache))
}

/// The generation-start restore outcome.
pub(super) struct PrismAstroSsdPromptCacheRestoreOutcome {
    pub(super) restored_token_count: u32,
    pub(super) last_restored_astro_ssd_prompt_cache_block_key: Option<AstroSsdPromptCacheBlockKey>,
}

/// Looks the prompt up in the SSD store, restores the matched complete-block
/// prefix into the fresh `request_decoder_state`, and reports the restored
/// cursor the caller's next position starts from.
///
/// The terminal block is dropped when it would consume the whole prompt: the
/// final forward must process at least one prompt token to produce the first
/// logits, exactly like every family that restores complete blocks only.
pub(super) fn lookup_and_restore_prompt_prefix(
    model: &PrismHadamardTextModel,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    prompt_token_ids: &[u32],
    request_decoder_state: &mut RequestDecoderStateStack,
    astro_ssd_prompt_cache_counters: &mut AstroSsdPromptCacheCounters,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<PrismAstroSsdPromptCacheRestoreOutcome, InferenceEngineError> {
    let model_contract = astro_ssd_prompt_cache.model_contract_ref();
    let block_token_count = model_contract.block_token_count();
    let lookup_result = performance_attribution.measure_operation(
        PerformanceOperation::AstroSsdPromptCachePrefixLookup,
        |_performance_attribution| {
            AstroSsdPromptCachePrefixLookup::for_prompt(
                model_contract,
                prompt_token_ids,
                |block_hash| astro_ssd_prompt_cache.has_kv_block(block_hash),
                |block_hash| astro_ssd_prompt_cache.has_recurrent_snapshot(block_hash),
            )
        },
    );
    let mut restored_token_count = lookup_result.restored_token_count();
    if restored_token_count >= prompt_token_ids.len() && block_token_count > 0 {
        restored_token_count = restored_token_count.saturating_sub(block_token_count);
    }
    if restored_token_count == 0 {
        astro_ssd_prompt_cache_counters.record_astro_ssd_prompt_cache_miss();
        return Ok(PrismAstroSsdPromptCacheRestoreOutcome {
            restored_token_count: 0,
            last_restored_astro_ssd_prompt_cache_block_key: None,
        });
    }
    let LoadedAstroSsdPromptCacheBlocks {
        loaded_block_tensors,
        last_restored_block_key,
    } = load_complete_kv_blocks(
        model,
        astro_ssd_prompt_cache,
        model_contract,
        prompt_token_ids,
        restored_token_count,
        block_token_count,
        performance_attribution,
    )?;
    restore_state_and_absorb_boundary_snapshot(
        model,
        astro_ssd_prompt_cache,
        loaded_block_tensors,
        last_restored_block_key.as_ref(),
        restored_token_count,
        request_decoder_state,
        performance_attribution,
    )?;
    astro_ssd_prompt_cache_counters.record_astro_ssd_prompt_cache_hit(restored_token_count);
    Ok(PrismAstroSsdPromptCacheRestoreOutcome {
        restored_token_count: u32::try_from(restored_token_count).unwrap_or(u32::MAX),
        last_restored_astro_ssd_prompt_cache_block_key: last_restored_block_key,
    })
}

/// Seats every full-attention slab from the complete block set with one
/// concatenation per tensor, then validates the restored length.
fn restore_full_attention_layers_from_blocks(
    model: &PrismHadamardTextModel,
    loaded_block_tensors: &[HashMap<String, MlxArray>],
    restored_token_count: usize,
    request_decoder_state: &mut RequestDecoderStateStack,
) -> Result<(), InferenceEngineError> {
    for (decoder_layer_index, decoder_layer_state) in
        request_decoder_state.layers_mut().iter_mut().enumerate()
    {
        if let AstroDecoderStateCacheState::AppendOnlyAttention { attention } = decoder_layer_state
        {
            restore_full_attention_layer(
                &model.runtime,
                decoder_layer_index,
                loaded_block_tensors,
                restored_token_count,
                attention,
            )?;
        }
    }
    Ok(())
}

/// Concatenates one full-attention layer's restored key and value spans and
/// installs them as the layer's slabs.
fn restore_full_attention_layer(
    runtime: &MlxRuntime,
    decoder_layer_index: usize,
    loaded_block_tensors: &[HashMap<String, MlxArray>],
    restored_token_count: usize,
    key_value_state: &mut FullAttentionKeyValueState,
) -> Result<(), InferenceEngineError> {
    let keys_tensor_name = format!("layer_{decoder_layer_index}_attention.keys");
    let values_tensor_name = format!("layer_{decoder_layer_index}_attention.values");
    let keys_slices = block_tensors_by_name(loaded_block_tensors, &keys_tensor_name)?;
    let values_slices = block_tensors_by_name(loaded_block_tensors, &values_tensor_name)?;
    let restored_keys =
        runtime
            .concatenate_axis(&keys_slices, 2)
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary restored {keys_tensor_name} concat failed: {error}"
                ),
            })?;
    let restored_values = runtime
        .concatenate_axis(&values_slices, 2)
        .map_err(|error| InferenceEngineError::Fatal {
            reason: format!("folded-ternary restored {values_tensor_name} concat failed: {error}"),
        })?;
    let restored_key_token_count = restored_keys.shape()[2];
    if restored_key_token_count != restored_token_count as i32 {
        return Err(InferenceEngineError::Fatal {
            reason: format!(
                "folded-ternary restored {keys_tensor_name} covers {restored_key_token_count} tokens but the prefix is {restored_token_count}"
            ),
        });
    }
    key_value_state
        .restore_from_blocks(restored_keys, restored_values)
        .map_err(|error| InferenceEngineError::Fatal {
            reason: format!("folded-ternary restored full-attention state is invalid: {error}"),
        })
}

/// Routes the boundary snapshot's convolution and gated-delta tensors into
/// every linear-attention layer's state.
fn absorb_boundary_snapshot(
    snapshot_tensors: &HashMap<String, MlxArray>,
    request_decoder_state: &mut RequestDecoderStateStack,
) -> Result<(), InferenceEngineError> {
    for (decoder_layer_index, decoder_layer_state) in
        request_decoder_state.layers_mut().iter_mut().enumerate()
    {
        if let AstroDecoderStateCacheState::Composite {
            convolution,
            recurrent,
        } = decoder_layer_state
        {
            absorb_linear_layer_snapshot(
                decoder_layer_index,
                snapshot_tensors,
                convolution,
                recurrent,
            )?;
        }
    }
    Ok(())
}

/// Installs one linear-attention layer's restored convolution buffer, whose
/// shape is validated against the layer's configured geometry, and its
/// gated-delta recurrent accumulator.
fn absorb_linear_layer_snapshot(
    decoder_layer_index: usize,
    snapshot_tensors: &HashMap<String, MlxArray>,
    convolution_state: &mut ConvolutionState,
    recurrent_state: &mut GatedDeltaRecurrentState,
) -> Result<(), InferenceEngineError> {
    let convolution_tensor_name = format!("layer_{decoder_layer_index}_linear.convolution");
    let convolution_tensor = snapshot_tensors
        .get(&convolution_tensor_name)
        .ok_or_else(|| missing_snapshot_tensor(&convolution_tensor_name))?;
    convolution_state
        .replace_state(convolution_tensor.retain().map_err(|error| {
            InferenceEngineError::Fatal {
                reason: format!("folded-ternary snapshot retain failed: {error}"),
            }
        })?)
        .map_err(|error| InferenceEngineError::Fatal {
            reason: format!("folded-ternary restored convolution state is invalid: {error}"),
        })?;
    let recurrent_tensor_name = format!("layer_{decoder_layer_index}_linear.gated_delta_recurrent");
    let recurrent_tensor = snapshot_tensors
        .get(&recurrent_tensor_name)
        .ok_or_else(|| missing_snapshot_tensor(&recurrent_tensor_name))?;
    recurrent_state.restore_from_snapshot(recurrent_tensor.retain().map_err(|error| {
        InferenceEngineError::Fatal {
            reason: format!("folded-ternary snapshot retain failed: {error}"),
        }
    })?);
    Ok(())
}

fn missing_snapshot_tensor(tensor_name: &str) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: format!(
            "folded-ternary astro-ssd-prompt-cache boundary snapshot is missing {tensor_name}"
        ),
    }
}

/// Every live array a restored prefix must materialize before the next
/// forward: the KV slabs plus the convolution and recurrent accumulators.
/// The roots keep their per-kind order (every key/value pair first, then each
/// linear layer's convolution and recurrent pair), the order the restored
/// evaluation always used.
fn restored_evaluation_arrays(request_decoder_state: &RequestDecoderStateStack) -> Vec<&MlxArray> {
    let decoder_layer_states = request_decoder_state.layers();
    let mut restored_arrays = Vec::with_capacity(2 * decoder_layer_states.len());
    for decoder_layer_state in decoder_layer_states {
        if let AstroDecoderStateCacheState::AppendOnlyAttention { attention } = decoder_layer_state
            && let (Some(keys), Some(values)) = (attention.keys_state(), attention.values_state())
        {
            restored_arrays.push(keys);
            restored_arrays.push(values);
        }
    }
    for decoder_layer_state in decoder_layer_states {
        if let AstroDecoderStateCacheState::Composite {
            convolution,
            recurrent,
        } = decoder_layer_state
        {
            restored_arrays.extend(convolution.state());
            restored_arrays.extend(recurrent.state());
        }
    }
    restored_arrays
}

fn block_tensors_by_name<'tensors>(
    loaded_block_tensors: &'tensors [HashMap<String, MlxArray>],
    tensor_name: &str,
) -> Result<Vec<&'tensors MlxArray>, InferenceEngineError> {
    loaded_block_tensors
        .iter()
        .map(|block_tensors| {
            block_tensors
                .get(tensor_name)
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: format!(
                        "folded-ternary astro-ssd-prompt-cache restored block is missing {tensor_name}"
                    ),
                })
        })
        .collect()
}

/// Publishes the completed block whose boundary the just-finished forward
/// landed on: the KV span plus the boundary snapshot taken at that forward.
///
/// Capture failures stop the request (typed `InvalidRequest`), matching the
/// required-persistence policy every family with mandatory capture follows.
pub(super) fn capture_completed_astro_ssd_prompt_cache_block(
    runtime: &MlxRuntime,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    let block_token_count = astro_ssd_prompt_cache
        .model_contract_ref()
        .block_token_count();
    let block_end = active_generation.processed_prompt_token_count as usize;
    if block_end == 0 || !block_end.is_multiple_of(block_token_count) {
        return Ok(());
    }
    let block_start = block_end - block_token_count;
    let model_contract = astro_ssd_prompt_cache.model_contract_ref();
    let block_tokens = active_generation.prompt_token_ids[block_start..block_end].to_vec();
    let block_key = chain_block_key(
        model_contract,
        &block_tokens,
        active_generation.last_published_block_key.as_ref(),
    )?;
    let sequence_state_tensors = performance_attribution.measure_operation(
        PerformanceOperation::AstroSsdPromptCacheStateExtraction,
        |_performance_attribution| {
            extract_sequence_block_tensors(
                runtime,
                &active_generation.request_decoder_state,
                block_start,
                block_end,
            )
        },
    )?;
    let boundary_state_tensors = performance_attribution.measure_operation(
        PerformanceOperation::AstroSsdPromptCacheStateExtraction,
        |_performance_attribution| {
            extract_boundary_snapshot_tensors(&active_generation.request_decoder_state)
        },
    )?;
    let publication_outcome = astro_ssd_prompt_cache
        .publish_block_with_performance_attribution(
            runtime,
            &block_key,
            active_generation.last_published_block_key.as_ref(),
            &sequence_state_tensors,
            &boundary_state_tensors,
            performance_attribution,
        )
        .map_err(|publish_error| InferenceEngineError::InvalidRequest {
            reason: format!(
                "astro SSD prompt cache failed during required persistent prompt-state capture; the request was stopped: {publish_error}"
            ),
        })?;
    if matches!(
        publication_outcome,
        AstroSsdPromptCachePublicationOutcome::Published
            | AstroSsdPromptCachePublicationOutcome::AlreadyPublished
    ) {
        active_generation.last_published_block_key = Some(block_key);
    }
    Ok(())
}

/// Slices one block's KV span out of every full-attention slab.
fn extract_sequence_block_tensors(
    runtime: &MlxRuntime,
    request_decoder_state: &RequestDecoderStateStack,
    block_start: usize,
    block_end: usize,
) -> Result<HashMap<String, MlxArray>, InferenceEngineError> {
    let mut sequence_state_tensors = HashMap::new();
    for (decoder_layer_index, decoder_layer_state) in
        request_decoder_state.layers().iter().enumerate()
    {
        let AstroDecoderStateCacheState::AppendOnlyAttention { attention } = decoder_layer_state
        else {
            continue;
        };
        let (Some(keys_state), Some(values_state)) =
            (attention.keys_state(), attention.values_state())
        else {
            return Err(missing_capture_state(decoder_layer_index, "key/value"));
        };
        sequence_state_tensors.insert(
            format!("layer_{decoder_layer_index}_attention.keys"),
            slice_token_range(runtime, keys_state, block_start, block_end)?,
        );
        sequence_state_tensors.insert(
            format!("layer_{decoder_layer_index}_attention.values"),
            slice_token_range(runtime, values_state, block_start, block_end)?,
        );
    }
    Ok(sequence_state_tensors)
}

/// Retains the convolution and gated-delta accumulators captured at the
/// forward boundary the published block ends on.
fn extract_boundary_snapshot_tensors(
    request_decoder_state: &RequestDecoderStateStack,
) -> Result<HashMap<String, MlxArray>, InferenceEngineError> {
    let mut boundary_state_tensors = HashMap::new();
    for (decoder_layer_index, decoder_layer_state) in
        request_decoder_state.layers().iter().enumerate()
    {
        let AstroDecoderStateCacheState::Composite {
            convolution,
            recurrent,
        } = decoder_layer_state
        else {
            continue;
        };
        let convolution_state = convolution
            .state()
            .ok_or_else(|| missing_capture_state(decoder_layer_index, "convolution"))?;
        boundary_state_tensors.insert(
            format!("layer_{decoder_layer_index}_linear.convolution"),
            retain_captured_tensor(convolution_state)?,
        );
        let recurrent_state = recurrent
            .state()
            .ok_or_else(|| missing_capture_state(decoder_layer_index, "recurrent"))?;
        boundary_state_tensors.insert(
            format!("layer_{decoder_layer_index}_linear.gated_delta_recurrent"),
            retain_captured_tensor(recurrent_state)?,
        );
    }
    Ok(boundary_state_tensors)
}

fn missing_capture_state(decoder_layer_index: usize, state_role: &str) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: format!(
            "folded-ternary astro-ssd-prompt-cache capture is missing decoder layer {decoder_layer_index} {state_role} state"
        ),
    }
}

fn retain_captured_tensor(captured_tensor: &MlxArray) -> Result<MlxArray, InferenceEngineError> {
    captured_tensor
        .retain()
        .map_err(|error| InferenceEngineError::Fatal {
            reason: format!("folded-ternary capture retain failed: {error}"),
        })
}

fn slice_token_range(
    runtime: &MlxRuntime,
    tensor: &MlxArray,
    start_tokens: usize,
    end_tokens: usize,
) -> Result<MlxArray, InferenceEngineError> {
    let tensor_shape = tensor.shape();
    if tensor_shape.len() != 4 {
        return Err(InferenceEngineError::Fatal {
            reason: "folded-ternary astro decoder state cache tensors must have rank four"
                .to_owned(),
        });
    }
    let start = i32::try_from(start_tokens).unwrap_or(i32::MAX);
    let end = i32::try_from(end_tokens).unwrap_or(i32::MAX);
    runtime
        .slice(
            tensor,
            &[0, 0, start, 0],
            &[tensor_shape[0], tensor_shape[1], end, tensor_shape[3]],
            &[1, 1, 1, 1],
        )
        .map_err(|error| InferenceEngineError::Fatal {
            reason: format!("folded-ternary astro decoder state cache slice failed: {error}"),
        })
}

/// The complete KV blocks loaded for one restored prefix, plus the chain key
/// of the newest block — the boundary snapshot rides that block.
struct LoadedAstroSsdPromptCacheBlocks {
    loaded_block_tensors: Vec<HashMap<String, MlxArray>>,
    last_restored_block_key: Option<AstroSsdPromptCacheBlockKey>,
}

/// Walks the restored prefix's complete blocks in chain order, loading each
/// block's tensors under its own measured read. Chaining means each block's
/// key commits to its parent, so a prefix can only restore along the exact
/// chain it was written with.
#[allow(clippy::too_many_arguments)]
fn load_complete_kv_blocks(
    model: &PrismHadamardTextModel,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    model_contract: &AstroSsdPromptCacheModelContract,
    prompt_token_ids: &[u32],
    restored_token_count: usize,
    block_token_count: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<LoadedAstroSsdPromptCacheBlocks, InferenceEngineError> {
    let complete_block_count = restored_token_count / block_token_count;
    let mut loaded_block_tensors: Vec<HashMap<String, MlxArray>> =
        Vec::with_capacity(complete_block_count);
    let mut last_restored_block_key = None;
    for block_index in 0..complete_block_count {
        let block_start = block_index * block_token_count;
        let block_end = block_start + block_token_count;
        let block_key = chain_block_key(
            model_contract,
            &prompt_token_ids[block_start..block_end],
            last_restored_block_key.as_ref(),
        )?;
        let loaded_sequence_block = performance_attribution
            .measure_operation(
                PerformanceOperation::AstroSsdPromptCacheKvBlockRead,
                |attribution| {
                    astro_ssd_prompt_cache.load_kv_block(
                        &model.runtime,
                        &block_key,
                        attribution.positional_file_read_metrics(),
                    )
                },
            )
            .map_err(|store_error| InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary astro-ssd-prompt-cache sequence block could not load: {store_error}"
                ),
            })?
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason:
                    "folded-ternary astro-ssd-prompt-cache sequence block was reported present but missing"
                        .to_owned(),
            })?;
        loaded_block_tensors.push(loaded_sequence_block);
        last_restored_block_key = Some(block_key);
    }
    Ok(LoadedAstroSsdPromptCacheBlocks {
        loaded_block_tensors,
        last_restored_block_key,
    })
}

/// Rebuilds the decoder state from the loaded blocks, then absorbs the
/// boundary snapshot riding the newest block and evaluates the restored
/// arrays. The newest boundary snapshot is mandatory: without it the
/// linear-attention recurrence would restart from zero mid-prefix.
#[allow(clippy::too_many_arguments)]
fn restore_state_and_absorb_boundary_snapshot(
    model: &PrismHadamardTextModel,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    loaded_block_tensors: Vec<HashMap<String, MlxArray>>,
    last_restored_block_key: Option<&AstroSsdPromptCacheBlockKey>,
    restored_token_count: usize,
    request_decoder_state: &mut RequestDecoderStateStack,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    performance_attribution.measure_operation(
        PerformanceOperation::AstroSsdPromptCacheStateReconstruction,
        |_performance_attribution| {
            restore_full_attention_layers_from_blocks(
                model,
                &loaded_block_tensors,
                restored_token_count,
                request_decoder_state,
            )
        },
    )?;
    // Release the block tensors before the snapshot load: the reconstructed
    // state owns evaluated copies, so the source tensors are dead weight.
    drop(loaded_block_tensors);
    // The newest boundary snapshot rides the last complete block; without it
    // the linear-attention recurrence would restart from zero mid-prefix.
    if let Some(snapshot_block_key) = last_restored_block_key {
        let snapshot_tensors = astro_ssd_prompt_cache
            .load_recurrent_snapshot(
                &model.runtime,
                snapshot_block_key,
                performance_attribution.positional_file_read_metrics(),
            )
            .map_err(|store_error| InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary astro-ssd-prompt-cache boundary snapshot could not load: {store_error}"
                ),
            })?
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason:
                    "folded-ternary astro-ssd-prompt-cache boundary snapshot was reported present but missing"
                        .to_owned(),
            })?;
        absorb_boundary_snapshot(&snapshot_tensors, request_decoder_state)?;
        let restored_arrays = restored_evaluation_arrays(request_decoder_state);
        model
            .runtime
            .evaluate_arrays(&restored_arrays)
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary restored astro decoder state cache evaluation failed: {error}"
                ),
            })?;
    }
    Ok(())
}

fn chain_block_key(
    model_contract: &AstroSsdPromptCacheModelContract,
    block_tokens: &[u32],
    parent_block_key: Option<&AstroSsdPromptCacheBlockKey>,
) -> Result<AstroSsdPromptCacheBlockKey, InferenceEngineError> {
    match parent_block_key {
        None => AstroSsdPromptCacheBlockKey::for_root_block(model_contract, block_tokens),
        Some(parent_block_key) => parent_block_key.for_child_block(block_tokens),
    }
    .map_err(|block_key_error| InferenceEngineError::InvalidRequest {
        reason: format!(
            "astro SSD prompt cache failed during required persistent prompt-state capture; the request was stopped: {block_key_error}"
        ),
    })
}

impl PrismHadamardQwen35InferenceExecution {
    /// The persistent astro SSD prompt cache observability event, mirroring every other
    /// family with an open store.
    pub(super) fn astro_ssd_prompt_cache_stats_event(&self) -> Option<WorkerEvent> {
        let astro_ssd_prompt_cache = self.astro_ssd_prompt_cache.as_ref()?;
        let global_astro_ssd_prompt_cache_maximum_size_bytes = self
            .astro_ssd_prompt_cache_disk_store_config
            .as_ref()
            .map(AstroSsdPromptCacheDiskStoreConfig::global_astro_ssd_prompt_cache_maximum_size_bytes)
            .unwrap_or(0);
        Some(build_astro_ssd_prompt_cache_stats_event(
            &self.astro_ssd_prompt_cache_counters,
            u64::try_from(
                astro_ssd_prompt_cache
                    .model_contract_ref()
                    .block_token_count(),
            )
            .unwrap_or(u64::MAX),
            u64::try_from(astro_ssd_prompt_cache.sequence_state_block_count()).unwrap_or(u64::MAX),
            u64::try_from(astro_ssd_prompt_cache.boundary_state_snapshot_count())
                .unwrap_or(u64::MAX),
            u64::try_from(astro_ssd_prompt_cache.visual_embedding_count()).unwrap_or(u64::MAX),
            astro_ssd_prompt_cache.total_size_bytes(),
            astro_ssd_prompt_cache.visual_embedding_total_size_bytes(),
            global_astro_ssd_prompt_cache_maximum_size_bytes,
        ))
    }
}
