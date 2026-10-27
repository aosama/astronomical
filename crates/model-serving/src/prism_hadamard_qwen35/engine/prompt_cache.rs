//! Folded-ternary persistent prompt-cache open, restore, capture, and stats.
//!
//! The disk store, contract, and prefix lookup are family-neutral; this module
//! only maps the straight generation state onto that contract. Full-attention
//! KV spans persist per block, and each published block also carries the
//! convolution and gated-delta boundary snapshot taken at that block's forward
//! boundary, so a restored prefix resumes the linear-attention recurrence
//! exactly where capture stopped. Every chunk that ends on a block boundary is
//! clamped by the caller so one forward produces at most one durable boundary —
//! the snapshot is only valid at a forward edge.

use std::collections::HashMap;
use std::sync::Arc;

use astronomical_ipc_protocol::WorkerEvent;
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::MlxArray;

use crate::persistent_cache::PersistentPromptCacheCounters;
use crate::persistent_cache::prefix_lookup::PersistentPromptCachePrefixLookup;
use crate::persistent_cache::quota_bounded_contract::resolve_quota_bounded_prompt_cache_contract;
use crate::prism_hadamard_qwen35::cache_layout;
use crate::prism_hadamard_qwen35::engine::execution::{
    PrismHadamardQwen35ActiveGeneration, PrismHadamardQwen35InferenceExecution,
};
use crate::prism_hadamard_qwen35::model::{
    PrismHadamardTextModel, PrismStateSlotLayout, StraightGenerationState,
};
use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreConfig, PersistentPromptCacheModelContract,
    PersistentPromptCachePublicationOutcome, build_persistent_prompt_cache_stats_event,
};

#[allow(clippy::too_many_arguments)]
/// Opens the SSD store for one loaded folded-ternary revision.
pub(super) fn open_prompt_cache_store(
    effective_mlx_memory_ceiling_bytes: usize,
    model: &PrismHadamardTextModel,
    model_id: &str,
    model_revision: &str,
    configured_block_token_count: Option<usize>,
    common_prefix_checkpoint_stride_blocks: u32,
    disk_store_config: PersistentPromptCacheDiskStoreConfig,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Arc<PersistentPromptCacheDiskStore>, InferenceEngineError> {
    let decoder_cache_layout = cache_layout::prism_hadamard_qwen35_decoder_cache_layout(
        model.pack.config(),
    )
    .map_err(|layout_error| InferenceEngineError::Fatal {
        reason: format!("folded-ternary prompt-cache layout is invalid: {layout_error}"),
    })?;
    let model_contract = resolve_quota_bounded_prompt_cache_contract(
        model_id,
        model_revision,
        decoder_cache_layout,
        model.pack.config().text().max_position_embeddings as usize,
        effective_mlx_memory_ceiling_bytes as u64,
        disk_store_config.global_prompt_cache_maximum_size_bytes(),
        configured_block_token_count,
        common_prefix_checkpoint_stride_blocks.max(1),
    )
    .map_err(|contract_error| InferenceEngineError::Fatal {
        reason: format!("folded-ternary prompt-cache contract is invalid: {contract_error}"),
    })?;
    let persistent_prompt_cache = performance_attribution
        .measure_operation(
            PerformanceOperation::PersistentPromptCacheOpenAndScan,
            |_performance_attribution| {
                PersistentPromptCacheDiskStore::open(disk_store_config, model_contract)
            },
        )
        .map_err(|store_error| InferenceEngineError::Fatal {
            reason: format!("folded-ternary prompt-cache store could not open: {store_error}"),
        })?;
    Ok(Arc::new(persistent_prompt_cache))
}

/// The generation-start restore outcome.
pub(super) struct PrismPromptCacheRestoreOutcome {
    pub(super) restored_token_count: u32,
    pub(super) last_restored_persistent_prompt_cache_block_key:
        Option<PersistentPromptCacheBlockKey>,
}

/// Looks the prompt up in the SSD store, restores the matched complete-block
/// prefix into `generation_state`, and reports the restored cursor.
///
/// The terminal block is dropped when it would consume the whole prompt: the
/// final forward must process at least one prompt token to produce the first
/// logits, exactly like every family that restores complete blocks only.
pub(super) fn lookup_and_restore_prompt_prefix(
    model: &PrismHadamardTextModel,
    persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    prompt_token_ids: &[u32],
    generation_state: &mut StraightGenerationState,
    persistent_prompt_cache_counters: &mut PersistentPromptCacheCounters,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<PrismPromptCacheRestoreOutcome, InferenceEngineError> {
    let model_contract = persistent_prompt_cache.model_contract_ref();
    let block_token_count = model_contract.block_token_count();
    let lookup_result = performance_attribution.measure_operation(
        PerformanceOperation::PersistentPromptCachePrefixLookup,
        |_performance_attribution| {
            PersistentPromptCachePrefixLookup::for_prompt(
                model_contract,
                prompt_token_ids,
                |block_hash| persistent_prompt_cache.has_kv_block(block_hash),
                |block_hash| persistent_prompt_cache.has_recurrent_snapshot(block_hash),
            )
        },
    );
    let mut restored_token_count = lookup_result.restored_token_count();
    if restored_token_count >= prompt_token_ids.len() && block_token_count > 0 {
        restored_token_count = restored_token_count.saturating_sub(block_token_count);
    }
    if restored_token_count == 0 {
        persistent_prompt_cache_counters.record_cache_miss();
        return Ok(PrismPromptCacheRestoreOutcome {
            restored_token_count: 0,
            last_restored_persistent_prompt_cache_block_key: None,
        });
    }
    *generation_state = StraightGenerationState::restored_prefix_slot_owners(model);
    let slot_layout = PrismStateSlotLayout::for_model(model);
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
                PerformanceOperation::PersistentPromptCacheKvBlockRead,
                |attribution| {
                    persistent_prompt_cache.load_kv_block(
                        &model.runtime,
                        &block_key,
                        attribution.positional_file_read_metrics(),
                    )
                },
            )
            .map_err(|store_error| InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary prompt-cache sequence block could not load: {store_error}"
                ),
            })?
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason:
                    "folded-ternary prompt-cache sequence block was reported present but missing"
                        .to_owned(),
            })?;
        loaded_block_tensors.push(loaded_sequence_block);
        last_restored_block_key = Some(block_key);
    }
    performance_attribution.measure_operation(
        PerformanceOperation::PersistentPromptCacheStateReconstruction,
        |_performance_attribution| {
            restore_generation_state_from_blocks(
                model,
                &slot_layout,
                &loaded_block_tensors,
                restored_token_count,
                generation_state,
            )
        },
    )?;
    drop(loaded_block_tensors);
    // The newest boundary snapshot rides the last complete block; without it
    // the linear-attention recurrence would restart from zero mid-prefix.
    if let Some(snapshot_block_key) = last_restored_block_key.as_ref() {
        let snapshot_tensors = persistent_prompt_cache
            .load_recurrent_snapshot(
                &model.runtime,
                snapshot_block_key,
                performance_attribution.positional_file_read_metrics(),
            )
            .map_err(|store_error| InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary prompt-cache boundary snapshot could not load: {store_error}"
                ),
            })?
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason:
                    "folded-ternary prompt-cache boundary snapshot was reported present but missing"
                        .to_owned(),
            })?;
        absorb_boundary_snapshot(&slot_layout, &snapshot_tensors, generation_state)?;
        let restored_arrays = restored_evaluation_arrays(generation_state, &slot_layout);
        model
            .runtime
            .evaluate_arrays(&restored_arrays)
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!("folded-ternary restored cache evaluation failed: {error}"),
            })?;
    }
    generation_state.set_sequence_position_offset(restored_token_count as i32);
    persistent_prompt_cache_counters.record_cache_hit(restored_token_count);
    Ok(PrismPromptCacheRestoreOutcome {
        restored_token_count: u32::try_from(restored_token_count).unwrap_or(u32::MAX),
        last_restored_persistent_prompt_cache_block_key: last_restored_block_key,
    })
}

/// Seats every full-attention slab from the complete block set with one
/// concatenation per tensor, then validates the restored length.
fn restore_generation_state_from_blocks(
    model: &PrismHadamardTextModel,
    slot_layout: &PrismStateSlotLayout,
    loaded_block_tensors: &[HashMap<String, MlxArray>],
    restored_token_count: usize,
    generation_state: &mut StraightGenerationState,
) -> Result<(), InferenceEngineError> {
    let runtime = &model.runtime;
    for (slot, decoder_layer_index) in slot_layout
        .full_attention_decoder_layer_indices
        .iter()
        .enumerate()
    {
        let keys_tensor_name = format!("layer_{decoder_layer_index}_attention.keys");
        let values_tensor_name = format!("layer_{decoder_layer_index}_attention.values");
        let keys_slices = block_tensors_by_name(loaded_block_tensors, &keys_tensor_name)?;
        let values_slices = block_tensors_by_name(loaded_block_tensors, &values_tensor_name)?;
        let restored_keys = runtime.concatenate_axis(&keys_slices, 2).map_err(|error| {
            InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary restored {keys_tensor_name} concat failed: {error}"
                ),
            }
        })?;
        let restored_values = runtime
            .concatenate_axis(&values_slices, 2)
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary restored {values_tensor_name} concat failed: {error}"
                ),
            })?;
        let restored_key_token_count = restored_keys.shape()[2];
        if restored_key_token_count != restored_token_count as i32 {
            return Err(InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary restored {keys_tensor_name} covers {restored_key_token_count} tokens but the prefix is {restored_token_count}"
                ),
            });
        }
        generation_state
            .restore_key_value_slabs(slot, restored_keys, restored_values)
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!("folded-ternary restored full-attention state is invalid: {error}"),
            })?;
    }
    Ok(())
}

/// Routes the boundary snapshot's convolution and gated-delta tensors into
/// their per-kind slots.
fn absorb_boundary_snapshot(
    slot_layout: &PrismStateSlotLayout,
    snapshot_tensors: &HashMap<String, MlxArray>,
    generation_state: &mut StraightGenerationState,
) -> Result<(), InferenceEngineError> {
    for (slot, decoder_layer_index) in slot_layout
        .linear_attention_decoder_layer_indices
        .iter()
        .enumerate()
    {
        let convolution_tensor_name = format!("layer_{decoder_layer_index}_linear.convolution");
        let convolution_tensor = snapshot_tensors
            .get(&convolution_tensor_name)
            .ok_or_else(|| missing_snapshot_tensor(&convolution_tensor_name))?;
        generation_state
            .restore_convolution_state(
                slot,
                convolution_tensor
                    .retain()
                    .map_err(|error| InferenceEngineError::Fatal {
                        reason: format!("folded-ternary snapshot retain failed: {error}"),
                    })?,
            )
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!("folded-ternary restored convolution state is invalid: {error}"),
            })?;
        let recurrent_tensor_name =
            format!("layer_{decoder_layer_index}_linear.gated_delta_recurrent");
        let recurrent_tensor = snapshot_tensors
            .get(&recurrent_tensor_name)
            .ok_or_else(|| missing_snapshot_tensor(&recurrent_tensor_name))?;
        generation_state.restore_recurrent_state(
            slot,
            recurrent_tensor
                .retain()
                .map_err(|error| InferenceEngineError::Fatal {
                    reason: format!("folded-ternary snapshot retain failed: {error}"),
                })?,
        );
    }
    Ok(())
}

fn missing_snapshot_tensor(tensor_name: &str) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: format!("folded-ternary prompt-cache boundary snapshot is missing {tensor_name}"),
    }
}

/// Every live array a restored prefix must materialize before the next
/// forward: the KV slabs plus the convolution and recurrent accumulators.
fn restored_evaluation_arrays<'state>(
    generation_state: &'state StraightGenerationState,
    slot_layout: &PrismStateSlotLayout,
) -> Vec<&'state MlxArray> {
    let mut restored_arrays = Vec::new();
    for slot in 0..slot_layout.full_attention_decoder_layer_indices.len() {
        if let Some((keys, values)) = generation_state.key_value_slabs(slot) {
            restored_arrays.push(keys);
            restored_arrays.push(values);
        }
    }
    for slot in 0..slot_layout.linear_attention_decoder_layer_indices.len() {
        if let Some(convolution_state) = generation_state.convolution_state_array(slot) {
            restored_arrays.push(convolution_state);
        }
        if let Some(recurrent_state) = generation_state.recurrent_state_array(slot) {
            restored_arrays.push(recurrent_state);
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
                        "folded-ternary prompt-cache restored block is missing {tensor_name}"
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
pub(super) fn capture_completed_cache_block(
    runtime: &MlxRuntime,
    persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    slot_layout: &PrismStateSlotLayout,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    let block_token_count = persistent_prompt_cache
        .model_contract_ref()
        .block_token_count();
    let block_end = active_generation.processed_prompt_token_count as usize;
    if block_end == 0 || !block_end.is_multiple_of(block_token_count) {
        return Ok(());
    }
    let block_start = block_end - block_token_count;
    let model_contract = persistent_prompt_cache.model_contract_ref();
    let block_tokens = active_generation.prompt_token_ids[block_start..block_end].to_vec();
    let block_key = chain_block_key(
        model_contract,
        &block_tokens,
        active_generation.last_published_block_key.as_ref(),
    )?;
    let sequence_state_tensors = performance_attribution.measure_operation(
        PerformanceOperation::PersistentPromptCacheStateExtraction,
        |_performance_attribution| {
            extract_sequence_block_tensors(
                runtime,
                &active_generation.generation_state,
                slot_layout,
                block_start,
                block_end,
            )
        },
    )?;
    let boundary_state_tensors = performance_attribution.measure_operation(
        PerformanceOperation::PersistentPromptCacheStateExtraction,
        |_performance_attribution| {
            extract_boundary_snapshot_tensors(&active_generation.generation_state, slot_layout)
        },
    )?;
    let publication_outcome = persistent_prompt_cache
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
                "persistent prompt cache failed during required persistent prompt-state capture; the request was stopped: {publish_error}"
            ),
        })?;
    if matches!(
        publication_outcome,
        PersistentPromptCachePublicationOutcome::Published
            | PersistentPromptCachePublicationOutcome::AlreadyPublished
    ) {
        active_generation.last_published_block_key = Some(block_key);
    }
    Ok(())
}

/// Slices one block's KV span out of every full-attention slab.
fn extract_sequence_block_tensors(
    runtime: &MlxRuntime,
    generation_state: &StraightGenerationState,
    slot_layout: &PrismStateSlotLayout,
    block_start: usize,
    block_end: usize,
) -> Result<HashMap<String, MlxArray>, InferenceEngineError> {
    let mut sequence_state_tensors = HashMap::new();
    for (slot, decoder_layer_index) in slot_layout
        .full_attention_decoder_layer_indices
        .iter()
        .enumerate()
    {
        let (keys_state, values_state) = generation_state
            .key_value_slabs(slot)
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary prompt-cache capture is missing full-attention slot {slot} state"
                ),
            })?;
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
    generation_state: &StraightGenerationState,
    slot_layout: &PrismStateSlotLayout,
) -> Result<HashMap<String, MlxArray>, InferenceEngineError> {
    let mut boundary_state_tensors = HashMap::new();
    for (slot, decoder_layer_index) in slot_layout
        .linear_attention_decoder_layer_indices
        .iter()
        .enumerate()
    {
        let convolution_state = generation_state
            .convolution_state_array(slot)
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary prompt-cache capture is missing linear slot {slot} convolution state"
                ),
            })?;
        boundary_state_tensors.insert(
            format!("layer_{decoder_layer_index}_linear.convolution"),
            convolution_state
                .retain()
                .map_err(|error| InferenceEngineError::Fatal {
                    reason: format!("folded-ternary capture retain failed: {error}"),
                })?,
        );
        let recurrent_state = generation_state
            .recurrent_state_array(slot)
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary prompt-cache capture is missing linear slot {slot} recurrent state"
                ),
            })?;
        boundary_state_tensors.insert(
            format!("layer_{decoder_layer_index}_linear.gated_delta_recurrent"),
            recurrent_state
                .retain()
                .map_err(|error| InferenceEngineError::Fatal {
                    reason: format!("folded-ternary capture retain failed: {error}"),
                })?,
        );
    }
    Ok(boundary_state_tensors)
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
            reason: "folded-ternary cache tensors must have rank four".to_owned(),
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
            reason: format!("folded-ternary cache slice failed: {error}"),
        })
}

fn chain_block_key(
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

impl PrismHadamardQwen35InferenceExecution {
    /// The persistent prompt-cache observability event, mirroring every other
    /// family with an open store.
    pub(super) fn persistent_prompt_cache_stats_event(&self) -> Option<WorkerEvent> {
        let persistent_prompt_cache = self.persistent_prompt_cache.as_ref()?;
        let global_prompt_cache_maximum_size_bytes = self
            .persistent_prompt_cache_disk_store_config
            .as_ref()
            .map(PersistentPromptCacheDiskStoreConfig::global_prompt_cache_maximum_size_bytes)
            .unwrap_or(0);
        Some(build_persistent_prompt_cache_stats_event(
            &self.persistent_prompt_cache_counters,
            u64::try_from(
                persistent_prompt_cache
                    .model_contract_ref()
                    .block_token_count(),
            )
            .unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.sequence_state_block_count()).unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.boundary_state_snapshot_count())
                .unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.visual_embedding_count()).unwrap_or(u64::MAX),
            persistent_prompt_cache.total_size_bytes(),
            persistent_prompt_cache.visual_embedding_total_size_bytes(),
            global_prompt_cache_maximum_size_bytes,
        ))
    }
}
