//! Assembles restored full-attention KV from persistent prompt-cache blocks.
//!
//! The restored sequence length is known before the first block is loaded.
//! Allocate that destination once, write each block into its token range with
//! `slice_update`, drop the source block, and evaluate. MLX donates a uniquely
//! held destination buffer, so each write copies the incoming block instead of
//! recopying the growing prefix.

use std::collections::HashMap;

use super::persistent_state_bridge::{
    PersistentPromptCacheStateBridgeError, take_named_block_tensor,
};
use super::request_decoder_state::RequestDecoderStateStack;
use crate::decoder_cache::DecoderCacheState;
use astronomical_runtime_integration::{MlxArray, MlxRuntime};

const FULL_ATTENTION_TOKEN_AXIS: usize = 2;

impl RequestDecoderStateStack {
    /// Restores live decoder state from split persistent prompt-cache tensors.
    ///
    /// Each KV block is absorbed into a final-length destination and
    /// materialized before the next block is required. Holding every loaded
    /// block until one terminal concatenation would pin a full-prefix source
    /// copy beside the seated model.
    pub fn restore_from_persistent_prompt_cache_blocks(
        &mut self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_kv_block_tensors: &mut [HashMap<String, MlxArray>],
        persistent_prompt_cache_recurrent_snapshot_tensors: &mut HashMap<String, MlxArray>,
    ) -> Result<(), PersistentPromptCacheStateBridgeError> {
        if persistent_prompt_cache_kv_block_tensors.is_empty() {
            return Ok(());
        }
        let restored_token_count =
            restored_sequence_token_count(persistent_prompt_cache_kv_block_tensors)?;
        let mut sequence_start_tokens = 0_usize;
        for kv_block_tensors in persistent_prompt_cache_kv_block_tensors.iter_mut() {
            let block_token_count = full_attention_block_sequence_token_count(kv_block_tensors)?;
            self.absorb_persistent_prompt_cache_kv_block(
                runtime,
                kv_block_tensors,
                sequence_start_tokens,
                restored_token_count,
            )?;
            sequence_start_tokens = sequence_start_tokens.checked_add(block_token_count).ok_or(
                PersistentPromptCacheStateBridgeError::InvalidRestoredSequenceTokenCount {
                    restored_token_count,
                },
            )?;
        }
        self.absorb_persistent_prompt_cache_recurrent_snapshot(
            runtime,
            persistent_prompt_cache_recurrent_snapshot_tensors,
        )
    }

    /// Writes one sequence-state block into the final-length KV destination.
    ///
    /// `sequence_start_tokens == 0` allocates that destination from the first
    /// block's geometry. Later blocks must fit the remaining token range.
    pub fn absorb_persistent_prompt_cache_kv_block(
        &mut self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_kv_block_tensors: &mut HashMap<String, MlxArray>,
        sequence_start_tokens: usize,
        restored_token_count: usize,
    ) -> Result<(), PersistentPromptCacheStateBridgeError> {
        if sequence_start_tokens == 0 {
            self.allocate_persistent_prompt_cache_kv_restore_destination(
                runtime,
                restored_token_count,
                persistent_prompt_cache_kv_block_tensors,
            )?;
        }
        for layer_index in 0..self.layer_count() {
            match self.layer_mut(layer_index) {
                Some(DecoderCacheState::AppendOnlyAttention { attention }) => {
                    let incoming_keys = take_named_block_tensor(
                        persistent_prompt_cache_kv_block_tensors,
                        layer_index,
                        format!("layer_{layer_index}_attention.keys"),
                    )?;
                    let incoming_values = take_named_block_tensor(
                        persistent_prompt_cache_kv_block_tensors,
                        layer_index,
                        format!("layer_{layer_index}_attention.values"),
                    )?;
                    let (existing_keys, existing_values) = attention
                        .take_key_value_storage()
                        .ok_or(PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                            layer_index,
                            tensor_role: "keys",
                        })?;
                    let updated_keys_and_values = write_kv_block_into_restore_destination(
                        runtime,
                        layer_index,
                        &existing_keys,
                        &existing_values,
                        &incoming_keys,
                        &incoming_values,
                        sequence_start_tokens,
                    );
                    match updated_keys_and_values {
                        Ok((updated_keys, updated_values)) => {
                            drop(existing_keys);
                            drop(existing_values);
                            attention.restore_from_blocks(updated_keys, updated_values).map_err(
                                |source| {
                                    PersistentPromptCacheStateBridgeError::RestoreFullAttentionState {
                                        layer_index,
                                        source,
                                    }
                                },
                            )?;
                        }
                        Err(write_error) => {
                            attention
                                .restore_from_blocks(existing_keys, existing_values)
                                .map_err(|source| {
                                    PersistentPromptCacheStateBridgeError::RestoreFullAttentionState {
                                        layer_index,
                                        source,
                                    }
                                })?;
                            return Err(write_error);
                        }
                    }
                }
                Some(DecoderCacheState::Composite { .. }) => {}
                None => {
                    return Err(PersistentPromptCacheStateBridgeError::MissingLayer {
                        layer_index,
                    });
                }
            }
        }
        materialize_restored_full_attention_tensors(self, runtime)
    }

    fn allocate_persistent_prompt_cache_kv_restore_destination(
        &mut self,
        runtime: &MlxRuntime,
        restored_token_count: usize,
        prototype_kv_block_tensors: &HashMap<String, MlxArray>,
    ) -> Result<(), PersistentPromptCacheStateBridgeError> {
        if self.has_allocated_full_attention_restore_destination() {
            return Ok(());
        }
        let restored_sequence_token_count_i32 =
            i32::try_from(restored_token_count).map_err(|_| {
                PersistentPromptCacheStateBridgeError::InvalidRestoredSequenceTokenCount {
                    restored_token_count,
                }
            })?;
        if restored_sequence_token_count_i32 <= 0 {
            return Err(
                PersistentPromptCacheStateBridgeError::InvalidRestoredSequenceTokenCount {
                    restored_token_count,
                },
            );
        }
        for layer_index in 0..self.layer_count() {
            match self.layer_mut(layer_index) {
                Some(DecoderCacheState::AppendOnlyAttention { attention }) => {
                    let prototype_keys = prototype_kv_block_tensors
                        .get(&format!("layer_{layer_index}_attention.keys"))
                        .ok_or(PersistentPromptCacheStateBridgeError::MissingBlockTensor {
                            layer_index,
                            tensor_name: format!("layer_{layer_index}_attention.keys"),
                        })?;
                    let prototype_values = prototype_kv_block_tensors
                        .get(&format!("layer_{layer_index}_attention.values"))
                        .ok_or(PersistentPromptCacheStateBridgeError::MissingBlockTensor {
                            layer_index,
                            tensor_name: format!("layer_{layer_index}_attention.values"),
                        })?;
                    let destination_keys = zeros_restore_destination(
                        runtime,
                        layer_index,
                        "keys",
                        prototype_keys,
                        restored_sequence_token_count_i32,
                    )?;
                    let destination_values = zeros_restore_destination(
                        runtime,
                        layer_index,
                        "values",
                        prototype_values,
                        restored_sequence_token_count_i32,
                    )?;
                    attention
                        .restore_from_blocks(destination_keys, destination_values)
                        .map_err(|source| {
                            PersistentPromptCacheStateBridgeError::RestoreFullAttentionState {
                                layer_index,
                                source,
                            }
                        })?;
                }
                Some(DecoderCacheState::Composite { .. }) => {}
                None => {
                    return Err(PersistentPromptCacheStateBridgeError::MissingLayer {
                        layer_index,
                    });
                }
            }
        }
        materialize_restored_full_attention_tensors(self, runtime)
    }

    fn has_allocated_full_attention_restore_destination(&self) -> bool {
        (0..self.layer_count()).any(|layer_index| {
            matches!(
                self.layer(layer_index),
                Some(DecoderCacheState::AppendOnlyAttention { attention })
                    if attention.keys_state().is_some()
            )
        })
    }
}

fn restored_sequence_token_count(
    persistent_prompt_cache_kv_block_tensors: &[HashMap<String, MlxArray>],
) -> Result<usize, PersistentPromptCacheStateBridgeError> {
    let mut restored_token_count = 0_usize;
    for kv_block_tensors in persistent_prompt_cache_kv_block_tensors {
        restored_token_count = restored_token_count
            .checked_add(full_attention_block_sequence_token_count(kv_block_tensors)?)
            .ok_or(
                PersistentPromptCacheStateBridgeError::InvalidRestoredSequenceTokenCount {
                    restored_token_count: usize::MAX,
                },
            )?;
    }
    if restored_token_count == 0 {
        return Err(
            PersistentPromptCacheStateBridgeError::InvalidRestoredSequenceTokenCount {
                restored_token_count,
            },
        );
    }
    Ok(restored_token_count)
}

fn full_attention_block_sequence_token_count(
    kv_block_tensors: &HashMap<String, MlxArray>,
) -> Result<usize, PersistentPromptCacheStateBridgeError> {
    let (_, keys) = kv_block_tensors
        .iter()
        .find(|(tensor_name, _)| tensor_name.ends_with("_attention.keys"))
        .ok_or(PersistentPromptCacheStateBridgeError::MissingBlockTensor {
            layer_index: 0,
            tensor_name: "attention.keys".to_owned(),
        })?;
    sequence_token_count_from_rank_four_tensor(keys, 0, "keys")
}

fn sequence_token_count_from_rank_four_tensor(
    tensor: &MlxArray,
    layer_index: usize,
    tensor_role: &'static str,
) -> Result<usize, PersistentPromptCacheStateBridgeError> {
    let tensor_shape = tensor.shape();
    if tensor_shape.len() != 4 || tensor_shape[FULL_ATTENTION_TOKEN_AXIS] <= 0 {
        return Err(
            PersistentPromptCacheStateBridgeError::InvalidLayerTensorShape {
                layer_index,
                tensor_role,
                actual_shape: tensor_shape,
            },
        );
    }
    usize::try_from(tensor_shape[FULL_ATTENTION_TOKEN_AXIS]).map_err(|_| {
        PersistentPromptCacheStateBridgeError::InvalidLayerTensorShape {
            layer_index,
            tensor_role,
            actual_shape: tensor_shape,
        }
    })
}

fn zeros_restore_destination(
    runtime: &MlxRuntime,
    layer_index: usize,
    tensor_role: &'static str,
    prototype: &MlxArray,
    restored_sequence_token_count: i32,
) -> Result<MlxArray, PersistentPromptCacheStateBridgeError> {
    let prototype_shape = prototype.shape();
    if prototype_shape.len() != 4 {
        return Err(
            PersistentPromptCacheStateBridgeError::InvalidLayerTensorShape {
                layer_index,
                tensor_role,
                actual_shape: prototype_shape,
            },
        );
    }
    let mut destination_shape = prototype_shape;
    destination_shape[FULL_ATTENTION_TOKEN_AXIS] = restored_sequence_token_count;
    runtime
        .zeros(&destination_shape, prototype.dtype())
        .map_err(
            |source| PersistentPromptCacheStateBridgeError::AllocateRestoreDestination {
                layer_index,
                source,
            },
        )
}

fn write_kv_block_into_restore_destination(
    runtime: &MlxRuntime,
    layer_index: usize,
    existing_keys: &MlxArray,
    existing_values: &MlxArray,
    incoming_keys: &MlxArray,
    incoming_values: &MlxArray,
    sequence_start_tokens: usize,
) -> Result<(MlxArray, MlxArray), PersistentPromptCacheStateBridgeError> {
    let updated_keys = write_block_into_restore_destination(
        runtime,
        layer_index,
        "keys",
        existing_keys,
        incoming_keys,
        sequence_start_tokens,
    )?;
    let updated_values = write_block_into_restore_destination(
        runtime,
        layer_index,
        "values",
        existing_values,
        incoming_values,
        sequence_start_tokens,
    )?;
    Ok((updated_keys, updated_values))
}

fn write_block_into_restore_destination(
    runtime: &MlxRuntime,
    layer_index: usize,
    tensor_role: &'static str,
    destination: &MlxArray,
    incoming: &MlxArray,
    sequence_start_tokens: usize,
) -> Result<MlxArray, PersistentPromptCacheStateBridgeError> {
    let destination_shape = destination.shape();
    let incoming_shape = incoming.shape();
    if destination_shape.len() != 4 {
        return Err(
            PersistentPromptCacheStateBridgeError::InvalidLayerTensorShape {
                layer_index,
                tensor_role,
                actual_shape: destination_shape,
            },
        );
    }
    let incoming_token_count = incoming_shape
        .get(FULL_ATTENTION_TOKEN_AXIS)
        .copied()
        .unwrap_or(0);
    if incoming_shape.len() != 4
        || incoming_shape[0] != destination_shape[0]
        || incoming_shape[1] != destination_shape[1]
        || incoming_shape[3] != destination_shape[3]
    {
        return Err(
            PersistentPromptCacheStateBridgeError::KvBlockShapeDoesNotMatchRestoreDestination {
                layer_index,
                tensor_role,
                actual_shape: incoming_shape,
                expected_shape: expected_incoming_block_shape(
                    &destination_shape,
                    incoming_token_count,
                ),
            },
        );
    }
    let destination_token_count = destination_shape[FULL_ATTENTION_TOKEN_AXIS];
    let sequence_start_tokens_i32 = i32::try_from(sequence_start_tokens).map_err(|_| {
        PersistentPromptCacheStateBridgeError::KvBlockDoesNotFitRestoreDestination {
            sequence_start_tokens,
            destination_token_count: usize::try_from(destination_token_count).unwrap_or(usize::MAX),
        }
    })?;
    let sequence_end_tokens_i32 = sequence_start_tokens_i32
        .checked_add(incoming_token_count)
        .ok_or(
            PersistentPromptCacheStateBridgeError::KvBlockDoesNotFitRestoreDestination {
                sequence_start_tokens,
                destination_token_count: usize::try_from(destination_token_count)
                    .unwrap_or(usize::MAX),
            },
        )?;
    if sequence_end_tokens_i32 > destination_token_count {
        return Err(
            PersistentPromptCacheStateBridgeError::KvBlockDoesNotFitRestoreDestination {
                sequence_start_tokens,
                destination_token_count: usize::try_from(destination_token_count)
                    .unwrap_or(usize::MAX),
            },
        );
    }
    let mut slice_starts = vec![0_i32; destination_shape.len()];
    slice_starts[FULL_ATTENTION_TOKEN_AXIS] = sequence_start_tokens_i32;
    let mut slice_stops = destination_shape;
    slice_stops[FULL_ATTENTION_TOKEN_AXIS] = sequence_end_tokens_i32;
    let slice_strides = vec![1_i32; slice_starts.len()];
    runtime
        .slice_update(
            destination,
            incoming,
            &slice_starts,
            &slice_stops,
            &slice_strides,
        )
        .map_err(
            |source| PersistentPromptCacheStateBridgeError::WriteRestoreDestination {
                layer_index,
                tensor_name: format!("layer_{layer_index}_attention.{tensor_role}"),
                source,
            },
        )
}

fn expected_incoming_block_shape(destination_shape: &[i32], incoming_token_count: i32) -> Vec<i32> {
    let mut expected_shape = destination_shape.to_vec();
    if expected_shape.len() > FULL_ATTENTION_TOKEN_AXIS {
        expected_shape[FULL_ATTENTION_TOKEN_AXIS] = incoming_token_count;
    }
    expected_shape
}

fn materialize_restored_full_attention_tensors(
    request_decoder_state: &RequestDecoderStateStack,
    runtime: &MlxRuntime,
) -> Result<(), PersistentPromptCacheStateBridgeError> {
    let mut restored_tensors = Vec::new();
    for layer_index in 0..request_decoder_state.layer_count() {
        match request_decoder_state.layer(layer_index) {
            Some(DecoderCacheState::AppendOnlyAttention { attention }) => {
                restored_tensors.push(attention.keys_state().ok_or(
                    PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                        layer_index,
                        tensor_role: "keys",
                    },
                )?);
                restored_tensors.push(attention.values_state().ok_or(
                    PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                        layer_index,
                        tensor_role: "values",
                    },
                )?);
            }
            Some(DecoderCacheState::Composite { .. }) => {}
            None => {
                return Err(PersistentPromptCacheStateBridgeError::MissingLayer { layer_index });
            }
        }
    }
    if restored_tensors.is_empty() {
        return Ok(());
    }
    runtime
        .evaluate_arrays(&restored_tensors)
        .map_err(PersistentPromptCacheStateBridgeError::EvaluateRestoredPersistentPromptCacheState)
}
