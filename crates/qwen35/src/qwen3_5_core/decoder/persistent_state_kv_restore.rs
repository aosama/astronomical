//! Assembles restored full-attention KV from persistent astro SSD prompt cache blocks in
//! one concatenation per layer.
//!
//! Concatenating every restored block slice along the sequence axis exactly
//! once is O(restored tokens). The per-block `slice_update` assembly this
//! replaces recopied the whole final-length destination for every block,
//! O(tokens × blocks), and forced one GPU synchronization per block. Holding
//! the complete source block set beside the destination is the deliberate
//! memory-for-speed tradeoff; the driver charges the full block set into
//! restore admission before loading anything.

use std::collections::HashMap;

use super::persistent_state_bridge::AstroSsdPromptCacheStateBridgeError;
use super::request_state::RequestDecoderStateStack;
use crate::astro_decoder_state_cache::AstroDecoderStateCacheState;
use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;

const FULL_ATTENTION_TOKEN_AXIS: usize = 2;

impl RequestDecoderStateStack {
    /// Restores full-attention KV by concatenating every restored block slice
    /// along the sequence axis in one pass per layer, then materializing the
    /// result so the source blocks release before the recurrent snapshot
    /// loads. Every full-attention layer must concatenate to exactly
    /// `restored_token_count`.
    pub fn restore_full_attention_kv_concat(
        &mut self,
        runtime: &MlxRuntime,
        astro_ssd_prompt_cache_kv_block_tensors: &[HashMap<String, MlxArray>],
        restored_token_count: usize,
    ) -> Result<(), AstroSsdPromptCacheStateBridgeError> {
        if astro_ssd_prompt_cache_kv_block_tensors.is_empty() {
            return Err(
                AstroSsdPromptCacheStateBridgeError::InvalidRestoredSequenceTokenCount {
                    restored_token_count,
                },
            );
        }
        let restored_token_count_i32 = i32::try_from(restored_token_count).map_err(|_| {
            AstroSsdPromptCacheStateBridgeError::InvalidRestoredSequenceTokenCount {
                restored_token_count,
            }
        })?;
        for layer_index in 0..self.layer_count() {
            match self.layer_mut(layer_index) {
                Some(AstroDecoderStateCacheState::AppendOnlyAttention { attention }) => {
                    let keys_tensor_name = format!("layer_{layer_index}_attention.keys");
                    let values_tensor_name = format!("layer_{layer_index}_attention.values");
                    let keys_slices = block_slice_tensors_by_name(
                        astro_ssd_prompt_cache_kv_block_tensors,
                        layer_index,
                        &keys_tensor_name,
                    )?;
                    let values_slices = block_slice_tensors_by_name(
                        astro_ssd_prompt_cache_kv_block_tensors,
                        layer_index,
                        &values_tensor_name,
                    )?;
                    let full_keys = runtime
                        .concatenate_axis(&keys_slices, FULL_ATTENTION_TOKEN_AXIS as i32)
                        .map_err(|source| {
                            AstroSsdPromptCacheStateBridgeError::ConcatenateRestoreDestination {
                                layer_index,
                                tensor_name: keys_tensor_name.clone(),
                                source: source.into(),
                            }
                        })?;
                    let full_values = runtime
                        .concatenate_axis(&values_slices, FULL_ATTENTION_TOKEN_AXIS as i32)
                        .map_err(|source| {
                            AstroSsdPromptCacheStateBridgeError::ConcatenateRestoreDestination {
                                layer_index,
                                tensor_name: values_tensor_name,
                                source: source.into(),
                            }
                        })?;
                    validate_concatenated_token_count(
                        &full_keys,
                        layer_index,
                        "keys",
                        restored_token_count_i32,
                        restored_token_count,
                    )?;
                    validate_concatenated_token_count(
                        &full_values,
                        layer_index,
                        "values",
                        restored_token_count_i32,
                        restored_token_count,
                    )?;
                    attention
                        .restore_from_blocks(full_keys, full_values)
                        .map_err(|source| {
                            AstroSsdPromptCacheStateBridgeError::RestoreFullAttentionState {
                                layer_index,
                                source,
                            }
                        })?;
                }
                Some(AstroDecoderStateCacheState::Composite { .. }) => {}
                None => {
                    return Err(AstroSsdPromptCacheStateBridgeError::MissingLayer { layer_index });
                }
            }
        }
        materialize_restored_full_attention_tensors(self, runtime)
    }
}

fn block_slice_tensors_by_name<'a>(
    astro_ssd_prompt_cache_kv_block_tensors: &'a [HashMap<String, MlxArray>],
    layer_index: usize,
    tensor_name: &str,
) -> Result<Vec<&'a MlxArray>, AstroSsdPromptCacheStateBridgeError> {
    astro_ssd_prompt_cache_kv_block_tensors
        .iter()
        .map(|block_tensors| {
            block_tensors.get(tensor_name).ok_or(
                AstroSsdPromptCacheStateBridgeError::MissingBlockTensor {
                    layer_index,
                    tensor_name: tensor_name.to_owned(),
                },
            )
        })
        .collect()
}

fn validate_concatenated_token_count(
    concatenated_tensor: &MlxArray,
    layer_index: usize,
    tensor_role: &'static str,
    restored_token_count_i32: i32,
    restored_token_count: usize,
) -> Result<(), AstroSsdPromptCacheStateBridgeError> {
    let concatenated_shape = concatenated_tensor.shape();
    if concatenated_shape.len() != 4 {
        return Err(
            AstroSsdPromptCacheStateBridgeError::InvalidLayerTensorShape {
                layer_index,
                tensor_role,
                actual_shape: concatenated_shape,
            },
        );
    }
    if concatenated_shape[FULL_ATTENTION_TOKEN_AXIS] != restored_token_count_i32 {
        return Err(
            AstroSsdPromptCacheStateBridgeError::ConcatenatedTokenCountMismatch {
                layer_index,
                concatenated_token_count: concatenated_shape[FULL_ATTENTION_TOKEN_AXIS] as usize,
                restored_token_count,
            },
        );
    }
    Ok(())
}

fn materialize_restored_full_attention_tensors(
    request_decoder_state: &RequestDecoderStateStack,
    runtime: &MlxRuntime,
) -> Result<(), AstroSsdPromptCacheStateBridgeError> {
    let mut restored_tensors = Vec::new();
    for layer_index in 0..request_decoder_state.layer_count() {
        match request_decoder_state.layer(layer_index) {
            Some(AstroDecoderStateCacheState::AppendOnlyAttention { attention }) => {
                restored_tensors.push(attention.keys_state().ok_or(
                    AstroSsdPromptCacheStateBridgeError::MissingLayerTensor {
                        layer_index,
                        tensor_role: "keys",
                    },
                )?);
                restored_tensors.push(attention.values_state().ok_or(
                    AstroSsdPromptCacheStateBridgeError::MissingLayerTensor {
                        layer_index,
                        tensor_role: "values",
                    },
                )?);
            }
            Some(AstroDecoderStateCacheState::Composite { .. }) => {}
            None => {
                return Err(AstroSsdPromptCacheStateBridgeError::MissingLayer { layer_index });
            }
        }
    }
    if restored_tensors.is_empty() {
        return Ok(());
    }
    runtime
        .evaluate_arrays(&restored_tensors)
        .map_err(|captured_error| {
            AstroSsdPromptCacheStateBridgeError::EvaluateRestoredAstroSsdPromptCacheState(
                runtime_integration::MlxRuntimeError::from(captured_error),
            )
        })
}
