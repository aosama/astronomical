use std::collections::HashMap;

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::persistent_state_bridge::{
    retain_block_tensor, retain_layer_tensor, slice_full_attention_block,
};
use super::{PersistentPromptCacheStateBridgeError, RequestDecoderStateStack};
use crate::decoder_cache::DecoderCacheState;

impl RequestDecoderStateStack {
    /// Extracts the complete compact decoder state produced by sparse target prefill.
    pub fn extract_speculative_prefill_target_state_tensors(
        &self,
        runtime: &MlxRuntime,
    ) -> Result<HashMap<String, MlxArray>, PersistentPromptCacheStateBridgeError> {
        let mut target_state_tensors = HashMap::with_capacity(self.layer_count() * 2);
        for layer_index in 0..self.layer_count() {
            match self.layer(layer_index) {
                Some(DecoderCacheState::AppendOnlyAttention { attention }) => {
                    let compact_target_token_count = usize::try_from(attention.offset_tokens())
                        .map_err(
                            |_| PersistentPromptCacheStateBridgeError::InvalidBlockRange {
                                block_start_tokens: 0,
                                block_end_tokens: 0,
                            },
                        )?;
                    for (tensor_role, target_state_tensor) in [
                        ("keys", attention.keys_state()),
                        ("values", attention.values_state()),
                    ] {
                        let target_state_tensor = target_state_tensor.ok_or(
                            PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                                layer_index,
                                tensor_role,
                            },
                        )?;
                        target_state_tensors.insert(
                            format!("layer_{layer_index}_attention.{tensor_role}"),
                            slice_full_attention_block(
                                runtime,
                                target_state_tensor,
                                layer_index,
                                tensor_role,
                                0,
                                compact_target_token_count,
                            )?,
                        );
                    }
                }
                Some(DecoderCacheState::Composite {
                    convolution,
                    recurrent,
                }) => {
                    let convolution_state = convolution.state().ok_or(
                        PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                            layer_index,
                            tensor_role: "convolution",
                        },
                    )?;
                    let recurrent_state = recurrent.state().ok_or(
                        PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                            layer_index,
                            tensor_role: "recurrent",
                        },
                    )?;
                    target_state_tensors.insert(
                        format!("layer_{layer_index}_linear.convolution"),
                        retain_layer_tensor(layer_index, "convolution", convolution_state)?,
                    );
                    target_state_tensors.insert(
                        format!("layer_{layer_index}_linear.gated_delta_recurrent"),
                        retain_layer_tensor(layer_index, "recurrent", recurrent_state)?,
                    );
                }
                None => {
                    return Err(PersistentPromptCacheStateBridgeError::MissingLayer {
                        layer_index,
                    });
                }
            }
        }
        Ok(target_state_tensors)
    }

    /// Restores one selection-bound compact target state without dense target recomputation.
    pub fn restore_speculative_prefill_target_state_tensors(
        &mut self,
        target_state_tensors: &HashMap<String, MlxArray>,
        expected_compact_target_token_count: usize,
    ) -> Result<(), PersistentPromptCacheStateBridgeError> {
        for layer_index in 0..self.layer_count() {
            match self.layer_mut(layer_index) {
                Some(DecoderCacheState::AppendOnlyAttention { attention }) => {
                    let keys_tensor_name = format!("layer_{layer_index}_attention.keys");
                    let values_tensor_name = format!("layer_{layer_index}_attention.values");
                    let restored_keys = target_state_tensors.get(&keys_tensor_name).ok_or(
                        PersistentPromptCacheStateBridgeError::MissingBlockTensor {
                            layer_index,
                            tensor_name: keys_tensor_name.clone(),
                        },
                    )?;
                    let restored_values = target_state_tensors.get(&values_tensor_name).ok_or(
                        PersistentPromptCacheStateBridgeError::MissingBlockTensor {
                            layer_index,
                            tensor_name: values_tensor_name.clone(),
                        },
                    )?;
                    attention
                        .restore_from_blocks(
                            retain_block_tensor(layer_index, &keys_tensor_name, restored_keys)?,
                            retain_block_tensor(layer_index, &values_tensor_name, restored_values)?,
                        )
                        .map_err(|source| {
                            PersistentPromptCacheStateBridgeError::RestoreFullAttentionState {
                                layer_index,
                                source,
                            }
                        })?;
                    if usize::try_from(attention.offset_tokens()).ok()
                        != Some(expected_compact_target_token_count)
                    {
                        return Err(PersistentPromptCacheStateBridgeError::InconsistentSpeculativePrefillTargetTokenCount {
                            layer_index,
                            expected_token_count: expected_compact_target_token_count,
                            actual_token_count: attention.offset_tokens(),
                        });
                    }
                }
                Some(DecoderCacheState::Composite {
                    convolution,
                    recurrent,
                }) => {
                    let convolution_tensor_name = format!("layer_{layer_index}_linear.convolution");
                    let recurrent_tensor_name =
                        format!("layer_{layer_index}_linear.gated_delta_recurrent");
                    let restored_convolution =
                        target_state_tensors.get(&convolution_tensor_name).ok_or(
                            PersistentPromptCacheStateBridgeError::MissingBlockTensor {
                                layer_index,
                                tensor_name: convolution_tensor_name.clone(),
                            },
                        )?;
                    let restored_recurrent = target_state_tensors
                        .get(&recurrent_tensor_name)
                        .ok_or(PersistentPromptCacheStateBridgeError::MissingBlockTensor {
                            layer_index,
                            tensor_name: recurrent_tensor_name.clone(),
                        })?;
                    convolution.restore_from_snapshot(retain_block_tensor(
                        layer_index,
                        &convolution_tensor_name,
                        restored_convolution,
                    )?);
                    recurrent.restore_from_snapshot(retain_block_tensor(
                        layer_index,
                        &recurrent_tensor_name,
                        restored_recurrent,
                    )?);
                }
                None => {
                    return Err(PersistentPromptCacheStateBridgeError::MissingLayer {
                        layer_index,
                    });
                }
            }
        }
        Ok(())
    }
}
