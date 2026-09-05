use super::RequestDecoderStateStack;
use crate::DecoderCacheState;
use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};
use std::collections::HashMap;
/// Bridges persistent prompt-cache block tensors and the live in-memory
/// request decoder state. The in-memory owners decide how restored tensors
/// become live state; this module owns only SSD block extraction and assembly.
impl RequestDecoderStateStack {
    /// Extracts one full-attention key/value block into the persistent tensor map.
    pub fn extract_persistent_prompt_cache_kv_block_tensors(
        &self,
        runtime: &MlxRuntime,
        block_start_tokens: usize,
        block_end_tokens: usize,
        persistent_prompt_cache_block_token_count: usize,
    ) -> Result<HashMap<String, MlxArray>, PersistentPromptCacheStateBridgeError> {
        validate_persistent_prompt_cache_block_range(
            block_start_tokens,
            block_end_tokens,
            persistent_prompt_cache_block_token_count,
        )?;
        // Sequence state is sliced only at a contract-derived complete boundary. A partial
        // slice would produce a valid-looking tensor that cannot participate in the hash chain
        // or be concatenated with other blocks during a later restore.
        let mut kv_block_tensors = HashMap::with_capacity(self.layer_count() * 2);
        for layer_index in 0..self.layer_count() {
            match self.layer(layer_index) {
                Some(DecoderCacheState::AppendOnlyAttention { attention }) => {
                    let attention_keys = attention.keys_state().ok_or(
                        PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                            layer_index,
                            tensor_role: "keys",
                        },
                    )?;
                    let attention_values = attention.values_state().ok_or(
                        PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                            layer_index,
                            tensor_role: "values",
                        },
                    )?;
                    kv_block_tensors.insert(
                        format!("layer_{layer_index}_attention.keys"),
                        slice_full_attention_block(
                            runtime,
                            attention_keys,
                            layer_index,
                            "keys",
                            block_start_tokens,
                            block_end_tokens,
                        )?,
                    );
                    kv_block_tensors.insert(
                        format!("layer_{layer_index}_attention.values"),
                        slice_full_attention_block(
                            runtime,
                            attention_values,
                            layer_index,
                            "values",
                            block_start_tokens,
                            block_end_tokens,
                        )?,
                    );
                }
                Some(DecoderCacheState::Composite { .. }) => {}
                None => {
                    return Err(PersistentPromptCacheStateBridgeError::MissingLayer {
                        layer_index,
                    });
                }
            }
        }
        Ok(kv_block_tensors)
    }

    /// Extracts the current gated-delta recurrent and convolution snapshots.
    pub fn extract_persistent_prompt_cache_recurrent_snapshot_tensors(
        &self,
    ) -> Result<HashMap<String, MlxArray>, PersistentPromptCacheStateBridgeError> {
        let mut recurrent_snapshot_tensors = HashMap::with_capacity(self.layer_count() * 2);
        for layer_index in 0..self.layer_count() {
            match self.layer(layer_index) {
                Some(DecoderCacheState::AppendOnlyAttention { .. }) => {}
                Some(DecoderCacheState::Composite {
                    convolution,
                    recurrent,
                }) => {
                    let convolution = convolution.state().ok_or(
                        PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                            layer_index,
                            tensor_role: "convolution",
                        },
                    )?;
                    let recurrent = recurrent.state().ok_or(
                        PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                            layer_index,
                            tensor_role: "recurrent",
                        },
                    )?;
                    recurrent_snapshot_tensors.insert(
                        format!("layer_{layer_index}_linear.convolution"),
                        retain_layer_tensor(layer_index, "convolution", convolution)?,
                    );
                    recurrent_snapshot_tensors.insert(
                        format!("layer_{layer_index}_linear.gated_delta_recurrent"),
                        retain_layer_tensor(layer_index, "recurrent", recurrent)?,
                    );
                }
                None => {
                    return Err(PersistentPromptCacheStateBridgeError::MissingLayer {
                        layer_index,
                    });
                }
            }
        }
        Ok(recurrent_snapshot_tensors)
    }

    /// Installs the newest complete recurrent snapshot after KV blocks are live.
    pub fn absorb_persistent_prompt_cache_recurrent_snapshot(
        &mut self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_recurrent_snapshot_tensors: &mut HashMap<String, MlxArray>,
    ) -> Result<(), PersistentPromptCacheStateBridgeError> {
        for layer_index in 0..self.layer_count() {
            match self.layer_mut(layer_index) {
                Some(DecoderCacheState::AppendOnlyAttention { .. }) => {}
                Some(DecoderCacheState::Composite {
                    convolution,
                    recurrent,
                }) => {
                    let convolution_tensor_name = format!("layer_{layer_index}_linear.convolution");
                    let loaded_convolution = take_named_block_tensor(
                        persistent_prompt_cache_recurrent_snapshot_tensors,
                        layer_index,
                        convolution_tensor_name,
                    )?;
                    let recurrent_tensor_name =
                        format!("layer_{layer_index}_linear.gated_delta_recurrent");
                    let loaded_recurrent = take_named_block_tensor(
                        persistent_prompt_cache_recurrent_snapshot_tensors,
                        layer_index,
                        recurrent_tensor_name,
                    )?;
                    convolution.restore_from_snapshot(loaded_convolution);
                    recurrent.restore_from_snapshot(loaded_recurrent);
                    materialize_restored_layer_tensors(
                        runtime,
                        layer_index,
                        convolution.state(),
                        recurrent.state(),
                        "convolution",
                        "recurrent",
                    )?;
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

    /// Materializes all restored state before the first new prefill chunk.
    pub fn materialize_restored_persistent_prompt_cache_state(
        &self,
        runtime: &MlxRuntime,
    ) -> Result<(), PersistentPromptCacheStateBridgeError> {
        let mut restored_tensors = Vec::with_capacity(self.layer_count() * 2);
        for layer_index in 0..self.layer_count() {
            match self.layer(layer_index) {
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
                Some(DecoderCacheState::Composite {
                    convolution,
                    recurrent,
                }) => {
                    restored_tensors.push(convolution.state().ok_or(
                        PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                            layer_index,
                            tensor_role: "convolution",
                        },
                    )?);
                    restored_tensors.push(recurrent.state().ok_or(
                        PersistentPromptCacheStateBridgeError::MissingLayerTensor {
                            layer_index,
                            tensor_role: "recurrent",
                        },
                    )?);
                }
                None => {
                    return Err(PersistentPromptCacheStateBridgeError::MissingLayer {
                        layer_index,
                    });
                }
            }
        }

        // Evaluate every retained or concatenated array before the next prefill forward. This
        // places malformed restored state at a single explicit GPU boundary instead of letting
        // lazy MLX evaluation attribute a later request failure to unrelated model work.
        runtime.evaluate_arrays(&restored_tensors).map_err(
            PersistentPromptCacheStateBridgeError::EvaluateRestoredPersistentPromptCacheState,
        )
    }
}

pub(super) fn take_named_block_tensor(
    block_tensors: &mut HashMap<String, MlxArray>,
    layer_index: usize,
    tensor_name: String,
) -> Result<MlxArray, PersistentPromptCacheStateBridgeError> {
    block_tensors.remove(&tensor_name).ok_or(
        PersistentPromptCacheStateBridgeError::MissingBlockTensor {
            layer_index,
            tensor_name,
        },
    )
}

fn materialize_restored_layer_tensors(
    runtime: &MlxRuntime,
    layer_index: usize,
    first_tensor: Option<&MlxArray>,
    second_tensor: Option<&MlxArray>,
    first_tensor_role: &'static str,
    second_tensor_role: &'static str,
) -> Result<(), PersistentPromptCacheStateBridgeError> {
    let first_tensor =
        first_tensor.ok_or(PersistentPromptCacheStateBridgeError::MissingLayerTensor {
            layer_index,
            tensor_role: first_tensor_role,
        })?;
    let second_tensor =
        second_tensor.ok_or(PersistentPromptCacheStateBridgeError::MissingLayerTensor {
            layer_index,
            tensor_role: second_tensor_role,
        })?;
    runtime
        .evaluate_arrays(&[first_tensor, second_tensor])
        .map_err(PersistentPromptCacheStateBridgeError::EvaluateRestoredPersistentPromptCacheState)
}

fn validate_persistent_prompt_cache_block_range(
    block_start_tokens: usize,
    block_end_tokens: usize,
    persistent_prompt_cache_block_token_count: usize,
) -> Result<(), PersistentPromptCacheStateBridgeError> {
    let block_token_count = block_end_tokens.checked_sub(block_start_tokens).ok_or(
        PersistentPromptCacheStateBridgeError::InvalidBlockRange {
            block_start_tokens,
            block_end_tokens,
        },
    )?;
    if block_token_count != persistent_prompt_cache_block_token_count
        || persistent_prompt_cache_block_token_count == 0
    {
        return Err(PersistentPromptCacheStateBridgeError::InvalidBlockRange {
            block_start_tokens,
            block_end_tokens,
        });
    }
    Ok(())
}

pub(super) fn slice_full_attention_block(
    runtime: &MlxRuntime,
    tensor: &MlxArray,
    layer_index: usize,
    tensor_role: &'static str,
    block_start_tokens: usize,
    block_end_tokens: usize,
) -> Result<MlxArray, PersistentPromptCacheStateBridgeError> {
    let tensor_shape = tensor.shape();
    if tensor_shape.len() != 4 {
        return Err(
            PersistentPromptCacheStateBridgeError::InvalidLayerTensorShape {
                layer_index,
                tensor_role,
                actual_shape: tensor_shape,
            },
        );
    }
    let block_start_tokens_i32 = i32::try_from(block_start_tokens).map_err(|_| {
        PersistentPromptCacheStateBridgeError::InvalidBlockRange {
            block_start_tokens,
            block_end_tokens,
        }
    })?;
    let block_end_tokens_i32 = i32::try_from(block_end_tokens).map_err(|_| {
        PersistentPromptCacheStateBridgeError::InvalidBlockRange {
            block_start_tokens,
            block_end_tokens,
        }
    })?;
    if tensor_shape[2] < block_end_tokens_i32 {
        return Err(
            PersistentPromptCacheStateBridgeError::BlockRangeExceedsLayerTensor {
                layer_index,
                tensor_role,
                requested_end_tokens: block_end_tokens_i32,
                available_tokens: tensor_shape[2],
            },
        );
    }
    let mut slice_stops = tensor_shape;
    slice_stops[2] = block_end_tokens_i32;
    runtime
        .slice(
            tensor,
            &[0, 0, block_start_tokens_i32, 0],
            &slice_stops,
            &[1, 1, 1, 1],
        )
        .map_err(
            |source| PersistentPromptCacheStateBridgeError::SliceLayerTensor {
                layer_index,
                tensor_role,
                source,
            },
        )
}
pub(super) fn retain_layer_tensor(
    layer_index: usize,
    tensor_role: &'static str,
    tensor: &MlxArray,
) -> Result<MlxArray, PersistentPromptCacheStateBridgeError> {
    tensor.retain().map_err(
        |source| PersistentPromptCacheStateBridgeError::RetainLayerTensor {
            layer_index,
            tensor_role,
            source,
        },
    )
}
pub(super) fn retain_block_tensor(
    layer_index: usize,
    tensor_name: &str,
    tensor: &MlxArray,
) -> Result<MlxArray, PersistentPromptCacheStateBridgeError> {
    tensor.retain().map_err(
        |source| PersistentPromptCacheStateBridgeError::RetainBlockTensor {
            layer_index,
            tensor_name: tensor_name.to_owned(),
            source,
        },
    )
}
/// Persistent prompt-cache block extraction or restoration could not bridge
/// to the live in-memory request decoder state.
#[derive(Debug, thiserror::Error)]
pub enum PersistentPromptCacheStateBridgeError {
    #[error(
        "qwen3.5-moe persistent prompt-cache block range [{block_start_tokens}, {block_end_tokens}) is invalid"
    )]
    InvalidBlockRange {
        block_start_tokens: usize,
        block_end_tokens: usize,
    },
    #[error("qwen3.5-moe request decoder state is missing layer {layer_index}")]
    MissingLayer { layer_index: usize },
    #[error("qwen3.5-moe request decoder layer {layer_index} is missing its {tensor_role} tensor")]
    MissingLayerTensor {
        layer_index: usize,
        tensor_role: &'static str,
    },
    #[error(
        "qwen3.5-moe persistent prompt-cache block tensor {tensor_name} for layer {layer_index} is missing"
    )]
    MissingBlockTensor {
        layer_index: usize,
        tensor_name: String,
    },
    #[error(
        "qwen3.5-moe request decoder layer {layer_index} has unknown attention tensor role {tensor_role}"
    )]
    UnknownAttentionTensorRole {
        layer_index: usize,
        tensor_role: &'static str,
    },
    #[error(
        "qwen3.5-moe request decoder layer {layer_index} {tensor_role} tensor has invalid shape {actual_shape:?}"
    )]
    InvalidLayerTensorShape {
        layer_index: usize,
        tensor_role: &'static str,
        actual_shape: Vec<i32>,
    },
    #[error(
        "qwen3.5-moe request decoder layer {layer_index} {tensor_role} tensor has only {available_tokens} tokens, cannot extract through {requested_end_tokens}"
    )]
    BlockRangeExceedsLayerTensor {
        layer_index: usize,
        tensor_role: &'static str,
        requested_end_tokens: i32,
        available_tokens: i32,
    },
    #[error("failed to slice qwen3.5-moe request decoder layer {layer_index} {tensor_role} tensor")]
    SliceLayerTensor {
        layer_index: usize,
        tensor_role: &'static str,
        #[source]
        source: MlxRuntimeError,
    },
    #[error(
        "failed to retain qwen3.5-moe request decoder layer {layer_index} {tensor_role} tensor"
    )]
    RetainLayerTensor {
        layer_index: usize,
        tensor_role: &'static str,
        #[source]
        source: MlxRuntimeError,
    },
    #[error(
        "failed to retain qwen3.5-moe persistent prompt-cache block tensor {tensor_name} for layer {layer_index}"
    )]
    RetainBlockTensor {
        layer_index: usize,
        tensor_name: String,
        #[source]
        source: MlxRuntimeError,
    },
    #[error(
        "qwen3.5-moe persistent prompt-cache KV restore destination token count {restored_token_count} is invalid"
    )]
    InvalidRestoredSequenceTokenCount { restored_token_count: usize },
    #[error(
        "qwen3.5-moe persistent prompt-cache KV block at token offset {sequence_start_tokens} does not fit the restored destination of {destination_token_count} tokens"
    )]
    KvBlockDoesNotFitRestoreDestination {
        sequence_start_tokens: usize,
        destination_token_count: usize,
    },
    #[error(
        "qwen3.5-moe persistent prompt-cache KV block layer {layer_index} {tensor_role} shape {actual_shape:?} does not match restore destination {expected_shape:?}"
    )]
    KvBlockShapeDoesNotMatchRestoreDestination {
        layer_index: usize,
        tensor_role: &'static str,
        actual_shape: Vec<i32>,
        expected_shape: Vec<i32>,
    },
    #[error(
        "failed to allocate the qwen3.5-moe persistent prompt-cache KV restore destination for layer {layer_index}"
    )]
    AllocateRestoreDestination {
        layer_index: usize,
        #[source]
        source: MlxRuntimeError,
    },
    #[error(
        "failed to write qwen3.5-moe persistent prompt-cache block tensor {tensor_name} into the restore destination for layer {layer_index}"
    )]
    WriteRestoreDestination {
        layer_index: usize,
        tensor_name: String,
        #[source]
        source: MlxRuntimeError,
    },
    #[error("failed to restore the in-memory full-attention KV state for layer {layer_index}")]
    RestoreFullAttentionState {
        layer_index: usize,
        #[source]
        source: MlxRuntimeError,
    },
    #[error(
        "restored sparse target state layer {layer_index} has {actual_token_count} rows; expected {expected_token_count}"
    )]
    InconsistentSpeculativePrefillTargetTokenCount {
        layer_index: usize,
        expected_token_count: usize,
        actual_token_count: i32,
    },
    #[error("failed to materialize restored qwen3.5-moe persistent prompt-cache state")]
    EvaluateRestoredPersistentPromptCacheState(#[source] MlxRuntimeError),
}
