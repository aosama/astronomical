//! Extracts and restores Laguna decoder state for persistent prompt-cache blocks.

use std::collections::HashMap;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use super::super::error::LagunaExecutionError;
use super::{LagunaDecoderState, LagunaLayerCacheState};

impl LagunaDecoderState {
    /// Extracts one append-only sequence block plus the current rotating snapshot.
    pub fn extract_cache_block_tensors(
        &self,
        runtime: &MlxRuntime,
        block_start_tokens: usize,
        block_end_tokens: usize,
    ) -> Result<(HashMap<String, MlxArray>, HashMap<String, MlxArray>), LagunaExecutionError> {
        let mut sequence_state_tensors = HashMap::new();
        let mut boundary_state_tensors = HashMap::new();
        for (layer_index, layer_state) in self.layers.iter().enumerate() {
            match layer_state {
                LagunaLayerCacheState::AppendOnly(attention) => {
                    let keys = attention.keys_state().ok_or_else(|| {
                        LagunaExecutionError::invalid_geometry(
                            "append-only cache is missing keys during capture",
                        )
                    })?;
                    let values = attention.values_state().ok_or_else(|| {
                        LagunaExecutionError::invalid_geometry(
                            "append-only cache is missing values during capture",
                        )
                    })?;
                    sequence_state_tensors.insert(
                        format!("layer_{layer_index}_attention.keys"),
                        slice_token_range(runtime, keys, block_start_tokens, block_end_tokens)?,
                    );
                    sequence_state_tensors.insert(
                        format!("layer_{layer_index}_attention.values"),
                        slice_token_range(runtime, values, block_start_tokens, block_end_tokens)?,
                    );
                }
                LagunaLayerCacheState::Rotating(attention) => {
                    let keys = attention.keys().ok_or_else(|| {
                        LagunaExecutionError::invalid_geometry(
                            "rotating cache is missing keys during capture",
                        )
                    })?;
                    let values = attention.values().ok_or_else(|| {
                        LagunaExecutionError::invalid_geometry(
                            "rotating cache is missing values during capture",
                        )
                    })?;
                    boundary_state_tensors.insert(
                        format!("layer_{layer_index}_attention.keys"),
                        pad_committed_tokens_to_window(runtime, keys, attention.window_size())?,
                    );
                    boundary_state_tensors.insert(
                        format!("layer_{layer_index}_attention.values"),
                        pad_committed_tokens_to_window(runtime, values, attention.window_size())?,
                    );
                    boundary_state_tensors.insert(
                        format!("layer_{layer_index}_attention.absolute_position"),
                        runtime.array_from_f32(&[attention.absolute_position() as f32], &[1])?,
                    );
                    boundary_state_tensors.insert(
                        format!("layer_{layer_index}_attention.ring_write_index"),
                        runtime.array_from_f32(&[attention.ring_write_index() as f32], &[1])?,
                    );
                }
            }
        }
        Ok((sequence_state_tensors, boundary_state_tensors))
    }

    /// Restores append-only layers from concatenated blocks and rotating layers from the newest snapshot.
    pub fn restore_from_cache_blocks(
        &mut self,
        runtime: &MlxRuntime,
        sequence_blocks: &mut [HashMap<String, MlxArray>],
        boundary_snapshot: &mut HashMap<String, MlxArray>,
    ) -> Result<(), LagunaExecutionError> {
        for (layer_index, layer_state) in self.layers.iter_mut().enumerate() {
            match layer_state {
                LagunaLayerCacheState::AppendOnly(attention) => {
                    let restored_keys = concatenate_taken_blocks(
                        runtime,
                        sequence_blocks,
                        &format!("layer_{layer_index}_attention.keys"),
                    )?;
                    let restored_values = concatenate_taken_blocks(
                        runtime,
                        sequence_blocks,
                        &format!("layer_{layer_index}_attention.values"),
                    )?;
                    attention.restore_from_blocks(restored_keys, restored_values)?;
                    evaluate_restored_pair(
                        runtime,
                        attention.keys_state(),
                        attention.values_state(),
                    )?;
                }
                LagunaLayerCacheState::Rotating(attention) => {
                    let persisted_keys = boundary_snapshot
                        .remove(&format!("layer_{layer_index}_attention.keys"))
                        .ok_or_else(|| {
                            LagunaExecutionError::invalid_geometry(
                                "rotating restore is missing keys",
                            )
                        })?;
                    let persisted_values = boundary_snapshot
                        .remove(&format!("layer_{layer_index}_attention.values"))
                        .ok_or_else(|| {
                            LagunaExecutionError::invalid_geometry(
                                "rotating restore is missing values",
                            )
                        })?;
                    let absolute_position = take_scalar_counter(
                        boundary_snapshot,
                        &format!("layer_{layer_index}_attention.absolute_position"),
                    )?;
                    let ring_write_index = take_scalar_counter(
                        boundary_snapshot,
                        &format!("layer_{layer_index}_attention.ring_write_index"),
                    )?;
                    let live_token_count = absolute_position.min(attention.window_size());
                    attention.restore_from_blocks(
                        slice_leading_tokens(runtime, &persisted_keys, live_token_count)?,
                        slice_leading_tokens(runtime, &persisted_values, live_token_count)?,
                        absolute_position,
                        ring_write_index,
                    )?;
                    evaluate_restored_pair(runtime, attention.keys(), attention.values())?;
                }
            }
        }
        Ok(())
    }
}

fn slice_token_range(
    runtime: &MlxRuntime,
    tensor: &MlxArray,
    start_tokens: usize,
    end_tokens: usize,
) -> Result<MlxArray, LagunaExecutionError> {
    let shape = tensor.shape();
    if shape.len() != 4 {
        return Err(LagunaExecutionError::invalid_geometry(
            "cache tensors must have rank four",
        ));
    }
    let start = i32::try_from(start_tokens).unwrap_or(i32::MAX);
    let end = i32::try_from(end_tokens).unwrap_or(i32::MAX);
    Ok(runtime.slice(
        tensor,
        &[0, 0, start, 0],
        &[shape[0], shape[1], end, shape[3]],
        &[1, 1, 1, 1],
    )?)
}

fn pad_committed_tokens_to_window(
    runtime: &MlxRuntime,
    tensor: &MlxArray,
    window_size: i32,
) -> Result<MlxArray, LagunaExecutionError> {
    let shape = tensor.shape();
    if shape.len() != 4 {
        return Err(LagunaExecutionError::invalid_geometry(
            "rotating tensors must have rank four",
        ));
    }
    let committed_token_count = shape[2];
    if committed_token_count == window_size {
        return Ok(tensor.retain()?);
    }
    if committed_token_count > window_size {
        return Ok(runtime.slice(
            tensor,
            &[0, 0, committed_token_count - window_size, 0],
            &[shape[0], shape[1], committed_token_count, shape[3]],
            &[1, 1, 1, 1],
        )?);
    }
    let pad_token_count = window_size - committed_token_count;
    let padding = runtime.zeros(
        &[shape[0], shape[1], pad_token_count, shape[3]],
        tensor.dtype(),
    )?;
    Ok(runtime.concatenate_axis(&[tensor, &padding], 2)?)
}

fn slice_leading_tokens(
    runtime: &MlxRuntime,
    tensor: &MlxArray,
    live_token_count: i32,
) -> Result<MlxArray, LagunaExecutionError> {
    let shape = tensor.shape();
    if shape.len() != 4 || live_token_count <= 0 {
        return Err(LagunaExecutionError::invalid_geometry(
            "restored rotating tensors must contain at least one token",
        ));
    }
    if shape[2] == live_token_count {
        return Ok(tensor.retain()?);
    }
    Ok(runtime.slice(
        tensor,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], live_token_count, shape[3]],
        &[1, 1, 1, 1],
    )?)
}

fn concatenate_taken_blocks(
    runtime: &MlxRuntime,
    sequence_blocks: &mut [HashMap<String, MlxArray>],
    tensor_name: &str,
) -> Result<MlxArray, LagunaExecutionError> {
    let mut block_tensors = Vec::new();
    for sequence_block in sequence_blocks {
        let block_tensor = sequence_block.remove(tensor_name).ok_or_else(|| {
            LagunaExecutionError::invalid_geometry("a sequence cache block is missing a tensor")
        })?;
        block_tensors.push(block_tensor);
    }
    if block_tensors.len() == 1 {
        return block_tensors.pop().ok_or_else(|| {
            LagunaExecutionError::invalid_geometry("a sequence cache block is missing a tensor")
        });
    }
    let block_tensor_refs = block_tensors.iter().collect::<Vec<_>>();
    Ok(runtime.concatenate_axis(&block_tensor_refs, 2)?)
}

fn take_scalar_counter(
    boundary_snapshot: &mut HashMap<String, MlxArray>,
    tensor_name: &str,
) -> Result<i32, LagunaExecutionError> {
    let counter = boundary_snapshot.remove(tensor_name).ok_or_else(|| {
        LagunaExecutionError::invalid_geometry("a rotating counter tensor is missing")
    })?;
    if counter.dtype() != MlxDtype::Float32 {
        return Err(LagunaExecutionError::invalid_geometry(
            "rotating counters must be float32 scalars",
        ));
    }
    let host_values = counter.to_vec_f32()?;
    Ok(host_values.first().copied().unwrap_or(0.0) as i32)
}

fn evaluate_restored_pair(
    runtime: &MlxRuntime,
    restored_keys: Option<&MlxArray>,
    restored_values: Option<&MlxArray>,
) -> Result<(), LagunaExecutionError> {
    let restored_keys = restored_keys
        .ok_or_else(|| LagunaExecutionError::invalid_geometry("restored cache is missing keys"))?;
    let restored_values = restored_values.ok_or_else(|| {
        LagunaExecutionError::invalid_geometry("restored cache is missing values")
    })?;
    runtime.evaluate_arrays(&[restored_keys, restored_values])?;
    Ok(())
}
