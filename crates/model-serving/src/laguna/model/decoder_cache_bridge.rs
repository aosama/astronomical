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

    /// Begins the incremental restore: allocates each append-only layer's
    /// final-length destination from the first block and writes it. Rotating
    /// layers ignore sequence blocks; they restore from the boundary snapshot
    /// at finish time.
    pub fn begin_incremental_cache_block_restore(
        &mut self,
        runtime: &MlxRuntime,
        first_block_tensors: &mut HashMap<String, MlxArray>,
        restored_token_count: usize,
    ) -> Result<(), LagunaExecutionError> {
        let restored_token_count = restored_token_count_i32(restored_token_count)?;
        for (layer_index, layer_state) in self.layers.iter_mut().enumerate() {
            if let LagunaLayerCacheState::AppendOnly(attention) = layer_state {
                let first_block_keys = take_block_tensor(first_block_tensors, layer_index, "keys")?;
                let first_block_values =
                    take_block_tensor(first_block_tensors, layer_index, "values")?;
                // Headroom stays zero: Laguna restores exact-length slabs.
                attention.begin_incremental_block_restore(
                    runtime,
                    first_block_keys,
                    first_block_values,
                    restored_token_count,
                    0,
                )?;
                evaluate_restored_pair(runtime, attention.keys_state(), attention.values_state())?;
            }
        }
        Ok(())
    }

    /// Writes one sequence block into every append-only layer's
    /// preallocated destination at the block's token range.
    pub fn absorb_incremental_cache_block(
        &mut self,
        runtime: &MlxRuntime,
        block_tensors: &mut HashMap<String, MlxArray>,
        block_start_tokens: usize,
    ) -> Result<(), LagunaExecutionError> {
        let block_start_tokens = restored_token_count_i32(block_start_tokens)?;
        for (layer_index, layer_state) in self.layers.iter_mut().enumerate() {
            if let LagunaLayerCacheState::AppendOnly(attention) = layer_state {
                let block_keys = take_block_tensor(block_tensors, layer_index, "keys")?;
                let block_values = take_block_tensor(block_tensors, layer_index, "values")?;
                attention.absorb_incremental_restore_block(
                    runtime,
                    block_keys,
                    block_values,
                    block_start_tokens,
                )?;
                evaluate_restored_pair(runtime, attention.keys_state(), attention.values_state())?;
            }
        }
        Ok(())
    }

    /// Completes the restore: append-only layers record the restored offset,
    /// and rotating layers restore from the newest boundary snapshot.
    pub fn finish_incremental_cache_block_restore(
        &mut self,
        runtime: &MlxRuntime,
        restored_token_count: usize,
        boundary_snapshot: &mut HashMap<String, MlxArray>,
    ) -> Result<(), LagunaExecutionError> {
        let restored_token_count = restored_token_count_i32(restored_token_count)?;
        for (layer_index, layer_state) in self.layers.iter_mut().enumerate() {
            match layer_state {
                LagunaLayerCacheState::AppendOnly(attention) => {
                    attention.finish_incremental_block_restore(restored_token_count)?;
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

fn restored_token_count_i32(token_count: usize) -> Result<i32, LagunaExecutionError> {
    i32::try_from(token_count).map_err(|_| {
        LagunaExecutionError::invalid_geometry("restored token count exceeds the i32 range")
    })
}

fn take_block_tensor(
    block_tensors: &mut HashMap<String, MlxArray>,
    layer_index: usize,
    tensor_role: &'static str,
) -> Result<MlxArray, LagunaExecutionError> {
    let tensor_name = format!("layer_{layer_index}_attention.{tensor_role}");
    block_tensors
        .remove(&tensor_name)
        .ok_or_else(|| LagunaExecutionError::RuntimeOperation {
            description: format!("a sequence cache block is missing a tensor: {tensor_name}"),
        })
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
