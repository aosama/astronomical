//! SSD prompt-cache block restoration for append-only KV states, one
//! concatenation per tensor.
//!
//! Every restored block slice is concatenated along the token axis exactly
//! once, O(restored tokens). The per-block `slice_update` assembly this
//! replaces recopied the whole final-length destination for every block,
//! O(tokens × blocks), and each materialization forced a GPU synchronization.
//! Restored blocks are disk-backed lazy handles loaded through the retained
//! file-descriptor safetensors reader, so holding the complete block set
//! costs handles rather than payload: the bytes cross disk → destination
//! during the single concat materialization.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::append_only_attention_state::{FullAttentionKeyValueState, STATE_DIMENSION_TOKEN_AXIS};
use super::quantized_full_attention_state::{QuantizedFullAttentionKeyValueState, QuantizedSlab};
use astronomical_mlx_c_rust::MlxArray;

const RESTORE_OPERATION: &str =
    "restore the in-memory KV state from persistent prompt-cache blocks";

impl FullAttentionKeyValueState {
    /// Seats the restored K/V by concatenating every block slice along the
    /// token axis once, plus `growth_headroom_tokens` of zero capacity beyond
    /// the restored tokens so the first post-restore append splices into the
    /// slab instead of growing it.
    pub fn restore_from_block_slices(
        &mut self,
        runtime: &MlxRuntime,
        block_keys: &[&MlxArray],
        block_values: &[&MlxArray],
        restored_token_count: i32,
        growth_headroom_tokens: i32,
    ) -> Result<(), MlxRuntimeError> {
        validate_restore_bounds(restored_token_count, growth_headroom_tokens)?;
        let keys_slab = concatenated_block_slab(
            runtime,
            block_keys,
            restored_token_count,
            growth_headroom_tokens,
        )?;
        let values_slab = concatenated_block_slab(
            runtime,
            block_values,
            restored_token_count,
            growth_headroom_tokens,
        )?;
        self.keys = Some(keys_slab);
        self.values = Some(values_slab);
        self.offset_tokens = restored_token_count;
        Ok(())
    }

    /// Physical growth headroom configured for this state's slabs.
    #[must_use]
    pub fn growth_headroom_tokens(&self) -> i32 {
        self.full_attention_kv_state_growth_tokens
    }
}

impl QuantizedFullAttentionKeyValueState {
    /// Seats the restored K/V by concatenating every bfloat16 block slice
    /// once, quantizing the whole restored prefix in a single affine pass —
    /// affine groups never straddle a token boundary, so this matches
    /// per-block quantization — and appending `growth_headroom_tokens` of
    /// zero slab capacity beyond the restored tokens.
    pub fn restore_from_bf16_block_slices(
        &mut self,
        runtime: &MlxRuntime,
        block_keys: &[&MlxArray],
        block_values: &[&MlxArray],
        restored_token_count: i32,
        growth_headroom_tokens: i32,
    ) -> Result<(), MlxRuntimeError> {
        validate_restore_bounds(restored_token_count, growth_headroom_tokens)?;
        let restored_keys = concatenated_block_tensor(runtime, block_keys, restored_token_count)?;
        let restored_values =
            concatenated_block_tensor(runtime, block_values, restored_token_count)?;
        let (keys_packed, keys_scales, keys_biases) =
            runtime.quantize_affine(&restored_keys, self.group_size(), self.bits())?;
        let (values_packed, values_scales, values_biases) =
            runtime.quantize_affine(&restored_values, self.group_size(), self.bits())?;
        self.keys = Some(QuantizedSlab {
            packed: zero_padded_slab(runtime, keys_packed, growth_headroom_tokens)?,
            scales: zero_padded_slab(runtime, keys_scales, growth_headroom_tokens)?,
            biases: zero_padded_slab(runtime, keys_biases, growth_headroom_tokens)?,
        });
        self.values = Some(QuantizedSlab {
            packed: zero_padded_slab(runtime, values_packed, growth_headroom_tokens)?,
            scales: zero_padded_slab(runtime, values_scales, growth_headroom_tokens)?,
            biases: zero_padded_slab(runtime, values_biases, growth_headroom_tokens)?,
        });
        self.offset_tokens = restored_token_count;
        Ok(())
    }

    /// Physical growth headroom configured for this state's slabs.
    #[must_use]
    pub fn growth_headroom_tokens(&self) -> i32 {
        self.full_attention_kv_state_growth_tokens
    }
}

/// Concatenates block slices plus a zero headroom tail into one
/// capacity-sized slab.
fn concatenated_block_slab(
    runtime: &MlxRuntime,
    block_tensors: &[&MlxArray],
    restored_token_count: i32,
    growth_headroom_tokens: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let restored_tensor = concatenated_block_tensor(runtime, block_tensors, restored_token_count)?;
    if growth_headroom_tokens == 0 {
        return Ok(restored_tensor);
    }
    let restored_shape = restored_tensor.shape();
    let mut headroom_shape = restored_shape.clone();
    headroom_shape[STATE_DIMENSION_TOKEN_AXIS] = growth_headroom_tokens;
    let headroom = runtime.zeros(&headroom_shape, restored_tensor.dtype())?;
    runtime.concatenate_axis(
        &[&restored_tensor, &headroom],
        STATE_DIMENSION_TOKEN_AXIS as i32,
    )
}

/// Concatenates block slices along the token axis and validates the result
/// against the restored token count at rank four.
fn concatenated_block_tensor(
    runtime: &MlxRuntime,
    block_tensors: &[&MlxArray],
    restored_token_count: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    if block_tensors.is_empty() {
        return Err(restore_error("restore requires at least one block slice"));
    }
    let block_references: Vec<&MlxArray> = block_tensors.iter().copied().collect();
    let concatenated =
        runtime.concatenate_axis(&block_references, STATE_DIMENSION_TOKEN_AXIS as i32)?;
    let concatenated_shape = concatenated.shape();
    if concatenated_shape.len() != 4
        || concatenated_shape[STATE_DIMENSION_TOKEN_AXIS] != restored_token_count
    {
        return Err(restore_error(
            "restored blocks must concatenate to the restored token count at rank four",
        ));
    }
    Ok(concatenated)
}

/// Appends `headroom_tokens` of zero slab capacity beyond the restored
/// prefix, matching the zero-filled destinations the quantized restore
/// previously preallocated.
fn zero_padded_slab(
    runtime: &MlxRuntime,
    slab: MlxArray,
    headroom_tokens: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    if headroom_tokens == 0 {
        return Ok(slab);
    }
    let slab_shape = slab.shape();
    let mut headroom_shape = slab_shape.clone();
    headroom_shape[STATE_DIMENSION_TOKEN_AXIS] = headroom_tokens;
    let headroom = runtime.zeros(&headroom_shape, slab.dtype())?;
    runtime.concatenate_axis(&[&slab, &headroom], STATE_DIMENSION_TOKEN_AXIS as i32)
}

fn validate_restore_bounds(
    restored_token_count: i32,
    growth_headroom_tokens: i32,
) -> Result<(), MlxRuntimeError> {
    if restored_token_count <= 0 {
        return Err(restore_error("restored token count must be positive"));
    }
    if growth_headroom_tokens < 0 {
        return Err(restore_error(
            "restore growth headroom must not be negative",
        ));
    }
    Ok(())
}

fn restore_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: RESTORE_OPERATION,
        description: description.to_owned(),
    }
}
