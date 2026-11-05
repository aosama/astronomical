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
use mlx_c_rust::MlxArray;

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
    Ok(runtime.concatenate_axis(
        &[&restored_tensor, &headroom],
        STATE_DIMENSION_TOKEN_AXIS as i32,
    )?)
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
    let block_references: Vec<&MlxArray> = block_tensors.to_vec();
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
