//! Incremental SSD prompt-cache block restore for append-only KV states.
//!
//! The restored sequence length is known before the first block is loaded.
//! `begin` allocates that final-length destination once, `absorb` writes each
//! block into its token range with `slice_update`, and `finish` checks the
//! restored offset. MLX donates a uniquely held destination buffer, so each
//! write copies only the incoming block instead of recopying the growing
//! prefix — the restore peak stays at destination + one block, never
//! destination + every block.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use super::append_only_attention_state::{FullAttentionKeyValueState, STATE_DIMENSION_TOKEN_AXIS};
use super::quantized_full_attention_state::{QuantizedFullAttentionKeyValueState, QuantizedSlab};

const INCREMENTAL_RESTORE_OPERATION: &str =
    "restore the in-memory KV state from persistent prompt-cache blocks";

impl FullAttentionKeyValueState {
    /// Allocates the final-length K/V destination from the first block's
    /// geometry, including `growth_headroom_tokens` of zero capacity beyond
    /// the restored length, and writes the first block at the sequence start.
    ///
    /// The headroom lets the first post-restore update splice into the slab
    /// instead of recopying the complete restored prefix.
    pub fn begin_incremental_block_restore(
        &mut self,
        runtime: &MlxRuntime,
        first_block_keys: MlxArray,
        first_block_values: MlxArray,
        restored_token_count: i32,
        growth_headroom_tokens: i32,
    ) -> Result<(), MlxRuntimeError> {
        validate_block_pair(&first_block_keys, &first_block_values)?;
        validate_restore_bounds(restored_token_count, growth_headroom_tokens)?;
        let first_block_token_count = first_block_keys.shape()[STATE_DIMENSION_TOKEN_AXIS];
        let destination_token_count = restored_token_count
            .checked_add(growth_headroom_tokens)
            .ok_or_else(|| incremental_restore_error("restore destination size overflowed"))?;
        let mut destination_keys =
            zeros_restore_destination(runtime, &first_block_keys, destination_token_count)?;
        let mut destination_values =
            zeros_restore_destination(runtime, &first_block_values, destination_token_count)?;
        destination_keys =
            write_block_into_token_range(runtime, &destination_keys, &first_block_keys, 0)?;
        destination_values =
            write_block_into_token_range(runtime, &destination_values, &first_block_values, 0)?;
        self.keys = Some(destination_keys);
        self.values = Some(destination_values);
        self.offset_tokens = first_block_token_count;
        Ok(())
    }

    /// Writes one block into the preallocated destination at its token range.
    /// Blocks must arrive in order and contiguously: `block_start_tokens`
    /// must equal the tokens written so far.
    pub fn absorb_incremental_restore_block(
        &mut self,
        runtime: &MlxRuntime,
        block_keys: MlxArray,
        block_values: MlxArray,
        block_start_tokens: i32,
    ) -> Result<(), MlxRuntimeError> {
        validate_block_pair(&block_keys, &block_values)?;
        if block_start_tokens != self.offset_tokens {
            return Err(incremental_restore_error(
                "restore blocks must arrive in order at the written offset",
            ));
        }
        let Some(destination_keys) = self.keys.take() else {
            return Err(incremental_restore_error(
                "incremental restore began without a K/V destination",
            ));
        };
        let Some(destination_values) = self.values.take() else {
            self.keys = Some(destination_keys);
            return Err(incremental_restore_error(
                "incremental restore began without a K/V destination",
            ));
        };
        let written_keys = write_block_into_token_range(
            runtime,
            &destination_keys,
            &block_keys,
            block_start_tokens,
        );
        let written_values = write_block_into_token_range(
            runtime,
            &destination_values,
            &block_values,
            block_start_tokens,
        );
        match (written_keys, written_values) {
            (Ok(written_keys), Ok(written_values)) => {
                drop(destination_keys);
                drop(destination_values);
                self.keys = Some(written_keys);
                self.values = Some(written_values);
                self.offset_tokens = block_start_tokens
                    .checked_add(block_keys.shape()[STATE_DIMENSION_TOKEN_AXIS])
                    .ok_or_else(|| {
                        incremental_restore_error("restore block token range overflowed")
                    })?;
                Ok(())
            }
            (keys_outcome, values_outcome) => {
                self.keys = Some(destination_keys);
                self.values = Some(destination_values);
                keys_outcome.and(values_outcome).map(|_| ())
            }
        }
    }

    /// Completes the restore by checking the destination holds exactly the
    /// restored prefix. The slab keeps its growth headroom capacity.
    pub fn finish_incremental_block_restore(
        &mut self,
        restored_token_count: i32,
    ) -> Result<(), MlxRuntimeError> {
        if self.offset_tokens != restored_token_count {
            return Err(incremental_restore_error(
                "incremental restore must absorb every restored token before finishing",
            ));
        }
        Ok(())
    }

    /// Physical growth headroom configured for this state's slabs.
    #[must_use]
    pub fn growth_headroom_tokens(&self) -> i32 {
        self.full_attention_kv_state_growth_tokens
    }
}

impl QuantizedFullAttentionKeyValueState {
    /// Quantizes the first bfloat16 block, allocates final-length packed,
    /// scale, and bias destinations including `growth_headroom_tokens`, and
    /// writes the first block at the sequence start.
    ///
    /// Affine groups never straddle a token boundary, so quantizing each
    /// block separately produces the same slabs as quantizing the whole
    /// concatenated prefix.
    pub fn begin_incremental_bf16_block_restore(
        &mut self,
        runtime: &MlxRuntime,
        first_block_keys: MlxArray,
        first_block_values: MlxArray,
        restored_token_count: i32,
        growth_headroom_tokens: i32,
    ) -> Result<(), MlxRuntimeError> {
        validate_block_pair(&first_block_keys, &first_block_values)?;
        validate_restore_bounds(restored_token_count, growth_headroom_tokens)?;
        let first_block_token_count = first_block_keys.shape()[STATE_DIMENSION_TOKEN_AXIS];
        let (keys_slab, values_slab) = quantize_block_pair(
            runtime,
            self.group_size(),
            self.bits(),
            &first_block_keys,
            &first_block_values,
        )?;
        let destination_capacity_tokens = restored_token_count
            .checked_add(growth_headroom_tokens)
            .ok_or_else(|| incremental_restore_error("restore destination size overflowed"))?;
        let mut keys_destination =
            zeros_slab_destination(runtime, &keys_slab, destination_capacity_tokens)?;
        let mut values_destination =
            zeros_slab_destination(runtime, &values_slab, destination_capacity_tokens)?;
        write_slab_into_token_range(runtime, &mut keys_destination, &keys_slab, 0)?;
        write_slab_into_token_range(runtime, &mut values_destination, &values_slab, 0)?;
        self.keys = Some(keys_destination);
        self.values = Some(values_destination);
        self.offset_tokens = first_block_token_count;
        Ok(())
    }

    /// Quantizes one bfloat16 block and writes its packed, scale, and bias
    /// slabs into the preallocated destinations at the block's token range.
    /// Blocks must arrive in order and contiguously.
    pub fn absorb_incremental_bf16_restore_block(
        &mut self,
        runtime: &MlxRuntime,
        block_keys: MlxArray,
        block_values: MlxArray,
        block_start_tokens: i32,
    ) -> Result<(), MlxRuntimeError> {
        validate_block_pair(&block_keys, &block_values)?;
        if block_start_tokens != self.offset_tokens {
            return Err(incremental_restore_error(
                "restore blocks must arrive in order at the written offset",
            ));
        }
        let (keys_slab, values_slab) = quantize_block_pair(
            runtime,
            self.group_size(),
            self.bits(),
            &block_keys,
            &block_values,
        )?;
        let Some(mut keys_destination) = self.keys.take() else {
            return Err(incremental_restore_error(
                "incremental restore began without a K/V destination",
            ));
        };
        let Some(mut values_destination) = self.values.take() else {
            self.keys = Some(keys_destination);
            return Err(incremental_restore_error(
                "incremental restore began without a K/V destination",
            ));
        };
        let keys_outcome = write_slab_into_token_range(
            runtime,
            &mut keys_destination,
            &keys_slab,
            block_start_tokens,
        );
        let values_outcome = write_slab_into_token_range(
            runtime,
            &mut values_destination,
            &values_slab,
            block_start_tokens,
        );
        match (keys_outcome, values_outcome) {
            (Ok(()), Ok(())) => {
                self.keys = Some(keys_destination);
                self.values = Some(values_destination);
                self.offset_tokens = block_start_tokens
                    .checked_add(block_keys.shape()[STATE_DIMENSION_TOKEN_AXIS])
                    .ok_or_else(|| {
                        incremental_restore_error("restore block token range overflowed")
                    })?;
                Ok(())
            }
            (keys_outcome, values_outcome) => {
                self.keys = Some(keys_destination);
                self.values = Some(values_destination);
                keys_outcome.and(values_outcome)
            }
        }
    }

    /// Completes the restore by checking the slabs hold exactly the restored
    /// prefix. The slabs keep their growth headroom capacity.
    pub fn finish_incremental_bf16_block_restore(
        &mut self,
        restored_token_count: i32,
    ) -> Result<(), MlxRuntimeError> {
        if self.offset_tokens != restored_token_count {
            return Err(incremental_restore_error(
                "incremental restore must absorb every restored token before finishing",
            ));
        }
        Ok(())
    }

    /// Physical growth headroom configured for this state's slabs.
    #[must_use]
    pub fn growth_headroom_tokens(&self) -> i32 {
        self.full_attention_kv_state_growth_tokens
    }
}

fn quantize_block_pair(
    runtime: &MlxRuntime,
    group_size: i32,
    bits: i32,
    block_keys: &MlxArray,
    block_values: &MlxArray,
) -> Result<(QuantizedSlab, QuantizedSlab), MlxRuntimeError> {
    let (keys_packed, keys_scales, keys_biases) =
        runtime.quantize_affine(block_keys, group_size, bits)?;
    let (values_packed, values_scales, values_biases) =
        runtime.quantize_affine(block_values, group_size, bits)?;
    Ok((
        QuantizedSlab {
            packed: keys_packed,
            scales: keys_scales,
            biases: keys_biases,
        },
        QuantizedSlab {
            packed: values_packed,
            scales: values_scales,
            biases: values_biases,
        },
    ))
}

fn validate_block_pair(
    block_keys: &MlxArray,
    block_values: &MlxArray,
) -> Result<(), MlxRuntimeError> {
    let block_key_shape = block_keys.shape();
    if block_key_shape.len() != 4
        || block_key_shape != block_values.shape()
        || block_key_shape[STATE_DIMENSION_TOKEN_AXIS] <= 0
    {
        return Err(incremental_restore_error(
            "restored K and V blocks must have identical rank-four nonempty shapes",
        ));
    }
    Ok(())
}

fn validate_restore_bounds(
    restored_token_count: i32,
    growth_headroom_tokens: i32,
) -> Result<(), MlxRuntimeError> {
    if restored_token_count <= 0 {
        return Err(incremental_restore_error(
            "restored token count must be positive",
        ));
    }
    if growth_headroom_tokens < 0 {
        return Err(incremental_restore_error(
            "restore growth headroom must not be negative",
        ));
    }
    Ok(())
}

fn zeros_restore_destination(
    runtime: &MlxRuntime,
    prototype: &MlxArray,
    destination_token_count: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let mut destination_shape = prototype.shape();
    destination_shape[STATE_DIMENSION_TOKEN_AXIS] = destination_token_count;
    runtime.zeros(&destination_shape, prototype.dtype())
}

fn zeros_slab_destination(
    runtime: &MlxRuntime,
    prototype: &QuantizedSlab,
    destination_token_count: i32,
) -> Result<QuantizedSlab, MlxRuntimeError> {
    Ok(QuantizedSlab {
        packed: zeros_restore_destination(runtime, &prototype.packed, destination_token_count)?,
        scales: zeros_restore_destination(runtime, &prototype.scales, destination_token_count)?,
        biases: zeros_restore_destination(runtime, &prototype.biases, destination_token_count)?,
    })
}

fn write_block_into_token_range(
    runtime: &MlxRuntime,
    destination: &MlxArray,
    incoming: &MlxArray,
    block_start_tokens: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let destination_shape = destination.shape();
    let incoming_shape = incoming.shape();
    if incoming_shape.len() != 4
        || incoming_shape[0] != destination_shape[0]
        || incoming_shape[1] != destination_shape[1]
        || incoming_shape[3] != destination_shape[3]
    {
        return Err(incremental_restore_error(
            "restored block geometry must match the restore destination",
        ));
    }
    let block_token_count = incoming_shape[STATE_DIMENSION_TOKEN_AXIS];
    if block_start_tokens < 0 {
        return Err(incremental_restore_error(
            "restore block start token must not be negative",
        ));
    }
    let block_end_tokens = block_start_tokens
        .checked_add(block_token_count)
        .ok_or_else(|| incremental_restore_error("restore block token range overflowed"))?;
    if block_end_tokens > destination_shape[STATE_DIMENSION_TOKEN_AXIS] {
        return Err(incremental_restore_error(
            "restore block does not fit the restore destination",
        ));
    }
    let mut slice_starts = vec![0_i32; destination_shape.len()];
    slice_starts[STATE_DIMENSION_TOKEN_AXIS] = block_start_tokens;
    let mut slice_stops = destination_shape;
    slice_stops[STATE_DIMENSION_TOKEN_AXIS] = block_end_tokens;
    let slice_strides = vec![1_i32; slice_starts.len()];
    runtime.slice_update(
        destination,
        incoming,
        &slice_starts,
        &slice_stops,
        &slice_strides,
    )
}

fn write_slab_into_token_range(
    runtime: &MlxRuntime,
    destination: &mut QuantizedSlab,
    incoming: &QuantizedSlab,
    block_start_tokens: i32,
) -> Result<(), MlxRuntimeError> {
    destination.packed = write_block_into_token_range(
        runtime,
        &destination.packed,
        &incoming.packed,
        block_start_tokens,
    )?;
    destination.scales = write_block_into_token_range(
        runtime,
        &destination.scales,
        &incoming.scales,
        block_start_tokens,
    )?;
    destination.biases = write_block_into_token_range(
        runtime,
        &destination.biases,
        &incoming.biases,
        block_start_tokens,
    )?;
    Ok(())
}

fn incremental_restore_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: INCREMENTAL_RESTORE_OPERATION,
        description: description.to_owned(),
    }
}
