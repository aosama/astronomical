use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

/// Default capacity-growth granularity for the full-attention KV slab. Matches
/// the 256-token capacity-growth policy.
pub const DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS: i32 = 256;

const STATE_DIMENSION_TOKEN_AXIS: usize = 2;
const FULL_ATTENTION_OPERATION: &str = "update the in-memory full-attention KV state";

/// Single owner for one full-attention layer's keys and values.
///
/// Holds both the current storage tensors and the logical token offset, so the
/// attention layer no longer has to re-derive capacity or repeat slice/concat
/// math per step. The empty state allocates nothing; storage is grown lazily on
/// the first `update_and_fetch` call and then in configured token steps.
pub struct FullAttentionKeyValueState {
    keys: Option<MlxArray>,
    values: Option<MlxArray>,
    /// Number of tokens written into the K and V slabs so far. This is the
    /// logical length of the KV state; the physical capacity may be larger
    /// because of step-bounded over-allocation.
    offset_tokens: i32,
    full_attention_kv_state_growth_tokens: i32,
}

/// Physical owner checkpoint for retrying one full-attention update.
pub struct FullAttentionKeyValueStateAllocationCheckpoint {
    keys: Option<MlxArray>,
    values: Option<MlxArray>,
    offset_tokens: i32,
    full_attention_kv_state_growth_tokens: i32,
}

impl FullAttentionKeyValueState {
    /// Creates empty KV state without allocating MLX arrays.
    #[must_use]
    pub fn empty() -> Self {
        Self::empty_with_validated_growth_tokens(DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS)
    }

    pub(crate) fn empty_with_validated_growth_tokens(
        full_attention_kv_state_growth_tokens: i32,
    ) -> Self {
        Self {
            keys: None,
            values: None,
            offset_tokens: 0,
            full_attention_kv_state_growth_tokens,
        }
    }

    /// Creates empty KV state with an explicit capacity-growth granularity.
    pub fn empty_with_growth_tokens(
        full_attention_kv_state_growth_tokens: i32,
    ) -> Result<Self, MlxRuntimeError> {
        if full_attention_kv_state_growth_tokens <= 0 {
            return Err(full_attention_error(
                "full-attention KV-state growth tokens must be positive",
            ));
        }
        Ok(Self::empty_with_validated_growth_tokens(
            full_attention_kv_state_growth_tokens,
        ))
    }

    /// Returns the current physical capacity of the K and V slabs in tokens.
    /// The capacity is the size of the over-allocated slab, not the number of
    /// tokens written so far (use `offset_tokens()` for that).
    #[must_use]
    pub fn capacity_tokens(&self) -> i32 {
        self.keys
            .as_ref()
            .map_or(0, |keys| keys.shape()[STATE_DIMENSION_TOKEN_AXIS])
    }

    /// Returns the number of tokens written into the K and V slabs so far.
    #[must_use]
    pub fn offset_tokens(&self) -> i32 {
        self.offset_tokens
    }

    /// Returns exact additional physical slab capacity required by one update.
    ///
    /// Logical tokens and allocated tokens differ because storage grows in fixed
    /// steps. Admission uses this physical projection so a 128-token update that
    /// allocates a 256-token slab is charged for all 256 tokens.
    pub fn projected_capacity_growth_tokens(
        &self,
        update_token_count: usize,
    ) -> Result<usize, MlxRuntimeError> {
        let update_token_count = i32::try_from(update_token_count).map_err(|_| {
            full_attention_error("full-attention KV update token count exceeds the i32 range")
        })?;
        let projected_capacity_tokens = projected_capacity_tokens(
            self.capacity_tokens(),
            self.keys.is_some(),
            self.offset_tokens,
            update_token_count,
            self.full_attention_kv_state_growth_tokens,
        )?;
        let capacity_growth_tokens = projected_capacity_tokens
            .checked_sub(self.capacity_tokens())
            .ok_or_else(|| full_attention_error("projected KV capacity moved backwards"))?;
        usize::try_from(capacity_growth_tokens).map_err(|_| {
            full_attention_error("full-attention KV capacity growth exceeds the usize range")
        })
    }

    /// Returns exact slab growth across updates that execute as separate forwards.
    pub fn projected_sequential_capacity_growth_tokens(
        &self,
        sequential_update_token_counts: &[usize],
    ) -> Result<usize, MlxRuntimeError> {
        let initial_capacity_tokens = self.capacity_tokens();
        let mut projected_capacity_token_count = initial_capacity_tokens;
        let mut projected_offset_token_count = self.offset_tokens;
        let mut has_projected_storage = self.keys.is_some();
        for sequential_update_token_count in sequential_update_token_counts {
            let sequential_update_token_count = i32::try_from(*sequential_update_token_count)
                .map_err(|_| {
                    full_attention_error(
                        "sequential full-attention KV update token count exceeds the i32 range",
                    )
                })?;
            projected_capacity_token_count = projected_capacity_tokens(
                projected_capacity_token_count,
                has_projected_storage,
                projected_offset_token_count,
                sequential_update_token_count,
                self.full_attention_kv_state_growth_tokens,
            )?;
            projected_offset_token_count = projected_offset_token_count
                .checked_add(sequential_update_token_count)
                .ok_or_else(|| {
                    full_attention_error("sequential full-attention KV offset overflowed")
                })?;
            has_projected_storage = true;
        }
        let capacity_growth_tokens = projected_capacity_token_count
            .checked_sub(initial_capacity_tokens)
            .ok_or_else(|| full_attention_error("projected KV capacity moved backwards"))?;
        usize::try_from(capacity_growth_tokens).map_err(|_| {
            full_attention_error("full-attention KV capacity growth exceeds the usize range")
        })
    }

    /// Grows capacity when needed, writes the new K and V into the slab, and
    /// returns views over `[0..offset_tokens]` for the attention call. This is
    /// the single entry point for capacity decisions and in-place writes; the
    /// attention layer receives the fetched views and runs only the math.
    ///
    /// `previous_token_count` is the rope offset for this step (the number of
    /// tokens already processed before this update). It must match
    /// `offset_tokens()` for normal continuation, but is taken as a parameter so
    /// the caller is explicit about the position being appended at.
    pub fn update_and_fetch(
        &mut self,
        runtime: &MlxRuntime,
        new_keys: &MlxArray,
        new_values: &MlxArray,
        previous_token_count: i32,
    ) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
        let update_shape =
            validate_key_value_update(self, new_keys, new_values, previous_token_count)?;
        let update_token_count = update_shape[STATE_DIMENSION_TOKEN_AXIS];
        let next_token_count = previous_token_count
            .checked_add(update_token_count)
            .ok_or_else(|| full_attention_error("full-attention KV offset overflowed"))?;
        let next_keys = build_updated_storage(
            runtime,
            self.keys.as_ref(),
            new_keys,
            previous_token_count,
            self.full_attention_kv_state_growth_tokens,
        )?;
        let next_values = build_updated_storage(
            runtime,
            self.values.as_ref(),
            new_values,
            previous_token_count,
            self.full_attention_kv_state_growth_tokens,
        )?;
        let active_keys = active_view(runtime, &next_keys, next_token_count)?;
        let active_values = active_view(runtime, &next_values, next_token_count)?;

        // Commit K, V, and offset together only after both storage updates and
        // both active views have been built successfully.
        self.keys = Some(next_keys.retain()?);
        self.values = Some(next_values.retain()?);
        self.offset_tokens = next_token_count;
        Ok((active_keys, active_values))
    }

    /// Read-only access to the current K storage tensor. Used by the SSD
    /// prompt-cache bridge to extract full-attention KV blocks for persistence.
    #[must_use]
    pub fn keys_state(&self) -> Option<&MlxArray> {
        self.keys.as_ref()
    }

    /// Read-only access to the current V storage tensor. Used by the SSD
    /// prompt-cache bridge to extract full-attention KV blocks for persistence.
    #[must_use]
    pub fn values_state(&self) -> Option<&MlxArray> {
        self.values.as_ref()
    }

    #[must_use]
    /// Returns the logical payload bytes owned by the physical key/value slabs.
    pub fn payload_byte_count(&self) -> u64 {
        self.keys
            .as_ref()
            .into_iter()
            .chain(self.values.as_ref())
            .map(|state| state.byte_count() as u64)
            .sum()
    }

    /// Retains the current physical K/V owners and logical offset for a retry.
    pub fn allocation_checkpoint(
        &self,
    ) -> Result<FullAttentionKeyValueStateAllocationCheckpoint, MlxRuntimeError> {
        if self.keys.is_some() != self.values.is_some() {
            return Err(full_attention_error(
                "in-memory K and V storage must both be present or absent",
            ));
        }
        Ok(FullAttentionKeyValueStateAllocationCheckpoint {
            keys: self.keys.as_ref().map(MlxArray::retain).transpose()?,
            values: self.values.as_ref().map(MlxArray::retain).transpose()?,
            offset_tokens: self.offset_tokens,
            full_attention_kv_state_growth_tokens: self.full_attention_kv_state_growth_tokens,
        })
    }

    /// Restores physical K/V owners and the logical offset from a retry checkpoint.
    pub fn restore_allocation_checkpoint(
        &mut self,
        allocation_checkpoint: FullAttentionKeyValueStateAllocationCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        if allocation_checkpoint.keys.is_some() != allocation_checkpoint.values.is_some() {
            return Err(full_attention_error(
                "allocation checkpoint K and V storage must both be present or absent",
            ));
        }
        if allocation_checkpoint.offset_tokens < 0
            || allocation_checkpoint.keys.as_ref().is_some_and(|keys| {
                keys.shape()[STATE_DIMENSION_TOKEN_AXIS] < allocation_checkpoint.offset_tokens
            })
        {
            return Err(full_attention_error(
                "allocation checkpoint offset does not fit its physical K/V storage",
            ));
        }
        if allocation_checkpoint.full_attention_kv_state_growth_tokens <= 0 {
            return Err(full_attention_error(
                "allocation checkpoint KV-state growth tokens must be positive",
            ));
        }
        if let (Some(checkpoint_keys), Some(checkpoint_values)) = (
            allocation_checkpoint.keys.as_ref(),
            allocation_checkpoint.values.as_ref(),
        ) && checkpoint_keys.shape() != checkpoint_values.shape()
        {
            return Err(full_attention_error(
                "allocation checkpoint K and V storage shapes must match",
            ));
        }
        self.keys = allocation_checkpoint.keys;
        self.values = allocation_checkpoint.values;
        self.offset_tokens = allocation_checkpoint.offset_tokens;
        self.full_attention_kv_state_growth_tokens =
            allocation_checkpoint.full_attention_kv_state_growth_tokens;
        Ok(())
    }

    /// Truncates the logical KV offset back to a previous checkpoint.
    ///
    /// This restores the state to an earlier logical length without freeing the
    /// physical slab capacity. The caller must pass a checkpoint offset that was
    /// recorded before the MTP forward that is being rolled back. This is
    /// the single rollback primitive for MTP transactional decode: it does not
    /// re-allocate, it only moves the logical offset back so subsequent updates
    /// overwrite the discarded suffix.
    pub fn truncate_to_offset(
        &mut self,
        checkpoint_offset_tokens: i32,
    ) -> Result<(), MlxRuntimeError> {
        if checkpoint_offset_tokens < 0 {
            return Err(full_attention_error(
                "truncate checkpoint offset must not be negative",
            ));
        }
        if checkpoint_offset_tokens > self.offset_tokens {
            return Err(full_attention_error(
                "truncate checkpoint offset must not exceed the current KV offset",
            ));
        }
        self.offset_tokens = checkpoint_offset_tokens;
        Ok(())
    }

    /// Replaces the K and V storage from a restored SSD prompt-cache prefix.
    /// Called by the SSD bridge after it has concatenated the block tensors
    /// into a single slab; the owner takes ownership and updates its offset.
    pub fn restore_from_blocks(
        &mut self,
        restored_keys: MlxArray,
        restored_values: MlxArray,
    ) -> Result<(), MlxRuntimeError> {
        let restored_key_shape = restored_keys.shape();
        let restored_value_shape = restored_values.shape();
        if restored_key_shape.len() != 4
            || restored_key_shape != restored_value_shape
            || restored_key_shape[STATE_DIMENSION_TOKEN_AXIS] <= 0
        {
            return Err(full_attention_error(
                "restored K and V slabs must have identical rank-four nonempty shapes",
            ));
        }
        let restored_token_count = restored_key_shape[STATE_DIMENSION_TOKEN_AXIS];
        self.keys = Some(restored_keys);
        self.values = Some(restored_values);
        self.offset_tokens = restored_token_count;
        Ok(())
    }
}

fn validate_key_value_update(
    state: &FullAttentionKeyValueState,
    new_keys: &MlxArray,
    new_values: &MlxArray,
    previous_token_count: i32,
) -> Result<Vec<i32>, MlxRuntimeError> {
    let key_shape = new_keys.shape();
    if key_shape.len() != 4
        || key_shape != new_values.shape()
        || key_shape[STATE_DIMENSION_TOKEN_AXIS] <= 0
    {
        return Err(full_attention_error(
            "new K and V tensors must have identical rank-four nonempty shapes",
        ));
    }
    if previous_token_count != state.offset_tokens {
        return Err(full_attention_error(
            "append position does not match the in-memory KV state offset",
        ));
    }
    if state.keys.is_some() != state.values.is_some() {
        return Err(full_attention_error(
            "in-memory K and V storage must both be present or absent",
        ));
    }
    if let (Some(existing_keys), Some(existing_values)) =
        (state.keys.as_ref(), state.values.as_ref())
    {
        let existing_key_shape = existing_keys.shape();
        let existing_value_shape = existing_values.shape();
        if existing_key_shape.len() != 4
            || existing_key_shape != existing_value_shape
            || existing_key_shape[0] != key_shape[0]
            || existing_key_shape[1] != key_shape[1]
            || existing_key_shape[3] != key_shape[3]
            || existing_key_shape[STATE_DIMENSION_TOKEN_AXIS] < previous_token_count
        {
            return Err(full_attention_error(
                "new K and V tensors are incompatible with the existing KV storage",
            ));
        }
    }
    Ok(key_shape)
}

fn build_updated_storage(
    runtime: &MlxRuntime,
    current_storage: Option<&MlxArray>,
    state_update: &MlxArray,
    previous_token_count: i32,
    full_attention_kv_state_growth_tokens: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let state_update_shape = state_update.shape();
    let update_token_count = state_update_shape[STATE_DIMENSION_TOKEN_AXIS];
    let next_token_count = previous_token_count
        .checked_add(update_token_count)
        .ok_or_else(|| full_attention_error("full-attention state token count overflowed"))?;
    let current_capacity_tokens =
        current_storage.map_or(0, |state| state.shape()[STATE_DIMENSION_TOKEN_AXIS]);

    let projected_capacity_tokens = projected_capacity_tokens(
        current_capacity_tokens,
        current_storage.is_some(),
        previous_token_count,
        update_token_count,
        full_attention_kv_state_growth_tokens,
    )?;
    let grown_state = if projected_capacity_tokens > current_capacity_tokens {
        // A partially used slab may contain unused tail capacity. Once an update
        // no longer fits, retain only the written prefix before appending a newly
        // rounded extension; otherwise the unused gap would become logical state.
        let retained_capacity_tokens = if current_storage.is_some()
            && previous_token_count % full_attention_kv_state_growth_tokens != 0
        {
            previous_token_count
        } else {
            current_capacity_tokens
        };
        let extension_capacity_tokens = projected_capacity_tokens
            .checked_sub(retained_capacity_tokens)
            .ok_or_else(|| full_attention_error("full-attention state growth underflowed"))?;
        let mut extension_shape = state_update_shape.clone();
        extension_shape[STATE_DIMENSION_TOKEN_AXIS] = extension_capacity_tokens;
        let state_extension = runtime.zeros(&extension_shape, state_update.dtype())?;
        match current_storage {
            Some(previous_state) => {
                let retained_prefix =
                    if previous_token_count % full_attention_kv_state_growth_tokens != 0 {
                        let mut retained_stops = previous_state.shape();
                        retained_stops[STATE_DIMENSION_TOKEN_AXIS] = previous_token_count;
                        Some(runtime.slice(
                            previous_state,
                            &[0, 0, 0, 0],
                            &retained_stops,
                            &[1, 1, 1, 1],
                        )?)
                    } else {
                        None
                    };
                let retained_state = retained_prefix.as_ref().unwrap_or(previous_state);
                Some(runtime.concatenate_axis(&[retained_state, &state_extension], 2)?)
            }
            None => Some(state_extension),
        }
    } else {
        None
    };
    let state_storage = grown_state
        .as_ref()
        .or(current_storage)
        .ok_or_else(|| full_attention_error("full-attention state storage is unavailable"))?;

    let mut update_starts = vec![0; state_update_shape.len()];
    update_starts[STATE_DIMENSION_TOKEN_AXIS] = previous_token_count;
    let mut update_stops = state_update_shape;
    update_stops[STATE_DIMENSION_TOKEN_AXIS] = next_token_count;
    let update_strides = vec![1; update_starts.len()];
    let updated_state = runtime.slice_update(
        state_storage,
        state_update,
        &update_starts,
        &update_stops,
        &update_strides,
    )?;
    Ok(updated_state)
}

fn projected_capacity_tokens(
    current_capacity_tokens: i32,
    has_current_storage: bool,
    previous_token_count: i32,
    update_token_count: i32,
    full_attention_kv_state_growth_tokens: i32,
) -> Result<i32, MlxRuntimeError> {
    let next_token_count = previous_token_count
        .checked_add(update_token_count)
        .ok_or_else(|| full_attention_error("full-attention state token count overflowed"))?;
    if next_token_count <= current_capacity_tokens {
        // The existing over-allocated slab already has enough unused room.
        return Ok(current_capacity_tokens);
    }
    // Round the incoming update, not the total context, to the configured growth
    // step. This mirrors the actual extension allocated by `build_updated_storage`.
    let rounded_update_tokens = update_token_count
        .checked_add(full_attention_kv_state_growth_tokens - 1)
        .and_then(|rounded_token_count| {
            rounded_token_count
                .checked_div(full_attention_kv_state_growth_tokens)
                .and_then(|growth_steps| {
                    growth_steps.checked_mul(full_attention_kv_state_growth_tokens)
                })
        })
        .ok_or_else(|| full_attention_error("full-attention state growth overflowed"))?;
    let retained_capacity_tokens = if has_current_storage
        && previous_token_count % full_attention_kv_state_growth_tokens != 0
    {
        previous_token_count
    } else {
        current_capacity_tokens
    };
    retained_capacity_tokens
        .checked_add(rounded_update_tokens)
        .ok_or_else(|| full_attention_error("full-attention state capacity overflowed"))
}

fn active_view(
    runtime: &MlxRuntime,
    updated_state: &MlxArray,
    active_token_count: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let mut active_state_stops = updated_state.shape();
    active_state_stops[STATE_DIMENSION_TOKEN_AXIS] = active_token_count;
    runtime.slice(
        updated_state,
        &[0, 0, 0, 0],
        &active_state_stops,
        &[1, 1, 1, 1],
    )
}

fn full_attention_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: FULL_ATTENTION_OPERATION,
        description: description.to_owned(),
    }
}
