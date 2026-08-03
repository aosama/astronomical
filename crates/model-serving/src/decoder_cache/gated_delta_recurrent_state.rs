use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError};

const GATED_DELTA_RECURRENT_OPERATION: &str =
    "fetch or allocate the in-memory gated-delta recurrent state";

/// Single owner for one gated-delta layer's recurrent state.
///
/// The recurrent state is always float32 and has shape
/// `[1, value_head_count, value_dimension, key_dimension]`. The empty state
/// allocates nothing; the first `current_or_zero` call materializes a zero
/// tensor of the requested shape. After the recurrent step runs, the layer
/// stores the next state with `set_next`.
pub struct GatedDeltaRecurrentState {
    state: Option<MlxArray>,
    linear_value_head_count: i32,
    linear_value_head_dimension: i32,
    linear_key_head_dimension: i32,
}

/// Retained checkpoint of one gated-delta recurrent state tensor.
pub struct GatedDeltaRecurrentStateCheckpoint {
    state: Option<MlxArray>,
}

impl GatedDeltaRecurrentState {
    /// Creates empty recurrent state without allocating MLX arrays.
    ///
    /// Uses the certified Qwen3.5-MoE shape `[1, 32, 128, 128]` for the two GPU
    /// unit tests that exercise this owner in isolation. Production code
    /// should use `empty_with_shape` with config-derived dimensions.
    #[must_use]
    pub fn empty() -> Self {
        Self::empty_with_shape(32, 128, 128)
    }

    /// Creates empty recurrent state with config-derived dimensions, without
    /// allocating MLX arrays. The first `current_or_zero` call materializes a
    /// zero tensor of shape
    /// `[1, linear_value_head_count, linear_value_head_dimension, linear_key_head_dimension]`.
    #[must_use]
    pub fn empty_with_shape(
        linear_value_head_count: i32,
        linear_value_head_dimension: i32,
        linear_key_head_dimension: i32,
    ) -> Self {
        Self {
            state: None,
            linear_value_head_count,
            linear_value_head_dimension,
            linear_key_head_dimension,
        }
    }

    /// True when no MLX array has been allocated or restored yet.
    #[must_use]
    pub fn is_unallocated(&self) -> bool {
        self.state.is_none()
    }

    /// Returns the current recurrent state, allocating a zero tensor of the
    /// requested shape on first use. The caller takes a logical view: when the
    /// state already exists, the stored tensor is returned (it must be
    /// retained before mutation); when the state is absent, a fresh zero
    /// tensor is returned and stored.
    pub fn current_or_zero(&mut self, runtime: &MlxRuntime) -> Result<MlxArray, MlxRuntimeError> {
        match self.state.as_ref() {
            Some(existing_state) => {
                existing_state
                    .retain()
                    .map_err(|source| MlxRuntimeError::RuntimeOperation {
                        operation: GATED_DELTA_RECURRENT_OPERATION,
                        description: format!(
                            "failed to retain the existing recurrent state: {source}"
                        ),
                    })
            }
            None => {
                let recurrent_state_shape = [
                    1,
                    self.linear_value_head_count,
                    self.linear_value_head_dimension,
                    self.linear_key_head_dimension,
                ];
                let zero_state = runtime.zeros(&recurrent_state_shape, MlxDtype::Float32)?;
                self.state = Some(zero_state.retain()?);
                Ok(zero_state)
            }
        }
    }

    /// Stores the recurrent state produced by the gated-delta step. Replaces
    /// the previous state unconditionally.
    pub fn set_next(&mut self, next_state: MlxArray) {
        self.state = Some(next_state);
    }

    /// Read-only access to the current recurrent state. Used by the SSD
    /// prompt-cache bridge to snapshot recurrent state for persistence.
    #[must_use]
    pub fn state(&self) -> Option<&MlxArray> {
        self.state.as_ref()
    }

    #[must_use]
    /// Returns the logical payload bytes owned by the live recurrent state.
    pub fn payload_byte_count(&self) -> u64 {
        self.state
            .as_ref()
            .map_or(0, |state| state.byte_count() as u64)
    }

    /// Retains the current recurrent tensor for MTP rollback.
    pub fn checkpoint(&self) -> Result<GatedDeltaRecurrentStateCheckpoint, MlxRuntimeError> {
        Ok(GatedDeltaRecurrentStateCheckpoint {
            state: self.state.as_ref().map(MlxArray::retain).transpose()?,
        })
    }

    /// Restores a retained MTP checkpoint.
    pub fn restore_checkpoint(&mut self, checkpoint: GatedDeltaRecurrentStateCheckpoint) {
        self.state = checkpoint.state;
    }

    /// Replaces the recurrent state from a restored SSD prompt-cache snapshot.
    /// Called by the SSD bridge after it has loaded the snapshot tensor.
    pub fn restore_from_snapshot(&mut self, restored_state: MlxArray) {
        self.state = Some(restored_state);
    }
}
