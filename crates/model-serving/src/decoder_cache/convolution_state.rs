use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

const CONVOLUTION_STATE_OPERATION: &str = "roll the in-memory convolution rolling buffer";

/// Single owner for one gated-delta layer's convolution rolling buffer.
///
/// The convolution state is a fixed-shape `[1, 3, convolution_dimension]`
/// rolling buffer that holds the last 3 tokens of the mixed query/key/value
/// input for the next conv1d call. The empty state allocates nothing; the
/// first `update` call materializes a zero buffer of the config dimension,
/// prepends it to the new input, and stores the trailing 3-token slice.
pub struct ConvolutionState {
    state: Option<MlxArray>,
    linear_convolution_kernel_dimension: i32,
    linear_convolution_dimension: i32,
}

/// Retained checkpoint of one convolution rolling buffer.
pub struct ConvolutionStateCheckpoint {
    state: Option<MlxArray>,
}

/// One convolution update with exact rolling-state views at requested boundaries.
pub struct ConvolutionStateBoundaryCheckpointUpdate {
    pub convolution_input: MlxArray,
    pub boundary_convolution_states: Vec<MlxArray>,
}

impl ConvolutionState {
    /// Creates empty convolution state without allocating MLX arrays.
    ///
    /// Uses the Qwen3.5-MoE dimension `8192` and kernel dimension `4` for
    /// the two GPU unit tests that exercise this owner in isolation. Production
    /// code should use `empty_with_shape` with config-derived dimensions.
    #[must_use]
    pub fn empty() -> Self {
        Self::empty_with_shape(4, 8_192)
    }

    /// Creates empty convolution state with config-derived dimensions, without
    /// allocating MLX arrays. The first `update` call materializes a zero
    /// rolling buffer of shape `[1, kernel_dimension - 1, linear_convolution_dimension]`.
    #[must_use]
    pub fn empty_with_shape(
        linear_convolution_kernel_dimension: i32,
        linear_convolution_dimension: i32,
    ) -> Self {
        Self {
            state: None,
            linear_convolution_kernel_dimension,
            linear_convolution_dimension,
        }
    }

    /// True when no MLX array has been allocated or restored yet.
    #[must_use]
    pub fn is_unallocated(&self) -> bool {
        self.state.is_none()
    }

    /// Prepends the existing (or freshly-allocated zero) rolling buffer to the
    /// new mixed query/key/value input, stores the trailing 3-token slice as
    /// the next buffer, and returns the full concatenated convolution input
    /// for the conv1d call.
    ///
    /// `token_count` is the number of new tokens in `mixed_queries_keys_values`.
    /// The rolling buffer keeps exactly `kernel_dimension - 1` trailing
    /// tokens across calls (3 for the Qwen3.5-MoE config).
    pub fn update(
        &mut self,
        runtime: &MlxRuntime,
        mixed_queries_keys_values: &MlxArray,
        token_count: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        Ok(self
            .update_internal(runtime, mixed_queries_keys_values, token_count, &[])?
            .convolution_input)
    }

    /// Updates final state and returns exact rolling-state views at local token boundaries.
    pub fn update_with_boundary_checkpoints(
        &mut self,
        runtime: &MlxRuntime,
        mixed_queries_keys_values: &MlxArray,
        token_count: i32,
        completed_prefill_chunk_tokens: &[i32],
    ) -> Result<ConvolutionStateBoundaryCheckpointUpdate, MlxRuntimeError> {
        if completed_prefill_chunk_tokens.is_empty() {
            return Err(convolution_error(
                "boundary checkpoint positions must not be empty".to_owned(),
            ));
        }
        self.update_internal(
            runtime,
            mixed_queries_keys_values,
            token_count,
            completed_prefill_chunk_tokens,
        )
    }

    fn update_internal(
        &mut self,
        runtime: &MlxRuntime,
        mixed_queries_keys_values: &MlxArray,
        token_count: i32,
        completed_prefill_chunk_tokens: &[i32],
    ) -> Result<ConvolutionStateBoundaryCheckpointUpdate, MlxRuntimeError> {
        let input_shape = mixed_queries_keys_values.shape();
        let expected_input_shape = [1, token_count, self.linear_convolution_dimension];
        if token_count <= 0 || input_shape != expected_input_shape {
            return Err(convolution_error(format!(
                "the convolution input must have shape [1, token_count, {}]",
                self.linear_convolution_dimension
            )));
        }
        let rolling_buffer_tokens = self.linear_convolution_kernel_dimension.saturating_sub(1);
        let mut previous_completed_prefill_chunk_tokens = 0;
        for current_completed_prefill_chunk_tokens in completed_prefill_chunk_tokens {
            if *current_completed_prefill_chunk_tokens <= previous_completed_prefill_chunk_tokens
                || *current_completed_prefill_chunk_tokens >= token_count
            {
                return Err(convolution_error(
                    "boundary checkpoint positions must be positive, strictly increasing, and less than token_count"
                        .to_owned(),
                ));
            }
            previous_completed_prefill_chunk_tokens = *current_completed_prefill_chunk_tokens;
        }
        if self.state.as_ref().is_some_and(|state| {
            state.shape() != [1, rolling_buffer_tokens, self.linear_convolution_dimension]
        }) {
            return Err(convolution_error(format!(
                "the existing convolution state must have shape [1, {}, {}]",
                rolling_buffer_tokens, self.linear_convolution_dimension
            )));
        }
        let zero_buffer_shape = [1, rolling_buffer_tokens, self.linear_convolution_dimension];

        // The zero buffer must outlive the concatenate call when no existing
        // state is present, so it is materialized in a local binding.
        let zero_state;
        let initial_state: &MlxArray = match self.state.as_ref() {
            Some(existing_state) => existing_state,
            None => {
                zero_state =
                    runtime.zeros(&zero_buffer_shape, mixed_queries_keys_values.dtype())?;
                &zero_state
            }
        };

        let convolution_input =
            runtime.concatenate_axis(&[initial_state, mixed_queries_keys_values], 1)?;

        let mut boundary_convolution_states =
            Vec::with_capacity(completed_prefill_chunk_tokens.len());
        for current_completed_prefill_chunk_tokens in completed_prefill_chunk_tokens {
            boundary_convolution_states.push(runtime.slice(
                &convolution_input,
                &[0, *current_completed_prefill_chunk_tokens, 0],
                &[
                    1,
                    current_completed_prefill_chunk_tokens + rolling_buffer_tokens,
                    self.linear_convolution_dimension,
                ],
                &[1, 1, 1],
            )?);
        }

        let next_state = runtime.slice(
            &convolution_input,
            &[0, token_count, 0],
            &[
                1,
                token_count + rolling_buffer_tokens,
                self.linear_convolution_dimension,
            ],
            &[1, 1, 1],
        )?;
        self.state = Some(next_state);
        Ok(ConvolutionStateBoundaryCheckpointUpdate {
            convolution_input,
            boundary_convolution_states,
        })
    }

    /// Read-only access to the current rolling buffer. Used by the SSD
    /// prompt-cache bridge to snapshot convolution state for persistence.
    #[must_use]
    pub fn state(&self) -> Option<&MlxArray> {
        self.state.as_ref()
    }

    #[must_use]
    /// Returns the logical payload bytes owned by the live rolling buffer.
    pub fn payload_byte_count(&self) -> u64 {
        self.state
            .as_ref()
            .map_or(0, |state| state.byte_count() as u64)
    }

    /// Retains the current rolling buffer for MTP rollback.
    pub fn checkpoint(&self) -> Result<ConvolutionStateCheckpoint, MlxRuntimeError> {
        Ok(ConvolutionStateCheckpoint {
            state: self.state.as_ref().map(MlxArray::retain).transpose()?,
        })
    }

    /// Restores a retained MTP checkpoint.
    pub fn restore_checkpoint(&mut self, checkpoint: ConvolutionStateCheckpoint) {
        self.state = checkpoint.state;
    }

    /// Replaces the convolution rolling buffer from a restored SSD
    /// prompt-cache snapshot. Called by the SSD bridge after it has loaded the
    /// snapshot tensor.
    pub fn restore_from_snapshot(&mut self, restored_state: MlxArray) {
        self.state = Some(restored_state);
    }
}

fn convolution_error(description: String) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: CONVOLUTION_STATE_OPERATION,
        description,
    }
}
