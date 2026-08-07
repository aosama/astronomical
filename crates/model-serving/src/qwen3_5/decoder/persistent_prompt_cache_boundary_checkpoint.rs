use std::collections::HashMap;

use astronomical_runtime_integration::{MlxArray, MlxRuntimeError};

const BOUNDARY_CHECKPOINT_COLLECTOR_OPERATION: &str =
    "collect Qwen3.5 persistent prompt-cache boundary state";

/// Exact recurrent tensors for one local prefill boundary.
pub struct Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
    pub completed_prefill_chunck_tokens: usize,
    pub recurrent_snapshot_tensors: HashMap<String, MlxArray>,
}

/// Request-local collector for intermediate persistent prompt-cache boundaries.
pub struct Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector {
    checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
    completed_prefill_chunck_tokens: Vec<i32>,
    checkpoint_interval_token_count: i32,
    expected_boundary_tensor_count: usize,
}

impl Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector {
    pub fn new(
        completed_prefill_chunck_tokens: Vec<usize>,
        expected_boundary_tensor_count: usize,
        checkpoint_interval_token_count: usize,
    ) -> Result<Self, MlxRuntimeError> {
        if completed_prefill_chunck_tokens.is_empty()
            || expected_boundary_tensor_count == 0
            || checkpoint_interval_token_count == 0
        {
            return Err(collector_error(
                "checkpoint positions and expected tensor count must not be empty",
            ));
        }
        let checkpoint_interval_token_count = i32::try_from(checkpoint_interval_token_count)
            .map_err(|_| collector_error("checkpoint interval exceeds the Int32 range"))?;
        let mut signed_completed_prefill_chunck_tokens =
            Vec::with_capacity(completed_prefill_chunck_tokens.len());
        let mut previous_completed_prefill_chunck_tokens = 0_usize;
        for current_completed_prefill_chunck_tokens in &completed_prefill_chunck_tokens {
            if *current_completed_prefill_chunck_tokens <= previous_completed_prefill_chunck_tokens
            {
                return Err(collector_error(
                    "checkpoint positions must be positive and strictly increasing",
                ));
            }
            previous_completed_prefill_chunck_tokens = *current_completed_prefill_chunck_tokens;
            signed_completed_prefill_chunck_tokens.push(
                i32::try_from(*current_completed_prefill_chunck_tokens)
                    .map_err(|_| collector_error("checkpoint position exceeds the Int32 range"))?,
            );
        }
        Ok(Self {
            checkpoints: completed_prefill_chunck_tokens
                .into_iter()
                .map(|completed_prefill_chunck_tokens| {
                    Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
                        completed_prefill_chunck_tokens,
                        recurrent_snapshot_tensors: HashMap::with_capacity(
                            expected_boundary_tensor_count,
                        ),
                    }
                })
                .collect(),
            completed_prefill_chunck_tokens: signed_completed_prefill_chunck_tokens,
            checkpoint_interval_token_count,
            expected_boundary_tensor_count,
        })
    }

    #[must_use]
    pub fn completed_prefill_chunck_tokens(&self) -> &[i32] {
        &self.completed_prefill_chunck_tokens
    }

    #[must_use]
    pub const fn checkpoint_interval_token_count(&self) -> i32 {
        self.checkpoint_interval_token_count
    }

    #[must_use]
    pub fn evaluation_arrays(&self) -> Vec<&MlxArray> {
        self.checkpoints
            .iter()
            .flat_map(|checkpoint| checkpoint.recurrent_snapshot_tensors.values())
            .collect()
    }

    pub fn record_linear_attention_layer(
        &mut self,
        decoder_layer_index: usize,
        boundary_convolution_states: Vec<MlxArray>,
        boundary_recurrent_states: Vec<MlxArray>,
    ) -> Result<(), MlxRuntimeError> {
        if boundary_convolution_states.len() != self.checkpoints.len()
            || boundary_recurrent_states.len() != self.checkpoints.len()
        {
            return Err(collector_error(
                "linear-attention boundary vectors must match every checkpoint position",
            ));
        }
        let convolution_tensor_name = format!("layer_{decoder_layer_index}_linear.convolution");
        let recurrent_tensor_name =
            format!("layer_{decoder_layer_index}_linear.gated_delta_recurrent");
        if self.checkpoints.iter().any(|checkpoint| {
            checkpoint
                .recurrent_snapshot_tensors
                .contains_key(&convolution_tensor_name)
                || checkpoint
                    .recurrent_snapshot_tensors
                    .contains_key(&recurrent_tensor_name)
        }) {
            return Err(collector_error(
                "linear-attention boundary tensor names must not be recorded twice",
            ));
        }
        for ((checkpoint, convolution_state), recurrent_state) in self
            .checkpoints
            .iter_mut()
            .zip(boundary_convolution_states)
            .zip(boundary_recurrent_states)
        {
            checkpoint
                .recurrent_snapshot_tensors
                .insert(convolution_tensor_name.clone(), convolution_state);
            checkpoint
                .recurrent_snapshot_tensors
                .insert(recurrent_tensor_name.clone(), recurrent_state);
        }
        Ok(())
    }

    pub fn complete(
        self,
    ) -> Result<Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>, MlxRuntimeError> {
        if self.checkpoints.iter().any(|checkpoint| {
            checkpoint.recurrent_snapshot_tensors.len() != self.expected_boundary_tensor_count
        }) {
            return Err(collector_error(
                "each boundary checkpoint must contain the complete decoder-cache tensor layout",
            ));
        }
        Ok(self.checkpoints)
    }
}

fn collector_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: BOUNDARY_CHECKPOINT_COLLECTOR_OPERATION,
        description: description.to_owned(),
    }
}
