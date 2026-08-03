use astronomical_runtime_integration::MlxRuntimeError;

use crate::decoder_cache::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, FullAttentionKeyValueState,
    FullAttentionKeyValueStateAllocationCheckpoint,
};

/// Request-local state for Qwen's one-layer multi-token prediction head.
///
/// MTP has its own causal full-attention history. It never shares, persists, or
/// pages target decoder state because its fused hidden states are a different
/// sequence from the target model's decoder inputs.
pub struct Qwen3_5MtpRequestState {
    full_attention_key_value_state: FullAttentionKeyValueState,
}

/// Retained physical owner checkpoint for one retryable MTP prompt update.
pub struct Qwen3_5MtpRequestStateAllocationCheckpoint {
    full_attention_key_value_state: FullAttentionKeyValueStateAllocationCheckpoint,
}

impl Qwen3_5MtpRequestState {
    /// Creates empty MTP state using the standard full-attention slab-growth policy.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            full_attention_key_value_state: FullAttentionKeyValueState::empty(),
        }
    }

    /// Creates empty MTP state with an explicit full-attention slab-growth policy.
    pub fn empty_with_growth_tokens(
        full_attention_kv_state_growth_tokens: i32,
    ) -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            full_attention_key_value_state: FullAttentionKeyValueState::empty_with_growth_tokens(
                full_attention_kv_state_growth_tokens,
            )?,
        })
    }

    pub(crate) fn empty_with_validated_growth_tokens(
        full_attention_kv_state_growth_tokens: i32,
    ) -> Self {
        Self {
            full_attention_key_value_state:
                FullAttentionKeyValueState::empty_with_validated_growth_tokens(
                    full_attention_kv_state_growth_tokens,
                ),
        }
    }

    /// Returns the MTP full-attention history's logical token length.
    #[must_use]
    pub fn committed_token_count(&self) -> i32 {
        self.full_attention_key_value_state.offset_tokens()
    }

    /// Returns exact additional physical MTP KV capacity required for an update.
    pub fn projected_capacity_growth_tokens(
        &self,
        update_token_count: usize,
    ) -> Result<usize, MlxRuntimeError> {
        self.full_attention_key_value_state
            .projected_capacity_growth_tokens(update_token_count)
    }

    /// Returns the current MTP KV slab payload bytes.
    #[must_use]
    pub fn payload_byte_count(&self) -> u64 {
        self.full_attention_key_value_state.payload_byte_count()
    }

    /// Projects physical rounded MTP KV growth in bytes for admission.
    ///
    /// Uses the model's per-layer-token byte cost so admission charges the same
    /// way it charges target full-attention layers.
    pub fn projected_capacity_growth_bytes(
        &self,
        mtp_full_attention_bytes_per_layer_token: usize,
        update_token_count: usize,
    ) -> Result<usize, MlxRuntimeError> {
        let growth_tokens = self.projected_capacity_growth_tokens(update_token_count)?;
        growth_tokens
            .checked_mul(mtp_full_attention_bytes_per_layer_token)
            .ok_or_else(|| MlxRuntimeError::RuntimeOperation {
                operation: "project MTP KV-state growth",
                description: "MTP KV-state growth bytes overflowed".to_owned(),
            })
    }

    /// Projects physical growth for MTP updates executed by separate forwards.
    pub fn projected_sequential_capacity_growth_bytes(
        &self,
        mtp_full_attention_bytes_per_layer_token: usize,
        sequential_update_token_counts: &[usize],
    ) -> Result<usize, MlxRuntimeError> {
        let growth_tokens = self
            .full_attention_key_value_state
            .projected_sequential_capacity_growth_tokens(sequential_update_token_counts)?;
        growth_tokens
            .checked_mul(mtp_full_attention_bytes_per_layer_token)
            .ok_or_else(|| MlxRuntimeError::RuntimeOperation {
                operation: "project sequential MTP KV-state growth",
                description: "sequential MTP KV-state growth bytes overflowed".to_owned(),
            })
    }

    /// Replaces all MTP history with empty state using the configured growth policy.
    pub fn reset_with_growth_tokens(
        &mut self,
        full_attention_kv_state_growth_tokens: i32,
    ) -> Result<(), MlxRuntimeError> {
        *self = Self::empty_with_growth_tokens(full_attention_kv_state_growth_tokens)?;
        Ok(())
    }

    /// Retains the MTP K/V owners and logical offset for a retryable prompt attempt.
    pub fn allocation_checkpoint(
        &self,
    ) -> Result<Qwen3_5MtpRequestStateAllocationCheckpoint, MlxRuntimeError> {
        Ok(Qwen3_5MtpRequestStateAllocationCheckpoint {
            full_attention_key_value_state: self
                .full_attention_key_value_state
                .allocation_checkpoint()?,
        })
    }

    /// Restores MTP K/V ownership after a failed prompt attempt.
    pub fn restore_allocation_checkpoint(
        &mut self,
        allocation_checkpoint: Qwen3_5MtpRequestStateAllocationCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        self.full_attention_key_value_state
            .restore_allocation_checkpoint(allocation_checkpoint.full_attention_key_value_state)
    }

    pub(in crate::qwen3_5) fn full_attention_key_value_state_mut(
        &mut self,
    ) -> &mut FullAttentionKeyValueState {
        &mut self.full_attention_key_value_state
    }

    /// Exposes the owned attention state to direct-MLX crate-root contracts.
    #[doc(hidden)]
    pub fn full_attention_key_value_state_mut_for_tests(
        &mut self,
    ) -> &mut FullAttentionKeyValueState {
        &mut self.full_attention_key_value_state
    }

    pub(in crate::qwen3_5) fn full_attention_key_value_state(&self) -> &FullAttentionKeyValueState {
        &self.full_attention_key_value_state
    }
}

impl Default for Qwen3_5MtpRequestState {
    fn default() -> Self {
        Self::empty_with_validated_growth_tokens(DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS)
    }
}

/// Bounded reason why MTP is unavailable for one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen3_5MtpUnavailableReason {
    /// The model has no compatible MTP head.
    NoCompatibleHead,
}

impl std::fmt::Display for Qwen3_5MtpUnavailableReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCompatibleHead => formatter.write_str("no compatible MTP head"),
        }
    }
}
