use astronomical_runtime_integration::MlxArray;
use astronomical_runtime_integration::MlxRuntimeError;

use super::verified_emission_queue::{VerifiedEmissionQueue, VerifiedTargetFrontier};
use crate::decoder_cache::{
    FullAttentionKeyValueState, FullAttentionKeyValueStateAllocationCheckpoint,
};
use crate::memory::MtpDraftDepth;

pub(crate) type MultiTokenPredictionRequestAllocationCheckpoint =
    Qwen3_5MtpRequestStateAllocationCheckpoint;

/// Optional request-local MTP session owned outside the standard target request state.
pub(crate) struct Qwen3_5MultiTokenPredictionRequest {
    request_state: Qwen3_5MtpRequestState,
    requested_depth: MtpDraftDepth,
    target_hidden_states: Option<MlxArray>,
    verified_emission_queue: Option<VerifiedEmissionQueue>,
    force_next_draft_rejection_for_tests: bool,
}

impl Qwen3_5MultiTokenPredictionRequest {
    /// Creates the optional MTP session only when the resolved runtime and request permit it.
    pub(crate) fn new_if_eligible(
        mtp_enabled: bool,
        mtp_runtime_is_active: bool,
        model_has_mtp_weights: bool,
        has_precomputed_visual_embeddings: bool,
        has_processed_visual_images: bool,
        persistent_prompt_cache_is_available: bool,
        sparse_experts_are_paged: bool,
        prompt_token_count: usize,
        restored_prompt_token_count: u32,
        full_attention_kv_state_growth_tokens: i32,
        requested_depth: Option<MtpDraftDepth>,
    ) -> Result<Option<Self>, MlxRuntimeError> {
        let is_eligible = super::qwen3_5_mtp_request_is_eligible(
            mtp_enabled,
            mtp_runtime_is_active,
            model_has_mtp_weights,
            has_precomputed_visual_embeddings,
            has_processed_visual_images,
            persistent_prompt_cache_is_available,
            sparse_experts_are_paged,
            prompt_token_count,
            restored_prompt_token_count,
        );
        if !is_eligible {
            return Ok(None);
        }
        let requested_depth = requested_depth.ok_or_else(|| MlxRuntimeError::RuntimeOperation {
            operation: "create Qwen3.5 MTP request",
            description: "active MTP runtime has no resolved draft depth".to_owned(),
        })?;
        Ok(Some(Self {
            request_state: Qwen3_5MtpRequestState::empty_with_growth_tokens(
                full_attention_kv_state_growth_tokens,
            )?,
            requested_depth,
            target_hidden_states: None,
            verified_emission_queue: None,
            force_next_draft_rejection_for_tests: false,
        }))
    }

    pub(crate) fn request_state_mut(&mut self) -> &mut Qwen3_5MtpRequestState {
        &mut self.request_state
    }

    pub(crate) const fn requested_depth(&self) -> MtpDraftDepth {
        self.requested_depth
    }

    pub(crate) fn target_hidden_states(&self) -> Option<&MlxArray> {
        self.target_hidden_states.as_ref()
    }

    pub(crate) fn take_target_hidden_states(&mut self) -> Option<MlxArray> {
        self.target_hidden_states.take()
    }

    pub(crate) fn set_target_hidden_states(&mut self, target_hidden_states: Option<MlxArray>) {
        self.target_hidden_states = target_hidden_states;
    }

    pub(crate) fn has_verified_generated_token_ids(&self) -> bool {
        self.verified_emission_queue
            .as_ref()
            .is_some_and(|queue| !queue.is_empty())
    }

    pub(crate) fn take_verified_generated_token_id(&mut self) -> Option<u32> {
        let queued_token_id = self
            .verified_emission_queue
            .as_mut()
            .and_then(VerifiedEmissionQueue::pop_front);
        if self
            .verified_emission_queue
            .as_ref()
            .is_some_and(VerifiedEmissionQueue::is_empty)
        {
            self.verified_emission_queue = None;
        }
        queued_token_id
    }

    pub(crate) fn set_verified_emission_queue(&mut self, queue: VerifiedEmissionQueue) {
        self.verified_emission_queue = Some(queue);
    }

    pub(crate) fn take_public_verified_frontier(&mut self) -> Option<VerifiedTargetFrontier> {
        self.verified_emission_queue
            .as_mut()
            .and_then(VerifiedEmissionQueue::take_public_frontier)
    }

    pub(crate) fn force_next_draft_rejection_for_tests(&mut self) {
        self.force_next_draft_rejection_for_tests = true;
    }

    pub(crate) fn take_forced_draft_rejection_for_tests(&mut self) -> bool {
        std::mem::take(&mut self.force_next_draft_rejection_for_tests)
    }

    pub(crate) fn allocation_checkpoint(
        &self,
    ) -> Result<Qwen3_5MtpRequestStateAllocationCheckpoint, MlxRuntimeError> {
        self.request_state.allocation_checkpoint()
    }

    pub(crate) fn restore_allocation_checkpoint(
        &mut self,
        allocation_checkpoint: Qwen3_5MtpRequestStateAllocationCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        self.request_state
            .restore_allocation_checkpoint(allocation_checkpoint)
    }

    pub(crate) fn context_state_payload_byte_count(&self) -> u64 {
        self.request_state.payload_byte_count().saturating_add(
            self.verified_emission_queue
                .as_ref()
                .map_or(0, VerifiedEmissionQueue::payload_byte_count),
        )
    }

    pub(crate) fn projected_full_attention_growth_bytes(
        &self,
        full_attention_bytes_per_layer_token: usize,
        update_token_count: usize,
    ) -> Result<usize, MlxRuntimeError> {
        self.request_state.projected_capacity_growth_bytes(
            full_attention_bytes_per_layer_token,
            update_token_count,
        )
    }

    pub(crate) fn projected_sequential_full_attention_growth_bytes(
        &self,
        full_attention_bytes_per_layer_token: usize,
        sequential_update_token_counts: &[usize],
    ) -> Result<usize, MlxRuntimeError> {
        self.request_state
            .projected_sequential_capacity_growth_bytes(
                full_attention_bytes_per_layer_token,
                sequential_update_token_counts,
            )
    }
}
pub(crate) fn create_optional_prediction_session(
    user_enabled_optional_prediction: bool,
    optional_prediction_runtime_is_active: bool,
    model_has_optional_prediction_weights: bool,
    has_precomputed_visual_embeddings: bool,
    has_processed_visual_images: bool,
    persistent_prompt_cache_is_available: bool,
    sparse_experts_are_paged: bool,
    prompt_token_count: usize,
    restored_prompt_token_count: u32,
    full_attention_kv_state_growth_tokens: i32,
    requested_depth: Option<MtpDraftDepth>,
) -> Result<Option<Qwen3_5MultiTokenPredictionRequest>, MlxRuntimeError> {
    Qwen3_5MultiTokenPredictionRequest::new_if_eligible(
        user_enabled_optional_prediction,
        optional_prediction_runtime_is_active,
        model_has_optional_prediction_weights,
        has_precomputed_visual_embeddings,
        has_processed_visual_images,
        persistent_prompt_cache_is_available,
        sparse_experts_are_paged,
        prompt_token_count,
        restored_prompt_token_count,
        full_attention_kv_state_growth_tokens,
        requested_depth,
    )
}

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
