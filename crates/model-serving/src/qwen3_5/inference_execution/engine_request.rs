use std::collections::VecDeque;

use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::{MlxArray, MlxRuntimeError};

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockKey, Qwen3_5SamplingStrategy,
};

use super::super::text::sampler::build_qwen3_5_sampled_token;
use super::{fatal_engine_error, qwen3_5_runtime_error};
use crate::expert_paging::ExpertWeightMemoryCacheStatistics;
use crate::qwen3_5::{
    Qwen3_5Model, Qwen3_5MtpRequestState, Qwen3_5MtpRequestStateAllocationCheckpoint,
    RequestDecoderStateStack, RequestDecoderStateStackAllocationCheckpoint,
};

pub(super) struct AcceptedMtpDraftRollback {
    pub(super) verified_prefix_boundary_checkpoint:
        crate::Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    pub(super) verified_prefix_position_tokens: u32,
}

/// Retained request state needed to retry one rejected prompt-processing attempt.
pub(super) struct Qwen3_5PrefillRequestCheckpoint {
    request_decoder_state_allocation_checkpoint: RequestDecoderStateStackAllocationCheckpoint,
    mtp_request_state_allocation_checkpoint: Option<Qwen3_5MtpRequestStateAllocationCheckpoint>,
    prefill_cursor: usize,
    next_position_tokens: u32,
    consumed_visual_embedding_count: usize,
}

pub(in crate::qwen3_5) struct Qwen3_5EngineRequest {
    pub(super) request_decoder_state: RequestDecoderStateStack,
    pub(super) generated_token_count: u16,
    pub(super) input_token_ids: Vec<u32>,
    pub(super) last_restored_persistent_prompt_cache_block_key:
        Option<PersistentPromptCacheBlockKey>,
    pub(super) can_use_persistent_prompt_cache: bool,
    pub(super) maximum_output_tokens: u16,
    pub(super) ordered_image_sha256_digests: Vec<[u8; 32]>,
    pub(super) next_position_tokens: u32,
    pub(super) pending_generated_token: Option<MlxArray>,
    pub(super) persistent_prompt_cache_capture_has_stopped: bool,
    pub(super) prefill_cursor: usize,
    /// Largest chunk proven to fit after a capacity-driven retry in this request.
    pub(super) maximum_successful_prefill_chunck_tokens: Option<usize>,
    pub(super) random_state: Option<MlxArray>,
    pub(super) request_id: RequestId,
    pub(super) sampling_strategy: Qwen3_5SamplingStrategy,
    /// Pre-uploaded visual embeddings for image prompts; `None` for text-only.
    pub(super) visual_embeddings: Option<MlxArray>,
    /// How many visual embeddings earlier prefill chunks have already consumed.
    pub(super) consumed_visual_embedding_count: usize,
    /// Token ID used for image-pad placeholders in the input prompt.
    pub(super) image_pad_token_id: u32,
    /// Maximum tokens the model may spend inside the thinking block.
    /// `None` means no budget.
    pub(super) thinking_budget: Option<u16>,
    /// Count of tokens generated so far inside the current thinking block.
    pub(super) thinking_token_count: u16,
    /// Whether the model is currently inside a thinking block.
    pub(super) is_inside_thinking: bool,
    pub(super) expert_weight_memory_cache_statistics_at_request_start:
        ExpertWeightMemoryCacheStatistics,
    pub(super) performance_attribution: PerformanceAttribution,
    pub(super) mtp_request_state: Option<Qwen3_5MtpRequestState>,
    pub(super) mtp_target_hidden_states: Option<MlxArray>,
    pub(super) verified_mtp_generated_token_ids: VecDeque<u32>,
    pub(super) accepted_mtp_draft_rollback: Option<AcceptedMtpDraftRollback>,
    pub(super) force_next_mtp_draft_rejection_for_tests: bool,
    pub(super) force_next_prefill_capacity_rejection_for_tests: bool,
}

impl Qwen3_5EngineRequest {
    /// Retains mutable prompt state before an attempt that can hit MLX's hard ceiling.
    pub(super) fn prefill_request_checkpoint(
        &self,
    ) -> Result<Qwen3_5PrefillRequestCheckpoint, MlxRuntimeError> {
        Ok(Qwen3_5PrefillRequestCheckpoint {
            request_decoder_state_allocation_checkpoint: self
                .request_decoder_state
                .allocation_checkpoint()?,
            mtp_request_state_allocation_checkpoint: self
                .mtp_request_state
                .as_ref()
                .map(Qwen3_5MtpRequestState::allocation_checkpoint)
                .transpose()?,
            prefill_cursor: self.prefill_cursor,
            next_position_tokens: self.next_position_tokens,
            consumed_visual_embedding_count: self.consumed_visual_embedding_count,
        })
    }

    /// Restores mutable prompt state after MLX rejected an allocation before a retry.
    pub(super) fn restore_prefill_request_checkpoint(
        &mut self,
        prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        if self.mtp_request_state.is_some()
            != prefill_request_checkpoint
                .mtp_request_state_allocation_checkpoint
                .is_some()
        {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "restore Qwen3.5 prefill request checkpoint",
                description:
                    "MTP request-state availability changed during a retryable prompt attempt"
                        .to_owned(),
            });
        }
        self.request_decoder_state.restore_allocation_checkpoint(
            prefill_request_checkpoint.request_decoder_state_allocation_checkpoint,
        )?;
        if let (Some(mtp_request_state), Some(mtp_request_state_allocation_checkpoint)) = (
            self.mtp_request_state.as_mut(),
            prefill_request_checkpoint.mtp_request_state_allocation_checkpoint,
        ) {
            mtp_request_state
                .restore_allocation_checkpoint(mtp_request_state_allocation_checkpoint)?;
        }
        self.prefill_cursor = prefill_request_checkpoint.prefill_cursor;
        self.next_position_tokens = prefill_request_checkpoint.next_position_tokens;
        self.consumed_visual_embedding_count =
            prefill_request_checkpoint.consumed_visual_embedding_count;
        Ok(())
    }

    #[must_use]
    pub(super) fn clamped_prefill_chunck_token_count(
        &self,
        requested_prefill_chunck_token_count: usize,
    ) -> usize {
        self.maximum_successful_prefill_chunck_tokens.map_or(
            requested_prefill_chunck_token_count,
            |maximum_successful_prefill_chunck_tokens| {
                requested_prefill_chunck_token_count.min(maximum_successful_prefill_chunck_tokens)
            },
        )
    }

    pub(super) fn record_successful_capacity_prefill_chunck(
        &mut self,
        successful_prefill_chunck_token_count: usize,
    ) {
        self.maximum_successful_prefill_chunck_tokens = Some(successful_prefill_chunck_token_count);
    }

    pub(super) fn measure_operation_with_request<OperationOutput>(
        &mut self,
        performance_operation: PerformanceOperation,
        operation: impl FnOnce(&mut Self) -> OperationOutput,
    ) -> OperationOutput {
        if !self.performance_attribution.is_enabled() {
            return operation(self);
        }

        let mut request_performance_attribution = std::mem::replace(
            &mut self.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        let operation_output = request_performance_attribution
            .measure_operation(performance_operation, |_performance_attribution| {
                operation(self)
            });
        self.performance_attribution = request_performance_attribution;
        operation_output
    }

    pub(super) fn build_generated_token(
        &mut self,
        model: &Qwen3_5Model,
        logits: &MlxArray,
    ) -> Result<MlxArray, InferenceEngineError> {
        let sampling_strategy = self.sampling_strategy;
        let mut sampling_random_state = self.random_state.take();
        let generated_token_outcome = self.performance_attribution.measure_operation(
            PerformanceOperation::TokenSamplingGraphConstruction,
            |_performance_attribution| match sampling_strategy {
                Qwen3_5SamplingStrategy::Greedy => model
                    .build_greedy_token(logits)
                    .map_err(qwen3_5_runtime_error),
                Qwen3_5SamplingStrategy::TopKTopP {
                    temperature_thousandths,
                    top_k,
                    top_p_thousandths,
                    ..
                } => build_qwen3_5_sampled_token(
                    model,
                    logits,
                    temperature_thousandths,
                    top_p_thousandths,
                    top_k,
                    sampling_random_state.as_mut().ok_or_else(|| {
                        fatal_engine_error("sampled request lost its random state")
                    })?,
                ),
            },
        );
        self.random_state = sampling_random_state;
        generated_token_outcome
    }

    pub(super) fn advance_position(
        &mut self,
        forwarded_token_count: usize,
    ) -> Result<(), InferenceEngineError> {
        let forwarded_token_count = u32::try_from(forwarded_token_count)
            .map_err(|_| fatal_engine_error("forwarded token count exceeds the u32 range"))?;
        self.next_position_tokens = self
            .next_position_tokens
            .checked_add(forwarded_token_count)
            .ok_or_else(|| fatal_engine_error("model position counter overflowed"))?;
        Ok(())
    }
}
