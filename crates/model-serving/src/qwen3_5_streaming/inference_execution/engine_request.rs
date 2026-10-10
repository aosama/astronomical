//! Mutable state for one serial Qwen3.5 generation journey.
//!
//! This owner is the transaction boundary for prompt retries, sampling state,
//! and performance attribution. State that can be rewound is deliberately kept
//! here rather than hidden in the model owner.

use astronomical_ipc_protocol::{
    RequestId, WorkerPersistentPromptCacheRequestDiagnostics, WorkerPromptWorkReuse,
};
use astronomical_runtime_integration::MlxRuntimeError;

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockCausalInput, PersistentPromptCacheBlockKey, Qwen3_5SamplingStrategy,
    Qwen3_5ThinkingBudgetState,
};

use super::qwen3_5_runtime_error;
use crate::qwen3_5::{
    Qwen3_5Model, RequestDecoderStateStack, RequestDecoderStateStackAllocationCheckpoint,
};
use crate::qwen3_5_core::text::sampler;
use astronomical_mlx_c_rust::MlxArray;

/// Retained request state needed to retry one rejected prompt-processing attempt.
pub(super) struct Qwen3_5PrefillRequestCheckpoint {
    request_decoder_state_allocation_checkpoint: RequestDecoderStateStackAllocationCheckpoint,
    prefill_cursor: usize,
    next_position_tokens: u32,
    consumed_visual_embedding_count: usize,
}

pub(in crate::qwen3_5_streaming) struct Qwen3_5EngineRequest {
    pub(super) request_decoder_state: RequestDecoderStateStack,
    pub(super) generated_token_count: u16,
    pub(super) input_token_ids: Vec<u32>,
    pub(super) last_restored_persistent_prompt_cache_block_key:
        Option<PersistentPromptCacheBlockKey>,
    pub(super) can_use_persistent_prompt_cache: bool,
    pub(super) maximum_output_tokens: u16,
    /// Qwen-owned visual identity aligned to ordinary target prompt-cache blocks.
    pub(super) persistent_prompt_cache_block_causal_inputs:
        Vec<PersistentPromptCacheBlockCausalInput>,
    pub(super) next_position_tokens: u32,
    /// One-token-ahead successor submitted during the previous advancement.
    pub(super) pending_generated_token: Option<MlxArray>,
    pub(super) prefill_cursor: usize,
    /// Largest chunk proven to fit after a capacity-driven retry in this request.
    pub(super) maximum_successful_prefill_chunk_tokens: Option<usize>,
    pub(super) random_state: Option<MlxArray>,
    pub(super) request_id: RequestId,
    pub(super) sampling_strategy: Qwen3_5SamplingStrategy,
    /// Pre-uploaded visual embeddings for image prompts; `None` for text-only.
    pub(super) visual_embeddings: Option<MlxArray>,
    /// How many visual embeddings earlier prefill chunks have already consumed.
    pub(super) consumed_visual_embedding_count: usize,
    /// Whether the request was constructed with an image source. Ordinary text may
    /// contain a vocabulary collision with the artifact's image-pad ID; only genuine
    /// image requests may demand visual embedding rows.
    pub(super) has_visual_inputs: bool,
    /// Token ID used for image-pad placeholders in the input prompt.
    pub(super) image_pad_token_id: u32,
    pub(super) thinking_budget_state: Qwen3_5ThinkingBudgetState,
    pub(super) performance_attribution: PerformanceAttribution,
    pub(super) prompt_work_reuse: WorkerPromptWorkReuse,
    pub(super) persistent_prompt_cache_diagnostics:
        Option<WorkerPersistentPromptCacheRequestDiagnostics>,
    pub(super) force_next_prefill_capacity_rejection_for_tests: bool,
    /// One-shot guard for no-I/O prefill-to-decode residency reconciliation.
    /// Mandatory decode reads populate any remaining elastic route ownership.
    pub(super) generation_residency_preparation_attempted: bool,
    /// First decode-forward latency waiting to be emitted with its generated token.
    pub(super) first_decode_forward_elapsed_millis: Option<u64>,
    /// Ensures the worker observes generation preparation before blocking handoff work.
    pub(super) generation_preparation_announced: bool,
    pub(super) structured_generation:
        Option<crate::structured_generation::StructuredTokenConstraint>,
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
        self.request_decoder_state.restore_allocation_checkpoint(
            prefill_request_checkpoint.request_decoder_state_allocation_checkpoint,
        )?;
        self.prefill_cursor = prefill_request_checkpoint.prefill_cursor;
        self.next_position_tokens = prefill_request_checkpoint.next_position_tokens;
        self.consumed_visual_embedding_count =
            prefill_request_checkpoint.consumed_visual_embedding_count;
        Ok(())
    }

    #[must_use]
    pub(super) fn clamped_prefill_chunk_token_count(
        &self,
        requested_prefill_chunk_token_count: usize,
        remaining_prompt_token_count: usize,
    ) -> usize {
        // A folded paged stub is the rest of the prompt and may exceed the last
        // proven size by less than one configured chunk. Capacity recovery still
        // halves if that forward cannot fit.
        if requested_prefill_chunk_token_count >= remaining_prompt_token_count {
            return remaining_prompt_token_count;
        }
        self.maximum_successful_prefill_chunk_tokens.map_or(
            requested_prefill_chunk_token_count,
            |maximum_successful_prefill_chunk_tokens| {
                requested_prefill_chunk_token_count.min(maximum_successful_prefill_chunk_tokens)
            },
        )
    }

    #[must_use]
    pub(super) const fn maximum_successful_prefill_chunk_tokens(&self) -> Option<usize> {
        self.maximum_successful_prefill_chunk_tokens
    }

    pub(super) fn record_successful_capacity_prefill_chunk(
        &mut self,
        successful_prefill_chunk_token_count: usize,
    ) {
        self.maximum_successful_prefill_chunk_tokens = Some(successful_prefill_chunk_token_count);
    }

    pub(crate) fn measure_operation_with_request<OperationOutput>(
        &mut self,
        performance_operation: PerformanceOperation,
        operation: impl FnOnce(&mut Self) -> OperationOutput,
    ) -> OperationOutput {
        if !self.performance_attribution.is_enabled() {
            return operation(self);
        }

        // Temporarily move attribution out of `self` so the measured closure
        // can borrow the complete mutable request. The disabled placeholder
        // prevents nested request code from recording the same outer span as a
        // leaf operation; the original accumulator is always restored.
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

    pub(in crate::qwen3_5_streaming) fn build_generated_token(
        &mut self,
        model: &Qwen3_5Model,
        logits: &MlxArray,
    ) -> Result<MlxArray, InferenceEngineError> {
        let sampling_strategy = self.sampling_strategy;
        let mut sampling_random_state = self.random_state.take();
        let outside_thinking = !self.is_inside_thinking() && !self.is_forcing_thinking_transition();
        let masked_logits = if outside_thinking
            && let Some(structured_generation) = self.structured_generation.as_mut()
        {
            let logit_bias_values = self.performance_attribution.measure_operation(
                PerformanceOperation::StructuredLogitMaskComputation,
                |_performance_attribution| structured_generation.logit_bias_values(),
            );
            Some(crate::gpu_token_sampling::add_token_logit_bias(
                &model.runtime,
                logits,
                &logit_bias_values,
            )?)
        } else {
            None
        };
        let logits = masked_logits.as_ref().unwrap_or(logits);
        let generated_token_outcome = self.performance_attribution.measure_operation(
            PerformanceOperation::TokenSamplingGraphConstruction,
            |_performance_attribution| match sampling_strategy {
                Qwen3_5SamplingStrategy::HighestLogit => model
                    .select_highest_logit_token(logits)
                    .map_err(qwen3_5_runtime_error),
                Qwen3_5SamplingStrategy::TopKTopP {
                    temperature_thousandths,
                    top_k,
                    top_p_thousandths,
                    ..
                } => sampler::build_qwen3_5_sampled_token(
                    model,
                    logits,
                    temperature_thousandths,
                    top_p_thousandths,
                    top_k,
                    sampling_random_state.as_mut().ok_or_else(|| {
                        super::fatal_engine_error("sampled request lost its random state")
                    })?,
                ),
            },
        );
        self.random_state = sampling_random_state;
        generated_token_outcome
    }

    pub(crate) fn advance_position(
        &mut self,
        forwarded_token_count: usize,
    ) -> Result<(), InferenceEngineError> {
        let forwarded_token_count = u32::try_from(forwarded_token_count).map_err(|_| {
            super::fatal_engine_error("forwarded token count exceeds the u32 range")
        })?;
        self.next_position_tokens = self
            .next_position_tokens
            .checked_add(forwarded_token_count)
            .ok_or_else(|| super::fatal_engine_error("model position counter overflowed"))?;
        Ok(())
    }

    pub(crate) fn is_inside_thinking(&self) -> bool {
        self.thinking_budget_state.is_inside_thinking()
    }

    pub(super) fn next_forced_thinking_transition_token_id(
        &mut self,
    ) -> Result<Option<u32>, InferenceEngineError> {
        self.thinking_budget_state
            .next_forced_transition_token_id()
            .map_err(|source| {
                super::fatal_engine_error(format!(
                    "invalid Qwen3.5 thinking-budget state: {source}"
                ))
            })
    }

    pub(super) fn observe_committed_thinking_token(
        &mut self,
        committed_token_id: u32,
    ) -> Result<bool, InferenceEngineError> {
        self.thinking_budget_state
            .observe_committed_token(committed_token_id)
            .map_err(|source| {
                super::fatal_engine_error(format!(
                    "invalid Qwen3.5 thinking-budget state: {source}"
                ))
            })
    }

    pub(super) fn is_forcing_thinking_transition(&self) -> bool {
        self.thinking_budget_state.is_forcing_transition()
    }

    pub(crate) fn set_pending_generated_token(&mut self, pending_generated_token: MlxArray) {
        self.pending_generated_token = Some(pending_generated_token);
    }
}
