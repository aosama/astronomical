//! Mutable state for one serial Qwen3.5 generation journey.
//!
//! This owner is the transaction boundary for prompt retries, sampling state,
//! and performance attribution. State that can be rewound is deliberately kept
//! here rather than hidden in the model owner.

use astronomical_ipc_protocol::{
    RequestId, WorkerPersistentPromptCacheRequestDiagnostics, WorkerPromptWorkReuse,
};

/// Deterministic failure points used by isolated SpecPrefill acceptance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub enum Qwen3_5SpeculativePrefillFailureStageForTests {
    DrafterLoading,
    DraftScoring,
    Selection,
    DrafterPromptStatePersistence,
    SelectionPersistence,
    SparseTargetInputAssembly,
    SparseTargetActiveMemoryLimitRejection,
    SparseTargetExecution,
    SparseTargetStatePersistence,
}
use astronomical_runtime_integration::{MlxArray, MlxRuntimeError};

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockCausalInput, PersistentPromptCacheBlockKey, Qwen3_5SamplingStrategy,
    Qwen3_5ThinkingBudgetState,
};

use super::super::text::sampler::build_qwen3_5_sampled_token;
use super::{fatal_engine_error, qwen3_5_runtime_error};
use crate::expert_paging::ExpertWeightMemoryCacheStatistics;
use crate::qwen3_5::multi_token_prediction::{
    MultiTokenPredictionRequestAllocationCheckpoint, Qwen3_5MultiTokenPredictionRequest,
};
use crate::qwen3_5::{
    Qwen3_5Model, Qwen3_5ProcessedImage, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint,
};

/// Retained request state needed to retry one rejected prompt-processing attempt.
pub(super) struct Qwen3_5PrefillRequestCheckpoint {
    request_decoder_state_allocation_checkpoint: RequestDecoderStateStackAllocationCheckpoint,
    optional_prediction_session_allocation_checkpoint:
        Option<MultiTokenPredictionRequestAllocationCheckpoint>,
    prefill_cursor: usize,
    next_position_tokens: u32,
    consumed_visual_embedding_count: usize,
}

pub(in crate::qwen3_5) struct Qwen3_5EngineRequest {
    pub(super) request_decoder_state: RequestDecoderStateStack,
    pub(super) generated_token_count: u16,
    pub(super) input_token_ids: Vec<u32>,
    /// Complete leading system-and-tool tokens that must use ordinary target prefill.
    pub(super) ordinary_target_prefill_control_span_token_count: usize,
    pub(super) last_restored_persistent_prompt_cache_block_key:
        Option<PersistentPromptCacheBlockKey>,
    pub(super) can_use_persistent_prompt_cache: bool,
    pub(super) maximum_output_tokens: u16,
    pub(super) ordered_image_sha256_digests: Vec<[u8; 32]>,
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
    pub(super) expert_weight_memory_cache_statistics_at_request_start:
        ExpertWeightMemoryCacheStatistics,
    pub(super) performance_attribution: PerformanceAttribution,
    pub(super) optional_prediction_session: Option<Qwen3_5MultiTokenPredictionRequest>,
    /// Whether this request may use draft-assisted sparse prompt prefill.
    pub(super) should_use_speculative_prefill: bool,
    /// Marks the one-time draft scoring attempt for this request.
    pub(super) speculative_prefill_scoring_attempted: bool,
    /// Whether the worker has been told that a confirmed drafter phase will begin.
    pub(super) speculative_prefill_draft_phase_announced: bool,
    /// Original prompt positions retained for target sparse prefill.
    pub(super) speculative_prefill_selected_token_positions: Option<Vec<usize>>,
    /// Complete exact target prefix processed before sparse conversation positions.
    pub(super) speculative_prefill_dense_target_prefix_token_count: usize,
    /// Full prompt token indices retained on the MLX device for sparse gathers.
    pub(super) speculative_prefill_prompt_token_indices: Option<MlxArray>,
    /// CPU-processed source images retained only while visual draft scoring needs them.
    pub(super) speculative_prefill_processed_visual_images: Vec<Qwen3_5ProcessedImage>,
    /// GPU-resident selected rows restored with an earlier sparse target prompt prefix.
    pub(super) speculative_prefill_restored_target_token_positions: Option<MlxArray>,
    /// Target expert payload retained immediately after the request-scoped draft release.
    pub(super) speculative_prefill_target_expert_payload_bytes_after_draft_release: Option<u64>,
    /// Active MLX telemetry captured before the request-scoped drafter is released.
    pub(super) speculative_prefill_draft_memory_telemetry: Option<crate::MlxMemoryTelemetry>,
    pub(super) prompt_work_reuse: WorkerPromptWorkReuse,
    pub(super) persistent_prompt_cache_diagnostics:
        Option<WorkerPersistentPromptCacheRequestDiagnostics>,
    pub(super) force_next_speculative_prefill_draft_prefix_restore_failure_for_tests: bool,
    pub(super) forced_speculative_prefill_failure_stage_for_tests:
        Option<Qwen3_5SpeculativePrefillFailureStageForTests>,
    pub(super) force_next_prefill_capacity_rejection_for_tests: bool,
    /// One-shot guard for no-I/O prefill-to-decode residency reconciliation.
    /// Mandatory decode reads populate any remaining elastic route ownership.
    pub(super) generation_residency_preparation_attempted: bool,
    /// First decode-forward latency waiting to be emitted with its generated token.
    pub(super) first_decode_forward_elapsed_millis: Option<u64>,
    /// Ensures the worker observes generation preparation before blocking handoff work.
    pub(super) generation_preparation_announced: bool,
}

impl Qwen3_5EngineRequest {
    pub(super) fn take_forced_speculative_prefill_failure_for_tests(
        &mut self,
        expected_failure_stage: Qwen3_5SpeculativePrefillFailureStageForTests,
    ) -> bool {
        if self.forced_speculative_prefill_failure_stage_for_tests != Some(expected_failure_stage) {
            return false;
        }
        self.forced_speculative_prefill_failure_stage_for_tests = None;
        true
    }

    /// Retains mutable prompt state before an attempt that can hit MLX's hard ceiling.
    pub(super) fn prefill_request_checkpoint(
        &self,
    ) -> Result<Qwen3_5PrefillRequestCheckpoint, MlxRuntimeError> {
        Ok(Qwen3_5PrefillRequestCheckpoint {
            request_decoder_state_allocation_checkpoint: self
                .request_decoder_state
                .allocation_checkpoint()?,
            optional_prediction_session_allocation_checkpoint: self
                .optional_prediction_session
                .as_ref()
                .map(Qwen3_5MultiTokenPredictionRequest::allocation_checkpoint)
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
        if self.optional_prediction_session.is_some()
            != prefill_request_checkpoint
                .optional_prediction_session_allocation_checkpoint
                .is_some()
        {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "restore Qwen3.5 prefill request checkpoint",
                description:
                    "multi-token prediction request-state availability changed during a retryable prompt attempt"
                        .to_owned(),
            });
        }
        self.request_decoder_state.restore_allocation_checkpoint(
            prefill_request_checkpoint.request_decoder_state_allocation_checkpoint,
        )?;
        if let (Some(optional_prediction_session), Some(allocation_checkpoint)) = (
            self.optional_prediction_session.as_mut(),
            prefill_request_checkpoint.optional_prediction_session_allocation_checkpoint,
        ) {
            optional_prediction_session.restore_allocation_checkpoint(allocation_checkpoint)?;
        }
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

    pub(in crate::qwen3_5) fn build_generated_token(
        &mut self,
        model: &Qwen3_5Model,
        logits: &MlxArray,
    ) -> Result<MlxArray, InferenceEngineError> {
        let sampling_strategy = self.sampling_strategy;
        let mut sampling_random_state = self.random_state.take();
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

    /// Returns the sampling strategy this request resolves its tokens with.
    pub(in crate::qwen3_5) fn sampling_strategy(&self) -> Qwen3_5SamplingStrategy {
        self.sampling_strategy
    }

    /// Hands the keyed sampling stream to decode operations that need their own
    /// random-key splits, such as sampled multi-token-prediction verification.
    pub(in crate::qwen3_5) fn take_sampling_random_state(
        &mut self,
    ) -> Result<MlxArray, InferenceEngineError> {
        self.random_state
            .take()
            .ok_or_else(|| fatal_engine_error("sampled request lost its random state"))
    }

    /// Returns the keyed sampling stream after a decode operation used it.
    pub(in crate::qwen3_5) fn restore_sampling_random_state(
        &mut self,
        sampling_random_state: MlxArray,
    ) {
        self.random_state = Some(sampling_random_state);
    }

    pub(crate) fn advance_position(
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

    pub(crate) fn take_optional_prediction_session(
        &mut self,
    ) -> Option<Qwen3_5MultiTokenPredictionRequest> {
        self.optional_prediction_session.take()
    }

    pub(crate) fn restore_optional_prediction_session(
        &mut self,
        optional_prediction_session: Qwen3_5MultiTokenPredictionRequest,
    ) {
        self.optional_prediction_session = Some(optional_prediction_session);
    }

    pub(crate) fn clear_optional_prediction_session(&mut self) {
        self.optional_prediction_session = None;
    }

    pub(crate) fn optional_prediction_session_mut(
        &mut self,
    ) -> Option<&mut Qwen3_5MultiTokenPredictionRequest> {
        self.optional_prediction_session.as_mut()
    }

    pub(crate) fn optional_prediction_session(
        &self,
    ) -> Option<&Qwen3_5MultiTokenPredictionRequest> {
        self.optional_prediction_session.as_ref()
    }

    pub(crate) fn has_optional_prediction_session(&self) -> bool {
        self.optional_prediction_session.is_some()
    }

    pub(crate) fn has_queued_prediction_tokens(&self) -> bool {
        self.optional_prediction_session
            .as_ref()
            .is_some_and(|prediction_request| prediction_request.has_verified_generated_token_ids())
    }

    pub(crate) fn additional_context_state_payload_bytes(&self) -> u64 {
        self.optional_prediction_session
            .as_ref()
            .map_or(0, |optional_prediction_session| {
                optional_prediction_session.context_state_payload_byte_count()
            })
    }

    pub(crate) fn request_decoder_state(&self) -> &RequestDecoderStateStack {
        &self.request_decoder_state
    }

    pub(crate) fn request_decoder_state_mut(&mut self) -> &mut RequestDecoderStateStack {
        &mut self.request_decoder_state
    }

    pub(crate) fn next_position_tokens(&self) -> u32 {
        self.next_position_tokens
    }

    pub(crate) fn generated_token_count(&self) -> u16 {
        self.generated_token_count
    }

    pub(crate) fn maximum_output_tokens(&self) -> u16 {
        self.maximum_output_tokens
    }

    pub(crate) fn is_inside_thinking(&self) -> bool {
        self.thinking_budget_state.is_inside_thinking()
    }

    pub(crate) fn thinking_token_count(&self) -> u16 {
        self.thinking_budget_state.thinking_token_count()
    }

    pub(crate) fn thinking_budget(&self) -> Option<u16> {
        self.thinking_budget_state.thinking_budget()
    }

    pub(super) fn next_forced_thinking_transition_token_id(
        &mut self,
    ) -> Result<Option<u32>, InferenceEngineError> {
        self.thinking_budget_state
            .next_forced_transition_token_id()
            .map_err(|source| {
                fatal_engine_error(format!("invalid Qwen3.5 thinking-budget state: {source}"))
            })
    }

    pub(super) fn observe_committed_thinking_token(
        &mut self,
        committed_token_id: u32,
    ) -> Result<bool, InferenceEngineError> {
        self.thinking_budget_state
            .observe_committed_token(committed_token_id)
            .map_err(|source| {
                fatal_engine_error(format!("invalid Qwen3.5 thinking-budget state: {source}"))
            })
    }

    pub(super) fn is_forcing_thinking_transition(&self) -> bool {
        self.thinking_budget_state.is_forcing_transition()
    }

    pub(crate) fn set_next_position_tokens(&mut self, next_position_tokens: u32) {
        self.next_position_tokens = next_position_tokens;
    }

    pub(crate) fn set_pending_generated_token(&mut self, pending_generated_token: MlxArray) {
        self.pending_generated_token = Some(pending_generated_token);
    }

    pub(crate) fn clear_pending_generated_token(&mut self) {
        self.pending_generated_token = None;
    }

    pub(crate) fn performance_attribution_mut(&mut self) -> &mut PerformanceAttribution {
        &mut self.performance_attribution
    }

    pub(crate) fn with_decoder_state_and_performance_attribution<OperationOutput>(
        &mut self,
        operation: impl FnOnce(
            &mut RequestDecoderStateStack,
            &mut PerformanceAttribution,
        ) -> OperationOutput,
    ) -> OperationOutput {
        let Self {
            request_decoder_state,
            performance_attribution,
            ..
        } = self;
        operation(request_decoder_state, performance_attribution)
    }

    pub(crate) fn with_optional_prediction_session_and_performance_attribution<OperationOutput>(
        &mut self,
        operation: impl FnOnce(
            &mut Qwen3_5MultiTokenPredictionRequest,
            &mut PerformanceAttribution,
        ) -> OperationOutput,
    ) -> Option<OperationOutput> {
        let Self {
            optional_prediction_session,
            performance_attribution,
            ..
        } = self;
        optional_prediction_session
            .as_mut()
            .map(|optional_prediction_session| {
                operation(optional_prediction_session, performance_attribution)
            })
    }

    pub(crate) fn with_input_token_range_and_optional_prediction_session_and_performance_attribution<
        OperationOutput,
    >(
        &mut self,
        input_token_start: usize,
        input_token_end: usize,
        operation: impl FnOnce(
            &[u32],
            &mut Qwen3_5MultiTokenPredictionRequest,
            &mut PerformanceAttribution,
        ) -> OperationOutput,
    ) -> Option<OperationOutput> {
        let Self {
            input_token_ids,
            optional_prediction_session,
            performance_attribution,
            ..
        } = self;
        let input_token_range = input_token_ids.get(input_token_start..input_token_end)?;
        optional_prediction_session
            .as_mut()
            .map(|optional_prediction_session| {
                operation(
                    input_token_range,
                    optional_prediction_session,
                    performance_attribution,
                )
            })
    }

    pub(crate) fn with_input_token_range_and_decoder_state_and_performance_attribution<
        OperationOutput,
    >(
        &mut self,
        input_token_start: usize,
        input_token_end: usize,
        operation: impl FnOnce(
            &[u32],
            u32,
            &mut RequestDecoderStateStack,
            &mut PerformanceAttribution,
        ) -> OperationOutput,
    ) -> Option<OperationOutput> {
        let Self {
            input_token_ids,
            next_position_tokens,
            request_decoder_state,
            performance_attribution,
            ..
        } = self;
        let input_token_range = input_token_ids.get(input_token_start..input_token_end)?;
        Some(operation(
            input_token_range,
            *next_position_tokens,
            request_decoder_state,
            performance_attribution,
        ))
    }
}
