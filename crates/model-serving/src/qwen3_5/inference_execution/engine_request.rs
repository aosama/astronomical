use astronomical_ipc_protocol::{RequestId, WorkerPromptWorkReuse};
use astronomical_runtime_integration::{MlxArray, MlxRuntimeError};

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockKey, Qwen3_5SamplingStrategy,
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
    pub(super) optional_prediction_session: Option<Qwen3_5MultiTokenPredictionRequest>,
    /// Whether this request may use draft-assisted sparse prompt prefill.
    pub(super) should_use_speculative_prefill: bool,
    /// Marks the one-time draft scoring attempt for this request.
    pub(super) speculative_prefill_scoring_attempted: bool,
    /// Original prompt positions retained for target sparse prefill.
    pub(super) speculative_prefill_selected_token_positions: Option<Vec<usize>>,
    /// Full prompt token indices retained on the MLX device for sparse gathers.
    pub(super) speculative_prefill_prompt_token_indices: Option<MlxArray>,
    /// CPU-processed source images retained only while visual draft scoring needs them.
    pub(super) speculative_prefill_processed_visual_images: Vec<Qwen3_5ProcessedImage>,
    /// GPU-resident selected rows restored with an earlier sparse target prompt prefix.
    pub(super) speculative_prefill_restored_target_token_positions: Option<MlxArray>,
    /// Target expert payload retained immediately after the request-scoped draft release.
    pub(super) speculative_prefill_target_expert_payload_bytes_after_draft_release: Option<u64>,
    pub(super) prompt_work_reuse: WorkerPromptWorkReuse,
    pub(super) force_next_speculative_prefill_draft_prefix_restore_failure_for_tests: bool,
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

    pub(crate) fn measure_operation_with_request<OperationOutput>(
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
        self.is_inside_thinking
    }

    pub(crate) fn thinking_token_count(&self) -> u16 {
        self.thinking_token_count
    }

    pub(crate) fn thinking_budget(&self) -> Option<u16> {
        self.thinking_budget
    }

    pub(crate) fn set_next_position_tokens(&mut self, next_position_tokens: u32) {
        self.next_position_tokens = next_position_tokens;
    }

    pub(crate) fn set_pending_generated_token(&mut self, pending_generated_token: MlxArray) {
        self.pending_generated_token = Some(pending_generated_token);
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
