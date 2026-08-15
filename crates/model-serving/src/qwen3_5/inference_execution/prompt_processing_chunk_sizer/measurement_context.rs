//! Maps Qwen request conditions into model-independent optimizer contexts.
//!
//! The low 32 bits encode prompt position range. Higher bits encode execution
//! conditions whose forward costs cannot safely share evidence. Clearing only
//! the position bits yields the profile used for cross-range measurement reuse.

use super::Qwen3_5PromptProcessingChunkSizer;
use crate::qwen3_5::inference_execution::prefill_execution_context::{
    CAPACITY_REDUCED_CONTEXT_FLAG, Qwen3_5PrefillExecutionContext,
};
use crate::{PromptProcessingChunkOptimizationContext, PromptProcessingMeasurementContext};

const RESTORED_PREFIX_CONTEXT_FLAG: u64 = 1 << 32;
const FIRST_CHUNK_AFTER_RESTORE_CONTEXT_FLAG: u64 = 1 << 33;
const POSITION_RANGE_IDENTIFIER_MASK: u64 = (1_u64 << 32) - 1;

impl Qwen3_5PromptProcessingChunkSizer {
    /// Returns the encoded exact context used to isolate optimizer measurements.
    #[must_use]
    pub(in crate::qwen3_5::inference_execution) fn exact_measurement_context_identifier(
        &self,
        chunk_start_token_position: usize,
        prompt_processing_execution_context: Qwen3_5PrefillExecutionContext,
    ) -> u64 {
        let position_range_identifier = self
            .position_range_size_tokens
            .map(|position_range_size_tokens| {
                u64::try_from(chunk_start_token_position / position_range_size_tokens)
                    .unwrap_or(u64::MAX >> 2)
            })
            .unwrap_or(0)
            .min(POSITION_RANGE_IDENTIFIER_MASK);
        let mut exact_measurement_context_identifier = position_range_identifier
            | prompt_processing_execution_context.context_identifier_flags();
        if self.active_request_restored_token_count > 0 {
            exact_measurement_context_identifier |= RESTORED_PREFIX_CONTEXT_FLAG;
        }
        if self.active_request_restored_token_count > 0
            && !self.has_completed_prompt_processing_chunk_in_active_request
        {
            exact_measurement_context_identifier |= FIRST_CHUNK_AFTER_RESTORE_CONTEXT_FLAG;
        }
        if self.active_request_has_observed_capacity_reduction {
            exact_measurement_context_identifier |= CAPACITY_REDUCED_CONTEXT_FLAG;
        }
        exact_measurement_context_identifier
    }

    pub(super) fn measurement_context_for_chunk_start(
        &self,
        chunk_start_token_position: usize,
        prompt_processing_execution_context: Qwen3_5PrefillExecutionContext,
    ) -> PromptProcessingMeasurementContext {
        let exact_measurement_context_identifier = self.exact_measurement_context_identifier(
            chunk_start_token_position,
            prompt_processing_execution_context,
        );
        PromptProcessingMeasurementContext::with_position_independent_execution_profile(
            exact_measurement_context_identifier,
            exact_measurement_context_identifier & !POSITION_RANGE_IDENTIFIER_MASK,
        )
    }

    /// Builds the position range and execution conditions exposed through telemetry.
    pub(super) fn optimization_context_for_chunk_start(
        &self,
        chunk_start_token_position: usize,
        prompt_processing_execution_context: Qwen3_5PrefillExecutionContext,
    ) -> PromptProcessingChunkOptimizationContext {
        let (position_range_start_token_position, position_range_end_token_position_exclusive) =
            self.position_range_size_tokens
                .map(|position_range_size_tokens| {
                    let position_range_start_token_position = (chunk_start_token_position
                        / position_range_size_tokens)
                        * position_range_size_tokens;
                    let position_range_end_token_position_exclusive =
                        position_range_start_token_position
                            .saturating_add(position_range_size_tokens);
                    (
                        position_range_start_token_position,
                        position_range_end_token_position_exclusive,
                    )
                })
                .unwrap_or((0, usize::MAX));
        PromptProcessingChunkOptimizationContext {
            chunk_start_token_position,
            position_range_start_token_position,
            position_range_end_token_position_exclusive,
            has_restored_prefix: self.active_request_restored_token_count > 0,
            is_first_chunk_after_restore: self.active_request_restored_token_count > 0
                && !self.has_completed_prompt_processing_chunk_in_active_request,
            has_visual_embeddings: prompt_processing_execution_context.has_visual_embeddings(),
            is_mtp_active: prompt_processing_execution_context.has_optional_prediction_session(),
            are_sparse_experts_paged: prompt_processing_execution_context
                .are_sparse_experts_paged(),
            is_prompt_cache_capture_eligible: prompt_processing_execution_context
                .is_prompt_cache_capture_eligible(),
            has_prior_capacity_reduction: self.active_request_has_observed_capacity_reduction,
        }
    }
}
