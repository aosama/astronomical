//! Maps Laguna request conditions into model-independent optimizer contexts.

use super::{LagunaPromptProcessingChunkSizer, LagunaPromptProcessingExecutionProfile};
use crate::{PromptProcessingChunkOptimizationContext, PromptProcessingMeasurementContext};

const RESTORED_PREFIX_CONTEXT_FLAG: u64 = 1 << 32;
const FIRST_CHUNK_AFTER_RESTORE_CONTEXT_FLAG: u64 = 1 << 33;
const PAGED_EXPERTS_CONTEXT_FLAG: u64 = 1 << 34;
const PROMPT_CACHE_CAPTURE_ELIGIBLE_CONTEXT_FLAG: u64 = 1 << 35;
const CAPACITY_REDUCED_CONTEXT_FLAG: u64 = 1 << 36;
const POSITION_RANGE_IDENTIFIER_MASK: u64 = (1_u64 << 32) - 1;
const EXECUTION_PROFILE_DIGEST_SHIFT: u32 = 40;

impl LagunaPromptProcessingChunkSizer {
    /// Encodes position range, request flags, and the canonical execution digest.
    #[must_use]
    pub fn exact_measurement_context_identifier(
        &self,
        chunk_start_token_position: usize,
        execution_profile: LagunaPromptProcessingExecutionProfile,
    ) -> u64 {
        let position_range_identifier = self
            .position_range_size_tokens()
            .map(|position_range_size_tokens| {
                u64::try_from(chunk_start_token_position / position_range_size_tokens)
                    .unwrap_or(u64::MAX >> 2)
            })
            .unwrap_or(0)
            .min(POSITION_RANGE_IDENTIFIER_MASK);
        let mut exact_measurement_context_identifier = position_range_identifier
            | ((execution_profile.execution_profile_digest() & 0x00FF_FFFF)
                << EXECUTION_PROFILE_DIGEST_SHIFT);
        if execution_profile.are_sparse_experts_paged() {
            exact_measurement_context_identifier |= PAGED_EXPERTS_CONTEXT_FLAG;
        }
        if execution_profile.is_prompt_cache_capture_eligible() {
            exact_measurement_context_identifier |= PROMPT_CACHE_CAPTURE_ELIGIBLE_CONTEXT_FLAG;
        }
        if self.active_request_restored_token_count() > 0 {
            exact_measurement_context_identifier |= RESTORED_PREFIX_CONTEXT_FLAG;
        }
        if self.active_request_restored_token_count() > 0
            && !self.has_completed_prompt_processing_chunk_in_active_request()
        {
            exact_measurement_context_identifier |= FIRST_CHUNK_AFTER_RESTORE_CONTEXT_FLAG;
        }
        if self.active_request_has_observed_capacity_reduction() {
            exact_measurement_context_identifier |= CAPACITY_REDUCED_CONTEXT_FLAG;
        }
        exact_measurement_context_identifier
    }

    pub(super) fn measurement_context_for_chunk_start(
        &self,
        chunk_start_token_position: usize,
        execution_profile: LagunaPromptProcessingExecutionProfile,
    ) -> PromptProcessingMeasurementContext {
        let exact_measurement_context_identifier = self
            .exact_measurement_context_identifier(chunk_start_token_position, execution_profile);
        PromptProcessingMeasurementContext::with_position_independent_execution_profile(
            exact_measurement_context_identifier,
            exact_measurement_context_identifier & !POSITION_RANGE_IDENTIFIER_MASK,
        )
    }

    pub(super) fn optimization_context_for_chunk_start(
        &self,
        chunk_start_token_position: usize,
        execution_profile: LagunaPromptProcessingExecutionProfile,
    ) -> PromptProcessingChunkOptimizationContext {
        let (position_range_start_token_position, position_range_end_token_position_exclusive) =
            self.position_range_size_tokens()
                .map(|position_range_size_tokens| {
                    let position_range_start_token_position = (chunk_start_token_position
                        / position_range_size_tokens)
                        * position_range_size_tokens;
                    (
                        position_range_start_token_position,
                        position_range_start_token_position
                            .saturating_add(position_range_size_tokens),
                    )
                })
                .unwrap_or((0, usize::MAX));
        PromptProcessingChunkOptimizationContext {
            chunk_start_token_position,
            position_range_start_token_position,
            position_range_end_token_position_exclusive,
            has_restored_prefix: self.active_request_restored_token_count() > 0,
            is_first_chunk_after_restore: self.active_request_restored_token_count() > 0
                && !self.has_completed_prompt_processing_chunk_in_active_request(),
            has_visual_embeddings: false,
            is_mtp_active: false,
            are_sparse_experts_paged: execution_profile.are_sparse_experts_paged(),
            is_prompt_cache_capture_eligible: execution_profile.is_prompt_cache_capture_eligible(),
            has_prior_capacity_reduction: self.active_request_has_observed_capacity_reduction(),
        }
    }
}
