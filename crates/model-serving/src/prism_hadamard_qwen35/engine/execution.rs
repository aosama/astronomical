//! Owner-thread folded-ternary execution state vocabulary.
//!
//! The execution structs and the pending-startup constructor live here; the
//! `MlxInferenceExecution` engine implementation that drives them lives in
//! `execution::engine_trait_impl`.

mod attribution;
mod engine_trait_impl;
mod input_injection;
mod sampling;

use astronomical_ipc_protocol::RequestId;
use astronomical_mlx_c_rust::MlxArray;

use crate::Qwen3_5InferenceRequest;
use crate::Qwen3_5ThinkingBudgetState;
use crate::prism_hadamard_qwen35::artifacts::ValidatedFoldedPack;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::prism_hadamard_qwen35::model::StraightGenerationState;
use crate::{PerformanceAttribution, PerformanceAttributionLog, PerformanceOperation};

/// Everything `load` needs, captured on the caller thread before the owner
/// thread starts.
pub struct PrismHadamardQwen35PendingStartup {
    pub validated_pack: ValidatedFoldedPack,
    pub effective_mlx_memory_ceiling_bytes: usize,
    pub allocator_cache_memory_limit_bytes: usize,
    pub maximum_context_tokens: u32,
    pub prompt_processing_chunk_tokens: u32,
    pub performance_attribution: PerformanceAttribution,
    pub performance_attribution_log: PerformanceAttributionLog,
    pub attribution_model_id: String,
    pub attribution_model_revision: String,
}

/// The synchronous folded-ternary execution owned by the MLX owner thread.
pub struct PrismHadamardQwen35InferenceExecution {
    pending_startup: Option<PrismHadamardQwen35PendingStartup>,
    pub(super) model: Option<PrismHadamardTextModel>,
    pub(super) active: Option<PrismHadamardQwen35ActiveGeneration>,
    pub(super) ceiling_bytes: u64,
    pub(super) allocator_cache_bytes: u64,
    pub(super) minimum_ceiling_bytes: u64,
    pub(super) maximum_context_tokens: u32,
    prompt_processing_chunk_tokens: u32,
    pub(super) performance_attribution_log: PerformanceAttributionLog,
    pub(super) attribution_model_id: Option<String>,
    pub(super) attribution_model_revision: Option<String>,
}

/// One active straight generation: chunked prefill plus one-token decode.
pub(super) struct PrismHadamardQwen35ActiveGeneration {
    pub(super) request_id: RequestId,
    pub(super) remaining_output_tokens: u32,
    pub(super) remaining_prompt_token_ids: Vec<u32>,
    pub(super) generation_state: StraightGenerationState,
    /// The next sampled token as a lazy `[1, 1]` device array whose graph
    /// is already submitted, or `None` once generation finished. Its value
    /// reaches the host only when the next step synchronizes it.
    pub(super) pending_sampled_token: Option<MlxArray>,
    /// Sorted unique ids the repetition penalty dampens.
    pub(super) penalized_token_ids: Vec<u32>,
    pub(super) random_state: MlxArray,
    pub(super) request: Qwen3_5InferenceRequest,
    pub(super) thinking_budget: Qwen3_5ThinkingBudgetState,
    pub(super) structured_generation:
        Option<crate::structured_generation::StructuredTokenConstraint>,
    pub(super) performance_attribution: PerformanceAttribution,
}

impl PrismHadamardQwen35ActiveGeneration {
    pub(super) fn measure_operation<OperationOutput>(
        &mut self,
        performance_operation: PerformanceOperation,
        operation: impl FnOnce(&mut Self) -> OperationOutput,
    ) -> OperationOutput {
        if !self.performance_attribution.is_enabled() {
            return operation(self);
        }
        let mut performance_attribution = std::mem::replace(
            &mut self.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        let operation_output = performance_attribution
            .measure_operation(performance_operation, |_performance_attribution| {
                operation(self)
            });
        self.performance_attribution = performance_attribution;
        operation_output
    }
}

impl PrismHadamardQwen35InferenceExecution {
    pub(crate) fn pending(pending_startup: PrismHadamardQwen35PendingStartup) -> Self {
        let ceiling_bytes = pending_startup.effective_mlx_memory_ceiling_bytes as u64;
        let allocator_cache_bytes = pending_startup.allocator_cache_memory_limit_bytes as u64;
        let maximum_context_tokens = pending_startup.maximum_context_tokens;
        let prompt_processing_chunk_tokens = pending_startup.prompt_processing_chunk_tokens.max(1);
        Self {
            pending_startup: Some(pending_startup),
            model: None,
            active: None,
            ceiling_bytes,
            allocator_cache_bytes,
            minimum_ceiling_bytes: 1,
            maximum_context_tokens,
            prompt_processing_chunk_tokens,
            performance_attribution_log: PerformanceAttributionLog::disabled(),
            attribution_model_id: None,
            attribution_model_revision: None,
        }
    }
}

/// Whether a freshly sampled successor should stay queued: only while output
/// budget remains. An end-of-sequence successor is recognized when its value
/// is synchronized, so it is queued like any other token here.
pub(super) fn successor_stays_pending(remaining_output_tokens: u32) -> bool {
    remaining_output_tokens > 0
}

pub(super) fn record_emitted_token(remaining_output_tokens: &mut u32) -> bool {
    *remaining_output_tokens = (*remaining_output_tokens).saturating_sub(1);
    *remaining_output_tokens == 0
}

#[cfg(test)]
mod tests {
    #[test]
    fn should_count_a_prefill_sample_only_when_it_is_emitted() {
        let mut remaining_output_tokens = 1;
        assert!(super::successor_stays_pending(remaining_output_tokens));
        assert_eq!(remaining_output_tokens, 1);

        assert!(super::record_emitted_token(&mut remaining_output_tokens));
        assert_eq!(remaining_output_tokens, 0);
        assert!(
            !super::successor_stays_pending(remaining_output_tokens),
            "the successor beyond a one-token output budget must not be queued"
        );
    }
}
