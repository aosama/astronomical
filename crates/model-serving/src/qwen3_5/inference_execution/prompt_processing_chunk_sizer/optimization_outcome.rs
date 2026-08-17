//! Projects optimizer evidence into the architecture-neutral engine telemetry DTO.
//!
//! This Qwen-owned builder supplies model execution context. It intentionally
//! stops before IPC conversion so transport types never enter model execution.

use crate::{
    PromptProcessingChunkCandidateMeasurementSummary, PromptProcessingChunkOptimizationContext,
    PromptProcessingChunkOptimizationOutcome, PromptProcessingChunkSizeOptimizer,
    PromptProcessingChunkSizeSelectionReason, PromptProcessingMeasurementContext,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn prompt_processing_chunk_optimization_outcome(
    prompt_processing_chunk_size_optimizer: &PromptProcessingChunkSizeOptimizer,
    measurement_context: PromptProcessingMeasurementContext,
    selected_candidate_chunk_size_tokens: usize,
    processed_prompt_token_count: usize,
    forward_elapsed_millis: u64,
    selection_reason: PromptProcessingChunkSizeSelectionReason,
    was_reduced_by_memory_capacity: bool,
    was_accepted_for_learning: bool,
    optimizer_context: PromptProcessingChunkOptimizationContext,
) -> PromptProcessingChunkOptimizationOutcome {
    let candidate_measurement_summaries =
        prompt_processing_chunk_size_optimizer.candidate_measurement_summaries(measurement_context);
    PromptProcessingChunkOptimizationOutcome {
        selected_candidate_chunk_size_tokens,
        processed_prompt_token_count,
        forward_elapsed_millis,
        was_reduced_by_memory_capacity,
        was_accepted_for_learning,
        selection_reason,
        measurement_context: optimizer_context,
        all_candidates_have_measurements: candidate_measurement_summaries
            .all_candidates_have_measurements,
        is_execution_profile_converged: candidate_measurement_summaries
            .all_candidates_have_measurements,
        candidate_measurement_summaries: candidate_measurement_summaries
            .candidate_measurement_summaries
            .into_iter()
            .map(
                |candidate_measurement_summary| PromptProcessingChunkCandidateMeasurementSummary {
                    candidate_chunk_size_tokens: candidate_measurement_summary
                        .candidate_chunk_size_tokens,
                    measurement_source: candidate_measurement_summary.measurement_source,
                    measurement_count: candidate_measurement_summary.measurement_count,
                    average_processed_prompt_token_count: candidate_measurement_summary
                        .average_processed_prompt_token_count,
                    average_forward_elapsed_millis: candidate_measurement_summary
                        .average_forward_elapsed_millis,
                },
            )
            .collect(),
    }
}
