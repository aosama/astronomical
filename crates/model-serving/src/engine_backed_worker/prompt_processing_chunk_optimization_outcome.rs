//! Translates model-serving optimizer outcome types into IPC protocol types.

use astronomical_ipc_protocol::{
    WorkerPromptProcessingChunkCandidateMeasurementSummary,
    WorkerPromptProcessingChunkMeasurementSource, WorkerPromptProcessingChunkOptimizationContext,
    WorkerPromptProcessingChunkOptimizationOutcome, WorkerPromptProcessingChunkSelectionReason,
};

use super::WorkerRuntimeError;
use crate::{
    CandidateMeasurementSource, PromptProcessingChunkOptimizationOutcome,
    PromptProcessingChunkSizeSelectionReason,
};

pub(super) fn to_worker_prompt_processing_chunk_optimization_outcome(
    optimization_outcome: PromptProcessingChunkOptimizationOutcome,
) -> Result<WorkerPromptProcessingChunkOptimizationOutcome, WorkerRuntimeError> {
    Ok(WorkerPromptProcessingChunkOptimizationOutcome {
        selected_candidate_chunk_size_tokens: bounded_token_count(
            optimization_outcome.selected_candidate_chunk_size_tokens,
        )?,
        processed_prompt_token_count: bounded_token_count(
            optimization_outcome.processed_prompt_token_count,
        )?,
        forward_elapsed_millis: optimization_outcome.forward_elapsed_millis,
        was_reduced_by_memory_capacity: optimization_outcome.was_reduced_by_memory_capacity,
        was_accepted_for_learning: optimization_outcome.was_accepted_for_learning,
        selection_reason: match optimization_outcome.selection_reason {
            PromptProcessingChunkSizeSelectionReason::ExploreUnmeasuredCandidate => {
                WorkerPromptProcessingChunkSelectionReason::ExploreUnmeasuredCandidate
            }
            PromptProcessingChunkSizeSelectionReason::MinimizeProjectedRemainingPromptLatency => {
                WorkerPromptProcessingChunkSelectionReason::MinimizeProjectedRemainingPromptLatency
            }
            PromptProcessingChunkSizeSelectionReason::RemainingTokensBelowSmallestCandidate => {
                WorkerPromptProcessingChunkSelectionReason::RemainingTokensBelowSmallestCandidate
            }
        },
        measurement_context: WorkerPromptProcessingChunkOptimizationContext {
            chunk_start_token_position: bounded_token_count(
                optimization_outcome
                    .measurement_context
                    .chunk_start_token_position,
            )?,
            position_range_start_token_position: bounded_token_count(
                optimization_outcome
                    .measurement_context
                    .position_range_start_token_position,
            )?,
            position_range_end_token_position_exclusive: bounded_token_count(
                optimization_outcome
                    .measurement_context
                    .position_range_end_token_position_exclusive,
            )?,
            has_restored_prefix: optimization_outcome.measurement_context.has_restored_prefix,
            is_first_chunk_after_restore: optimization_outcome
                .measurement_context
                .is_first_chunk_after_restore,
            has_visual_embeddings: optimization_outcome
                .measurement_context
                .has_visual_embeddings,
            is_mtp_active: optimization_outcome.measurement_context.is_mtp_active,
            are_sparse_experts_paged: optimization_outcome
                .measurement_context
                .are_sparse_experts_paged,
            is_prompt_cache_capture_eligible: optimization_outcome
                .measurement_context
                .is_prompt_cache_capture_eligible,
            has_prior_capacity_reduction: optimization_outcome
                .measurement_context
                .has_prior_capacity_reduction,
        },
        all_candidates_have_measurements: optimization_outcome.all_candidates_have_measurements,
        is_execution_profile_converged: optimization_outcome.is_execution_profile_converged,
        candidate_measurement_summaries: optimization_outcome
            .candidate_measurement_summaries
            .into_iter()
            .map(|candidate_summary| {
                Ok(WorkerPromptProcessingChunkCandidateMeasurementSummary {
                    candidate_chunk_size_tokens: bounded_token_count(
                        candidate_summary.candidate_chunk_size_tokens,
                    )?,
                    measurement_source: match candidate_summary.measurement_source {
                        CandidateMeasurementSource::ExecutionProfile => {
                            WorkerPromptProcessingChunkMeasurementSource::ExecutionProfile
                        }
                        CandidateMeasurementSource::NoMeasurementsAvailable => {
                            WorkerPromptProcessingChunkMeasurementSource::NoMeasurementsAvailable
                        }
                    },
                    measurement_count: u32::try_from(candidate_summary.measurement_count)
                        .unwrap_or(u32::MAX),
                    average_processed_prompt_token_count: bounded_token_count(
                        candidate_summary.average_processed_prompt_token_count,
                    )?,
                    average_forward_elapsed_millis: candidate_summary
                        .average_forward_elapsed_millis,
                })
            })
            .collect::<Result<Vec<_>, WorkerRuntimeError>>()?,
    })
}

fn bounded_token_count(token_count: usize) -> Result<u32, WorkerRuntimeError> {
    // Token positions are protocol-critical: fail rather than truncating them.
    // Measurement counts use saturation above because they are descriptive
    // telemetry and do not control model execution or progress accounting.
    u32::try_from(token_count).map_err(|_| WorkerRuntimeError::InferenceEngineGenerationFailed {
        reason: "prompt-processing chunk optimizer telemetry token count exceeds the u32 range"
            .to_owned(),
    })
}
