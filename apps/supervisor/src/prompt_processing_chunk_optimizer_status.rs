//! Records bounded prompt-processing chunk optimization outcomes and builds the status document.

use astronomical_config::PromptProcessingChunkSizingPolicy;
use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::WorkerPromptProcessingChunkOptimizationOutcome;
use serde_json::{Value, json};

use crate::WorkerHealthSnapshot;

const MAXIMUM_RECENT_PROMPT_PROCESSING_CHUNK_OPTIMIZATION_OUTCOMES: usize = 12;

/// Appends one measured outcome while bounding long-lived supervisor memory.
///
/// A poisoned health lock means the worker is already unavailable; dropping
/// telemetry here must not introduce a second process failure.
pub(crate) fn record_prompt_processing_chunk_optimization_outcome(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    prompt_processing_chunk_optimization_outcome: WorkerPromptProcessingChunkOptimizationOutcome,
) {
    let Ok(mut worker_health_snapshot) = health_snapshot.write() else {
        return;
    };
    if worker_health_snapshot
        .recent_prompt_processing_chunk_optimization_outcomes
        .len()
        >= MAXIMUM_RECENT_PROMPT_PROCESSING_CHUNK_OPTIMIZATION_OUTCOMES
    {
        // The bound is deliberately small, so removing the oldest entry keeps
        // chronological storage simple without meaningful request-path cost.
        worker_health_snapshot
            .recent_prompt_processing_chunk_optimization_outcomes
            .remove(0);
    }
    worker_health_snapshot
        .recent_prompt_processing_chunk_optimization_outcomes
        .push(prompt_processing_chunk_optimization_outcome);
}

/// Builds the stable status projection consumed by the Observatory and menu.
///
/// Full candidate evidence is published only for the latest outcome. Recent
/// history stays compact and is reversed so the newest decision renders first.
pub(crate) fn prompt_processing_chunk_optimizer_status_document(
    prompt_processing_chunk_sizing_policy: Option<&PromptProcessingChunkSizingPolicy>,
    recent_prompt_processing_chunk_optimization_outcomes: &[WorkerPromptProcessingChunkOptimizationOutcome],
) -> Value {
    let latest_chunk_outcome = recent_prompt_processing_chunk_optimization_outcomes.last();
    let (mode, candidate_chunk_size_token_counts, fixed_chunk_size_token_count) =
        match prompt_processing_chunk_sizing_policy {
            Some(PromptProcessingChunkSizingPolicy::Optimized {
                prompt_processing_chunk_size_optimizer_candidate_token_counts,
            }) => (
                "adaptive",
                prompt_processing_chunk_size_optimizer_candidate_token_counts.clone(),
                None,
            ),
            Some(PromptProcessingChunkSizingPolicy::Fixed {
                fixed_prompt_processing_chunk_size_tokens,
                fixed_ssd_streaming_prompt_processing_chunk_size_tokens: _,
            }) => (
                "fixed",
                Vec::new(),
                Some(*fixed_prompt_processing_chunk_size_tokens),
            ),
            None => (
                "unavailable",
                latest_chunk_outcome.map_or_else(Vec::new, |latest_outcome| {
                    latest_outcome
                        .candidate_measurement_summaries
                        .iter()
                        .map(|candidate_summary| candidate_summary.candidate_chunk_size_tokens)
                        .collect()
                }),
                None,
            ),
        };
    let latest_chunk_outcome_json = latest_chunk_outcome.map(|outcome| {
        json!({
            "selection": {
                "selected_candidate_chunk_size_tokens": outcome.selected_candidate_chunk_size_tokens,
                "reason": outcome.selection_reason,
            },
            "processed_prompt_token_count": outcome.processed_prompt_token_count,
            "forward_elapsed_millis": outcome.forward_elapsed_millis,
            "was_reduced_by_memory_capacity": outcome.was_reduced_by_memory_capacity,
            "measurement_context": {
                "chunk_start_token_position": outcome.measurement_context.chunk_start_token_position,
                "position_range_start_token_position": outcome.measurement_context.position_range_start_token_position,
                "position_range_end_token_position_exclusive": outcome.measurement_context.position_range_end_token_position_exclusive,
                "has_restored_prefix": outcome.measurement_context.has_restored_prefix,
                "is_first_chunk_after_restore": outcome.measurement_context.is_first_chunk_after_restore,
                "has_visual_embeddings": outcome.measurement_context.has_visual_embeddings,
                "is_mtp_active": outcome.measurement_context.is_mtp_active,
                "are_sparse_experts_paged": outcome.measurement_context.are_sparse_experts_paged,
                "is_prompt_cache_capture_eligible": outcome.measurement_context.is_prompt_cache_capture_eligible,
                "has_prior_capacity_reduction": outcome.measurement_context.has_prior_capacity_reduction,
            },
            "all_candidates_have_measurements": outcome.all_candidates_have_measurements,
            "candidate_measurement_summaries": outcome.candidate_measurement_summaries,
        })
    });
    let recent_chunk_outcomes = recent_prompt_processing_chunk_optimization_outcomes
        .iter()
        .rev()
        .map(|outcome| {
            json!({
                "selection": {
                    "selected_candidate_chunk_size_tokens": outcome.selected_candidate_chunk_size_tokens,
                    "reason": outcome.selection_reason,
                },
                "processed_prompt_token_count": outcome.processed_prompt_token_count,
                "forward_elapsed_millis": outcome.forward_elapsed_millis,
                "was_reduced_by_memory_capacity": outcome.was_reduced_by_memory_capacity,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "mode": mode,
        "candidate_chunk_size_token_counts": candidate_chunk_size_token_counts,
        "fixed_chunk_size_token_count": fixed_chunk_size_token_count,
        "latest_chunk_outcome": latest_chunk_outcome_json,
        "recent_chunk_outcomes": recent_chunk_outcomes,
    })
}
