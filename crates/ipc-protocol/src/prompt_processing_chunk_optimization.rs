//! Bounded worker-to-supervisor telemetry for prompt-processing chunk optimization.

use serde::{Deserialize, Serialize};

/// Why the worker's prompt-processing chunk optimizer selected one candidate.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPromptProcessingChunkSelectionReason {
    /// Largest feasible candidate without usable measurements.
    ExploreUnmeasuredCandidate,
    /// Feasible candidate whose retained evidence is stalest.
    RefreshStaleCandidateMeasurement,
    /// Candidate with the lowest predicted cumulative remaining-prompt latency.
    MinimizeProjectedRemainingPromptLatency,
    /// Remaining prompt is shorter than the smallest registered capacity.
    RemainingTokensBelowSmallestCandidate,
    /// Smallest registered candidate capable of labeling the final prompt segment.
    SmallestCandidateContainingFinalPromptSegment,
}

/// Measurement summary for one configured prompt-processing chunk candidate.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerPromptProcessingChunkCandidateMeasurementSummary {
    /// Registered candidate capacity in tokens.
    pub candidate_chunk_size_tokens: u32,
    /// Range/profile provenance for the retained measurements.
    pub measurement_source: WorkerPromptProcessingChunkMeasurementSource,
    /// Bounded number of retained measurements included in the averages.
    pub measurement_count: u32,
    /// Mean completed prompt-token advancement.
    pub average_processed_prompt_token_count: u32,
    /// Mean model-forward duration in milliseconds.
    pub average_forward_elapsed_millis: u64,
    /// Optimizer selections elapsed since this execution profile was measured.
    pub selections_since_last_measurement: Option<u64>,
}

/// Measurement source for one candidate's summary data.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPromptProcessingChunkMeasurementSource {
    /// Evidence was collected in the exact reported position range.
    CurrentPositionRange,
    /// Evidence came from another position with the same execution profile.
    OtherPositionRangesWithSameExecutionProfile,
    /// No retained measurement can represent this candidate and profile.
    NoMeasurementsAvailable,
}

/// Human-readable execution context that isolates prompt-processing measurements.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerPromptProcessingChunkOptimizationContext {
    /// Inclusive token position where the measured chunk began.
    pub chunk_start_token_position: u32,
    /// Inclusive start of the optimizer position range.
    pub position_range_start_token_position: u32,
    /// Exclusive end of the optimizer position range.
    pub position_range_end_token_position_exclusive: u32,
    /// Whether the request restored a persisted prompt prefix.
    pub has_restored_prefix: bool,
    /// Whether this was the first executed chunk after restoration.
    pub is_first_chunk_after_restore: bool,
    /// Whether visual embeddings participated in this profile.
    pub has_visual_embeddings: bool,
    /// Whether multi-token prediction state participated in this profile.
    pub is_mtp_active: bool,
    /// Whether sparse experts streamed rather than remaining complete-resident.
    pub are_sparse_experts_paged: bool,
    /// Whether the chunk could publish a persistent prompt-cache boundary.
    pub is_prompt_cache_capture_eligible: bool,
    /// Whether prior request work observed a memory-capacity reduction.
    pub has_prior_capacity_reduction: bool,
}

/// One optimizer selection, its measured outcome, and the available evidence.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerPromptProcessingChunkOptimizationOutcome {
    /// Candidate capacity selected before execution.
    pub selected_candidate_chunk_size_tokens: u32,
    /// Prompt tokens that actually completed.
    pub processed_prompt_token_count: u32,
    /// Model-forward duration in milliseconds, excluding cleanup and telemetry.
    pub forward_elapsed_millis: u64,
    /// Whether memory capacity reduced the executed work.
    pub was_reduced_by_memory_capacity: bool,
    /// Rule that selected the candidate.
    pub selection_reason: WorkerPromptProcessingChunkSelectionReason,
    /// Execution conditions associated with this measurement.
    pub measurement_context: WorkerPromptProcessingChunkOptimizationContext,
    /// Whether all configured candidates now have usable evidence.
    pub all_candidates_have_measurements: bool,
    /// Bounded evidence summaries in ascending candidate order.
    pub candidate_measurement_summaries:
        Vec<WorkerPromptProcessingChunkCandidateMeasurementSummary>,
}
