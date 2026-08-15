//! Architecture-neutral prompt-processing chunk optimization telemetry.

use crate::{CandidateMeasurementSource, PromptProcessingChunkSizeSelectionReason};

/// Measurement summary for one candidate in the latest optimizer selection context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromptProcessingChunkCandidateMeasurementSummary {
    /// Candidate capacity represented by this summary, in tokens.
    pub candidate_chunk_size_tokens: usize,
    /// Exact-range, equivalent-profile, or unavailable measurement provenance.
    pub measurement_source: CandidateMeasurementSource,
    /// Number of retained measurements included in the averages.
    pub measurement_count: usize,
    /// Mean completed token advancement for the candidate.
    pub average_processed_prompt_token_count: usize,
    /// Mean model-forward duration in milliseconds.
    pub average_forward_elapsed_millis: u64,
    /// Optimizer selections elapsed since the profile was measured.
    pub selections_since_last_measurement: Option<u64>,
}

/// Execution context that isolates prompt-processing measurements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromptProcessingChunkOptimizationContext {
    /// Inclusive token position where the measured chunk began.
    pub chunk_start_token_position: usize,
    /// Inclusive start of the optimizer position range.
    pub position_range_start_token_position: usize,
    /// Exclusive end of the optimizer position range.
    pub position_range_end_token_position_exclusive: usize,
    /// Whether this request reused any persisted prompt prefix.
    pub has_restored_prefix: bool,
    /// Whether this was the first executed chunk after that restored prefix.
    pub is_first_chunk_after_restore: bool,
    /// Whether visual embeddings participated in the execution profile.
    pub has_visual_embeddings: bool,
    /// Whether multi-token prediction state participated in this profile.
    pub is_mtp_active: bool,
    /// Whether sparse experts streamed rather than remaining complete-resident.
    pub are_sparse_experts_paged: bool,
    /// Whether this chunk could publish a persistent prompt-cache boundary.
    pub is_prompt_cache_capture_eligible: bool,
    /// Whether earlier request work observed a memory-capacity reduction.
    pub has_prior_capacity_reduction: bool,
}

/// Latest prompt-processing chunk selection, measured outcome, and measurement summaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptProcessingChunkOptimizationOutcome {
    /// Candidate capacity selected before the chunk executed.
    pub selected_candidate_chunk_size_tokens: usize,
    /// Prompt tokens that actually completed, including a shorter terminal tail.
    pub processed_prompt_token_count: usize,
    /// Model-forward duration, excluding cleanup and telemetry work.
    pub forward_elapsed_millis: u64,
    /// Whether memory admission reduced the executed work below its request.
    pub was_reduced_by_memory_capacity: bool,
    /// Optimizer rule that selected the candidate.
    pub selection_reason: PromptProcessingChunkSizeSelectionReason,
    /// Human-readable execution context for the completed measurement.
    pub measurement_context: PromptProcessingChunkOptimizationContext,
    /// Whether every configured candidate now has usable evidence.
    pub all_candidates_have_measurements: bool,
    /// Evidence summaries in ascending configured-candidate order.
    pub candidate_measurement_summaries: Vec<PromptProcessingChunkCandidateMeasurementSummary>,
}
