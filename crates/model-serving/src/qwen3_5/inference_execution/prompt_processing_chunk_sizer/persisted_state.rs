//! Persisted optimizer construction for warm process starts.
//!
//! Persistence accelerates convergence but is never a serving correctness gate.
//! The state is accepted only when the optimizer validates model identity,
//! revision, candidate set, and window configuration. Missing, stale, or corrupt
//! state produces a fresh optimizer with the same configured candidates.

use std::path::PathBuf;

use super::configuration::{
    configured_candidate_chunk_size_tokens, maximum_prompt_processing_chunk_size_tokens_from_u32,
};
use super::{
    OptimizerStatePersistence, PromptProcessingChunkSizingMode, Qwen3_5PromptProcessingChunkSizer,
    Qwen3_5PromptProcessingChunkSizerError,
};
use crate::PromptProcessingChunkSizeOptimizer;

impl Qwen3_5PromptProcessingChunkSizer {
    /// Creates the production optimizer with validated optional persisted state.
    ///
    /// After each accepted measurement, the parent owner saves the
    /// updated optimizer through `OptimizerStatePersistence`. Construction never
    /// changes serving behavior merely because local state cannot be read.
    #[allow(clippy::too_many_arguments)]
    pub fn for_optimized_production_with_persisted_state_and_behavior(
        maximum_prompt_processing_chunk_size_tokens: u32,
        configured_candidate_chunk_size_token_counts: Vec<u32>,
        optimizer_state_directory: PathBuf,
        model_id: String,
        model_revision: String,
        maximum_retained_measurements_per_candidate_and_context: u32,
        position_range_size_tokens: u32,
    ) -> Result<Self, Qwen3_5PromptProcessingChunkSizerError> {
        if maximum_retained_measurements_per_candidate_and_context == 0
            || position_range_size_tokens == 0
        {
            return Err(Qwen3_5PromptProcessingChunkSizerError::MustBePositive);
        }
        let maximum_retained_measurements_per_candidate_and_context =
            usize::try_from(maximum_retained_measurements_per_candidate_and_context)
                .map_err(|_| Qwen3_5PromptProcessingChunkSizerError::ExceedsPlatformRange)?;
        let position_range_size_tokens = usize::try_from(position_range_size_tokens)
            .map_err(|_| Qwen3_5PromptProcessingChunkSizerError::ExceedsPlatformRange)?;
        let maximum_prompt_processing_chunk_size_tokens =
            maximum_prompt_processing_chunk_size_tokens_from_u32(
                maximum_prompt_processing_chunk_size_tokens,
            )?;
        let candidate_chunk_size_tokens = configured_candidate_chunk_size_tokens(
            configured_candidate_chunk_size_token_counts,
            maximum_prompt_processing_chunk_size_tokens,
        )?;
        // The new optimizer persists to a new filename; the old filename is ignored.
        let optimizer_state_file_path =
            PromptProcessingChunkSizeOptimizer::persisted_state_file_path(
                &optimizer_state_directory,
                &model_id,
                &model_revision,
            );

        // Loading performs full contract validation. `None` means no usable file;
        // `Err` means local persistence was unreadable. Both safely fall forward
        // to a fresh optimizer because request execution owns no persisted state.
        let loaded_optimizer_state = PromptProcessingChunkSizeOptimizer::load_from_path(
            optimizer_state_file_path,
            model_id.clone(),
            model_revision.clone(),
            candidate_chunk_size_tokens.clone(),
            maximum_retained_measurements_per_candidate_and_context,
        );
        let prompt_processing_chunk_size_optimizer = match loaded_optimizer_state {
            Ok(Some(loaded_optimizer)) => {
                tracing::info!(
                    optimizer_state_directory = %optimizer_state_directory.display(),
                    model_id = %model_id,
                    model_revision = %model_revision,
                    "Loaded persisted prompt-processing chunk size optimizer state"
                );
                loaded_optimizer
            }
            Ok(None) => {
                tracing::info!(
                    optimizer_state_directory = %optimizer_state_directory.display(),
                    model_id = %model_id,
                    model_revision = %model_revision,
                    "No persisted optimizer state found; starting fresh"
                );
                fresh_optimizer(
                    candidate_chunk_size_tokens.clone(),
                    maximum_retained_measurements_per_candidate_and_context,
                )?
            }
            Err(persistence_error) => {
                tracing::warn!(
                    error = %persistence_error,
                    optimizer_state_directory = %optimizer_state_directory.display(),
                    "Failed to load persisted optimizer state; starting fresh"
                );
                fresh_optimizer(
                    candidate_chunk_size_tokens.clone(),
                    maximum_retained_measurements_per_candidate_and_context,
                )?
            }
        };

        // The optimizer's first validated candidate is the deterministic initial
        // action until existing or newly collected measurements select another size.
        let active_prompt_processing_chunk_size_tokens = prompt_processing_chunk_size_optimizer
            .candidate_chunk_size_tokens()
            .first()
            .copied()
            .ok_or(Qwen3_5PromptProcessingChunkSizerError::OptimizerRejectedCandidateSet)?;
        Ok(Self {
            maximum_prompt_processing_chunk_size_tokens,
            active_prompt_processing_chunk_size_tokens,
            ssd_streaming_prompt_processing_chunk_size_tokens: None,
            prompt_processing_chunk_sizing_mode: PromptProcessingChunkSizingMode::Optimized {
                prompt_processing_chunk_size_optimizer,
                pending_prompt_processing_chunk_selection: None,
                optimizer_state_persistence: Some(OptimizerStatePersistence {
                    optimizer_state_directory,
                    model_id,
                    model_revision,
                }),
            },
            active_request_restored_token_count: 0,
            has_completed_prompt_processing_chunk_in_active_request: false,
            active_request_has_observed_capacity_reduction: false,
            latest_prompt_processing_chunk_optimization_outcome: None,
            position_range_size_tokens: Some(position_range_size_tokens),
        })
    }
}

fn fresh_optimizer(
    candidate_chunk_size_tokens: Vec<usize>,
    maximum_retained_measurements_per_candidate_and_context: usize,
) -> Result<PromptProcessingChunkSizeOptimizer, Qwen3_5PromptProcessingChunkSizerError> {
    PromptProcessingChunkSizeOptimizer::new(
        candidate_chunk_size_tokens,
        maximum_retained_measurements_per_candidate_and_context,
    )
    .map_err(|_| Qwen3_5PromptProcessingChunkSizerError::OptimizerRejectedCandidateSet)
}
