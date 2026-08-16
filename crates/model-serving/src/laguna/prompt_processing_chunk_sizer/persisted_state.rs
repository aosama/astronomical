//! Optional persisted optimizer construction for warm Laguna process starts.

use std::path::PathBuf;

use super::configuration::LagunaPromptProcessingChunkSizerError;
use super::sizer::{LagunaPromptProcessingChunkSizer, OptimizerStatePersistence};
use crate::PromptProcessingChunkSizeOptimizer;

impl LagunaPromptProcessingChunkSizer {
    /// Creates the production optimizer and loads usable persisted state when present.
    #[allow(clippy::too_many_arguments)]
    pub fn for_optimized_production_with_persisted_state(
        maximum_prompt_processing_chunk_size_tokens: u32,
        configured_candidate_chunk_size_token_counts: Vec<u32>,
        optimizer_state_directory: PathBuf,
        model_id: String,
        model_revision: String,
        maximum_retained_measurements_per_candidate_and_context: u32,
        position_range_size_tokens: u32,
    ) -> Result<Self, LagunaPromptProcessingChunkSizerError> {
        Self::for_optimized_with_optional_persistence(
            maximum_prompt_processing_chunk_size_tokens,
            configured_candidate_chunk_size_token_counts,
            maximum_retained_measurements_per_candidate_and_context,
            position_range_size_tokens,
            Some(OptimizerStatePersistence {
                optimizer_state_directory,
                model_id,
                model_revision,
            }),
        )
    }
}

pub(super) fn load_or_create_optimizer(
    optimizer_state_persistence: &OptimizerStatePersistence,
    candidate_chunk_size_tokens: Vec<usize>,
    maximum_retained_measurements_per_candidate_and_context: usize,
) -> Result<PromptProcessingChunkSizeOptimizer, LagunaPromptProcessingChunkSizerError> {
    let optimizer_state_file_path = PromptProcessingChunkSizeOptimizer::persisted_state_file_path(
        &optimizer_state_persistence.optimizer_state_directory,
        &optimizer_state_persistence.model_id,
        &optimizer_state_persistence.model_revision,
    );
    match PromptProcessingChunkSizeOptimizer::load_from_path(
        optimizer_state_file_path,
        optimizer_state_persistence.model_id.clone(),
        optimizer_state_persistence.model_revision.clone(),
        candidate_chunk_size_tokens.clone(),
        maximum_retained_measurements_per_candidate_and_context,
    ) {
        Ok(Some(loaded_optimizer)) => Ok(loaded_optimizer),
        Ok(None) | Err(_) => PromptProcessingChunkSizeOptimizer::new(
            candidate_chunk_size_tokens,
            maximum_retained_measurements_per_candidate_and_context,
        )
        .map_err(|_| LagunaPromptProcessingChunkSizerError::OptimizerRejectedCandidateSet),
    }
}
