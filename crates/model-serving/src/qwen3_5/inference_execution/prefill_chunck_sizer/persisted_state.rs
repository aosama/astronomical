//! Persisted optimizer construction for warm process starts.
//!
//! Persistence accelerates convergence but is never a serving correctness gate.
//! The state is accepted only when the optimizer validates model identity,
//! revision, candidate set, and window configuration. Missing, stale, or corrupt
//! state produces a fresh optimizer with the same configured candidates.

use super::*;

impl Qwen3_5PrefillChunckSizer {
    /// Creates the production optimizer with validated optional persisted state.
    ///
    /// After each accepted full-chunk observation, the parent owner saves the
    /// updated optimizer through `OptimizerStatePersistence`. Construction never
    /// changes serving behavior merely because local state cannot be read.
    #[allow(clippy::too_many_arguments)]
    pub fn for_optimized_production_with_persisted_state_and_behavior(
        maximum_prefill_chunck_tokens: u32,
        optimizer_prefill_chunck_token_candidates: Vec<u32>,
        optimizer_state_directory: PathBuf,
        model_id: String,
        model_revision: String,
        sliding_window_observation_count: u32,
        prompt_position_context_bucket_tokens: u32,
    ) -> Result<Self, Qwen3_5PrefillChunckSizerError> {
        if sliding_window_observation_count == 0 || prompt_position_context_bucket_tokens == 0 {
            return Err(Qwen3_5PrefillChunckSizerError::MustBePositive);
        }
        let sliding_window_observation_count = usize::try_from(sliding_window_observation_count)
            .map_err(|_| Qwen3_5PrefillChunckSizerError::ExceedsPlatformRange)?;
        let prompt_position_context_bucket_tokens =
            usize::try_from(prompt_position_context_bucket_tokens)
                .map_err(|_| Qwen3_5PrefillChunckSizerError::ExceedsPlatformRange)?;
        let maximum_prefill_chunck_tokens =
            maximum_prefill_chunck_tokens_from_u32(maximum_prefill_chunck_tokens)?;
        let candidate_prefill_chunck_tokens = configured_candidate_prefill_chunck_tokens(
            optimizer_prefill_chunck_token_candidates,
            maximum_prefill_chunck_tokens,
        )?;
        let optimizer_state_file_path = optimizer_state_directory.join("prefill-chunck-size.json");

        // Loading performs full contract validation. `None` means no usable file;
        // `Err` means local persistence was unreadable. Both safely fall forward
        // to a fresh optimizer because request execution owns no persisted state.
        let prefill_chunck_size_optimizer = match PrefillChunckSizeOptimizer::load_from_path(
            optimizer_state_file_path,
            model_id.clone(),
            model_revision.clone(),
            candidate_prefill_chunck_tokens.clone(),
            sliding_window_observation_count,
        ) {
            Ok(Some(loaded_optimizer)) => {
                tracing::info!(
                    optimizer_state_directory = %optimizer_state_directory.display(),
                    model_id = %model_id,
                    model_revision = %model_revision,
                    "Loaded persisted prefill chunk size optimizer state"
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
                    candidate_prefill_chunck_tokens.clone(),
                    sliding_window_observation_count,
                )?
            }
            Err(persistence_error) => {
                tracing::warn!(
                    error = %persistence_error,
                    optimizer_state_directory = %optimizer_state_directory.display(),
                    "Failed to load persisted optimizer state; starting fresh"
                );
                fresh_optimizer(
                    candidate_prefill_chunck_tokens.clone(),
                    sliding_window_observation_count,
                )?
            }
        };

        // The optimizer's first validated candidate is the deterministic initial
        // action until existing or newly collected evidence selects another size.
        let active_prefill_chunck_tokens = prefill_chunck_size_optimizer
            .candidate_prefill_chunck_tokens()
            .first()
            .copied()
            .ok_or(Qwen3_5PrefillChunckSizerError::OptimizerRejectedCandidateSet)?;
        Ok(Self {
            maximum_prefill_chunck_tokens,
            active_prefill_chunck_tokens,
            ssd_streaming_prefill_chunck_tokens: None,
            prefill_chunck_sizing_mode: PrefillChunckSizingMode::Optimized {
                prefill_chunck_size_optimizer,
                pending_prefill_chunck_decision: None,
                optimizer_state_persistence: Some(OptimizerStatePersistence {
                    optimizer_state_directory,
                    model_id,
                    model_revision,
                }),
            },
            active_request_restored_token_count: 0,
            has_completed_prefill_chunck_in_active_request: false,
            active_request_has_observed_capacity_reduction: false,
            latest_prefill_optimizer_insight: None,
            prompt_position_context_bucket_tokens: Some(prompt_position_context_bucket_tokens),
        })
    }
}

fn fresh_optimizer(
    candidate_prefill_chunck_tokens: Vec<usize>,
    sliding_window_observation_count: usize,
) -> Result<PrefillChunckSizeOptimizer, Qwen3_5PrefillChunckSizerError> {
    PrefillChunckSizeOptimizer::new(
        candidate_prefill_chunck_tokens,
        sliding_window_observation_count,
    )
    .map_err(|_| Qwen3_5PrefillChunckSizerError::OptimizerRejectedCandidateSet)
}
