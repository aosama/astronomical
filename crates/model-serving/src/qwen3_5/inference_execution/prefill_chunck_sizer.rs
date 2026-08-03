use std::path::PathBuf;

use thiserror::Error;

use crate::{
    PrefillChunckSizeOptimizer, PrefillChunckSizeOptimizerContext,
    PrefillChunckSizeOptimizerObservation,
};

const TRUSTED_OBSERVATION_COUNT: usize = 3;
const SLIDING_WINDOW_OBSERVATION_COUNT: usize = 5;
const DRIFT_TRIGGER_FACTOR: u64 = 2;
const MINIMUM_OPTIMIZER_PREFILL_CHUNCK_TOKENS: usize = 128;
const PROMPT_POSITION_CONTEXT_BUCKET_TOKENS: usize = 32_768;
const RESTORED_PREFIX_CONTEXT_FLAG: u64 = 1 << 32;
const FIRST_CHUNCK_AFTER_RESTORE_CONTEXT_FLAG: u64 = 1 << 33;

/// Owns Qwen3.5 prompt-processing `prefill_chunck_tokens` selection and boundary calculation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5PrefillChunckSizer {
    maximum_prefill_chunck_tokens: usize,
    active_prefill_chunck_tokens: usize,
    prefill_chunck_sizing_mode: PrefillChunckSizingMode,
    active_request_restored_token_count: usize,
    has_completed_prefill_chunck_in_active_request: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PrefillChunckSizingMode {
    Fixed,
    Optimized {
        prefill_chunck_size_optimizer: PrefillChunckSizeOptimizer,
        pending_prefill_chunck_decision: Option<PendingPrefillChunckDecision>,
        optimizer_state_persistence: Option<OptimizerStatePersistence>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct OptimizerStatePersistence {
    optimizer_state_directory: PathBuf,
    model_id: String,
    model_revision: String,
}

impl Qwen3_5PrefillChunckSizer {
    /// Uses candidates through the validated model context maximum for prompt processing.
    pub fn production(
        maximum_prefill_chunck_tokens: u32,
    ) -> Result<Self, Qwen3_5PrefillChunckSizerError> {
        Self::for_optimized_with_maximum_prefill_chunck_tokens(maximum_prefill_chunck_tokens)
    }

    /// Creates the production optimizer with persisted state for warm starts.
    ///
    /// On construction, attempts to load persisted state from the given directory.
    /// If no state exists or the state is stale/corrupt, starts fresh (no error).
    /// After each full-chunk observation, persists the updated optimizer state.
    pub fn for_optimized_production_with_persisted_state(
        maximum_prefill_chunck_tokens: u32,
        optimizer_state_directory: PathBuf,
        model_id: String,
        model_revision: String,
    ) -> Result<Self, Qwen3_5PrefillChunckSizerError> {
        let prefill_chunck_tokens =
            maximum_prefill_chunck_tokens_from_u32(maximum_prefill_chunck_tokens)?;
        let candidate_prefill_chunck_tokens =
            optimizer_candidate_prefill_chunck_tokens(prefill_chunck_tokens);
        let state_file_path = optimizer_state_directory.join("prefill-chunck-size.json");

        let prefill_chunck_size_optimizer = match PrefillChunckSizeOptimizer::load_from_path(
            state_file_path,
            model_id.clone(),
            model_revision.clone(),
            candidate_prefill_chunck_tokens.clone(),
            TRUSTED_OBSERVATION_COUNT,
            SLIDING_WINDOW_OBSERVATION_COUNT,
            DRIFT_TRIGGER_FACTOR,
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
                PrefillChunckSizeOptimizer::new(
                    candidate_prefill_chunck_tokens.clone(),
                    TRUSTED_OBSERVATION_COUNT,
                    SLIDING_WINDOW_OBSERVATION_COUNT,
                    DRIFT_TRIGGER_FACTOR,
                )
                .map_err(|_| Qwen3_5PrefillChunckSizerError::OptimizerRejectedCandidateSet)?
            }
            Err(persistence_error) => {
                tracing::warn!(
                    error = %persistence_error,
                    optimizer_state_directory = %optimizer_state_directory.display(),
                    "Failed to load persisted optimizer state; starting fresh"
                );
                PrefillChunckSizeOptimizer::new(
                    candidate_prefill_chunck_tokens.clone(),
                    TRUSTED_OBSERVATION_COUNT,
                    SLIDING_WINDOW_OBSERVATION_COUNT,
                    DRIFT_TRIGGER_FACTOR,
                )
                .map_err(|_| Qwen3_5PrefillChunckSizerError::OptimizerRejectedCandidateSet)?
            }
        };

        let active_prefill_chunck_tokens = prefill_chunck_size_optimizer
            .candidate_prefill_chunck_tokens()
            .first()
            .copied()
            .ok_or(Qwen3_5PrefillChunckSizerError::OptimizerRejectedCandidateSet)?;
        Ok(Self {
            maximum_prefill_chunck_tokens: prefill_chunck_tokens,
            active_prefill_chunck_tokens,
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
        })
    }

    /// Creates fixed `prefill_chunck_tokens` that bypass optimizer selection and persistence.
    pub fn for_fixed_prefill_chunck_tokens(
        fixed_prefill_chunck_tokens: u32,
    ) -> Result<Self, Qwen3_5PrefillChunckSizerError> {
        let fixed_prefill_chunck_tokens = usize::try_from(fixed_prefill_chunck_tokens)
            .map_err(|_| Qwen3_5PrefillChunckSizerError::ExceedsPlatformRange)?;
        if fixed_prefill_chunck_tokens == 0 {
            return Err(Qwen3_5PrefillChunckSizerError::MustBePositive);
        }
        Ok(Self {
            maximum_prefill_chunck_tokens: fixed_prefill_chunck_tokens,
            active_prefill_chunck_tokens: fixed_prefill_chunck_tokens,
            prefill_chunck_sizing_mode: PrefillChunckSizingMode::Fixed,
            active_request_restored_token_count: 0,
            has_completed_prefill_chunck_in_active_request: false,
        })
    }

    /// Creates an adaptive optimizer bounded by explicit `prefill_chunck_tokens`.
    pub fn for_optimized_with_maximum_prefill_chunck_tokens(
        maximum_prefill_chunck_tokens: u32,
    ) -> Result<Self, Qwen3_5PrefillChunckSizerError> {
        let prefill_chunck_tokens =
            maximum_prefill_chunck_tokens_from_u32(maximum_prefill_chunck_tokens)?;
        let prefill_chunck_size_optimizer = PrefillChunckSizeOptimizer::new(
            optimizer_candidate_prefill_chunck_tokens(prefill_chunck_tokens),
            TRUSTED_OBSERVATION_COUNT,
            SLIDING_WINDOW_OBSERVATION_COUNT,
            DRIFT_TRIGGER_FACTOR,
        )
        .map_err(|_| Qwen3_5PrefillChunckSizerError::OptimizerRejectedCandidateSet)?;
        let active_prefill_chunck_tokens = prefill_chunck_size_optimizer
            .candidate_prefill_chunck_tokens()
            .first()
            .copied()
            .ok_or(Qwen3_5PrefillChunckSizerError::OptimizerRejectedCandidateSet)?;
        Ok(Self {
            maximum_prefill_chunck_tokens: prefill_chunck_tokens,
            active_prefill_chunck_tokens,
            prefill_chunck_sizing_mode: PrefillChunckSizingMode::Optimized {
                prefill_chunck_size_optimizer,
                pending_prefill_chunck_decision: None,
                optimizer_state_persistence: None,
            },
            active_request_restored_token_count: 0,
            has_completed_prefill_chunck_in_active_request: false,
        })
    }

    /// Returns the configured fixed size or optimized maximum `prefill_chunck_tokens`.
    #[must_use]
    pub const fn prefill_chunck_tokens(&self) -> usize {
        self.maximum_prefill_chunck_tokens
    }

    /// Returns the selected `prefill_chunck_tokens` for the next prompt-processing chunk.
    #[must_use]
    pub const fn active_prefill_chunck_tokens(&self) -> usize {
        self.active_prefill_chunck_tokens
    }

    /// Returns the same prompt-position identifier used for optimizer decisions.
    #[must_use]
    pub(super) fn prompt_processing_context_identifier(&self, prefill_chunck_start: usize) -> u64 {
        let raw_position_bucket_identifier =
            u64::try_from(prefill_chunck_start / PROMPT_POSITION_CONTEXT_BUCKET_TOKENS)
                .unwrap_or(u64::MAX >> 2);
        let mut context_identifier = raw_position_bucket_identifier.min((1_u64 << 32) - 1);
        if self.active_request_restored_token_count > 0 {
            context_identifier |= RESTORED_PREFIX_CONTEXT_FLAG;
        }
        if self.active_request_restored_token_count > 0
            && !self.has_completed_prefill_chunck_in_active_request
        {
            context_identifier |= FIRST_CHUNCK_AFTER_RESTORE_CONTEXT_FLAG;
        }
        context_identifier
    }

    /// Starts a fresh prompt-processing request with its restored prefix length.
    pub fn start_prompt_processing_request(&mut self, restored_token_count: usize) {
        self.active_request_restored_token_count = restored_token_count;
        self.has_completed_prefill_chunck_in_active_request = false;
        if let PrefillChunckSizingMode::Optimized {
            pending_prefill_chunck_decision,
            ..
        } = &mut self.prefill_chunck_sizing_mode
        {
            *pending_prefill_chunck_decision = None;
        }
        // The first optimized next_prefill_chunck_end call asks for an initial candidate.
    }

    /// Records a measured prompt-processing chunk for future optimizer choices.
    pub fn record_prefill_chunck_elapsed_millis(
        &mut self,
        actual_prefill_chunck_tokens: usize,
        elapsed_millis: u64,
    ) {
        let PrefillChunckSizingMode::Optimized {
            prefill_chunck_size_optimizer,
            pending_prefill_chunck_decision,
            optimizer_state_persistence,
        } = &mut self.prefill_chunck_sizing_mode
        else {
            return;
        };
        let Some(pending_prefill_chunck_decision) = pending_prefill_chunck_decision.take() else {
            return;
        };
        let prefill_chunck_observation = if actual_prefill_chunck_tokens
            == pending_prefill_chunck_decision.candidate_prefill_chunck_tokens
        {
            PrefillChunckSizeOptimizerObservation::full_prefill_chunck(
                actual_prefill_chunck_tokens,
                elapsed_millis,
            )
        } else {
            PrefillChunckSizeOptimizerObservation::partial_prefill_chunck(
                actual_prefill_chunck_tokens,
                elapsed_millis,
            )
        };
        let is_full_prefill_chunck = prefill_chunck_observation.is_full_candidate_prefill_chunck();
        if let Err(prefill_chunck_optimizer_error) = prefill_chunck_size_optimizer.tell(
            pending_prefill_chunck_decision.prompt_processing_context,
            pending_prefill_chunck_decision.candidate_prefill_chunck_tokens,
            prefill_chunck_observation,
        ) {
            tracing::warn!(
                error = %prefill_chunck_optimizer_error,
                "Qwen3.5 prefill_chunck size optimizer rejected an observation"
            );
        }
        // Persist optimizer state after each full-chunk observation.
        // Only full chunks carry throughput data that changes optimizer decisions.
        // Partial chunks are silently discarded by tell(), so no point saving.
        if is_full_prefill_chunck {
            Self::persist_optimizer_state(
                prefill_chunck_size_optimizer,
                optimizer_state_persistence.as_ref(),
            );
        }
        self.has_completed_prefill_chunck_in_active_request = true;
    }

    /// Persists optimizer state to disk if persistence is configured.
    /// Failures are logged at warn level and silently ignored — the optimizer
    /// is an accelerator, not a correctness gate.
    fn persist_optimizer_state(
        prefill_chunck_size_optimizer: &PrefillChunckSizeOptimizer,
        optimizer_state_persistence: Option<&OptimizerStatePersistence>,
    ) {
        let Some(optimizer_state_persistence) = optimizer_state_persistence else {
            return;
        };
        if let Err(persistence_error) = prefill_chunck_size_optimizer.save_to_directory(
            &optimizer_state_persistence.optimizer_state_directory,
            &optimizer_state_persistence.model_id,
            &optimizer_state_persistence.model_revision,
        ) {
            tracing::warn!(
                error = %persistence_error,
                optimizer_state_directory = %optimizer_state_persistence.optimizer_state_directory.display(),
                "Failed to persist optimizer state; will retry on next observation"
            );
        }
    }

    /// Returns the exclusive end of the next prompt-processing chunk.
    #[must_use]
    pub fn next_prefill_chunck_end(
        &mut self,
        prefill_chunck_start: usize,
        final_prompt_index: usize,
    ) -> usize {
        let prompt_processing_context = self.context_for_prefill_chunck_start(prefill_chunck_start);
        let (candidate_prefill_chunck_tokens, prefill_chunck_decision_reason) = {
            let PrefillChunckSizingMode::Optimized {
                prefill_chunck_size_optimizer,
                pending_prefill_chunck_decision,
                ..
            } = &mut self.prefill_chunck_sizing_mode
            else {
                return prefill_chunck_start
                    .saturating_add(self.active_prefill_chunck_tokens)
                    .min(final_prompt_index);
            };
            let prefill_chunck_decision =
                prefill_chunck_size_optimizer.ask(prompt_processing_context);
            let candidate_prefill_chunck_tokens = prefill_chunck_decision
                .candidate_prefill_chunck_tokens
                .min(self.maximum_prefill_chunck_tokens);
            *pending_prefill_chunck_decision = Some(PendingPrefillChunckDecision {
                prompt_processing_context,
                candidate_prefill_chunck_tokens,
            });
            (
                candidate_prefill_chunck_tokens,
                prefill_chunck_decision.reason,
            )
        };
        let previous_prefill_chunck_tokens = self.active_prefill_chunck_tokens;
        self.active_prefill_chunck_tokens = candidate_prefill_chunck_tokens;
        if candidate_prefill_chunck_tokens != previous_prefill_chunck_tokens {
            tracing::info!(
                previous_prefill_chunck_tokens,
                active_prefill_chunck_tokens = candidate_prefill_chunck_tokens,
                reason = ?prefill_chunck_decision_reason,
                context_bucket = prompt_processing_context.context_identifier(),
                prefill_chunck_start,
                "Qwen3.5 prefill chunk size changed"
            );
        }
        prefill_chunck_start
            .saturating_add(candidate_prefill_chunck_tokens)
            .min(final_prompt_index)
    }

    fn context_for_prefill_chunck_start(
        &self,
        prefill_chunck_start: usize,
    ) -> PrefillChunckSizeOptimizerContext {
        PrefillChunckSizeOptimizerContext::new(
            self.prompt_processing_context_identifier(prefill_chunck_start),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingPrefillChunckDecision {
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    candidate_prefill_chunck_tokens: usize,
}

fn optimizer_candidate_prefill_chunck_tokens(maximum_prefill_chunck_tokens: usize) -> Vec<usize> {
    if maximum_prefill_chunck_tokens <= MINIMUM_OPTIMIZER_PREFILL_CHUNCK_TOKENS {
        return vec![maximum_prefill_chunck_tokens];
    }
    let mut candidate_prefill_chunck_tokens = Vec::new();
    let mut next_candidate_prefill_chunck_tokens = MINIMUM_OPTIMIZER_PREFILL_CHUNCK_TOKENS;
    while next_candidate_prefill_chunck_tokens < maximum_prefill_chunck_tokens {
        candidate_prefill_chunck_tokens.push(next_candidate_prefill_chunck_tokens);
        next_candidate_prefill_chunck_tokens *= 2;
    }
    candidate_prefill_chunck_tokens.push(maximum_prefill_chunck_tokens);
    candidate_prefill_chunck_tokens
}

fn maximum_prefill_chunck_tokens_from_u32(
    maximum_prefill_chunck_tokens: u32,
) -> Result<usize, Qwen3_5PrefillChunckSizerError> {
    let prefill_chunck_tokens = usize::try_from(maximum_prefill_chunck_tokens)
        .map_err(|_| Qwen3_5PrefillChunckSizerError::ExceedsPlatformRange)?;
    if prefill_chunck_tokens == 0 {
        return Err(Qwen3_5PrefillChunckSizerError::MustBePositive);
    }
    Ok(prefill_chunck_tokens)
}

/// Invalid explicit Qwen3.5 prompt-processing `prefill_chunck_tokens`.
#[derive(Clone, Debug, Error)]
pub enum Qwen3_5PrefillChunckSizerError {
    #[error("prefill_chunck_tokens exceeds the platform integer range")]
    ExceedsPlatformRange,
    #[error("prefill_chunck_tokens must be positive")]
    MustBePositive,
    #[error("prefill_chunck_tokens optimizer rejected candidate set")]
    OptimizerRejectedCandidateSet,
}
