use super::{
    prefill_chunck_sizer_configuration::{
        Qwen3_5PrefillChunckSizerError, configured_candidate_prefill_chunck_tokens,
        maximum_prefill_chunck_tokens_from_u32,
    },
    prefill_execution_context::{CAPACITY_REDUCED_CONTEXT_FLAG, Qwen3_5PrefillExecutionContext},
    prefill_optimizer_insight::prefill_optimizer_insight,
};
use crate::{
    PrefillChunckOptimizerContextInsight, PrefillChunckOptimizerInsight,
    PrefillChunckSizeOptimizer, PrefillChunckSizeOptimizerContext,
    PrefillChunckSizeOptimizerObservation,
};
use std::path::PathBuf;
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
    active_request_has_observed_capacity_reduction: bool,
    latest_prefill_optimizer_insight: Option<PrefillChunckOptimizerInsight>,
    prompt_position_context_bucket_tokens: Option<usize>,
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
    /// Creates the production optimizer with explicit evidence and position boundaries.
    pub fn for_optimized_with_behavior(
        maximum_prefill_chunck_tokens: u32,
        optimizer_prefill_chunck_token_candidates: Vec<u32>,
        sliding_window_observation_count: u32,
        prompt_position_context_bucket_tokens: u32,
    ) -> Result<Self, Qwen3_5PrefillChunckSizerError> {
        if sliding_window_observation_count == 0 || prompt_position_context_bucket_tokens == 0 {
            return Err(Qwen3_5PrefillChunckSizerError::MustBePositive);
        }
        Self::for_optimized_with_maximum_prefill_chunck_tokens_and_behavior(
            maximum_prefill_chunck_tokens,
            optimizer_prefill_chunck_token_candidates,
            usize::try_from(sliding_window_observation_count)
                .map_err(|_| Qwen3_5PrefillChunckSizerError::ExceedsPlatformRange)?,
            usize::try_from(prompt_position_context_bucket_tokens)
                .map_err(|_| Qwen3_5PrefillChunckSizerError::ExceedsPlatformRange)?,
        )
    }

    /// Creates the production optimizer with persisted state for warm starts.
    ///
    /// On construction, attempts to load persisted state from the given directory.
    /// If no state exists or the state is stale/corrupt, starts fresh (no error).
    /// After each full-chunk observation, persists the updated optimizer state.
    /// Creates a persisted optimizer with explicit evidence and position boundaries.
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
        let prefill_chunck_tokens =
            maximum_prefill_chunck_tokens_from_u32(maximum_prefill_chunck_tokens)?;
        let candidate_prefill_chunck_tokens = configured_candidate_prefill_chunck_tokens(
            optimizer_prefill_chunck_token_candidates,
            prefill_chunck_tokens,
        )?;
        let state_file_path = optimizer_state_directory.join("prefill-chunck-size.json");

        let prefill_chunck_size_optimizer = match PrefillChunckSizeOptimizer::load_from_path(
            state_file_path,
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
                PrefillChunckSizeOptimizer::new(
                    candidate_prefill_chunck_tokens.clone(),
                    sliding_window_observation_count,
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
                    sliding_window_observation_count,
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
            active_request_has_observed_capacity_reduction: false,
            latest_prefill_optimizer_insight: None,
            prompt_position_context_bucket_tokens: Some(prompt_position_context_bucket_tokens),
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
            active_request_has_observed_capacity_reduction: false,
            latest_prefill_optimizer_insight: None,
            prompt_position_context_bucket_tokens: None,
        })
    }

    fn for_optimized_with_maximum_prefill_chunck_tokens_and_behavior(
        maximum_prefill_chunck_tokens: u32,
        optimizer_prefill_chunck_token_candidates: Vec<u32>,
        sliding_window_observation_count: usize,
        prompt_position_context_bucket_tokens: usize,
    ) -> Result<Self, Qwen3_5PrefillChunckSizerError> {
        let prefill_chunck_tokens =
            maximum_prefill_chunck_tokens_from_u32(maximum_prefill_chunck_tokens)?;
        let candidate_prefill_chunck_tokens = configured_candidate_prefill_chunck_tokens(
            optimizer_prefill_chunck_token_candidates,
            prefill_chunck_tokens,
        )?;
        let prefill_chunck_size_optimizer = PrefillChunckSizeOptimizer::new(
            candidate_prefill_chunck_tokens,
            sliding_window_observation_count,
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
            active_request_has_observed_capacity_reduction: false,
            latest_prefill_optimizer_insight: None,
            prompt_position_context_bucket_tokens: Some(prompt_position_context_bucket_tokens),
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
    pub(super) fn prompt_processing_context_identifier(
        &self,
        prefill_chunck_start: usize,
        prefill_execution_context: Qwen3_5PrefillExecutionContext,
    ) -> u64 {
        let raw_position_bucket_identifier = self
            .prompt_position_context_bucket_tokens
            .map(|prompt_position_context_bucket_tokens| {
                u64::try_from(prefill_chunck_start / prompt_position_context_bucket_tokens)
                    .unwrap_or(u64::MAX >> 2)
            })
            .unwrap_or(0);
        let mut context_identifier = raw_position_bucket_identifier.min((1_u64 << 32) - 1);
        context_identifier |= prefill_execution_context.context_identifier_flags();
        if self.active_request_restored_token_count > 0 {
            context_identifier |= RESTORED_PREFIX_CONTEXT_FLAG;
        }
        if self.active_request_restored_token_count > 0
            && !self.has_completed_prefill_chunck_in_active_request
        {
            context_identifier |= FIRST_CHUNCK_AFTER_RESTORE_CONTEXT_FLAG;
        }
        if self.active_request_has_observed_capacity_reduction {
            context_identifier |= CAPACITY_REDUCED_CONTEXT_FLAG;
        }
        context_identifier
    }

    /// Starts a fresh prompt-processing request with its restored prefix length.
    pub fn start_prompt_processing_request(&mut self, restored_token_count: usize) {
        self.active_request_restored_token_count = restored_token_count;
        self.has_completed_prefill_chunck_in_active_request = false;
        self.active_request_has_observed_capacity_reduction = false;
        if let PrefillChunckSizingMode::Optimized {
            pending_prefill_chunck_decision,
            ..
        } = &mut self.prefill_chunck_sizing_mode
        {
            *pending_prefill_chunck_decision = None;
        }
        // The first optimized next_prefill_chunck_end call asks for an initial candidate.
    }

    pub(super) fn discard_pending_prefill_chunck_decision(&mut self) {
        if let PrefillChunckSizingMode::Optimized {
            pending_prefill_chunck_decision,
            ..
        } = &mut self.prefill_chunck_sizing_mode
        {
            *pending_prefill_chunck_decision = None;
        }
        self.latest_prefill_optimizer_insight = None;
    }

    /// Records a measured prompt-processing chunk for future optimizer choices.
    pub fn record_prefill_chunck_elapsed_millis(
        &mut self,
        actual_prefill_chunck_tokens: usize,
        elapsed_millis: u64,
    ) {
        self.record_prefill_chunck_transition(
            actual_prefill_chunck_tokens,
            elapsed_millis,
            false,
            Qwen3_5PrefillExecutionContext::default(),
        );
    }

    /// Records one complete requested-action transition for future planning.
    pub fn record_prefill_chunck_transition(
        &mut self,
        actual_prefill_chunck_tokens: usize,
        elapsed_millis: u64,
        has_observed_prefill_capacity_constraint: bool,
        next_prefill_execution_context: Qwen3_5PrefillExecutionContext,
    ) {
        self.latest_prefill_optimizer_insight = None;
        let pending_prefill_chunck_decision = {
            let PrefillChunckSizingMode::Optimized {
                pending_prefill_chunck_decision,
                ..
            } = &mut self.prefill_chunck_sizing_mode
            else {
                return;
            };
            let Some(pending_prefill_chunck_decision) = pending_prefill_chunck_decision.take()
            else {
                return;
            };
            pending_prefill_chunck_decision
        };
        self.active_request_has_observed_capacity_reduction |=
            has_observed_prefill_capacity_constraint;
        self.has_completed_prefill_chunck_in_active_request = true;
        let next_prefill_chunck_start = pending_prefill_chunck_decision
            .prefill_chunck_start
            .saturating_add(actual_prefill_chunck_tokens);
        let next_prompt_processing_context = self.context_for_prefill_chunck_start(
            next_prefill_chunck_start,
            next_prefill_execution_context,
        );
        let prefill_chunck_observation = PrefillChunckSizeOptimizerObservation::transition(
            actual_prefill_chunck_tokens,
            elapsed_millis,
            next_prompt_processing_context,
        );
        let PrefillChunckSizingMode::Optimized {
            prefill_chunck_size_optimizer,
            optimizer_state_persistence,
            ..
        } = &mut self.prefill_chunck_sizing_mode
        else {
            return;
        };
        let observation_was_accepted = if let Err(prefill_chunck_optimizer_error) =
            prefill_chunck_size_optimizer.tell(
                pending_prefill_chunck_decision.prompt_processing_context,
                pending_prefill_chunck_decision.candidate_prefill_chunck_tokens,
                prefill_chunck_observation,
            ) {
            tracing::warn!(
                error = %prefill_chunck_optimizer_error,
                "Qwen3.5 prefill_chunck size optimizer rejected an observation"
            );
            false
        } else {
            true
        };
        tracing::trace!(
            requested_prefill_chunck_tokens = pending_prefill_chunck_decision
                .candidate_prefill_chunck_tokens,
            actual_prefill_chunck_tokens,
            elapsed_millis,
            has_observed_prefill_capacity_constraint,
            reason = ?pending_prefill_chunck_decision.reason,
            "Qwen3.5 prefill chunk optimizer recorded a transition"
        );
        Self::persist_optimizer_state(
            prefill_chunck_size_optimizer,
            optimizer_state_persistence.as_ref(),
        );
        if observation_was_accepted {
            self.latest_prefill_optimizer_insight = Some(prefill_optimizer_insight(
                prefill_chunck_size_optimizer,
                pending_prefill_chunck_decision.prompt_processing_context,
                pending_prefill_chunck_decision.candidate_prefill_chunck_tokens,
                actual_prefill_chunck_tokens,
                elapsed_millis,
                pending_prefill_chunck_decision.reason,
                has_observed_prefill_capacity_constraint,
                pending_prefill_chunck_decision.context_insight,
            ));
        }
    }

    /// Takes the latest optimized transition insight for worker telemetry.
    pub fn take_latest_prefill_optimizer_insight(
        &mut self,
    ) -> Option<PrefillChunckOptimizerInsight> {
        self.latest_prefill_optimizer_insight.take()
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
        self.next_prefill_chunck_end_for_execution_context(
            prefill_chunck_start,
            final_prompt_index,
            Qwen3_5PrefillExecutionContext::default(),
        )
    }

    /// Returns the exclusive end using the current execution-mode context.
    #[must_use]
    pub fn next_prefill_chunck_end_for_execution_context(
        &mut self,
        prefill_chunck_start: usize,
        final_prompt_index: usize,
        prefill_execution_context: Qwen3_5PrefillExecutionContext,
    ) -> usize {
        let prompt_processing_context =
            self.context_for_prefill_chunck_start(prefill_chunck_start, prefill_execution_context);
        let optimizer_context_insight = PrefillChunckOptimizerContextInsight {
            prompt_position_tokens: prefill_chunck_start,
            has_restored_prefix: self.active_request_restored_token_count > 0,
            is_first_chunck_after_restore: self.active_request_restored_token_count > 0
                && !self.has_completed_prefill_chunck_in_active_request,
            has_visual_embeddings: prefill_execution_context.has_visual_embeddings(),
            is_mtp_active: prefill_execution_context.has_optional_prediction_session(),
            are_sparse_experts_paged: prefill_execution_context.are_sparse_experts_paged(),
            is_prompt_cache_capture_eligible: prefill_execution_context
                .is_prompt_cache_capture_eligible(),
            has_prior_capacity_reduction: self.active_request_has_observed_capacity_reduction,
        };
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
            let remaining_prompt_tokens = final_prompt_index.saturating_sub(prefill_chunck_start);
            let prefill_chunck_decision = prefill_chunck_size_optimizer
                .ask_with_maximum_prefill_chunck_tokens(
                    prompt_processing_context,
                    remaining_prompt_tokens,
                );
            let candidate_prefill_chunck_tokens = prefill_chunck_decision
                .candidate_prefill_chunck_tokens
                .min(self.maximum_prefill_chunck_tokens);
            *pending_prefill_chunck_decision = Some(PendingPrefillChunckDecision {
                prompt_processing_context,
                candidate_prefill_chunck_tokens,
                prefill_chunck_start,
                reason: prefill_chunck_decision.reason,
                context_insight: optimizer_context_insight,
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
        prefill_execution_context: Qwen3_5PrefillExecutionContext,
    ) -> PrefillChunckSizeOptimizerContext {
        let context_identifier = self
            .prompt_processing_context_identifier(prefill_chunck_start, prefill_execution_context);
        PrefillChunckSizeOptimizerContext::new_with_fallback(
            context_identifier,
            context_identifier & !((1_u64 << 32) - 1),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingPrefillChunckDecision {
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    candidate_prefill_chunck_tokens: usize,
    prefill_chunck_start: usize,
    reason: crate::PrefillChunckSizeOptimizerDecisionReason,
    context_insight: PrefillChunckOptimizerContextInsight,
}
