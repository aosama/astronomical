//! Request-time Laguna chunk selection and pending-measurement state machine.

use std::path::PathBuf;

use super::configuration::{
    LagunaPromptProcessingChunkSizerError, configured_candidate_chunk_size_tokens,
    maximum_prompt_processing_chunk_size_tokens_from_u32,
};
use super::execution_profile::LagunaPromptProcessingExecutionProfile;
use super::optimization_outcome::prompt_processing_chunk_optimization_outcome;
use crate::{
    PromptProcessingChunkMeasurement, PromptProcessingChunkOptimizationContext,
    PromptProcessingChunkOptimizationOutcome, PromptProcessingChunkSizeOptimizer,
    PromptProcessingChunkSizeSelectionReason, PromptProcessingMeasurementContext,
};

/// Owns Laguna prompt-processing chunk-size selection and boundary calculation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaPromptProcessingChunkSizer {
    maximum_prompt_processing_chunk_size_tokens: usize,
    active_prompt_processing_chunk_size_tokens: usize,
    prompt_processing_chunk_sizing_mode: PromptProcessingChunkSizingMode,
    active_request_restored_token_count: usize,
    has_completed_prompt_processing_chunk_in_active_request: bool,
    active_request_has_observed_capacity_reduction: bool,
    latest_prompt_processing_chunk_optimization_outcome:
        Option<PromptProcessingChunkOptimizationOutcome>,
    position_range_size_tokens: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PromptProcessingChunkSizingMode {
    Fixed,
    Optimized {
        prompt_processing_chunk_size_optimizer: PromptProcessingChunkSizeOptimizer,
        pending_prompt_processing_chunk_selection: Option<PendingPromptProcessingChunkSelection>,
        optimizer_state_persistence: Option<OptimizerStatePersistence>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OptimizerStatePersistence {
    pub(super) optimizer_state_directory: PathBuf,
    pub(super) model_id: String,
    pub(super) model_revision: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingPromptProcessingChunkSelection {
    measurement_context: PromptProcessingMeasurementContext,
    selected_candidate_chunk_size_tokens: usize,
    chunk_start_token_position: usize,
    selection_reason: PromptProcessingChunkSizeSelectionReason,
    optimization_context: PromptProcessingChunkOptimizationContext,
}

impl LagunaPromptProcessingChunkSizer {
    /// Creates a fixed chunk size that never writes optimizer state.
    pub fn for_fixed_prompt_processing_chunk_size_tokens(
        fixed_prompt_processing_chunk_size_tokens: u32,
    ) -> Result<Self, LagunaPromptProcessingChunkSizerError> {
        let fixed_prompt_processing_chunk_size_tokens =
            maximum_prompt_processing_chunk_size_tokens_from_u32(
                fixed_prompt_processing_chunk_size_tokens,
            )?;
        Ok(Self {
            maximum_prompt_processing_chunk_size_tokens: fixed_prompt_processing_chunk_size_tokens,
            active_prompt_processing_chunk_size_tokens: fixed_prompt_processing_chunk_size_tokens,
            prompt_processing_chunk_sizing_mode: PromptProcessingChunkSizingMode::Fixed,
            active_request_restored_token_count: 0,
            has_completed_prompt_processing_chunk_in_active_request: false,
            active_request_has_observed_capacity_reduction: false,
            latest_prompt_processing_chunk_optimization_outcome: None,
            position_range_size_tokens: None,
        })
    }

    /// Creates the in-memory optimizer without a persistence directory.
    pub fn for_optimized_with_behavior(
        maximum_prompt_processing_chunk_size_tokens: u32,
        configured_candidate_chunk_size_token_counts: Vec<u32>,
        maximum_retained_measurements_per_candidate_and_context: u32,
        position_range_size_tokens: u32,
    ) -> Result<Self, LagunaPromptProcessingChunkSizerError> {
        Self::for_optimized_with_optional_persistence(
            maximum_prompt_processing_chunk_size_tokens,
            configured_candidate_chunk_size_token_counts,
            maximum_retained_measurements_per_candidate_and_context,
            position_range_size_tokens,
            None,
        )
    }

    pub(super) fn for_optimized_with_optional_persistence(
        maximum_prompt_processing_chunk_size_tokens: u32,
        configured_candidate_chunk_size_token_counts: Vec<u32>,
        maximum_retained_measurements_per_candidate_and_context: u32,
        position_range_size_tokens: u32,
        optimizer_state_persistence: Option<OptimizerStatePersistence>,
    ) -> Result<Self, LagunaPromptProcessingChunkSizerError> {
        if maximum_retained_measurements_per_candidate_and_context == 0
            || position_range_size_tokens == 0
        {
            return Err(LagunaPromptProcessingChunkSizerError::MustBePositive);
        }
        let maximum_prompt_processing_chunk_size_tokens =
            maximum_prompt_processing_chunk_size_tokens_from_u32(
                maximum_prompt_processing_chunk_size_tokens,
            )?;
        let candidate_chunk_size_tokens = configured_candidate_chunk_size_tokens(
            configured_candidate_chunk_size_token_counts,
            maximum_prompt_processing_chunk_size_tokens,
        )?;
        let maximum_retained_measurements_per_candidate_and_context =
            usize::try_from(maximum_retained_measurements_per_candidate_and_context)
                .map_err(|_| LagunaPromptProcessingChunkSizerError::ExceedsPlatformRange)?;
        let position_range_size_tokens = usize::try_from(position_range_size_tokens)
            .map_err(|_| LagunaPromptProcessingChunkSizerError::ExceedsPlatformRange)?;
        let prompt_processing_chunk_size_optimizer = match optimizer_state_persistence.as_ref() {
            Some(optimizer_state_persistence) => super::persisted_state::load_or_create_optimizer(
                optimizer_state_persistence,
                candidate_chunk_size_tokens,
                maximum_retained_measurements_per_candidate_and_context,
            )?,
            None => PromptProcessingChunkSizeOptimizer::new(
                candidate_chunk_size_tokens,
                maximum_retained_measurements_per_candidate_and_context,
            )
            .map_err(|_| LagunaPromptProcessingChunkSizerError::OptimizerRejectedCandidateSet)?,
        };
        let active_prompt_processing_chunk_size_tokens = prompt_processing_chunk_size_optimizer
            .candidate_chunk_size_tokens()
            .first()
            .copied()
            .ok_or(LagunaPromptProcessingChunkSizerError::OptimizerRejectedCandidateSet)?;
        Ok(Self {
            maximum_prompt_processing_chunk_size_tokens,
            active_prompt_processing_chunk_size_tokens,
            prompt_processing_chunk_sizing_mode: PromptProcessingChunkSizingMode::Optimized {
                prompt_processing_chunk_size_optimizer,
                pending_prompt_processing_chunk_selection: None,
                optimizer_state_persistence,
            },
            active_request_restored_token_count: 0,
            has_completed_prompt_processing_chunk_in_active_request: false,
            active_request_has_observed_capacity_reduction: false,
            latest_prompt_processing_chunk_optimization_outcome: None,
            position_range_size_tokens: Some(position_range_size_tokens),
        })
    }

    /// Starts a fresh prompt-processing request with its restored prefix length.
    pub fn start_prompt_processing_request(&mut self, restored_token_count: usize) {
        self.active_request_restored_token_count = restored_token_count;
        self.has_completed_prompt_processing_chunk_in_active_request = false;
        self.active_request_has_observed_capacity_reduction = false;
        if let PromptProcessingChunkSizingMode::Optimized {
            pending_prompt_processing_chunk_selection,
            ..
        } = &mut self.prompt_processing_chunk_sizing_mode
        {
            *pending_prompt_processing_chunk_selection = None;
        }
    }

    /// Returns the exclusive end of the next prompt-processing chunk.
    #[must_use]
    pub fn next_prompt_processing_chunk_end(
        &mut self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
        execution_profile: LagunaPromptProcessingExecutionProfile,
    ) -> usize {
        let measurement_context =
            self.measurement_context_for_chunk_start(chunk_start_token_position, execution_profile);
        let optimizer_context = self
            .optimization_context_for_chunk_start(chunk_start_token_position, execution_profile);
        let PromptProcessingChunkSizingMode::Optimized {
            prompt_processing_chunk_size_optimizer,
            pending_prompt_processing_chunk_selection,
            ..
        } = &mut self.prompt_processing_chunk_sizing_mode
        else {
            return chunk_start_token_position
                .saturating_add(self.maximum_prompt_processing_chunk_size_tokens)
                .min(final_prompt_end_token_position_exclusive);
        };
        let remaining_prompt_tokens =
            final_prompt_end_token_position_exclusive.saturating_sub(chunk_start_token_position);
        let chunk_selection = prompt_processing_chunk_size_optimizer
            .select_candidate_chunk_size_with_maximum(measurement_context, remaining_prompt_tokens);
        let selected_candidate_chunk_size_tokens = chunk_selection
            .selected_candidate_chunk_size_tokens
            .min(self.maximum_prompt_processing_chunk_size_tokens);
        *pending_prompt_processing_chunk_selection = Some(PendingPromptProcessingChunkSelection {
            measurement_context,
            selected_candidate_chunk_size_tokens,
            chunk_start_token_position,
            selection_reason: chunk_selection.reason,
            optimization_context: optimizer_context,
        });
        self.active_prompt_processing_chunk_size_tokens = selected_candidate_chunk_size_tokens;
        chunk_start_token_position
            .saturating_add(selected_candidate_chunk_size_tokens)
            .min(final_prompt_end_token_position_exclusive)
    }

    /// Records one complete selection-to-measurement transition.
    pub fn record_prompt_processing_chunk_transition(
        &mut self,
        processed_prompt_token_count: usize,
        forward_elapsed_millis: u64,
        was_reduced_by_memory_capacity: bool,
        next_execution_profile: LagunaPromptProcessingExecutionProfile,
    ) {
        self.latest_prompt_processing_chunk_optimization_outcome = None;
        let pending_prompt_processing_chunk_selection = {
            let PromptProcessingChunkSizingMode::Optimized {
                pending_prompt_processing_chunk_selection,
                ..
            } = &mut self.prompt_processing_chunk_sizing_mode
            else {
                return;
            };
            let Some(pending_prompt_processing_chunk_selection) =
                pending_prompt_processing_chunk_selection.take()
            else {
                return;
            };
            pending_prompt_processing_chunk_selection
        };
        self.active_request_has_observed_capacity_reduction |= was_reduced_by_memory_capacity;
        self.has_completed_prompt_processing_chunk_in_active_request = true;
        let next_chunk_start_token_position = pending_prompt_processing_chunk_selection
            .chunk_start_token_position
            .saturating_add(processed_prompt_token_count);
        let next_measurement_context = self.measurement_context_for_chunk_start(
            next_chunk_start_token_position,
            next_execution_profile,
        );
        let was_accepted_for_learning = processed_prompt_token_count
            == pending_prompt_processing_chunk_selection.selected_candidate_chunk_size_tokens;
        let PromptProcessingChunkSizingMode::Optimized {
            prompt_processing_chunk_size_optimizer,
            optimizer_state_persistence,
            ..
        } = &mut self.prompt_processing_chunk_sizing_mode
        else {
            return;
        };
        let measurement_was_accepted = if was_accepted_for_learning {
            let chunk_measurement = PromptProcessingChunkMeasurement::transition(
                processed_prompt_token_count,
                forward_elapsed_millis,
                next_measurement_context,
            );
            match prompt_processing_chunk_size_optimizer.record_measurement(
                pending_prompt_processing_chunk_selection.measurement_context,
                pending_prompt_processing_chunk_selection.selected_candidate_chunk_size_tokens,
                chunk_measurement,
            ) {
                Ok(()) => true,
                Err(chunk_optimizer_error) => {
                    tracing::warn!(
                        error = %chunk_optimizer_error,
                        "Laguna prompt-processing chunk size optimizer rejected a measurement"
                    );
                    false
                }
            }
        } else {
            false
        };
        self.latest_prompt_processing_chunk_optimization_outcome =
            Some(prompt_processing_chunk_optimization_outcome(
                prompt_processing_chunk_size_optimizer,
                pending_prompt_processing_chunk_selection.measurement_context,
                pending_prompt_processing_chunk_selection.selected_candidate_chunk_size_tokens,
                processed_prompt_token_count,
                forward_elapsed_millis,
                pending_prompt_processing_chunk_selection.selection_reason,
                was_reduced_by_memory_capacity,
                measurement_was_accepted,
                pending_prompt_processing_chunk_selection.optimization_context,
            ));
        if measurement_was_accepted
            && let Some(optimizer_state_persistence) = optimizer_state_persistence
        {
            if let Err(persistence_error) = prompt_processing_chunk_size_optimizer
                .save_to_directory(
                    &optimizer_state_persistence.optimizer_state_directory,
                    &optimizer_state_persistence.model_id,
                    &optimizer_state_persistence.model_revision,
                )
            {
                tracing::warn!(
                    error = %persistence_error,
                    "Failed to persist Laguna optimizer state; will retry on next learning sample"
                );
            }
        }
    }

    #[must_use]
    pub const fn maximum_prompt_processing_chunk_size_tokens(&self) -> usize {
        self.maximum_prompt_processing_chunk_size_tokens
    }

    #[must_use]
    pub const fn active_prompt_processing_chunk_size_tokens(&self) -> usize {
        self.active_prompt_processing_chunk_size_tokens
    }

    #[must_use]
    pub const fn is_optimized(&self) -> bool {
        matches!(
            self.prompt_processing_chunk_sizing_mode,
            PromptProcessingChunkSizingMode::Optimized { .. }
        )
    }

    #[must_use]
    pub fn latest_prompt_processing_chunk_optimization_outcome(
        &self,
    ) -> Option<&PromptProcessingChunkOptimizationOutcome> {
        self.latest_prompt_processing_chunk_optimization_outcome
            .as_ref()
    }

    /// Removes the latest outcome after it has been emitted to worker telemetry.
    pub fn take_latest_prompt_processing_chunk_optimization_outcome(
        &mut self,
    ) -> Option<PromptProcessingChunkOptimizationOutcome> {
        self.latest_prompt_processing_chunk_optimization_outcome
            .take()
    }

    #[must_use]
    pub const fn has_pending_selection(&self) -> bool {
        matches!(
            self.prompt_processing_chunk_sizing_mode,
            PromptProcessingChunkSizingMode::Optimized {
                pending_prompt_processing_chunk_selection: Some(_),
                ..
            }
        )
    }

    #[must_use]
    pub(super) const fn position_range_size_tokens(&self) -> Option<usize> {
        self.position_range_size_tokens
    }

    #[must_use]
    pub(super) const fn active_request_restored_token_count(&self) -> usize {
        self.active_request_restored_token_count
    }

    #[must_use]
    pub(super) const fn has_completed_prompt_processing_chunk_in_active_request(&self) -> bool {
        self.has_completed_prompt_processing_chunk_in_active_request
    }

    #[must_use]
    pub(super) const fn active_request_has_observed_capacity_reduction(&self) -> bool {
        self.active_request_has_observed_capacity_reduction
    }
}
