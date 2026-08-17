//! Qwen3.5 prompt-processing chunk selection and transition learning.
//!
//! Fixed mode bypasses learning. Optimized mode retains each selection until its
//! chunk completes, then learns only from full-capacity execution. Position remains
//! telemetry while material execution conditions isolate learning. Persistence stays
//! in its child module so loading policy cannot obscure this request-time state machine.

mod configuration;
mod measurement_context;
mod optimization_outcome;
mod pending_selection;
mod persisted_state;

use std::path::PathBuf;

use super::prefill_execution_context::Qwen3_5PrefillExecutionContext;
use crate::{
    PromptProcessingChunkMeasurement, PromptProcessingChunkOptimizationOutcome,
    PromptProcessingChunkSizeOptimizer,
};
use configuration::{
    configured_candidate_chunk_size_tokens, maximum_prompt_processing_chunk_size_tokens_from_u32,
};
use optimization_outcome::prompt_processing_chunk_optimization_outcome;
use pending_selection::PendingPromptProcessingChunkSelection;

pub use configuration::Qwen3_5PromptProcessingChunkSizerError;

/// Owns Qwen3.5 prompt-processing chunk-size selection and boundary calculation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5PromptProcessingChunkSizer {
    maximum_prompt_processing_chunk_size_tokens: usize,
    active_prompt_processing_chunk_size_tokens: usize,
    /// Fixed size used only while sparse experts stream from storage.
    ///
    /// Optimized mode and fixed mode without an SSD override leave this unset;
    /// both fall back to the complete-resident fixed size at selection time.
    ssd_streaming_prompt_processing_chunk_size_tokens: Option<usize>,
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
struct OptimizerStatePersistence {
    optimizer_state_directory: PathBuf,
    model_id: String,
    model_revision: String,
}

impl Qwen3_5PromptProcessingChunkSizer {
    /// Creates the production optimizer with explicit measurements and position boundaries.
    pub fn for_optimized_with_behavior(
        maximum_prompt_processing_chunk_size_tokens: u32,
        configured_candidate_chunk_size_token_counts: Vec<u32>,
        maximum_retained_measurements_per_candidate_and_context: u32,
        position_range_size_tokens: u32,
    ) -> Result<Self, Qwen3_5PromptProcessingChunkSizerError> {
        if maximum_retained_measurements_per_candidate_and_context == 0
            || position_range_size_tokens == 0
        {
            return Err(Qwen3_5PromptProcessingChunkSizerError::MustBePositive);
        }
        Self::for_optimized_with_maximum_chunk_size_and_behavior(
            maximum_prompt_processing_chunk_size_tokens,
            configured_candidate_chunk_size_token_counts,
            usize::try_from(maximum_retained_measurements_per_candidate_and_context)
                .map_err(|_| Qwen3_5PromptProcessingChunkSizerError::ExceedsPlatformRange)?,
            usize::try_from(position_range_size_tokens)
                .map_err(|_| Qwen3_5PromptProcessingChunkSizerError::ExceedsPlatformRange)?,
        )
    }

    /// Creates a fixed prompt-processing chunk size that bypasses optimizer selection and persistence.
    pub fn for_fixed_prompt_processing_chunk_size_tokens(
        fixed_prompt_processing_chunk_size_tokens: u32,
    ) -> Result<Self, Qwen3_5PromptProcessingChunkSizerError> {
        Self::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
            fixed_prompt_processing_chunk_size_tokens,
            None,
        )
    }

    /// Creates fixed complete-resident and optional SSD-streaming chunk sizes.
    ///
    /// When sparse experts are paged, the SSD-streaming size shortens each
    /// forward so activation pressure stays lower while expert pages stream.
    /// Complete-resident execution keeps the larger fixed size unchanged.
    pub fn for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
        fixed_prompt_processing_chunk_size_tokens: u32,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Option<u32>,
    ) -> Result<Self, Qwen3_5PromptProcessingChunkSizerError> {
        let fixed_prompt_processing_chunk_size_tokens =
            usize::try_from(fixed_prompt_processing_chunk_size_tokens)
                .map_err(|_| Qwen3_5PromptProcessingChunkSizerError::ExceedsPlatformRange)?;
        if fixed_prompt_processing_chunk_size_tokens == 0 {
            return Err(Qwen3_5PromptProcessingChunkSizerError::MustBePositive);
        }
        let ssd_streaming_prompt_processing_chunk_size_tokens =
            match fixed_ssd_streaming_prompt_processing_chunk_size_tokens {
                Some(0) => return Err(Qwen3_5PromptProcessingChunkSizerError::MustBePositive),
                Some(fixed_ssd_streaming_prompt_processing_chunk_size_tokens) => Some(
                    usize::try_from(fixed_ssd_streaming_prompt_processing_chunk_size_tokens)
                        .map_err(|_| {
                            Qwen3_5PromptProcessingChunkSizerError::ExceedsPlatformRange
                        })?,
                ),
                None => None,
            };
        Ok(Self {
            maximum_prompt_processing_chunk_size_tokens: fixed_prompt_processing_chunk_size_tokens,
            active_prompt_processing_chunk_size_tokens: fixed_prompt_processing_chunk_size_tokens,
            ssd_streaming_prompt_processing_chunk_size_tokens,
            prompt_processing_chunk_sizing_mode: PromptProcessingChunkSizingMode::Fixed,
            active_request_restored_token_count: 0,
            has_completed_prompt_processing_chunk_in_active_request: false,
            active_request_has_observed_capacity_reduction: false,
            latest_prompt_processing_chunk_optimization_outcome: None,
            position_range_size_tokens: None,
        })
    }

    fn for_optimized_with_maximum_chunk_size_and_behavior(
        maximum_prompt_processing_chunk_size_tokens: u32,
        configured_candidate_chunk_size_token_counts: Vec<u32>,
        maximum_retained_measurements_per_candidate_and_context: usize,
        position_range_size_tokens: usize,
    ) -> Result<Self, Qwen3_5PromptProcessingChunkSizerError> {
        let maximum_prompt_processing_chunk_size_tokens =
            maximum_prompt_processing_chunk_size_tokens_from_u32(
                maximum_prompt_processing_chunk_size_tokens,
            )?;
        let candidate_chunk_size_tokens = configured_candidate_chunk_size_tokens(
            configured_candidate_chunk_size_token_counts,
            maximum_prompt_processing_chunk_size_tokens,
        )?;
        let prompt_processing_chunk_size_optimizer = PromptProcessingChunkSizeOptimizer::new(
            candidate_chunk_size_tokens,
            maximum_retained_measurements_per_candidate_and_context,
        )
        .map_err(|_| Qwen3_5PromptProcessingChunkSizerError::OptimizerRejectedCandidateSet)?;
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
                optimizer_state_persistence: None,
            },
            active_request_restored_token_count: 0,
            has_completed_prompt_processing_chunk_in_active_request: false,
            active_request_has_observed_capacity_reduction: false,
            latest_prompt_processing_chunk_optimization_outcome: None,
            position_range_size_tokens: Some(position_range_size_tokens),
        })
    }

    /// Returns the configured fixed size or optimized maximum chunk capacity.
    ///
    /// This is an upper bound, not necessarily the capacity selected for the
    /// most recent chunk. Use [`Self::active_prompt_processing_chunk_size_tokens`]
    /// when reporting the selected candidate.
    #[must_use]
    pub const fn maximum_prompt_processing_chunk_size_tokens(&self) -> usize {
        self.maximum_prompt_processing_chunk_size_tokens
    }

    /// Returns the most recently selected prompt-processing chunk capacity.
    ///
    /// Fixed mode updates this value when paged experts select the optional
    /// storage-streaming override. Optimized mode updates it at selection time.
    #[must_use]
    pub const fn active_prompt_processing_chunk_size_tokens(&self) -> usize {
        self.active_prompt_processing_chunk_size_tokens
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
        // The first optimized next_prompt_processing_chunk_end call selects an initial candidate.
    }

    pub(super) fn discard_pending_prompt_processing_chunk_selection(&mut self) {
        if let PromptProcessingChunkSizingMode::Optimized {
            pending_prompt_processing_chunk_selection,
            ..
        } = &mut self.prompt_processing_chunk_sizing_mode
        {
            *pending_prompt_processing_chunk_selection = None;
        }
        self.latest_prompt_processing_chunk_optimization_outcome = None;
    }

    /// Records a measured prompt-processing chunk for future optimizer choices.
    pub fn record_prompt_processing_chunk_elapsed_millis(
        &mut self,
        processed_prompt_token_count: usize,
        forward_elapsed_millis: u64,
    ) {
        self.record_prompt_processing_chunk_transition(
            processed_prompt_token_count,
            forward_elapsed_millis,
            false,
            Qwen3_5PrefillExecutionContext::default(),
        );
    }

    /// Records one complete selection-to-measurement transition for future planning.
    ///
    /// The pending selection is removed before recording so a failed or duplicate
    /// completion cannot attribute the same executed chunk more than once.
    pub fn record_prompt_processing_chunk_transition(
        &mut self,
        processed_prompt_token_count: usize,
        forward_elapsed_millis: u64,
        was_reduced_by_memory_capacity: bool,
        next_prompt_processing_execution_context: Qwen3_5PrefillExecutionContext,
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
            next_prompt_processing_execution_context,
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
            if let Err(chunk_optimizer_error) = prompt_processing_chunk_size_optimizer
                .record_measurement(
                    pending_prompt_processing_chunk_selection.measurement_context,
                    pending_prompt_processing_chunk_selection.selected_candidate_chunk_size_tokens,
                    chunk_measurement,
                )
            {
                tracing::warn!(
                    error = %chunk_optimizer_error,
                    "Qwen3.5 prompt-processing chunk size optimizer rejected a measurement"
                );
                false
            } else {
                true
            }
        } else {
            false
        };
        tracing::trace!(
            selected_candidate_chunk_size_tokens = pending_prompt_processing_chunk_selection
                .selected_candidate_chunk_size_tokens,
            processed_prompt_token_count,
            forward_elapsed_millis,
            was_reduced_by_memory_capacity,
            reason = ?pending_prompt_processing_chunk_selection.selection_reason,
            measurement_was_accepted,
            "Qwen3.5 prompt-processing chunk optimizer observed a transition"
        );
        if measurement_was_accepted {
            Self::persist_optimizer_state(
                prompt_processing_chunk_size_optimizer,
                optimizer_state_persistence.as_ref(),
            );
        }
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
    }

    /// Takes the latest prompt-processing chunk optimization outcome for worker telemetry.
    pub fn take_latest_prompt_processing_chunk_optimization_outcome(
        &mut self,
    ) -> Option<PromptProcessingChunkOptimizationOutcome> {
        self.latest_prompt_processing_chunk_optimization_outcome
            .take()
    }

    /// Persists optimizer state to disk if persistence is configured.
    /// Failures are logged at warn level and silently ignored — the optimizer
    /// is an accelerator, not a correctness gate.
    fn persist_optimizer_state(
        prompt_processing_chunk_size_optimizer: &PromptProcessingChunkSizeOptimizer,
        optimizer_state_persistence: Option<&OptimizerStatePersistence>,
    ) {
        let Some(optimizer_state_persistence) = optimizer_state_persistence else {
            return;
        };
        if let Err(persistence_error) = prompt_processing_chunk_size_optimizer.save_to_directory(
            &optimizer_state_persistence.optimizer_state_directory,
            &optimizer_state_persistence.model_id,
            &optimizer_state_persistence.model_revision,
        ) {
            tracing::warn!(
                error = %persistence_error,
                optimizer_state_directory = %optimizer_state_persistence.optimizer_state_directory.display(),
                "Failed to persist optimizer state; will retry on next measurement"
            );
        }
    }

    /// Returns the exclusive end of the next prompt-processing chunk.
    #[must_use]
    pub fn next_prompt_processing_chunk_end(
        &mut self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
    ) -> usize {
        self.next_prompt_processing_chunk_end_for_execution_context(
            chunk_start_token_position,
            final_prompt_end_token_position_exclusive,
            Qwen3_5PrefillExecutionContext::default(),
        )
    }

    /// Returns the exclusive end using the current execution-mode context.
    #[must_use]
    pub fn next_prompt_processing_chunk_end_for_execution_context(
        &mut self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
        prompt_processing_execution_context: Qwen3_5PrefillExecutionContext,
    ) -> usize {
        self.next_prompt_processing_chunk_end_with_maximum_executable_capacity(
            chunk_start_token_position,
            final_prompt_end_token_position_exclusive,
            prompt_processing_execution_context,
            usize::MAX,
        )
    }

    pub fn next_prompt_processing_chunk_end_with_maximum_executable_capacity(
        &mut self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
        prompt_processing_execution_context: Qwen3_5PrefillExecutionContext,
        maximum_executable_chunk_size_tokens: usize,
    ) -> usize {
        let measurement_context = self.measurement_context_for_chunk_start(
            chunk_start_token_position,
            prompt_processing_execution_context,
        );
        let optimizer_context = self.optimization_context_for_chunk_start(
            chunk_start_token_position,
            prompt_processing_execution_context,
        );
        let (selected_candidate_chunk_size_tokens, selection_reason) = {
            let PromptProcessingChunkSizingMode::Optimized {
                prompt_processing_chunk_size_optimizer,
                pending_prompt_processing_chunk_selection,
                ..
            } = &mut self.prompt_processing_chunk_sizing_mode
            else {
                // Fixed mode may use a shorter SSD-streaming size while experts
                // page from storage. Complete-resident execution keeps the larger
                // configured fixed size so resident throughput is unchanged.
                let fixed_prompt_processing_chunk_size_tokens =
                    if prompt_processing_execution_context.are_sparse_experts_paged() {
                        self.ssd_streaming_prompt_processing_chunk_size_tokens
                            .unwrap_or(self.maximum_prompt_processing_chunk_size_tokens)
                    } else {
                        self.maximum_prompt_processing_chunk_size_tokens
                    };
                self.active_prompt_processing_chunk_size_tokens =
                    fixed_prompt_processing_chunk_size_tokens;
                return chunk_start_token_position
                    .saturating_add(fixed_prompt_processing_chunk_size_tokens)
                    .min(final_prompt_end_token_position_exclusive);
            };
            let remaining_prompt_tokens = final_prompt_end_token_position_exclusive
                .saturating_sub(chunk_start_token_position);
            let maximum_admissible_candidate_tokens =
                remaining_prompt_tokens.min(maximum_executable_chunk_size_tokens);
            let chunk_selection = prompt_processing_chunk_size_optimizer
                .select_candidate_chunk_size_with_maximum(
                    measurement_context,
                    maximum_admissible_candidate_tokens,
                );
            let selected_candidate_chunk_size_tokens = chunk_selection
                .selected_candidate_chunk_size_tokens
                .min(self.maximum_prompt_processing_chunk_size_tokens);
            *pending_prompt_processing_chunk_selection =
                Some(PendingPromptProcessingChunkSelection {
                    measurement_context,
                    selected_candidate_chunk_size_tokens,
                    chunk_start_token_position,
                    selection_reason: chunk_selection.reason,
                    optimization_context: optimizer_context,
                });
            (selected_candidate_chunk_size_tokens, chunk_selection.reason)
        };
        let previous_selected_candidate_chunk_size_tokens =
            self.active_prompt_processing_chunk_size_tokens;
        self.active_prompt_processing_chunk_size_tokens = selected_candidate_chunk_size_tokens;
        if selected_candidate_chunk_size_tokens != previous_selected_candidate_chunk_size_tokens {
            tracing::info!(
                previous_selected_candidate_chunk_size_tokens,
                selected_candidate_chunk_size_tokens,
                reason = ?selection_reason,
                exact_measurement_context_identifier = measurement_context.exact_measurement_context_identifier(),
                chunk_start_token_position,
                "Qwen3.5 prompt-processing chunk size changed"
            );
        }
        chunk_start_token_position
            .saturating_add(selected_candidate_chunk_size_tokens)
            .min(final_prompt_end_token_position_exclusive)
    }

    #[must_use]
    pub fn next_smaller_candidate_chunk_size_tokens(
        &self,
        attempted_chunk_size_tokens: usize,
    ) -> Option<usize> {
        let PromptProcessingChunkSizingMode::Optimized {
            prompt_processing_chunk_size_optimizer,
            ..
        } = &self.prompt_processing_chunk_sizing_mode
        else {
            return None;
        };
        prompt_processing_chunk_size_optimizer
            .candidate_chunk_size_tokens()
            .iter()
            .rev()
            .copied()
            .find(|candidate_chunk_size_tokens| {
                *candidate_chunk_size_tokens < attempted_chunk_size_tokens
            })
    }
}
