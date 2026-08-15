//! Online candidate selection over bounded, context-specific measurements.
//!
//! Selection deliberately prioritizes missing evidence, then stale evidence,
//! before exploiting cumulative-latency estimates. This ordering prevents a
//! fast early sample from permanently excluding another feasible capacity.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::context_statistics::ContextCandidateStatistics;
use super::episode_latency_planner::lowest_cumulative_latency_candidate_index;
use super::measurement_summary::{
    CandidateMeasurementSummaries, build_candidate_measurement_summaries,
};
use super::{
    PromptProcessingChunkMeasurement, PromptProcessingChunkSizeOptimizerError,
    PromptProcessingMeasurementContext, persistence,
};

const MINIMUM_MAXIMUM_RETAINED_MEASUREMENTS: usize = 1;
const STALE_MEASUREMENT_WINDOW_MULTIPLIER: u64 = 5;

/// Why the optimizer selected a particular candidate chunk size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptProcessingChunkSizeSelectionReason {
    /// Sampling the largest feasible candidate that has no measurements yet.
    ExploreUnmeasuredCandidate,
    /// Refreshing the least recently measured feasible candidate.
    RefreshStaleCandidateMeasurement,
    /// Minimizing predicted remaining prompt-processing latency.
    MinimizeProjectedRemainingPromptLatency,
    /// Remaining prompt tokens are fewer than the smallest registered candidate.
    RemainingTokensBelowSmallestCandidate,
    /// The smallest candidate that contains the final prompt segment.
    SmallestCandidateContainingFinalPromptSegment,
}

/// The candidate the optimizer selected and why.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromptProcessingChunkSizeSelection {
    pub selected_candidate_chunk_size_tokens: usize,
    pub reason: PromptProcessingChunkSizeSelectionReason,
}

/// Tabular online optimizer for cumulative prompt-processing latency.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptProcessingChunkSizeOptimizer {
    candidate_chunk_size_tokens: Vec<usize>,
    maximum_retained_measurements_per_candidate_and_context: usize,
    selection_sequence: u64,
    measurement_sequence: u64,
    context_statistics: BTreeMap<PromptProcessingMeasurementContext, ContextCandidateStatistics>,
}

impl PromptProcessingChunkSizeOptimizer {
    /// Creates an optimizer after sorting and deduplicating positive capacities.
    pub fn new(
        mut candidate_chunk_size_tokens: Vec<usize>,
        maximum_retained_measurements_per_candidate_and_context: usize,
    ) -> Result<Self, PromptProcessingChunkSizeOptimizerError> {
        if candidate_chunk_size_tokens.is_empty() {
            return Err(PromptProcessingChunkSizeOptimizerError::NoCandidateChunkSizeTokens);
        }
        if candidate_chunk_size_tokens.contains(&0) {
            return Err(
                PromptProcessingChunkSizeOptimizerError::CandidateChunkSizeTokensMustBePositive,
            );
        }
        candidate_chunk_size_tokens.sort_unstable();
        candidate_chunk_size_tokens.dedup();
        Ok(Self {
            candidate_chunk_size_tokens,
            maximum_retained_measurements_per_candidate_and_context:
                maximum_retained_measurements_per_candidate_and_context
                    .max(MINIMUM_MAXIMUM_RETAINED_MEASUREMENTS),
            selection_sequence: 0,
            measurement_sequence: 0,
            context_statistics: BTreeMap::new(),
        })
    }

    /// Selects the best candidate for the full prompt up to the largest configured capacity.
    #[must_use]
    pub fn select_candidate_chunk_size(
        &mut self,
        measurement_context: PromptProcessingMeasurementContext,
    ) -> PromptProcessingChunkSizeSelection {
        let maximum_candidate_chunk_size_tokens = self
            .candidate_chunk_size_tokens
            .last()
            .copied()
            .unwrap_or(1);
        self.select_candidate_chunk_size_with_maximum(
            measurement_context,
            maximum_candidate_chunk_size_tokens,
        )
    }

    /// Selects the best candidate for the prompt portion up to the given capacity.
    #[must_use]
    pub fn select_candidate_chunk_size_with_maximum(
        &mut self,
        measurement_context: PromptProcessingMeasurementContext,
        maximum_candidate_chunk_size_tokens: usize,
    ) -> PromptProcessingChunkSizeSelection {
        self.selection_sequence = self.selection_sequence.saturating_add(1);
        let eligible_candidate_count =
            self.candidate_chunk_size_tokens
                .partition_point(|candidate_chunk_size_tokens| {
                    *candidate_chunk_size_tokens <= maximum_candidate_chunk_size_tokens
                });
        if eligible_candidate_count == 0 {
            return PromptProcessingChunkSizeSelection {
                selected_candidate_chunk_size_tokens: self.candidate_chunk_size_tokens[0],
                reason:
                    PromptProcessingChunkSizeSelectionReason::RemainingTokensBelowSmallestCandidate,
            };
        }

        if let Some(candidate_index) = (0..eligible_candidate_count).rev().find(|candidate_index| {
            !self.has_measurements_in_same_execution_profile(measurement_context, *candidate_index)
        }) {
            return self.selection(
                candidate_index,
                PromptProcessingChunkSizeSelectionReason::ExploreUnmeasuredCandidate,
            );
        }

        let stale_after_selections =
            STALE_MEASUREMENT_WINDOW_MULTIPLIER.saturating_mul(eligible_candidate_count as u64);
        let stale_candidate_index = (0..eligible_candidate_count)
            .filter_map(|candidate_index| {
                self.last_measured_selection_sequence_in_same_execution_profile(
                    measurement_context,
                    candidate_index,
                )
                .map(|last_measured_selection_sequence| {
                    (candidate_index, last_measured_selection_sequence)
                })
            })
            .filter(|(_, last_measured_selection_sequence)| {
                self.selection_sequence
                    .saturating_sub(*last_measured_selection_sequence)
                    >= stale_after_selections
            })
            .min_by_key(|(candidate_index, last_measured_selection_sequence)| {
                (*last_measured_selection_sequence, *candidate_index)
            })
            .map(|(candidate_index, _)| candidate_index);
        if let Some(candidate_index) = stale_candidate_index {
            return self.selection(
                candidate_index,
                PromptProcessingChunkSizeSelectionReason::RefreshStaleCandidateMeasurement,
            );
        }

        let unknown_elapsed_millis_per_token = self
            .maximum_measured_elapsed_millis_per_token(measurement_context)
            .max(1);
        let selected_candidate_index = lowest_cumulative_latency_candidate_index(
            &self.candidate_chunk_size_tokens,
            maximum_candidate_chunk_size_tokens,
            measurement_context,
            unknown_elapsed_millis_per_token,
            &|future_measurement_context, candidate_index| {
                self.measurements_for_context_or_execution_profile(
                    future_measurement_context,
                    candidate_index,
                )
            },
        );
        self.selection(
            selected_candidate_index,
            PromptProcessingChunkSizeSelectionReason::MinimizeProjectedRemainingPromptLatency,
        )
    }

    /// Selects the smallest configured candidate that contains one terminal remainder.
    ///
    /// The returned candidate remains the selected capacity for optimizer measurements,
    /// while execution clamps the forward to the exact remaining token count.
    #[must_use]
    pub fn select_candidate_for_terminal_remainder(
        &mut self,
        terminal_remainder_tokens: usize,
        configured_maximum_candidate_chunk_size_tokens: usize,
    ) -> Option<PromptProcessingChunkSizeSelection> {
        let terminal_ceiling_candidate = self.candidate_chunk_size_tokens.iter().copied().find(
            |candidate_chunk_size_tokens| {
                *candidate_chunk_size_tokens >= terminal_remainder_tokens
                    && *candidate_chunk_size_tokens
                        <= configured_maximum_candidate_chunk_size_tokens
            },
        )?;
        self.selection_sequence = self.selection_sequence.saturating_add(1);
        Some(PromptProcessingChunkSizeSelection {
            selected_candidate_chunk_size_tokens: terminal_ceiling_candidate,
            reason: PromptProcessingChunkSizeSelectionReason::SmallestCandidateContainingFinalPromptSegment,
        })
    }

    /// Records one completed prompt-processing chunk measurement.
    pub fn record_measurement(
        &mut self,
        measurement_context: PromptProcessingMeasurementContext,
        selected_candidate_chunk_size_tokens: usize,
        chunk_measurement: PromptProcessingChunkMeasurement,
    ) -> Result<(), PromptProcessingChunkSizeOptimizerError> {
        let candidate_index = self.candidate_index(selected_candidate_chunk_size_tokens)?;
        if chunk_measurement.processed_prompt_token_count() == 0 {
            return Err(
                PromptProcessingChunkSizeOptimizerError::MeasurementProcessedTokenCountMustBePositive,
            );
        }
        if chunk_measurement.forward_elapsed_millis() == 0 {
            return Err(
                PromptProcessingChunkSizeOptimizerError::MeasurementForwardElapsedMillisMustBePositive,
            );
        }
        self.measurement_sequence = self.measurement_sequence.saturating_add(1);
        let candidate_measurement = CandidateChunkMeasurement {
            processed_prompt_token_count: chunk_measurement.processed_prompt_token_count(),
            forward_elapsed_millis: chunk_measurement.forward_elapsed_millis(),
            next_measurement_context: chunk_measurement.next_measurement_context(),
            measurement_sequence: self.measurement_sequence,
        };
        self.context_statistics
            .entry(measurement_context)
            .or_insert_with(|| {
                ContextCandidateStatistics::new(self.candidate_chunk_size_tokens.len())
            })
            .candidate_statistics[candidate_index]
            .record_measurement(
                candidate_measurement,
                self.maximum_retained_measurements_per_candidate_and_context,
                self.selection_sequence,
            );
        Ok(())
    }

    #[must_use]
    pub fn candidate_chunk_size_tokens(&self) -> &[usize] {
        &self.candidate_chunk_size_tokens
    }

    #[must_use]
    pub fn maximum_retained_measurements_per_candidate_and_context(&self) -> usize {
        self.maximum_retained_measurements_per_candidate_and_context
    }

    /// Summarizes the exact-or-execution-profile measurements used for selections in one context.
    #[must_use]
    pub fn candidate_measurement_summaries(
        &self,
        measurement_context: PromptProcessingMeasurementContext,
    ) -> CandidateMeasurementSummaries {
        build_candidate_measurement_summaries(
            &self.candidate_chunk_size_tokens,
            self.maximum_retained_measurements_per_candidate_and_context,
            self.selection_sequence,
            &self.context_statistics,
            measurement_context,
        )
    }

    pub(crate) fn selection_sequence(&self) -> u64 {
        self.selection_sequence
    }

    pub(crate) fn measurement_sequence(&self) -> u64 {
        self.measurement_sequence
    }

    pub(crate) fn context_statistics(
        &self,
    ) -> &BTreeMap<PromptProcessingMeasurementContext, ContextCandidateStatistics> {
        &self.context_statistics
    }

    pub(crate) fn new_from_persisted_state(
        candidate_chunk_size_tokens: Vec<usize>,
        maximum_retained_measurements_per_candidate_and_context: usize,
        selection_sequence: u64,
        measurement_sequence: u64,
        context_statistics: BTreeMap<
            PromptProcessingMeasurementContext,
            ContextCandidateStatistics,
        >,
    ) -> Self {
        Self {
            candidate_chunk_size_tokens,
            maximum_retained_measurements_per_candidate_and_context,
            selection_sequence,
            measurement_sequence,
            context_statistics,
        }
    }

    pub fn save_to_directory(
        &self,
        optimizer_directory: &std::path::Path,
        model_id: &str,
        model_revision: &str,
    ) -> Result<(), PromptProcessingChunkSizeOptimizerError> {
        persistence::save_optimizer_to_directory(
            self,
            optimizer_directory,
            model_id,
            model_revision,
        )
    }

    /// Returns the model-and-revision-specific state file beneath a shared
    /// optimizer root. The path contains no raw model identity components.
    #[must_use]
    pub fn persisted_state_file_path(
        optimizer_directory: &std::path::Path,
        model_id: &str,
        model_revision: &str,
    ) -> PathBuf {
        persistence::optimizer_state_file_path(optimizer_directory, model_id, model_revision)
    }

    pub fn load_from_path(
        state_file_path: PathBuf,
        model_id: String,
        model_revision: String,
        candidate_chunk_size_tokens: Vec<usize>,
        maximum_retained_measurements_per_candidate_and_context: usize,
    ) -> Result<Option<Self>, PromptProcessingChunkSizeOptimizerError> {
        persistence::load_optimizer_from_path(
            &state_file_path,
            &model_id,
            &model_revision,
            candidate_chunk_size_tokens,
            maximum_retained_measurements_per_candidate_and_context,
        )
    }

    fn selection(
        &self,
        candidate_index: usize,
        reason: PromptProcessingChunkSizeSelectionReason,
    ) -> PromptProcessingChunkSizeSelection {
        PromptProcessingChunkSizeSelection {
            selected_candidate_chunk_size_tokens: self.candidate_chunk_size_tokens[candidate_index],
            reason,
        }
    }

    fn candidate_index(
        &self,
        selected_candidate_chunk_size_tokens: usize,
    ) -> Result<usize, PromptProcessingChunkSizeOptimizerError> {
        self.candidate_chunk_size_tokens
            .binary_search(&selected_candidate_chunk_size_tokens)
            .map_err(|_| {
                PromptProcessingChunkSizeOptimizerError::UnregisteredCandidateChunkSizeTokens {
                    candidate_chunk_size_tokens: selected_candidate_chunk_size_tokens,
                }
            })
    }

    /// Returns true when any context sharing the same execution profile has measurements
    /// for the given candidate index.
    fn has_measurements_in_same_execution_profile(
        &self,
        measurement_context: PromptProcessingMeasurementContext,
        candidate_index: usize,
    ) -> bool {
        self.context_statistics
            .iter()
            .any(|(stored_context, statistics)| {
                stored_context.position_independent_execution_profile_identifier()
                    == measurement_context.position_independent_execution_profile_identifier()
                    && !statistics.candidate_statistics[candidate_index]
                        .measurements
                        .is_empty()
            })
    }

    /// Returns the most recent selection sequence number when any context sharing the same
    /// execution profile has a measurement for the given candidate index.
    fn last_measured_selection_sequence_in_same_execution_profile(
        &self,
        measurement_context: PromptProcessingMeasurementContext,
        candidate_index: usize,
    ) -> Option<u64> {
        self.context_statistics
            .iter()
            .filter(|(stored_context, _)| {
                stored_context.position_independent_execution_profile_identifier()
                    == measurement_context.position_independent_execution_profile_identifier()
            })
            .filter_map(|(_, statistics)| {
                statistics.candidate_statistics[candidate_index].last_measured_selection_sequence
            })
            .max()
    }

    /// Returns measurements from the exact context for the given candidate, or
    /// falls back to measurements from all contexts sharing the same execution profile.
    fn measurements_for_context_or_execution_profile(
        &self,
        measurement_context: PromptProcessingMeasurementContext,
        candidate_index: usize,
    ) -> Vec<CandidateChunkMeasurement> {
        let exact_measurements = self.exact_measurements(measurement_context, candidate_index);
        if !exact_measurements.is_empty() {
            return exact_measurements;
        }
        self.execution_profile_measurements(measurement_context, candidate_index)
    }

    /// Returns measurements from the exact measurement context for the given candidate.
    fn exact_measurements(
        &self,
        measurement_context: PromptProcessingMeasurementContext,
        candidate_index: usize,
    ) -> Vec<CandidateChunkMeasurement> {
        if let Some(exact_statistics) = self.context_statistics.get(&measurement_context)
            && !exact_statistics.candidate_statistics[candidate_index]
                .measurements
                .is_empty()
        {
            return exact_statistics.candidate_statistics[candidate_index]
                .measurements
                .iter()
                .copied()
                .collect();
        }
        Vec::new()
    }

    /// Returns measurements from all contexts sharing the same execution profile,
    /// sorted by measurement sequence and trimmed to the retained window.
    fn execution_profile_measurements(
        &self,
        measurement_context: PromptProcessingMeasurementContext,
        candidate_index: usize,
    ) -> Vec<CandidateChunkMeasurement> {
        let mut profile_measurements: Vec<CandidateChunkMeasurement> = self
            .context_statistics
            .iter()
            .filter(|(stored_context, _)| {
                stored_context.position_independent_execution_profile_identifier()
                    == measurement_context.position_independent_execution_profile_identifier()
            })
            .flat_map(|(_, statistics)| {
                statistics.candidate_statistics[candidate_index]
                    .measurements
                    .iter()
                    .copied()
            })
            .collect();
        profile_measurements.sort_unstable_by_key(|measurement| measurement.measurement_sequence);
        let retained_start = profile_measurements
            .len()
            .saturating_sub(self.maximum_retained_measurements_per_candidate_and_context);
        profile_measurements.drain(0..retained_start);
        profile_measurements
    }

    fn maximum_measured_elapsed_millis_per_token(
        &self,
        measurement_context: PromptProcessingMeasurementContext,
    ) -> u128 {
        (0..self.candidate_chunk_size_tokens.len())
            .flat_map(|candidate_index| {
                self.measurements_for_context_or_execution_profile(
                    measurement_context,
                    candidate_index,
                )
            })
            .map(|measurement| {
                u128::from(measurement.forward_elapsed_millis)
                    .div_ceil(measurement.processed_prompt_token_count as u128)
            })
            .max()
            .unwrap_or(1)
    }
}

/// Internal representation of one completed-chunk measurement stored in the optimizer table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CandidateChunkMeasurement {
    pub(crate) processed_prompt_token_count: usize,
    pub(crate) forward_elapsed_millis: u64,
    pub(crate) next_measurement_context: PromptProcessingMeasurementContext,
    pub(crate) measurement_sequence: u64,
}
