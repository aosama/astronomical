//! Online candidate selection over bounded, context-specific measurements.
//!
//! Selection explores missing evidence before exploiting cumulative-latency
//! estimates. Once a material execution profile is measured, it remains
//! converged until its admissible candidate set changes.

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
pub(crate) const MINIMUM_EXPLORATION_MEASUREMENTS_PER_CANDIDATE: usize = 2;

/// Why the optimizer selected a particular candidate chunk size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptProcessingChunkSizeSelectionReason {
    /// Sampling the largest feasible candidate that has no measurements yet.
    ExploreUnmeasuredCandidate,
    /// Minimizing predicted remaining prompt-processing latency.
    MinimizeProjectedRemainingPromptLatency,
    /// Remaining prompt tokens are fewer than the smallest registered candidate.
    RemainingTokensBelowSmallestCandidate,
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
        let execution_profile = measurement_context.execution_profile();
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

        let minimum_candidate_measurement_count = (0..eligible_candidate_count)
            .map(|candidate_index| self.measurement_count(execution_profile, candidate_index))
            .min()
            .unwrap_or(0);
        if minimum_candidate_measurement_count < MINIMUM_EXPLORATION_MEASUREMENTS_PER_CANDIDATE {
            let candidate_index = if minimum_candidate_measurement_count.is_multiple_of(2) {
                (0..eligible_candidate_count).rev().find(|candidate_index| {
                    self.measurement_count(execution_profile, *candidate_index)
                        == minimum_candidate_measurement_count
                })
            } else {
                (0..eligible_candidate_count).find(|candidate_index| {
                    self.measurement_count(execution_profile, *candidate_index)
                        == minimum_candidate_measurement_count
                })
            }
            .unwrap_or(0);
            return self.selection(
                candidate_index,
                PromptProcessingChunkSizeSelectionReason::ExploreUnmeasuredCandidate,
            );
        }

        let unknown_elapsed_millis_per_token = self
            .maximum_measured_elapsed_millis_per_token(execution_profile)
            .max(1);
        let selected_candidate_index = lowest_cumulative_latency_candidate_index(
            &self.candidate_chunk_size_tokens,
            maximum_candidate_chunk_size_tokens,
            execution_profile,
            unknown_elapsed_millis_per_token,
            &|future_measurement_context, candidate_index| {
                self.measurements(future_measurement_context, candidate_index)
            },
        );
        self.selection(
            selected_candidate_index,
            PromptProcessingChunkSizeSelectionReason::MinimizeProjectedRemainingPromptLatency,
        )
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
        if chunk_measurement.processed_prompt_token_count() != selected_candidate_chunk_size_tokens
        {
            return Err(
                PromptProcessingChunkSizeOptimizerError::MeasurementMustCompleteSelectedCandidateCapacity {
                    selected_candidate_chunk_size_tokens,
                    processed_prompt_token_count: chunk_measurement
                        .processed_prompt_token_count(),
                },
            );
        }
        self.measurement_sequence = self.measurement_sequence.saturating_add(1);
        let candidate_measurement = CandidateChunkMeasurement {
            processed_prompt_token_count: chunk_measurement.processed_prompt_token_count(),
            forward_elapsed_millis: chunk_measurement.forward_elapsed_millis(),
            next_measurement_context: chunk_measurement
                .next_measurement_context()
                .execution_profile(),
            measurement_sequence: self.measurement_sequence,
        };
        self.context_statistics
            .entry(measurement_context.execution_profile())
            .or_insert_with(|| {
                ContextCandidateStatistics::new(self.candidate_chunk_size_tokens.len())
            })
            .candidate_statistics[candidate_index]
            .record_measurement(
                candidate_measurement,
                self.maximum_retained_measurements_per_candidate_and_context,
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

    /// Summarizes measurements used for selections in one material execution profile.
    #[must_use]
    pub fn candidate_measurement_summaries(
        &self,
        measurement_context: PromptProcessingMeasurementContext,
    ) -> CandidateMeasurementSummaries {
        build_candidate_measurement_summaries(
            &self.candidate_chunk_size_tokens,
            &self.context_statistics,
            measurement_context.execution_profile(),
        )
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
        measurement_sequence: u64,
        context_statistics: BTreeMap<
            PromptProcessingMeasurementContext,
            ContextCandidateStatistics,
        >,
    ) -> Self {
        Self {
            candidate_chunk_size_tokens,
            maximum_retained_measurements_per_candidate_and_context,
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

    fn measurement_count(
        &self,
        execution_profile: PromptProcessingMeasurementContext,
        candidate_index: usize,
    ) -> usize {
        self.context_statistics
            .get(&execution_profile)
            .map_or(0, |statistics| {
                statistics.candidate_statistics[candidate_index]
                    .measurements
                    .len()
            })
    }

    fn measurements(
        &self,
        execution_profile: PromptProcessingMeasurementContext,
        candidate_index: usize,
    ) -> Vec<CandidateChunkMeasurement> {
        if let Some(profile_statistics) = self.context_statistics.get(&execution_profile) {
            return profile_statistics.candidate_statistics[candidate_index]
                .measurements
                .iter()
                .copied()
                .collect();
        }
        Vec::new()
    }

    fn maximum_measured_elapsed_millis_per_token(
        &self,
        measurement_context: PromptProcessingMeasurementContext,
    ) -> u128 {
        (0..self.candidate_chunk_size_tokens.len())
            .flat_map(|candidate_index| self.measurements(measurement_context, candidate_index))
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
