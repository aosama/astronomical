//! Presentation-neutral summaries of evidence available to one selection context.

use std::collections::BTreeMap;

use super::PromptProcessingMeasurementContext;
use super::context_statistics::ContextCandidateStatistics;
use super::optimizer::CandidateChunkMeasurement;

/// Identifies where the measurements for one candidate were collected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateMeasurementSource {
    /// Measurements belong to the exact position range and execution profile.
    CurrentPositionRange,
    /// Measurements belong to other positions with the same execution profile.
    OtherPositionRangesWithSameExecutionProfile,
    /// Neither the exact context nor its position-independent profile has measurements.
    NoMeasurementsAvailable,
}

/// Recent measurements for one candidate chunk capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateChunkMeasurementSummary {
    /// Registered candidate capacity represented by this summary.
    pub candidate_chunk_size_tokens: usize,
    /// Whether measurements came from this range, an equivalent range, or nowhere.
    pub measurement_source: CandidateMeasurementSource,
    /// Number of bounded recent measurements included in the averages.
    pub measurement_count: usize,
    /// Mean token advancement, which may be below candidate capacity at a prompt tail.
    pub average_processed_prompt_token_count: usize,
    /// Mean model-forward duration for the retained measurements.
    pub average_forward_elapsed_millis: u64,
    /// Number of optimizer selections since this execution profile was measured.
    pub selections_since_last_measurement: Option<u64>,
}

/// Candidate measurements available to one context-aware optimizer selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateMeasurementSummaries {
    /// True only when every registered candidate has usable evidence.
    pub all_candidates_have_measurements: bool,
    /// Candidate summaries in the optimizer's ascending capacity order.
    pub candidate_measurement_summaries: Vec<CandidateChunkMeasurementSummary>,
}

/// Builds candidate summaries without adding presentation logic to the optimizer owner.
pub(crate) fn build_candidate_measurement_summaries(
    candidate_chunk_size_tokens: &[usize],
    maximum_retained_measurements_per_candidate_and_context: usize,
    selection_sequence: u64,
    context_statistics: &BTreeMap<PromptProcessingMeasurementContext, ContextCandidateStatistics>,
    measurement_context: PromptProcessingMeasurementContext,
) -> CandidateMeasurementSummaries {
    let candidate_measurement_summaries = candidate_chunk_size_tokens
        .iter()
        .enumerate()
        .map(|(candidate_index, candidate_chunk_size_tokens)| {
            let exact_measurements =
                exact_measurements(context_statistics, measurement_context, candidate_index);
            let (measurement_source, candidate_measurements) = if !exact_measurements.is_empty() {
                (
                    CandidateMeasurementSource::CurrentPositionRange,
                    exact_measurements,
                )
            } else {
                let execution_profile_measurements = execution_profile_measurements(
                    context_statistics,
                    measurement_context,
                    candidate_index,
                    maximum_retained_measurements_per_candidate_and_context,
                );
                if execution_profile_measurements.is_empty() {
                    (
                        CandidateMeasurementSource::NoMeasurementsAvailable,
                        Vec::new(),
                    )
                } else {
                    (
                        CandidateMeasurementSource::OtherPositionRangesWithSameExecutionProfile,
                        execution_profile_measurements,
                    )
                }
            };
            summarize_candidate_measurements(
                *candidate_chunk_size_tokens,
                measurement_source,
                &candidate_measurements,
                selections_since_last_measurement(
                    context_statistics,
                    measurement_context,
                    candidate_index,
                    selection_sequence,
                ),
            )
        })
        .collect::<Vec<_>>();
    CandidateMeasurementSummaries {
        all_candidates_have_measurements: candidate_measurement_summaries
            .iter()
            .all(|candidate_summary| candidate_summary.measurement_count > 0),
        candidate_measurement_summaries,
    }
}

fn summarize_candidate_measurements(
    candidate_chunk_size_tokens: usize,
    measurement_source: CandidateMeasurementSource,
    candidate_measurements: &[CandidateChunkMeasurement],
    selections_since_last_measurement: Option<u64>,
) -> CandidateChunkMeasurementSummary {
    let measurement_count = candidate_measurements.len();
    let cumulative_processed_prompt_token_count =
        candidate_measurements
            .iter()
            .fold(0_u128, |cumulative_tokens, candidate_measurement| {
                cumulative_tokens
                    .saturating_add(candidate_measurement.processed_prompt_token_count as u128)
            });
    let cumulative_forward_elapsed_millis =
        candidate_measurements
            .iter()
            .fold(0_u128, |cumulative_millis, candidate_measurement| {
                cumulative_millis
                    .saturating_add(u128::from(candidate_measurement.forward_elapsed_millis))
            });
    let average_denominator = (measurement_count as u128).max(1);
    CandidateChunkMeasurementSummary {
        candidate_chunk_size_tokens,
        measurement_source,
        measurement_count,
        average_processed_prompt_token_count: usize::try_from(
            cumulative_processed_prompt_token_count / average_denominator,
        )
        .unwrap_or(usize::MAX),
        average_forward_elapsed_millis: u64::try_from(
            cumulative_forward_elapsed_millis / average_denominator,
        )
        .unwrap_or(u64::MAX),
        selections_since_last_measurement,
    }
}

fn exact_measurements(
    context_statistics: &BTreeMap<PromptProcessingMeasurementContext, ContextCandidateStatistics>,
    measurement_context: PromptProcessingMeasurementContext,
    candidate_index: usize,
) -> Vec<CandidateChunkMeasurement> {
    context_statistics
        .get(&measurement_context)
        .map_or_else(Vec::new, |exact_statistics| {
            exact_statistics.candidate_statistics[candidate_index]
                .measurements
                .iter()
                .copied()
                .collect()
        })
}

fn execution_profile_measurements(
    context_statistics: &BTreeMap<PromptProcessingMeasurementContext, ContextCandidateStatistics>,
    measurement_context: PromptProcessingMeasurementContext,
    candidate_index: usize,
    maximum_retained_measurements_per_candidate_and_context: usize,
) -> Vec<CandidateChunkMeasurement> {
    let mut execution_profile_measurements = context_statistics
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
        .collect::<Vec<_>>();
    execution_profile_measurements
        .sort_unstable_by_key(|measurement| measurement.measurement_sequence);
    let retained_start = execution_profile_measurements
        .len()
        .saturating_sub(maximum_retained_measurements_per_candidate_and_context);
    execution_profile_measurements.drain(0..retained_start);
    execution_profile_measurements
}

fn selections_since_last_measurement(
    context_statistics: &BTreeMap<PromptProcessingMeasurementContext, ContextCandidateStatistics>,
    measurement_context: PromptProcessingMeasurementContext,
    candidate_index: usize,
    selection_sequence: u64,
) -> Option<u64> {
    context_statistics
        .iter()
        .filter(|(stored_context, _)| {
            stored_context.position_independent_execution_profile_identifier()
                == measurement_context.position_independent_execution_profile_identifier()
        })
        .filter_map(|(_, statistics)| {
            statistics.candidate_statistics[candidate_index].last_measured_selection_sequence
        })
        .max()
        .map(|last_measured_selection_sequence| {
            selection_sequence.saturating_sub(last_measured_selection_sequence)
        })
}
