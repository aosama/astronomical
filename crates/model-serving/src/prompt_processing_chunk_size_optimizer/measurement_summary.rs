//! Presentation-neutral summaries of evidence available to one selection context.

use std::collections::BTreeMap;

use super::PromptProcessingMeasurementContext;
use super::context_statistics::ContextCandidateStatistics;
use super::optimizer::CandidateChunkMeasurement;
use super::optimizer::MINIMUM_EXPLORATION_MEASUREMENTS_PER_CANDIDATE;

/// Identifies whether one execution profile has measurements for a candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateMeasurementSource {
    /// Measurements belong to the material execution profile regardless of position.
    ExecutionProfile,
    /// The material execution profile has no measurements.
    NoMeasurementsAvailable,
}

/// Recent measurements for one candidate chunk capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateChunkMeasurementSummary {
    /// Registered candidate capacity represented by this summary.
    pub candidate_chunk_size_tokens: usize,
    /// Whether the execution profile has measurements for this candidate.
    pub measurement_source: CandidateMeasurementSource,
    /// Number of bounded recent measurements included in the averages.
    pub measurement_count: usize,
    /// Mean token advancement from accepted full-capacity executions.
    pub average_processed_prompt_token_count: usize,
    /// Mean model-forward duration for the retained measurements.
    pub average_forward_elapsed_millis: u64,
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
    context_statistics: &BTreeMap<PromptProcessingMeasurementContext, ContextCandidateStatistics>,
    execution_profile: PromptProcessingMeasurementContext,
) -> CandidateMeasurementSummaries {
    let candidate_measurement_summaries = candidate_chunk_size_tokens
        .iter()
        .enumerate()
        .map(|(candidate_index, candidate_chunk_size_tokens)| {
            let candidate_measurements =
                profile_measurements(context_statistics, execution_profile, candidate_index);
            let measurement_source = if candidate_measurements.is_empty() {
                CandidateMeasurementSource::NoMeasurementsAvailable
            } else {
                CandidateMeasurementSource::ExecutionProfile
            };
            summarize_candidate_measurements(
                *candidate_chunk_size_tokens,
                measurement_source,
                &candidate_measurements,
            )
        })
        .collect::<Vec<_>>();
    CandidateMeasurementSummaries {
        all_candidates_have_measurements: candidate_measurement_summaries.iter().all(
            |candidate_summary| {
                candidate_summary.measurement_count
                    >= MINIMUM_EXPLORATION_MEASUREMENTS_PER_CANDIDATE
            },
        ),
        candidate_measurement_summaries,
    }
}

fn summarize_candidate_measurements(
    candidate_chunk_size_tokens: usize,
    measurement_source: CandidateMeasurementSource,
    candidate_measurements: &[CandidateChunkMeasurement],
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
    }
}

fn profile_measurements(
    context_statistics: &BTreeMap<PromptProcessingMeasurementContext, ContextCandidateStatistics>,
    execution_profile: PromptProcessingMeasurementContext,
    candidate_index: usize,
) -> Vec<CandidateChunkMeasurement> {
    context_statistics
        .get(&execution_profile)
        .map_or_else(Vec::new, |profile_statistics| {
            profile_statistics.candidate_statistics[candidate_index]
                .measurements
                .iter()
                .copied()
                .collect()
        })
}
