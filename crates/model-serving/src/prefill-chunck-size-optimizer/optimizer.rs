use std::collections::BTreeMap;
use std::path::PathBuf;

use super::context_statistics::ContextCandidateStatistics;
use super::episode_latency_planner::lowest_cumulative_latency_candidate_index;
use super::{
    PrefillChunckOptimizerCandidateEvidence, PrefillChunckOptimizerContextEvidence,
    PrefillChunckSizeOptimizerContext, PrefillChunckSizeOptimizerError,
    PrefillChunckSizeOptimizerObservation, persistence,
};

const MINIMUM_SLIDING_WINDOW_OBSERVATION_COUNT: usize = 1;
const STALE_OBSERVATION_WINDOW_MULTIPLIER: u64 = 5;

/// Why the optimizer chose a particular candidate chunk size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefillChunckSizeOptimizerDecisionReason {
    /// Sampling the largest feasible requested action without evidence.
    InitialExploration,
    /// Refreshing the least recently observed feasible action.
    StaleObservationProbe,
    /// Minimizing predicted complete prompt-processing latency.
    CumulativeLatencyPlanning,
    /// The prompt tail is shorter than the smallest registered candidate.
    Fallback,
}

/// The candidate the optimizer chose and why.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillChunckSizeOptimizerDecision {
    pub candidate_prefill_chunck_tokens: usize,
    pub reason: PrefillChunckSizeOptimizerDecisionReason,
}

/// Tabular online optimizer for cumulative prompt-processing latency.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillChunckSizeOptimizer {
    candidate_prefill_chunck_tokens: Vec<usize>,
    sliding_window_observation_count: usize,
    decision_sequence: u64,
    observation_sequence: u64,
    context_statistics: BTreeMap<PrefillChunckSizeOptimizerContext, ContextCandidateStatistics>,
}

impl PrefillChunckSizeOptimizer {
    pub fn new(
        mut candidate_prefill_chunck_tokens: Vec<usize>,
        sliding_window_observation_count: usize,
    ) -> Result<Self, PrefillChunckSizeOptimizerError> {
        if candidate_prefill_chunck_tokens.is_empty() {
            return Err(PrefillChunckSizeOptimizerError::NoCandidatePrefillChunckTokens);
        }
        if candidate_prefill_chunck_tokens.contains(&0) {
            return Err(
                PrefillChunckSizeOptimizerError::CandidatePrefillChunckTokensMustBePositive,
            );
        }
        candidate_prefill_chunck_tokens.sort_unstable();
        candidate_prefill_chunck_tokens.dedup();
        Ok(Self {
            candidate_prefill_chunck_tokens,
            sliding_window_observation_count: sliding_window_observation_count
                .max(MINIMUM_SLIDING_WINDOW_OBSERVATION_COUNT),
            decision_sequence: 0,
            observation_sequence: 0,
            context_statistics: BTreeMap::new(),
        })
    }

    #[must_use]
    pub fn ask(
        &mut self,
        prompt_processing_context: PrefillChunckSizeOptimizerContext,
    ) -> PrefillChunckSizeOptimizerDecision {
        let maximum_prefill_chunck_tokens = self
            .candidate_prefill_chunck_tokens
            .last()
            .copied()
            .unwrap_or(1);
        self.ask_with_maximum_prefill_chunck_tokens(
            prompt_processing_context,
            maximum_prefill_chunck_tokens,
        )
    }

    #[must_use]
    pub fn ask_with_maximum_prefill_chunck_tokens(
        &mut self,
        prompt_processing_context: PrefillChunckSizeOptimizerContext,
        maximum_prefill_chunck_tokens: usize,
    ) -> PrefillChunckSizeOptimizerDecision {
        self.decision_sequence = self.decision_sequence.saturating_add(1);
        let eligible_candidate_count = self.candidate_prefill_chunck_tokens.partition_point(
            |candidate_prefill_chunck_tokens| {
                *candidate_prefill_chunck_tokens <= maximum_prefill_chunck_tokens
            },
        );
        if eligible_candidate_count == 0 {
            return PrefillChunckSizeOptimizerDecision {
                candidate_prefill_chunck_tokens: self.candidate_prefill_chunck_tokens[0],
                reason: PrefillChunckSizeOptimizerDecisionReason::Fallback,
            };
        }

        if let Some(candidate_index) = (0..eligible_candidate_count).rev().find(|candidate_index| {
            !self.has_observations_in_context_family(prompt_processing_context, *candidate_index)
        }) {
            return self.decision(
                candidate_index,
                PrefillChunckSizeOptimizerDecisionReason::InitialExploration,
            );
        }

        let stale_after_decisions =
            STALE_OBSERVATION_WINDOW_MULTIPLIER.saturating_mul(eligible_candidate_count as u64);
        let stale_candidate_index = (0..eligible_candidate_count)
            .filter_map(|candidate_index| {
                self.last_observed_decision_sequence_in_context_family(
                    prompt_processing_context,
                    candidate_index,
                )
                .map(|last_observed_decision_sequence| {
                    (candidate_index, last_observed_decision_sequence)
                })
            })
            .filter(|(_, last_observed_decision_sequence)| {
                self.decision_sequence
                    .saturating_sub(*last_observed_decision_sequence)
                    >= stale_after_decisions
            })
            .min_by_key(|(candidate_index, last_observed_decision_sequence)| {
                (*last_observed_decision_sequence, *candidate_index)
            })
            .map(|(candidate_index, _)| candidate_index);
        if let Some(candidate_index) = stale_candidate_index {
            return self.decision(
                candidate_index,
                PrefillChunckSizeOptimizerDecisionReason::StaleObservationProbe,
            );
        }

        let unknown_elapsed_millis_per_token = self
            .maximum_observed_elapsed_millis_per_token(prompt_processing_context)
            .max(1);
        let selected_candidate_index = lowest_cumulative_latency_candidate_index(
            &self.candidate_prefill_chunck_tokens,
            maximum_prefill_chunck_tokens,
            prompt_processing_context,
            unknown_elapsed_millis_per_token,
            &|future_prompt_processing_context, candidate_index| {
                self.observations_for_context_or_family(
                    future_prompt_processing_context,
                    candidate_index,
                )
            },
        );
        self.decision(
            selected_candidate_index,
            PrefillChunckSizeOptimizerDecisionReason::CumulativeLatencyPlanning,
        )
    }

    pub fn tell(
        &mut self,
        prompt_processing_context: PrefillChunckSizeOptimizerContext,
        candidate_prefill_chunck_tokens: usize,
        prefill_chunck_observation: PrefillChunckSizeOptimizerObservation,
    ) -> Result<(), PrefillChunckSizeOptimizerError> {
        let candidate_index = self.candidate_index(candidate_prefill_chunck_tokens)?;
        if prefill_chunck_observation.actual_prefill_chunck_tokens() == 0 {
            return Err(
                PrefillChunckSizeOptimizerError::ObservationPrefillChunckTokensMustBePositive,
            );
        }
        if prefill_chunck_observation.elapsed_millis() == 0 {
            return Err(PrefillChunckSizeOptimizerError::ObservationElapsedMillisMustBePositive);
        }
        self.observation_sequence = self.observation_sequence.saturating_add(1);
        let candidate_observation = CandidatePrefillChunckObservation {
            actual_prefill_chunck_tokens: prefill_chunck_observation.actual_prefill_chunck_tokens(),
            elapsed_millis: prefill_chunck_observation.elapsed_millis(),
            next_prompt_processing_context: prefill_chunck_observation
                .next_prompt_processing_context(),
            observation_sequence: self.observation_sequence,
        };
        self.context_statistics
            .entry(prompt_processing_context)
            .or_insert_with(|| {
                ContextCandidateStatistics::new(self.candidate_prefill_chunck_tokens.len())
            })
            .candidate_statistics[candidate_index]
            .record_observation(
                candidate_observation,
                self.sliding_window_observation_count,
                self.decision_sequence,
            );
        Ok(())
    }

    #[must_use]
    pub fn candidate_prefill_chunck_tokens(&self) -> &[usize] {
        &self.candidate_prefill_chunck_tokens
    }

    #[must_use]
    pub fn sliding_window_observation_count(&self) -> usize {
        self.sliding_window_observation_count
    }

    /// Summarizes the exact-or-family evidence used for decisions in one context.
    #[must_use]
    pub fn context_evidence(
        &self,
        prompt_processing_context: PrefillChunckSizeOptimizerContext,
    ) -> PrefillChunckOptimizerContextEvidence {
        let candidate_evidence = self
            .candidate_prefill_chunck_tokens
            .iter()
            .enumerate()
            .map(|(candidate_index, candidate_prefill_chunck_tokens)| {
                let candidate_observations = self
                    .observations_for_context_or_family(prompt_processing_context, candidate_index);
                let observation_count = candidate_observations.len();
                let cumulative_actual_prefill_chunck_tokens = candidate_observations.iter().fold(
                    0_u128,
                    |cumulative_tokens, candidate_observation| {
                        cumulative_tokens.saturating_add(
                            candidate_observation.actual_prefill_chunck_tokens as u128,
                        )
                    },
                );
                let cumulative_elapsed_millis = candidate_observations.iter().fold(
                    0_u128,
                    |cumulative_millis, candidate_observation| {
                        cumulative_millis
                            .saturating_add(u128::from(candidate_observation.elapsed_millis))
                    },
                );
                let average_denominator = (observation_count as u128).max(1);
                PrefillChunckOptimizerCandidateEvidence {
                    candidate_prefill_chunck_tokens: *candidate_prefill_chunck_tokens,
                    observation_count,
                    average_actual_prefill_chunck_tokens: usize::try_from(
                        cumulative_actual_prefill_chunck_tokens / average_denominator,
                    )
                    .unwrap_or(usize::MAX),
                    average_elapsed_millis: u64::try_from(
                        cumulative_elapsed_millis / average_denominator,
                    )
                    .unwrap_or(u64::MAX),
                    decisions_since_last_observation: self
                        .last_observed_decision_sequence_in_context_family(
                            prompt_processing_context,
                            candidate_index,
                        )
                        .map(|last_observed_decision_sequence| {
                            self.decision_sequence
                                .saturating_sub(last_observed_decision_sequence)
                        }),
                }
            })
            .collect::<Vec<_>>();
        PrefillChunckOptimizerContextEvidence {
            has_observations_for_every_candidate: candidate_evidence
                .iter()
                .all(|candidate_evidence| candidate_evidence.observation_count > 0),
            candidate_evidence,
        }
    }

    pub(crate) fn decision_sequence(&self) -> u64 {
        self.decision_sequence
    }

    pub(crate) fn observation_sequence(&self) -> u64 {
        self.observation_sequence
    }

    pub(crate) fn context_statistics(
        &self,
    ) -> &BTreeMap<PrefillChunckSizeOptimizerContext, ContextCandidateStatistics> {
        &self.context_statistics
    }

    pub(crate) fn new_from_persisted_state(
        candidate_prefill_chunck_tokens: Vec<usize>,
        sliding_window_observation_count: usize,
        decision_sequence: u64,
        observation_sequence: u64,
        context_statistics: BTreeMap<PrefillChunckSizeOptimizerContext, ContextCandidateStatistics>,
    ) -> Self {
        Self {
            candidate_prefill_chunck_tokens,
            sliding_window_observation_count,
            decision_sequence,
            observation_sequence,
            context_statistics,
        }
    }

    pub fn save_to_directory(
        &self,
        optimizer_directory: &std::path::Path,
        model_id: &str,
        model_revision: &str,
    ) -> Result<(), PrefillChunckSizeOptimizerError> {
        persistence::save_optimizer_to_directory(
            self,
            optimizer_directory,
            model_id,
            model_revision,
        )
    }

    pub fn load_from_path(
        state_file_path: PathBuf,
        model_id: String,
        model_revision: String,
        candidate_prefill_chunck_tokens: Vec<usize>,
        sliding_window_observation_count: usize,
    ) -> Result<Option<Self>, PrefillChunckSizeOptimizerError> {
        persistence::load_optimizer_from_path(
            &state_file_path,
            &model_id,
            &model_revision,
            candidate_prefill_chunck_tokens,
            sliding_window_observation_count,
        )
    }

    fn decision(
        &self,
        candidate_index: usize,
        reason: PrefillChunckSizeOptimizerDecisionReason,
    ) -> PrefillChunckSizeOptimizerDecision {
        PrefillChunckSizeOptimizerDecision {
            candidate_prefill_chunck_tokens: self.candidate_prefill_chunck_tokens[candidate_index],
            reason,
        }
    }

    fn candidate_index(
        &self,
        candidate_prefill_chunck_tokens: usize,
    ) -> Result<usize, PrefillChunckSizeOptimizerError> {
        self.candidate_prefill_chunck_tokens
            .binary_search(&candidate_prefill_chunck_tokens)
            .map_err(|_| {
                PrefillChunckSizeOptimizerError::UnregisteredCandidatePrefillChunckTokens {
                    candidate_prefill_chunck_tokens,
                }
            })
    }

    fn has_observations_in_context_family(
        &self,
        prompt_processing_context: PrefillChunckSizeOptimizerContext,
        candidate_index: usize,
    ) -> bool {
        self.context_statistics
            .iter()
            .any(|(stored_context, statistics)| {
                stored_context.fallback_context_identifier()
                    == prompt_processing_context.fallback_context_identifier()
                    && !statistics.candidate_statistics[candidate_index]
                        .observations
                        .is_empty()
            })
    }

    fn last_observed_decision_sequence_in_context_family(
        &self,
        prompt_processing_context: PrefillChunckSizeOptimizerContext,
        candidate_index: usize,
    ) -> Option<u64> {
        self.context_statistics
            .iter()
            .filter(|(stored_context, _)| {
                stored_context.fallback_context_identifier()
                    == prompt_processing_context.fallback_context_identifier()
            })
            .filter_map(|(_, statistics)| {
                statistics.candidate_statistics[candidate_index].last_observed_decision_sequence
            })
            .max()
    }

    fn observations_for_context_or_family(
        &self,
        prompt_processing_context: PrefillChunckSizeOptimizerContext,
        candidate_index: usize,
    ) -> Vec<CandidatePrefillChunckObservation> {
        if let Some(exact_statistics) = self.context_statistics.get(&prompt_processing_context)
            && !exact_statistics.candidate_statistics[candidate_index]
                .observations
                .is_empty()
        {
            return exact_statistics.candidate_statistics[candidate_index]
                .observations
                .iter()
                .copied()
                .collect();
        }
        let mut family_observations: Vec<CandidatePrefillChunckObservation> = self
            .context_statistics
            .iter()
            .filter(|(stored_context, _)| {
                stored_context.fallback_context_identifier()
                    == prompt_processing_context.fallback_context_identifier()
            })
            .flat_map(|(_, statistics)| {
                statistics.candidate_statistics[candidate_index]
                    .observations
                    .iter()
                    .copied()
            })
            .collect();
        family_observations.sort_unstable_by_key(|observation| observation.observation_sequence);
        let retained_start = family_observations
            .len()
            .saturating_sub(self.sliding_window_observation_count);
        family_observations.drain(0..retained_start);
        family_observations
    }

    fn maximum_observed_elapsed_millis_per_token(
        &self,
        prompt_processing_context: PrefillChunckSizeOptimizerContext,
    ) -> u128 {
        (0..self.candidate_prefill_chunck_tokens.len())
            .flat_map(|candidate_index| {
                self.observations_for_context_or_family(prompt_processing_context, candidate_index)
            })
            .map(|observation| {
                u128::from(observation.elapsed_millis)
                    .div_ceil(observation.actual_prefill_chunck_tokens as u128)
            })
            .max()
            .unwrap_or(1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CandidatePrefillChunckObservation {
    pub(crate) actual_prefill_chunck_tokens: usize,
    pub(crate) elapsed_millis: u64,
    pub(crate) next_prompt_processing_context: PrefillChunckSizeOptimizerContext,
    pub(crate) observation_sequence: u64,
}
