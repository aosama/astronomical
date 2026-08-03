use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;

use super::{
    PrefillChunckSizeOptimizerContext, PrefillChunckSizeOptimizerError,
    PrefillChunckSizeOptimizerObservation, persistence,
};

pub(crate) use super::context_statistics::ContextCandidateStatistics;

const MINIMUM_TRUSTED_OBSERVATION_COUNT: usize = 1;
const MINIMUM_DRIFT_TRIGGER_FACTOR: u64 = 2;

/// Why the optimizer chose a particular candidate chunk size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefillChunckSizeOptimizerDecisionReason {
    /// Exploring an untested or under-tested candidate to gather observations.
    Exploration,
    /// Exploiting the trusted candidate with the highest median throughput.
    Exploitation,
    /// Re-exploring all candidates after detecting drift on the previous best.
    ReExplorationAfterDrift,
    /// Fallback: no trusted candidates and nothing left to explore.
    Fallback,
}

/// The candidate the optimizer chose and why.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillChunckSizeOptimizerDecision {
    /// The chosen `prefill_chunck_tokens` for the next chunk.
    pub candidate_prefill_chunck_tokens: usize,
    /// Why this candidate was chosen.
    pub reason: PrefillChunckSizeOptimizerDecisionReason,
}

/// Online discrete optimizer for prompt pre-processing `prefill_chunck_tokens`.
///
/// Optimizes the highest median throughput per context bucket using a sliding
/// window of recent full-chunk observations, exploring each candidate until it
/// is trusted, then exploiting the best candidate until drift forces
/// re-exploration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillChunckSizeOptimizer {
    candidate_prefill_chunck_tokens: Vec<usize>,
    trusted_observation_count: usize,
    sliding_window_observation_count: usize,
    drift_trigger_factor: u64,
    context_statistics: BTreeMap<PrefillChunckSizeOptimizerContext, ContextCandidateStatistics>,
}

impl PrefillChunckSizeOptimizer {
    /// Creates an optimizer over positive discrete `prefill_chunck_tokens` candidates.
    pub fn new(
        mut candidate_prefill_chunck_tokens: Vec<usize>,
        trusted_observation_count: usize,
        sliding_window_observation_count: usize,
        drift_trigger_factor: u64,
    ) -> Result<Self, PrefillChunckSizeOptimizerError> {
        if candidate_prefill_chunck_tokens.is_empty() {
            return Err(PrefillChunckSizeOptimizerError::NoCandidatePrefillChunckTokens);
        }
        if candidate_prefill_chunck_tokens.contains(&0) {
            return Err(
                PrefillChunckSizeOptimizerError::CandidatePrefillChunckTokensMustBePositive,
            );
        }
        if drift_trigger_factor < MINIMUM_DRIFT_TRIGGER_FACTOR {
            return Err(PrefillChunckSizeOptimizerError::DriftTriggerFactorMustBeAtLeastTwo);
        }
        let trusted_observation_count =
            trusted_observation_count.max(MINIMUM_TRUSTED_OBSERVATION_COUNT);
        let sliding_window_observation_count =
            sliding_window_observation_count.max(trusted_observation_count);
        candidate_prefill_chunck_tokens.sort_unstable();
        candidate_prefill_chunck_tokens.dedup();
        Ok(Self {
            candidate_prefill_chunck_tokens,
            trusted_observation_count,
            sliding_window_observation_count,
            drift_trigger_factor,
            context_statistics: BTreeMap::new(),
        })
    }

    /// Picks the next candidate for a context bucket and explains why.
    #[must_use]
    pub fn ask(
        &mut self,
        prompt_processing_context: PrefillChunckSizeOptimizerContext,
    ) -> PrefillChunckSizeOptimizerDecision {
        let candidate_count = self.candidate_prefill_chunck_tokens.len();
        let context_candidate_statistics = self
            .context_statistics
            .entry(prompt_processing_context)
            .or_insert_with(|| ContextCandidateStatistics::new(candidate_count));

        if context_candidate_statistics.is_re_exploring {
            let candidate_prefill_chunck_tokens = &self.candidate_prefill_chunck_tokens;
            let chosen_prefill_chunck_tokens = context_candidate_statistics
                .round_robin_next_candidate(candidate_prefill_chunck_tokens);
            return PrefillChunckSizeOptimizerDecision {
                candidate_prefill_chunck_tokens: chosen_prefill_chunck_tokens,
                reason: PrefillChunckSizeOptimizerDecisionReason::ReExplorationAfterDrift,
            };
        }

        if let Some(candidate_index) = context_candidate_statistics
            .next_exploration_candidate_index(self.trusted_observation_count)
        {
            return PrefillChunckSizeOptimizerDecision {
                candidate_prefill_chunck_tokens: self.candidate_prefill_chunck_tokens
                    [candidate_index],
                reason: PrefillChunckSizeOptimizerDecisionReason::Exploration,
            };
        }

        if let Some(selected_candidate_index) = best_trusted_candidate_index(
            &self.candidate_prefill_chunck_tokens,
            context_candidate_statistics,
            self.trusted_observation_count,
        ) {
            return PrefillChunckSizeOptimizerDecision {
                candidate_prefill_chunck_tokens: self.candidate_prefill_chunck_tokens
                    [selected_candidate_index],
                reason: PrefillChunckSizeOptimizerDecisionReason::Exploitation,
            };
        }

        PrefillChunckSizeOptimizerDecision {
            candidate_prefill_chunck_tokens: self.candidate_prefill_chunck_tokens[0],
            reason: PrefillChunckSizeOptimizerDecisionReason::Fallback,
        }
    }

    /// Teaches the optimizer about one measured candidate outcome.
    pub fn tell(
        &mut self,
        prompt_processing_context: PrefillChunckSizeOptimizerContext,
        candidate_prefill_chunck_tokens: usize,
        prefill_chunck_observation: PrefillChunckSizeOptimizerObservation,
    ) -> Result<(), PrefillChunckSizeOptimizerError> {
        let candidate_index = self.candidate_index(candidate_prefill_chunck_tokens)?;
        if prefill_chunck_observation.elapsed_millis() == 0 {
            return Err(PrefillChunckSizeOptimizerError::ObservationElapsedMillisMustBePositive);
        }
        if !prefill_chunck_observation.is_full_candidate_prefill_chunck() {
            return Ok(());
        }
        let context_candidate_statistics = self
            .context_statistics
            .entry(prompt_processing_context)
            .or_insert_with(|| {
                ContextCandidateStatistics::new(self.candidate_prefill_chunck_tokens.len())
            });
        let candidate_observation = CandidatePrefillChunckObservation {
            actual_prefill_chunck_tokens: prefill_chunck_observation.actual_prefill_chunck_tokens(),
            elapsed_millis: prefill_chunck_observation.elapsed_millis(),
        };
        let previous_best_candidate_index = best_trusted_candidate_index(
            &self.candidate_prefill_chunck_tokens,
            context_candidate_statistics,
            self.trusted_observation_count,
        );
        let was_re_exploring = context_candidate_statistics.is_re_exploring;
        context_candidate_statistics.candidate_statistics[candidate_index]
            .record_observation(candidate_observation, self.sliding_window_observation_count);
        if was_re_exploring {
            context_candidate_statistics.advance_re_exploration();
        } else if let Some(previous_best_candidate_index) = previous_best_candidate_index
            && context_candidate_statistics.candidate_statistics[previous_best_candidate_index]
                .latest_observation_drifted_above(self.drift_trigger_factor)
        {
            context_candidate_statistics
                .begin_re_exploration(self.candidate_prefill_chunck_tokens.len());
        }
        Ok(())
    }

    /// Returns the sorted candidate set used by this optimizer.
    #[must_use]
    pub fn candidate_prefill_chunck_tokens(&self) -> &[usize] {
        &self.candidate_prefill_chunck_tokens
    }

    /// Returns the trusted observation count threshold.
    #[must_use]
    pub fn trusted_observation_count(&self) -> usize {
        self.trusted_observation_count
    }

    /// Returns the sliding window observation count.
    #[must_use]
    pub fn sliding_window_observation_count(&self) -> usize {
        self.sliding_window_observation_count
    }

    /// Returns the drift trigger factor.
    #[must_use]
    pub fn drift_trigger_factor(&self) -> u64 {
        self.drift_trigger_factor
    }

    /// Returns the context statistics map.
    #[must_use]
    pub(crate) fn context_statistics(
        &self,
    ) -> &BTreeMap<PrefillChunckSizeOptimizerContext, ContextCandidateStatistics> {
        &self.context_statistics
    }

    /// Reconstructs an optimizer from persisted state. Used by the persistence
    /// module to restore an optimizer from disk.
    #[must_use]
    pub(crate) fn new_from_persisted_state(
        candidate_prefill_chunck_tokens: Vec<usize>,
        trusted_observation_count: usize,
        sliding_window_observation_count: usize,
        drift_trigger_factor: u64,
        context_statistics: BTreeMap<PrefillChunckSizeOptimizerContext, ContextCandidateStatistics>,
    ) -> Self {
        Self {
            candidate_prefill_chunck_tokens,
            trusted_observation_count,
            sliding_window_observation_count,
            drift_trigger_factor,
            context_statistics,
        }
    }

    /// Persists the optimizer state to the given directory. The state file
    /// is written atomically (temp file + rename) to avoid partial writes.
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

    /// Loads an optimizer from a state file. Returns `Ok(None)` if the file
    /// doesn't exist, is corrupt, or doesn't match the current model or
    /// configuration — the optimizer starts fresh in all such cases.
    pub fn load_from_path(
        state_file_path: PathBuf,
        model_id: String,
        model_revision: String,
        candidate_prefill_chunck_tokens: Vec<usize>,
        trusted_observation_count: usize,
        sliding_window_observation_count: usize,
        drift_trigger_factor: u64,
    ) -> Result<Option<Self>, PrefillChunckSizeOptimizerError> {
        persistence::load_optimizer_from_path(
            &state_file_path,
            &model_id,
            &model_revision,
            candidate_prefill_chunck_tokens,
            trusted_observation_count,
            sliding_window_observation_count,
            drift_trigger_factor,
        )
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
}

fn best_trusted_candidate_index(
    candidate_prefill_chunck_tokens: &[usize],
    context_candidate_statistics: &ContextCandidateStatistics,
    trusted_observation_count: usize,
) -> Option<usize> {
    let mut best_candidate_index: Option<usize> = None;
    for candidate_index in 0..candidate_prefill_chunck_tokens.len() {
        let candidate_statistics =
            &context_candidate_statistics.candidate_statistics[candidate_index];
        if !candidate_statistics.is_trusted(trusted_observation_count) {
            continue;
        }
        match best_candidate_index {
            None => best_candidate_index = Some(candidate_index),
            Some(current_best_candidate_index) => {
                if candidate_statistics.has_higher_median_throughput_than(
                    &context_candidate_statistics.candidate_statistics
                        [current_best_candidate_index],
                ) {
                    best_candidate_index = Some(candidate_index);
                }
            }
        }
    }
    best_candidate_index
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CandidatePrefillChunckStatistics {
    pub(crate) observations: VecDeque<CandidatePrefillChunckObservation>,
}

impl CandidatePrefillChunckStatistics {
    fn record_observation(
        &mut self,
        candidate_observation: CandidatePrefillChunckObservation,
        sliding_window_observation_count: usize,
    ) {
        self.observations.push_back(candidate_observation);
        while self.observations.len() > sliding_window_observation_count {
            self.observations.pop_front();
        }
    }

    pub(crate) fn is_trusted(&self, trusted_observation_count: usize) -> bool {
        self.observations.len() >= trusted_observation_count
    }

    fn median_throughput_tokens_per_second(&self) -> Option<u128> {
        if self.observations.is_empty() {
            return None;
        }
        let mut throughputs: Vec<u128> = self
            .observations
            .iter()
            .map(|candidate_observation| candidate_observation.throughput_tokens_per_second())
            .collect();
        throughputs.sort_unstable();
        Some(median_u128_of_sorted(&throughputs))
    }

    fn latest_elapsed_millis(&self) -> Option<u64> {
        self.observations
            .back()
            .map(|candidate_observation| candidate_observation.elapsed_millis)
    }

    fn median_elapsed_millis(&self) -> Option<u64> {
        if self.observations.is_empty() {
            return None;
        }
        let mut elapsed_millis_values: Vec<u64> = self
            .observations
            .iter()
            .map(|candidate_observation| candidate_observation.elapsed_millis)
            .collect();
        elapsed_millis_values.sort_unstable();
        let median = median_u64_of_sorted(&elapsed_millis_values);
        Some(median)
    }

    /// Detects drift by comparing the latest observation's elapsed time against
    /// the median. Uses elapsed_millis rather than throughput because observations
    /// for the same candidate always have the same token count; within a candidate,
    /// higher latency directly corresponds to lower throughput.
    fn latest_observation_drifted_above(&self, drift_trigger_factor: u64) -> bool {
        let Some(latest_elapsed_millis) = self.latest_elapsed_millis() else {
            return false;
        };
        let Some(median_elapsed_millis) = self.median_elapsed_millis() else {
            return false;
        };
        if median_elapsed_millis == 0 {
            return false;
        }
        u128::from(latest_elapsed_millis)
            > u128::from(median_elapsed_millis) * u128::from(drift_trigger_factor)
    }

    fn has_higher_median_throughput_than(
        &self,
        other_candidate_statistics: &CandidatePrefillChunckStatistics,
    ) -> bool {
        match (
            self.median_throughput_tokens_per_second(),
            other_candidate_statistics.median_throughput_tokens_per_second(),
        ) {
            (Some(self_median_throughput), Some(other_median_throughput)) => {
                self_median_throughput > other_median_throughput
            }
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CandidatePrefillChunckObservation {
    pub(crate) actual_prefill_chunck_tokens: usize,
    pub(crate) elapsed_millis: u64,
}

impl CandidatePrefillChunckObservation {
    fn throughput_tokens_per_second(self) -> u128 {
        if self.elapsed_millis == 0 {
            return 0;
        }
        u128::from(self.actual_prefill_chunck_tokens as u64) * 1_000
            / u128::from(self.elapsed_millis)
    }
}

fn median_u128_of_sorted(sorted_throughputs: &[u128]) -> u128 {
    let observation_count = sorted_throughputs.len();
    if observation_count % 2 == 1 {
        sorted_throughputs[observation_count / 2]
    } else {
        (sorted_throughputs[observation_count / 2 - 1] + sorted_throughputs[observation_count / 2])
            / 2
    }
}

fn median_u64_of_sorted(sorted_elapsed_millis_values: &[u64]) -> u64 {
    let observation_count = sorted_elapsed_millis_values.len();
    if observation_count % 2 == 1 {
        sorted_elapsed_millis_values[observation_count / 2]
    } else {
        (sorted_elapsed_millis_values[observation_count / 2 - 1] / 2)
            + (sorted_elapsed_millis_values[observation_count / 2] / 2)
            + ((sorted_elapsed_millis_values[observation_count / 2 - 1] % 2)
                + (sorted_elapsed_millis_values[observation_count / 2] % 2))
                / 2
    }
}
