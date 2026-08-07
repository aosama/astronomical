use std::collections::VecDeque;

use super::optimizer::CandidatePrefillChunckObservation;

/// Per-context requested-action transition statistics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ContextCandidateStatistics {
    pub(crate) candidate_statistics: Vec<CandidatePrefillChunckStatistics>,
}

impl ContextCandidateStatistics {
    pub(crate) fn new(candidate_count: usize) -> Self {
        Self {
            candidate_statistics: vec![
                CandidatePrefillChunckStatistics::default();
                candidate_count
            ],
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CandidatePrefillChunckStatistics {
    pub(crate) observations: VecDeque<CandidatePrefillChunckObservation>,
    pub(crate) last_observed_decision_sequence: Option<u64>,
}

impl CandidatePrefillChunckStatistics {
    pub(crate) fn record_observation(
        &mut self,
        candidate_observation: CandidatePrefillChunckObservation,
        sliding_window_observation_count: usize,
        decision_sequence: u64,
    ) {
        self.observations.push_back(candidate_observation);
        while self.observations.len() > sliding_window_observation_count {
            self.observations.pop_front();
        }
        self.last_observed_decision_sequence = Some(decision_sequence);
    }
}
