//! Bounded measurement windows for each context and candidate capacity.

use std::collections::VecDeque;

use super::optimizer::CandidateChunkMeasurement;

/// Per-context candidate measurement statistics kept by the optimizer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ContextCandidateStatistics {
    pub(crate) candidate_statistics: Vec<CandidateChunkStatistics>,
}

impl ContextCandidateStatistics {
    /// Allocates one empty bounded-measurement slot per registered candidate.
    pub(crate) fn new(candidate_count: usize) -> Self {
        Self {
            candidate_statistics: vec![CandidateChunkStatistics::default(); candidate_count],
        }
    }
}

/// Sliding-window measurements for one candidate chunk capacity within one measurement context.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CandidateChunkStatistics {
    /// Recent completed measurements, oldest first.
    pub(crate) measurements: VecDeque<CandidateChunkMeasurement>,
    /// Global selection sequence used to decide when this candidate is stale.
    pub(crate) last_measured_selection_sequence: Option<u64>,
}

impl CandidateChunkStatistics {
    /// Records one completed-chunk measurement and trims to the retained window.
    pub(crate) fn record_measurement(
        &mut self,
        candidate_measurement: CandidateChunkMeasurement,
        maximum_retained_measurements: usize,
        selection_sequence: u64,
    ) {
        self.measurements.push_back(candidate_measurement);
        while self.measurements.len() > maximum_retained_measurements {
            self.measurements.pop_front();
        }
        self.last_measured_selection_sequence = Some(selection_sequence);
    }
}
