use super::optimizer::CandidatePrefillChunckStatistics;

/// Per-context exploration and candidate-observation state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ContextCandidateStatistics {
    pub(crate) candidate_statistics: Vec<CandidatePrefillChunckStatistics>,
    pub(crate) is_re_exploring: bool,
    pub(crate) re_exploration_remaining: usize,
    pub(crate) exploration_cursor: usize,
}

impl ContextCandidateStatistics {
    pub(crate) fn new(candidate_count: usize) -> Self {
        Self {
            candidate_statistics: vec![
                CandidatePrefillChunckStatistics::default();
                candidate_count
            ],
            is_re_exploring: false,
            re_exploration_remaining: 0,
            exploration_cursor: 0,
        }
    }

    pub(crate) fn next_exploration_candidate_index(
        &mut self,
        trusted_observation_count: usize,
    ) -> Option<usize> {
        let candidate_count = self.candidate_statistics.len();
        for _candidate_search_offset in 0..candidate_count {
            let candidate_index = self.exploration_cursor % candidate_count;
            self.exploration_cursor = (self.exploration_cursor + 1) % candidate_count;
            if !self.candidate_statistics[candidate_index].is_trusted(trusted_observation_count) {
                return Some(candidate_index);
            }
        }
        None
    }

    pub(crate) fn begin_re_exploration(&mut self, candidate_count: usize) {
        self.is_re_exploring = true;
        self.re_exploration_remaining = candidate_count;
        self.exploration_cursor = 0;
    }

    pub(crate) fn advance_re_exploration(&mut self) {
        self.re_exploration_remaining = self.re_exploration_remaining.saturating_sub(1);
        if self.re_exploration_remaining == 0 {
            self.is_re_exploring = false;
        }
    }

    pub(crate) fn round_robin_next_candidate(
        &mut self,
        candidate_prefill_chunck_tokens: &[usize],
    ) -> usize {
        if self.re_exploration_remaining == 0 {
            self.is_re_exploring = false;
            return candidate_prefill_chunck_tokens[0];
        }
        let candidate_count = candidate_prefill_chunck_tokens.len();
        let candidate_index = self.exploration_cursor % candidate_count;
        self.exploration_cursor += 1;
        self.re_exploration_remaining -= 1;
        if self.re_exploration_remaining == 0 {
            self.is_re_exploring = false;
        }
        candidate_prefill_chunck_tokens[candidate_index]
    }
}
