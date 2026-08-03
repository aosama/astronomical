/// One measured prompt pre-processing chunk outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillChunckSizeOptimizerObservation {
    actual_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
    is_full_candidate_prefill_chunck: bool,
}

impl PrefillChunckSizeOptimizerObservation {
    /// Records a full chunk that used the complete candidate prefill_chunck_tokens count.
    #[must_use]
    pub const fn full_prefill_chunck(
        actual_prefill_chunck_tokens: usize,
        elapsed_millis: u64,
    ) -> Self {
        Self {
            actual_prefill_chunck_tokens,
            elapsed_millis,
            is_full_candidate_prefill_chunck: true,
        }
    }

    /// Records a final prompt tail that did not use the complete candidate size.
    #[must_use]
    pub const fn partial_prefill_chunck(
        actual_prefill_chunck_tokens: usize,
        elapsed_millis: u64,
    ) -> Self {
        Self {
            actual_prefill_chunck_tokens,
            elapsed_millis,
            is_full_candidate_prefill_chunck: false,
        }
    }

    #[must_use]
    pub const fn actual_prefill_chunck_tokens(self) -> usize {
        self.actual_prefill_chunck_tokens
    }

    #[must_use]
    pub const fn elapsed_millis(self) -> u64 {
        self.elapsed_millis
    }

    #[must_use]
    pub const fn is_full_candidate_prefill_chunck(self) -> bool {
        self.is_full_candidate_prefill_chunck
    }
}
