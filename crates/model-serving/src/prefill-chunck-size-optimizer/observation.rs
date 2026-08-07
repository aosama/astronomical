/// One measured prompt pre-processing chunk outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillChunckSizeOptimizerObservation {
    actual_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
    next_prompt_processing_context: super::PrefillChunckSizeOptimizerContext,
}

impl PrefillChunckSizeOptimizerObservation {
    /// Records one completed requested-action transition.
    #[must_use]
    pub const fn transition(
        actual_prefill_chunck_tokens: usize,
        elapsed_millis: u64,
        next_prompt_processing_context: super::PrefillChunckSizeOptimizerContext,
    ) -> Self {
        Self {
            actual_prefill_chunck_tokens,
            elapsed_millis,
            next_prompt_processing_context,
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
    pub const fn next_prompt_processing_context(self) -> super::PrefillChunckSizeOptimizerContext {
        self.next_prompt_processing_context
    }
}
