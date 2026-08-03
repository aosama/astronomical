/// Stable context bucket used by the prompt pre-processing chunk-size optimizer.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PrefillChunckSizeOptimizerContext {
    context_identifier: u64,
}

impl PrefillChunckSizeOptimizerContext {
    /// Creates a context bucket identifier chosen by the caller's domain adapter.
    #[must_use]
    pub const fn new(context_identifier: u64) -> Self {
        Self { context_identifier }
    }

    /// Returns the opaque context bucket identifier.
    #[must_use]
    pub const fn context_identifier(self) -> u64 {
        self.context_identifier
    }
}
