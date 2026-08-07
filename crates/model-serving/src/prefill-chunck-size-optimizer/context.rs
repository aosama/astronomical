/// Stable context bucket used by the prompt pre-processing chunk-size optimizer.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PrefillChunckSizeOptimizerContext {
    context_identifier: u64,
    fallback_context_identifier: u64,
}

impl PrefillChunckSizeOptimizerContext {
    /// Creates a context bucket identifier chosen by the caller's domain adapter.
    #[must_use]
    pub const fn new(context_identifier: u64) -> Self {
        Self {
            context_identifier,
            fallback_context_identifier: context_identifier,
        }
    }

    /// Creates an exact context and its position-independent fallback family.
    #[must_use]
    pub const fn new_with_fallback(
        context_identifier: u64,
        fallback_context_identifier: u64,
    ) -> Self {
        Self {
            context_identifier,
            fallback_context_identifier,
        }
    }

    /// Returns the opaque context bucket identifier.
    #[must_use]
    pub const fn context_identifier(self) -> u64 {
        self.context_identifier
    }

    /// Returns the context family used only when the exact bucket has no evidence.
    #[must_use]
    pub const fn fallback_context_identifier(self) -> u64 {
        self.fallback_context_identifier
    }
}
