//! One completed selection-to-measurement transition recorded by the optimizer.

/// Measured work and the context reached after that work completed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromptProcessingChunkMeasurement {
    processed_prompt_token_count: usize,
    forward_elapsed_millis: u64,
    next_measurement_context: super::PromptProcessingMeasurementContext,
}

impl PromptProcessingChunkMeasurement {
    /// Records one completed candidate-selection transition.
    #[must_use]
    pub const fn transition(
        processed_prompt_token_count: usize,
        forward_elapsed_millis: u64,
        next_measurement_context: super::PromptProcessingMeasurementContext,
    ) -> Self {
        Self {
            processed_prompt_token_count,
            forward_elapsed_millis,
            next_measurement_context,
        }
    }

    /// Returns the number of prompt tokens that actually completed.
    #[must_use]
    pub const fn processed_prompt_token_count(self) -> usize {
        self.processed_prompt_token_count
    }

    /// Returns model-forward time, excluding allocator cleanup and telemetry.
    #[must_use]
    pub const fn forward_elapsed_millis(self) -> u64 {
        self.forward_elapsed_millis
    }

    /// Returns the context reached after the measured token advancement.
    #[must_use]
    pub const fn next_measurement_context(self) -> super::PromptProcessingMeasurementContext {
        self.next_measurement_context
    }
}
