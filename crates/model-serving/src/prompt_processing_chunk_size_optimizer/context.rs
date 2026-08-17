//! Opaque context identities supplied by model-specific chunk-sizing adapters.

/// Stable identifiers supplied by a model adapter for one execution profile.
///
/// The exact identifier remains available while adapters migrate position data
/// to observation telemetry. Optimizer learning always uses the material,
/// position-independent execution profile.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PromptProcessingMeasurementContext {
    exact_measurement_context_identifier: u64,
    position_independent_execution_profile_identifier: u64,
}

impl PromptProcessingMeasurementContext {
    /// Creates a context whose measurements cannot be reused by another position.
    #[must_use]
    pub const fn isolated(exact_measurement_context_identifier: u64) -> Self {
        Self {
            exact_measurement_context_identifier,
            position_independent_execution_profile_identifier: exact_measurement_context_identifier,
        }
    }

    /// Creates an exact context and the same execution profile without position.
    #[must_use]
    pub const fn with_position_independent_execution_profile(
        exact_measurement_context_identifier: u64,
        position_independent_execution_profile_identifier: u64,
    ) -> Self {
        Self {
            exact_measurement_context_identifier,
            position_independent_execution_profile_identifier,
        }
    }

    /// Returns the opaque identifier for the complete measurement context.
    #[must_use]
    pub const fn exact_measurement_context_identifier(self) -> u64 {
        self.exact_measurement_context_identifier
    }

    /// Returns the execution-profile identifier used to reuse measurements across positions.
    #[must_use]
    pub const fn position_independent_execution_profile_identifier(self) -> u64 {
        self.position_independent_execution_profile_identifier
    }

    /// Returns the canonical identity used for optimizer learning and planning.
    #[must_use]
    pub(crate) const fn execution_profile(self) -> Self {
        Self::isolated(self.position_independent_execution_profile_identifier)
    }
}
