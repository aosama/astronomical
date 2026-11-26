//! Whether the limit a model was loaded under holds it whole.
//!
//! Decided once at load by startup selection. One engine serves either way;
//! the placement only says how it started, for the status surface and logs.

use crate::admission::complete_residency::CompleteResidencyDecision;
use crate::reclamation::ExpertMemoryAdmissionError;

/// Why the loaded engine holds its current placement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnginePlacement {
    /// The startup limit holds the whole model: every expert of a sparse
    /// model, or every text weight of a dense one.
    CompleteResidencyFits,
    /// The startup limit does not hold the whole model, so part of it is
    /// read from the SSD as it is needed.
    CompleteResidencyDoesNotFit {
        /// Bytes short of complete residency plus its required headroom at
        /// the startup ceiling.
        shortfall_bytes: u64,
    },
}

impl EnginePlacement {
    /// Maps the startup complete-residency decision onto the placement
    /// lifecycle. `RejectInvalidObservation` has no placement: startup
    /// fails the load instead of placing an engine, so the observation
    /// error is returned for the caller to surface.
    ///
    /// # Errors
    ///
    /// Returns the invalid-observation error the decision carried.
    pub const fn from_startup_complete_residency_decision(
        decision: CompleteResidencyDecision,
    ) -> Result<Self, ExpertMemoryAdmissionError> {
        match decision {
            CompleteResidencyDecision::Admit { .. } => Ok(Self::CompleteResidencyFits),
            CompleteResidencyDecision::DoesNotFit {
                shortfall_bytes, ..
            } => Ok(Self::CompleteResidencyDoesNotFit { shortfall_bytes }),
            CompleteResidencyDecision::RejectInvalidObservation { error } => Err(error),
        }
    }

    /// Whether the startup limit holds the whole model.
    #[must_use]
    pub const fn holds_whole_model(self) -> bool {
        matches!(self, Self::CompleteResidencyFits)
    }

    /// Returns the established placement label for worker lifecycle logs.
    #[must_use]
    pub const fn log_label(self) -> &'static str {
        match self {
            Self::CompleteResidencyFits => "complete_residency_fits",
            Self::CompleteResidencyDoesNotFit { .. } => "complete_residency_does_not_fit",
        }
    }
}
