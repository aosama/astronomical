//! The typed engine-placement lifecycle for one loaded model worker.
//!
//! Placement is decided once at load by startup engine selection and then
//! only transitions through typed events: a request admission may demote the
//! complete-residency engine to the SSD-streaming engine, and a bounded
//! re-promotion gate may return a demoted worker to the resident engine once
//! the pressure that caused the demotion has resolved. Free-form strings
//! cannot express these states, so every transition is a typed variant an
//! exhaustive match must acknowledge.

use crate::admission::complete_residency::CompleteResidencyDecision;
use crate::forward_admission::StreamingEngineRequirement;
use crate::reclamation::ExpertMemoryAdmissionError;

/// Why the loaded engine holds its current placement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnginePlacement {
    /// Startup selection admitted complete expert residency under the
    /// startup ceiling; the resident engine owns every expert.
    CompleteResidencyFits,
    /// Startup selection could not admit complete expert residency; the
    /// streaming engine owns the experts and pages them from SSD.
    CompleteResidencyDoesNotFit {
        /// Bytes short of complete residency plus its required headroom at
        /// the startup ceiling.
        shortfall_bytes: u64,
    },
    /// A request's admission demoted an already-loaded resident engine to
    /// the streaming engine.
    DemotedForRequest {
        /// The typed admission cause that forced the demotion.
        cause: StreamingEngineRequirement,
        /// Total context token count (prompt plus maximum output) of the
        /// request whose admission demoted the engine. The re-promotion
        /// gate compares every later request against this count.
        demoted_total_context_token_count: usize,
    },
    /// A demoted worker swapped back to the resident engine after the
    /// memory pressure that caused the demotion resolved.
    PromotedAfterPressureResolved,
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

    /// Returns whether this placement runs the complete-residency resident
    /// engine rather than the SSD-streaming engine.
    #[must_use]
    pub const fn runs_resident_engine(self) -> bool {
        matches!(
            self,
            Self::CompleteResidencyFits | Self::PromotedAfterPressureResolved
        )
    }

    /// Returns whether a request demoted this worker to the streaming engine.
    #[must_use]
    pub const fn is_demoted_for_request(self) -> bool {
        matches!(self, Self::DemotedForRequest { .. })
    }

    /// Returns the typed cause carried by a request demotion, if any.
    #[must_use]
    pub const fn demotion_cause(self) -> Option<StreamingEngineRequirement> {
        match self {
            Self::DemotedForRequest { cause, .. } => Some(cause),
            _ => None,
        }
    }

    /// Returns the total context token count recorded at the request
    /// demotion, if any.
    #[must_use]
    pub const fn demoted_total_context_token_count(self) -> Option<usize> {
        match self {
            Self::DemotedForRequest {
                demoted_total_context_token_count,
                ..
            } => Some(demoted_total_context_token_count),
            _ => None,
        }
    }

    /// Returns the established placement label for worker lifecycle logs.
    #[must_use]
    pub const fn log_label(self) -> &'static str {
        match self {
            Self::CompleteResidencyFits => "complete_residency_fits",
            Self::CompleteResidencyDoesNotFit { .. } => "complete_residency_does_not_fit",
            Self::DemotedForRequest { .. } => "demoted_for_request",
            Self::PromotedAfterPressureResolved => "promoted_after_pressure_resolved",
        }
    }
}
