//! The one failure token sampling reports.
//!
//! Sampling lives below the engine, so it cannot name the engine's error
//! type; the serving crate turns this into its fatal engine error at the
//! boundary, keeping the reason text the client sees unchanged.

use thiserror::Error;

/// A sampling graph could not be built or its input was malformed.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("{reason}")]
pub struct TokenSamplingError {
    /// Context-rich reason, carried unchanged into the engine's fatal error.
    pub reason: String,
}
