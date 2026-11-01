//! Which serving phase one measurement belongs to.
//!
//! The variants are constructed only by the direct-MLX streaming engine and
//! its tests; the enum stays ungated because the always-compiled attribution
//! signatures name it, and the allow keeps a default build's dead-variant
//! lint from forcing the phase vocabulary behind a feature gate it does not
//! own.
//!
//! Prefill processes the prompt in multi-token chunks; decode generates one
//! token at a time (multi-token decode routes exist behind speculative
//! decoding, so a route's token count is not phase evidence — issue #1715).
//! The phase is engine state, passed into the accounting seams; it must never
//! be re-derived from route shape anywhere downstream.

/// The serving phase a storage stream, read-time window, or route served.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionStreamingPhase {
    Prefill,
    Decode,
}
