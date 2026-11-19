//! Which serving phase one measurement belongs to.
//!
//! The variants are constructed only by the direct-MLX streaming engine and
//! its tests; the enum stays ungated because the always-compiled attribution
//! signatures name it. It lives beside performance attribution, which records
//! it, so the attribution never depends on the engine that sets it.
//!
//! Prefill processes the prompt in multi-token chunks; decode generates one
//! token at a time (multi-token decode routes exist behind speculative
//! decoding, so a route's token count is not phase evidence — issue #1715).
//! The phase is engine state, passed into the accounting seams; it must never
//! be re-derived from route shape anywhere downstream.

/// The serving phase a storage stream, read-time window, or route served.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionStreamingPhase {
    Prefill,
    Decode,
}
