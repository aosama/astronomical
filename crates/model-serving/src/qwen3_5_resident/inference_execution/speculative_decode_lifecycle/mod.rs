//! Speculative decoding lifecycle for the resident Qwen3.5 engine.
//!
//! One DFlash draft model proposes token blocks in a single parallel pass;
//! the target verifies them in one multi-token forward and the accepted
//! prefix extends the context. The pure acceptance rules live in
//! `verification_decisions`; one orchestrated round lives in
//! `speculation_round`. This root only wires the children into the
//! inference-execution module.

pub mod engine_draft_runtime;
pub mod speculation_round;
pub mod speculation_session_admission;
pub mod speculative_decode_advance;
pub mod verification_decisions;

pub use engine_draft_runtime::Qwen3_5SpeculativeDraftRuntime;

pub use speculation_round::{
    SpeculationRoundOutcome, SpeculationRoundRequest, run_speculative_round,
};
pub use speculation_session_admission::{
    SpeculationCompatibleSampling, SpeculationSessionRequestProperties, draft_pairs_with_target,
    should_admit_speculation_for_session,
};
pub use verification_decisions::GreedyVerificationDecision;

#[cfg(test)]
mod speculation_round_tests;
#[cfg(test)]
mod speculation_session_admission_tests;
#[cfg(test)]
mod verification_decisions_tests;
