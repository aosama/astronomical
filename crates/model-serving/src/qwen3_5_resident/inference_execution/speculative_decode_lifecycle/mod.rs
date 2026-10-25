//! Speculative decoding lifecycle for the resident Qwen3.5 engine.
//!
//! One DFlash draft model proposes token blocks in a single parallel pass;
//! the target verifies them in one multi-token forward and the accepted
//! prefix extends the context. The pure acceptance rules live in
//! `verification_decisions`; one orchestrated round lives in
//! `speculation_round`. This root only wires the children into the
//! inference-execution module.

pub mod speculation_round;
pub mod verification_decisions;

pub use speculation_round::{
    SpeculationRoundOutcome, SpeculationRoundRequest, run_speculative_round,
};
pub use verification_decisions::GreedyVerificationDecision;

#[cfg(test)]
mod verification_decisions_tests;
