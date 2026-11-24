//! Speculative decoding lifecycle for the resident Qwen3.5 engine.
//!
//! When a DFlash drafter of either generation that pairs with the target is installed beside it
//! (`engine_core::speculative_decoding::drafter_discovery`), the engine hosts it (`engine_draft_runtime`) and
//! admits the sessions whose semantics drafting preserves
//! (`speculation_session_admission`). An admitted session decodes in rounds
//! (`draft_tree_round`): the drafter proposes a tree of continuations, the
//! target verifies all of it in one pass and keeps the chain its own picks
//! follow. `speculative_decode_advance` drains each round's tokens through
//! the ordinary emission machinery, one per engine advance.

pub mod draft_model_status_report;
pub mod draft_tree_round;
pub mod draft_tree_size;
pub mod engine_draft_runtime;
pub mod speculation_session_admission;
pub mod speculative_decode_advance;

pub use engine_draft_runtime::Qwen3_5SpeculativeDraftRuntime;
pub use speculation_session_admission::VerificationTokenChoice;

#[cfg(test)]
mod draft_tree_round_tests;
#[cfg(test)]
mod draft_tree_size_tests;
#[cfg(test)]
mod speculation_session_admission_tests;
