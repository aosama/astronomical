//! Decoding a reply in drafted rounds, shared by both engines.
//!
//! When a DFlash drafter of either generation that pairs with the target is
//! installed beside it (`engine_core::speculative_decoding::drafter_discovery`),
//! an engine hosts it (`engine_draft_runtime`) and admits the sessions whose
//! semantics drafting preserves (`speculation_session_admission`). An
//! admitted session decodes in rounds (`draft_tree_round`): the drafter
//! proposes a tree of continuations, the target verifies all of it in one
//! pass and keeps the chain its own picks follow.
//!
//! Everything here is written against `DraftRoundTarget`, so a speed-up
//! one engine has is one the other has: the engines differ only in how a
//! verified tree's pass is evaluated and how the target is made ready.

pub mod draft_model_status_report;
pub mod draft_round_target;
pub mod draft_tree_round;
pub mod draft_tree_size;
mod drafter_target_rows;
pub mod engine_draft_runtime;
pub mod speculation_session_admission;

#[cfg(test)]
mod draft_tree_round_tests;
#[cfg(test)]
mod draft_tree_size_tests;
#[cfg(test)]
mod speculation_session_admission_tests;
