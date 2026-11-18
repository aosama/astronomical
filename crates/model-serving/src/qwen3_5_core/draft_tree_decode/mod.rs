//! Speculative decoding with DFlash2 draft trees, for every Qwen3.5 engine.
//!
//! When a DFlash2 drafter that pairs with the target is installed beside it
//! (`drafter_discovery`), an engine hosts it (`engine_draft_runtime`) and
//! admits the sessions whose semantics drafting preserves
//! (`speculation_session_admission`). An admitted session decodes in rounds
//! (`draft_tree_round`): the drafter proposes a tree of continuations, the
//! target verifies all of it in one pass and keeps the chain its own picks
//! follow.
//!
//! These pieces sit in the shared core, not under one engine, because a
//! model that streams from storage must decode as fast as its Mac allows
//! too: each engine supplies only how a layer's feed-forward finds its
//! weights (`DraftTreeTarget`) and how a round's tokens reach its caller.

pub(crate) mod draft_request_state;
pub mod draft_tree_round;
pub mod draft_tree_size;
pub mod drafter_discovery;
pub mod engine_draft_runtime;
pub mod speculation_session_admission;

#[cfg(test)]
mod draft_tree_round_tests;
#[cfg(test)]
mod draft_tree_size_tests;
#[cfg(test)]
mod drafter_discovery_tests;
#[cfg(test)]
mod speculation_session_admission_tests;
