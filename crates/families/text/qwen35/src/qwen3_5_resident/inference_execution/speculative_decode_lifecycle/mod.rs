//! Speculative decoding lifecycle for the resident Qwen3.5 engine.
//!
//! When a DFlash drafter of either generation that pairs with the target is installed beside it
//! (`engine_core::speculative_decoding::drafter_discovery`), the engine hosts it (`engine_draft_runtime`) and
//! admits the sessions whose semantics drafting preserves
//! (`speculation_session_admission`). An admitted session decodes in rounds
//! (`draft_tree_round`; these three are the shared core's `draft_tree_decode`): the drafter proposes a tree of continuations, the
//! target verifies all of it in one pass and keeps the chain its own picks
//! follow. `speculative_decode_advance` drains each round's tokens through
//! the ordinary emission machinery, one per engine advance.

pub mod draft_model_status_report;
pub mod speculative_decode_advance;

pub use crate::qwen3_5_core::draft_tree_decode::{
    draft_tree_round, engine_draft_runtime, speculation_session_admission,
};

pub use engine_draft_runtime::Qwen3_5SpeculativeDraftRuntime;
pub use speculation_session_admission::VerificationTokenChoice;
