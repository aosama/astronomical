//! Speculative decoding lifecycle for the resident Qwen3.5 engine.
//!
//! Drafting itself (finding and hosting a drafter, admitting sessions, one
//! draft tree round) is the shared core's `draft_tree_decode`, which the
//! streaming engine uses too. What is the resident engine's own is
//! `speculative_decode_advance`: draining each round's tokens through this
//! engine's emission machinery, one per engine advance.

pub use crate::qwen3_5_core::draft_tree_decode::{
    draft_tree_round, draft_tree_size, drafter_discovery, engine_draft_runtime,
    speculation_session_admission,
};
pub mod speculative_decode_advance;

pub use engine_draft_runtime::Qwen3_5SpeculativeDraftRuntime;
pub use speculation_session_admission::{
    SpeculationSessionRequestProperties, VerificationTokenChoice,
    should_admit_speculation_for_session,
};
