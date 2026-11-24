//! The Laya decision family, a crate of its own beside the decision shell
//! (issue #2473, Step 7). Laya answers typed questions about a text from the
//! ModernBERT trunk the shell holds.
//!
//! The crates below this family are imported at the root, as in the
//! decision shell, so the family's `crate::...` paths into them read as they
//! did when it lived inside the shell.
#![cfg(feature = "direct-mlx")]

mod adapter_targets;
mod artifact;
mod configuration;
mod engine;
mod head;
mod loaded_state;
mod served_batches;
mod tokenizer;
mod whole_model_scorer;

pub use engine::{LayaDecisionEngine, laya_question_sequences};

use ::decision_models::*;
use ::model_primitives::*;
