//! The Clef decision family, a crate of its own beside the decision shell
//! (issue #2473, Step 7). Clef answers typed questions about a text with a
//! joint head on the Qwen 3.5 backbone the shell holds.
//!
//! The crates below this family are imported at the root, as in the
//! decision shell, so the family's `crate::...` paths into them read as they
//! did when it lived inside the shell.
#![cfg(feature = "direct-mlx")]

mod engine;
mod head_layers;
mod head_settings;
mod head_weights;
mod joint_head;
mod prompt;
#[cfg(test)]
mod test_tokenizer;
mod tokenizer;
mod whole_model_scorer;

pub use engine::{ClefDecisionEngine, clef_sequence_token_ids};

use ::decision_models::*;
use ::qwen35::*;
