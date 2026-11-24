//! The Kev decision family, a crate of its own beside the decision shell
//! (issue #2473, Step 7). Kev answers typed questions about a text with a
//! pointer head on the Qwen 3.5 backbone the shell holds.
//!
//! The crates below this family are imported at the root, as in the
//! decision shell, so the family's `crate::...` paths into them read as they
//! did when it lived inside the shell.
#![cfg(feature = "direct-mlx")]

mod engine;
mod pointer_head;
mod sequence;
mod settings;
#[cfg(test)]
mod test_tokenizer;
mod tokenizer;
mod whole_model_scorer;

pub use engine::{KevDecisionEngine, KevDecisionFamily, kev_question_row_layouts};
pub use sequence::KevRowLayout;

use ::decision_models::*;
use ::qwen35::*;
