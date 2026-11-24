//! The ModernBERT embedding family, a crate of its own beside the decision
//! shell (issue #2473, Step 7). It loads the mlx-community ModernBERT 8-bit
//! affine embedding checkpoint and pools one sequence's hidden states from
//! the shared ModernBERT trunk into a unit-norm vector (`embedding_pooling`).
//! Pooling and normalization live in Astronomical; no third-party embedding
//! package is linked.
//!
//! The crates below this family are imported at the root, as in the
//! decision shell, so the family's `crate::...` paths into them read as they
//! did when it lived inside the shell.
#![cfg(feature = "direct-mlx")]

mod artifact;
mod embedding_pooling;
mod engine;
mod tokenizer;

use ::decision_models::*;
use ::engine_core::*;
use ::model_primitives::*;

pub use engine::{ModernBertEmbeddingEngine, ModernBertEmbeddingFamily};
pub use tokenizer::{EncodedEmbeddingInput, encode_embedding_input};
