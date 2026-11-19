//! ModernBERT native inference: a shared trunk and the embedding engine over it.
//!
//! The trunk (`trunk`) turns a batch of padded token sequences into final
//! hidden states and knows nothing about what they are used for. The
//! embedding engine loads the mlx-community ModernBERT 8-bit affine embedding
//! checkpoint and pools one sequence's hidden states into a unit-norm vector
//! (`embedding_pooling`). Pooling and normalization live in Astronomical; no
//! third-party embedding package is linked.

mod artifact;
pub mod attention_mask;
pub mod configuration;
mod embedding_pooling;
mod engine;
pub mod token_batch;
mod tokenizer;
pub mod trunk;
pub mod weight_layout;

pub use engine::ModernBertEmbeddingEngine;
pub use tokenizer::{EncodedEmbeddingInput, encode_embedding_input};
