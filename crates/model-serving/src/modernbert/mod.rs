//! ModernBERT native embedding inference.
//!
//! The loaded artifact is the mlx-community ModernBERT 8-bit affine embedding
//! checkpoint: one bidirectional encoder forward pass, mean pooling over all
//! token positions, and L2 normalization. Pooling and normalization live in
//! Astronomical; no third-party embedding package is linked.

mod artifact;
mod configuration;
mod engine;
mod forward;
mod memory_utilization;
mod tokenizer;

pub use engine::ModernBertEmbeddingEngine;
