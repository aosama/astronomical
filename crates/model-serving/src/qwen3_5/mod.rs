//! Dense Qwen3.5 text-model public boundary.
//!
//! The current hybrid MLX executor is shared with the sparse Qwen3.5
//! specialization because full attention, Gated Delta, tokenization, and the
//! request state have identical contracts. A zero-expert configuration selects
//! only dense SwiGLU weights and never constructs a router, expert pager, or
//! vision model.

pub type Qwen3_5ArtifactValidator = super::qwen3_5_moe::Qwen3_5MoEArtifactValidator;
pub type Qwen3_5Config = super::qwen3_5_moe::Qwen3_5MoEConfig;
#[cfg(feature = "direct-mlx")]
pub type Qwen3_5Engine = super::qwen3_5_moe::Qwen3_5MoEEngine;
pub type Qwen3_5GenerationProcessor = super::qwen3_5_moe::Qwen3_5MoEGenerationProcessor;
#[cfg(feature = "direct-mlx")]
pub type Qwen3_5Model = super::qwen3_5_moe::Qwen3_5MoEModel;
#[cfg(feature = "direct-mlx")]
pub type Qwen3_5Weights = super::qwen3_5_moe::Qwen3_5MoEWeights;
pub type ValidatedQwen3_5Artifact = super::qwen3_5_moe::ValidatedQwen3_5MoEArtifact;
