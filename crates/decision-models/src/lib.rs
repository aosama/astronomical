//! The decision and embedding models of Astronomical (issue #2340): Laya,
//! Kev and Clef, which answer typed questions about a text, their
//! calibration and training, ModernBERT embeddings, and the decision and
//! embedding seams the worker drives.
//!
//! Kev and Clef score on a Qwen 3.5 backbone, so this crate sits above
//! `qwen35`. It is the one consumer of
//! `decision-training` among the model crates. The edge to the
//! serving crate is one way, which `cargo xtask check-architecture`
//! enforces. The serving crate re-exports this crate's root with a glob, so
//! every name at this root stays unique across the crates it re-exports.

pub mod decision_engine;

#[cfg(feature = "direct-mlx")]
pub mod decision_models;

pub mod embedding_engine;

#[cfg(feature = "direct-mlx")]
pub mod modernbert;

// The crates below these families, imported at the root so the families'
// `crate::...` paths into them read as they did inside the serving crate.
use engine_core::*;
// Only the direct-MLX families reach the model primitives and the Qwen
// backbone through these.
#[cfg(feature = "direct-mlx")]
use model_primitives::*;
#[cfg(feature = "direct-mlx")]
use qwen35::*;

pub use decision_engine::{
    DecisionEngine, DecisionEngineLoadResult, DecisionEngineOutput, DecisionTrainingStep,
    DecisionUnavailableEngine,
};

#[cfg(feature = "direct-mlx")]
pub use decision_models::clef::ClefDecisionEngine;

#[cfg(feature = "direct-mlx")]
pub use decision_models::kev::{KevDecisionEngine, KevRowLayout};

#[cfg(feature = "direct-mlx")]
pub use decision_models::laya::LayaDecisionEngine;

#[cfg(feature = "direct-mlx")]
pub use decision_models::recorded_training::RecordedTrainingEngine;

pub use embedding_engine::{
    EmbeddingEngine, EmbeddingEngineLoadResult, EmbeddingEngineOutput, EmbeddingUnavailableEngine,
};

#[cfg(feature = "direct-mlx")]
pub use modernbert::{EncodedEmbeddingInput, ModernBertEmbeddingEngine, encode_embedding_input};
