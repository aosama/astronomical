//! The folded ternary family's engine-backed serving execution.
//!
//! The engine wraps the straight `PrismHadamardTextModel` in the generic
//! owner-thread facade and reuses the Qwen3.5 text request: the folded pack
//! speaks the same Qwen chat dialect, so prompt preparation, thinking
//! budgets, tool markers, and output translation all come from the shared
//! text processor.

mod astro_ssd_prompt_cache;
mod execution;
#[cfg(test)]
mod fixture_engine_test_support;
mod startup;

pub use execution::{
    PrismHadamardQwen35InferenceExecution, PrismHadamardQwen35PendingLoad,
    PrismHadamardQwen35PendingStartup,
};
pub use startup::{
    PrismHadamardQwen35ServingSettings, PrismHadamardQwen35StartupError,
    initialize_prism_hadamard_qwen35_model_with_serving_settings,
};

use crate::MlxInferenceEngine;

/// The family engine served through the generic worker.
pub type PrismHadamardQwen35Engine = MlxInferenceEngine<PrismHadamardQwen35InferenceExecution>;
