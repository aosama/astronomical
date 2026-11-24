//! The chat engine the worker holds, whichever chat family it serves.

#[cfg(feature = "direct-mlx")]
mod chat_engine;

#[cfg(feature = "direct-mlx")]
pub use chat_engine::{ModelFamilyInferenceEngine, boxed_chat_engine};
