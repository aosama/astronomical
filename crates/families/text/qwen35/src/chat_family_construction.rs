//! How the worker builds a chat family's runtime, from the family's own type
//! (issue #2473). A chat line in the family list names a type implementing
//! [`ChatFamilyConstruction`], so the worker holds one chat builder and no
//! family's startup: Qwen 3.5's engine selection, dense placement and startup
//! failures live here with the family, as an image family's folder check
//! lives with that family.
//!
//! Every chat family today prepares the Qwen 3.5 request and serves through
//! the boxed chat engine, so the runtime parts are typed on them.

mod chat_engine;
#[cfg(feature = "direct-mlx")]
mod qwen3_5_dense_engine_placement;
#[cfg(all(test, feature = "direct-mlx"))]
mod qwen3_5_dense_engine_placement_tests;
#[cfg(feature = "direct-mlx")]
mod qwen3_5_engine_selection;
#[cfg(feature = "direct-mlx")]
mod qwen3_5_model_startup;
#[cfg(feature = "direct-mlx")]
mod qwen3_5_model_startup_error;
#[cfg(all(test, feature = "direct-mlx"))]
mod qwen3_5_model_startup_error_tests;
#[cfg(feature = "direct-mlx")]
mod qwen3_5_prompt_chunk_settlement;
#[cfg(feature = "direct-mlx")]
mod qwen3_5_startup_storage;

pub use chat_engine::{ModelFamilyInferenceEngine, boxed_chat_engine};
#[cfg(feature = "direct-mlx")]
pub use qwen3_5_model_startup_error::Qwen3_5ModelStartupError;

#[cfg(feature = "direct-mlx")]
mod family_types;

#[cfg(feature = "direct-mlx")]
pub use family_types::*;
