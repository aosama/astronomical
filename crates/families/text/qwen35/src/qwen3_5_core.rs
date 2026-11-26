//! The shared Qwen3.5 family kernel (issue #1132).
//!
//! This module owns everything the Qwen3.5 models share: configuration,
//! artifact validation and shard plumbing, OptiQ quantization profiles, the
//! text stack (tokenizer, prompt, output parser, sampler), the vision stack,
//! dense MLP math, decoder state and `astro_decoder_state_cache` layouts, and
//! the persistent `astro_ssd_prompt_cache` machinery. The streaming model
//! serves every text model; the in-memory model in `qwen3_5_resident` exists
//! only for the quantization sensitivity probe.
//!
//! Dependency direction: `qwen3_5_resident` and `qwen3_5_streaming` may import
//! from here; neither may import from the other. This module never imports
//! from the model modules.

/// The frozen Ornith 1.0 sparse-artifact model identity, shared by every
/// Qwen3.5 engine and the beside-source test fixtures.
pub const ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
pub const ORNITH_1_0_35B_OPTIQ_4BIT_REVISION: &str = "ce62c23d34b91d84f838e0b292d517dbe4b9b60f";

pub mod artifacts;
pub mod configuration;
pub mod decoder;
pub mod dense;
#[cfg(feature = "direct-mlx")]
pub mod draft_tree_decode;
#[cfg(test)]
pub mod frozen_config_test_support;
pub mod model;
pub mod model_math;
#[cfg(feature = "direct-mlx")]
pub mod neural_engine;
pub mod quantizations;
#[cfg(test)]
pub mod qwen4_exp_config_test_support;
#[cfg(feature = "direct-mlx")]
pub mod shared_execution;
pub use model_primitives::route_observation;
pub mod text;
pub mod vision;
