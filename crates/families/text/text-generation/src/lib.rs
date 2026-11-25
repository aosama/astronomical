//! The text shell (issue #2533): the generation loop every text family runs
//! through, and what that loop holds between steps.
//!
//! A text family brings its own model, kernels and
//! `astro_decoder_state_cache` layout, and plugs them into the loop through
//! the hook traits in [`generation_loop`]. The loop then owns the steps that
//! are the same for every family: request checks, thinking markers,
//! vocabulary boundaries, token emission and finalization; cancel, input
//! injection and live MLX ceiling changes; prompt reading and
//! `astro_ssd_prompt_cache` restore.
//!
//! It sits above the engine core, whose `InferenceEngine` contract the loop
//! serves, and below every text family. It depends on no family and never on
//! `model-serving`, which `cargo xtask check-architecture` enforces.

#[cfg(feature = "direct-mlx")]
pub mod astro_ssd_prompt_cache_capture;
pub mod decoder_state;
#[cfg(feature = "direct-mlx")]
pub mod engine_errors;
#[cfg(feature = "direct-mlx")]
pub mod engine_request;
#[cfg(any(all(test, feature = "direct-mlx"), feature = "test-support"))]
pub mod engine_request_test_support;
#[cfg(feature = "direct-mlx")]
pub mod execution_error;
#[cfg(feature = "direct-mlx")]
pub mod generation_loop;
#[cfg(feature = "direct-mlx")]
pub mod model;
pub mod request;

// The crates below the shell, imported at the root so the loop's
// `crate::...` paths into them read as they did inside the family crate.
#[allow(unused_imports)]
use artifact_validation::*;
#[allow(unused_imports)]
use engine_core::*;
#[allow(unused_imports)]
use model_primitives::*;

#[cfg(feature = "direct-mlx")]
pub use decoder_state::{
    AstroSsdPromptCacheBoundaryCheckpoint, AstroSsdPromptCacheBoundaryCheckpointCollector,
    AstroSsdPromptCacheStateBridgeError, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint, RequestDecoderStateStackCheckpoint,
};
#[cfg(feature = "direct-mlx")]
pub use execution_error::TextExecutionError;
#[cfg(any(test, feature = "direct-mlx"))]
pub use request::thinking_budget::ThinkingBudgetState;
pub use request::{ImageGrid, ProcessedImage, TextInferenceRequest, TextSamplingStrategy};
