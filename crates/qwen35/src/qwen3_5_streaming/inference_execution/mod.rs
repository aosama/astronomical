mod advance_generation;
mod astro_ssd_prompt_cache_capture_engine;
mod astro_ssd_prompt_cache_startup_logging;
mod astro_ssd_prompt_cache_visual_identity;
mod completed_forward_memory;
/// Prefill-to-decode no-I/O reconciliation: lift pressure and preserve topology.
mod decode_expert_memory_handoff;
/// How many layers a reply keeps whole, and giving up the rest.
mod decode_whole_layer_keep;
mod decoder_state_reuse;
mod engine_construction;
pub mod engine_request;
mod engine_state;
pub(in crate::qwen3_5_streaming) mod forward_memory_baseline;
pub(in crate::qwen3_5_streaming) mod generated_token_emission;
mod generation_finalization;
mod generation_request_validation;
mod inference_execution_bridge;
mod inject_input_tokens;
mod layer_at_a_time_prefill;
pub mod layer_major_span_plan;
mod memory_limit;
mod model_loading;
mod model_loading_finalization;
mod prefill_advance;
mod prefill_chunk_planning;
mod prefill_memory_recovery;
mod prompt_prefill_errors;
mod prompt_processing_chunk_sizer;
mod request_memory_release;
mod start_generation;
mod test_controls;

use crate::MlxInferenceEngine;

// Kept so the phase files keep resolving the model as `super::Qwen3_5Model`
// after the state and constructors moved into their own files.
use super::model::Qwen3_5Model;

pub use engine_state::{Qwen3_5EngineState, Qwen3_5InferenceExecution};
pub use memory_limit::safe_minimum_mlx_memory_ceiling_bytes;
pub use prompt_processing_chunk_sizer::{
    Qwen3_5StreamingPromptProcessingChunkSizer, Qwen3_5StreamingPromptProcessingChunkSizerError,
};

/// Qwen3.5 inference engine backed by the architecture-neutral MLX owner driver.
pub type Qwen3_5StreamingEngine = MlxInferenceEngine<Qwen3_5InferenceExecution>;

pub use crate::qwen3_5_core::engine_errors::{fatal_engine_error, qwen3_5_runtime_error};

/// The streaming engine keeps no decoder state between requests, so a
/// finalized measurement has no retained live prefix to charge to the
/// context owner.
pub(super) const NO_RETAINED_LIVE_PREFIX_BYTES: u64 = 0;
