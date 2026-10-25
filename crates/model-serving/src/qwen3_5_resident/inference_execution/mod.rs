/// Prefill-to-decode no-I/O reconciliation: lift pressure and preserve topology.
mod decode_lifecycle;
mod engine_construction;
mod engine_error_helpers;
pub(crate) mod engine_request;
mod engine_state;
mod generation_session_lifecycle;
mod inference_execution_bridge;
mod memory_governance;
mod model_loading;
mod model_loading_finalization;
mod persistent_prompt_cache;
mod prompt_prefill_lifecycle;
pub(crate) mod speculative_decode_lifecycle;
mod test_controls;
mod visual_embeddings_runtime;
// Downward bindings so the phase folders keep their pre-move one-level
// `super::<module>` paths into the persistent prompt-cache machinery.
use persistent_prompt_cache::decoder_state_reuse;
use persistent_prompt_cache::persistent_prompt_cache_capture;
use persistent_prompt_cache::persistent_prompt_cache_startup_logging;
use persistent_prompt_cache::persistent_prompt_cache_visual_identity;
use persistent_prompt_cache::retained_live_prefix;
// Downward binding so the phase folders keep their one-level
// `super::completed_forward_memory` paths into the memory-governance home.
use memory_governance::completed_forward_memory;
// The typed admission surface keeps its resident-engine-wide reach.
pub(in crate::qwen3_5_resident) use memory_governance::memory_admission;
// Re-exports at the original visibilities so children and crate-level paths
// keep resolving after the state and helpers moved into their own files.
use crate::MlxInferenceEngine;
pub(crate) use engine_error_helpers::fatal_engine_error;
pub(crate) use engine_error_helpers::qwen3_5_runtime_error;
pub(crate) use engine_state::Qwen3_5EngineState;
pub use engine_state::Qwen3_5InferenceExecution;

pub use memory_governance::memory_limit::safe_minimum_mlx_memory_ceiling_bytes;
pub use persistent_prompt_cache::persistent_prompt_cache_capture::persistent_prompt_cache_publication_advances_parent_chain;
pub use prompt_prefill_lifecycle::prefill_execution_context::Qwen3_5PrefillExecutionContext;
pub use prompt_prefill_lifecycle::prompt_processing_chunk_sizer::{
    Qwen3_5ResidentPromptProcessingChunkSizer, Qwen3_5ResidentPromptProcessingChunkSizerError,
};

/// Qwen3.5 inference engine backed by the architecture-neutral MLX owner driver.
pub type Qwen3_5ResidentEngine = MlxInferenceEngine<Qwen3_5InferenceExecution>;
