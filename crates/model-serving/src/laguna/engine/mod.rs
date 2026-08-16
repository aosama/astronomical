//! Laguna-owned inference engine used by the architecture-neutral worker.

mod active_generation;
mod attribution;
pub(in crate::laguna) mod execution;
mod live_memory_limit;
mod loading;
mod memory;
pub(in crate::laguna) mod prefill;
pub(in crate::laguna) mod prompt_cache;

use crate::MlxInferenceEngine;

pub use execution::LagunaInferenceExecution;

/// Worker-facing Laguna engine. MLX work stays on the owner thread.
pub type LagunaEngine = MlxInferenceEngine<LagunaInferenceExecution>;
