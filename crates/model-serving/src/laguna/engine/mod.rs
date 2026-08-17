//! Laguna-owned inference engine used by the architecture-neutral worker.

mod active_generation;
mod attribution;
pub(in crate::laguna) mod execution;
mod injected_tokens;
mod live_memory_limit;
mod loading;
mod memory;
mod memory_admission;
pub(in crate::laguna) mod prefill;
mod prefill_capacity_recovery;
pub(in crate::laguna) mod prompt_cache;

use crate::MlxInferenceEngine;

pub use execution::LagunaInferenceExecution;

/// Worker-facing Laguna engine. MLX work stays on the owner thread.
pub type LagunaEngine = MlxInferenceEngine<LagunaInferenceExecution>;
