//! How the streaming engine builds itself on the MLX owner thread. The
//! constructors it shares with the resident engine live in
//! `qwen3_5_core::shared_execution::engine_construction`.

use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;

use crate::qwen3_5_core::shared_execution::engine_construction::Qwen3_5EngineConstruction;
use crate::qwen3_5_core::shared_execution::execution_core::Qwen3_5ExecutionCore;

use super::super::model::Qwen3_5Model;
use super::Qwen3_5InferenceExecution;
use super::engine_request::StreamingRequestExtension;

/// What only the streaming engine is started with.
#[derive(Debug, Default)]
pub struct StreamingEngineSettings {
    /// The decode policy of the `astro_expert_cache`; only controlled
    /// benchmarks select anything but the baseline.
    pub decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
}

impl Qwen3_5EngineConstruction for Qwen3_5InferenceExecution {
    type EngineSettings = StreamingEngineSettings;

    fn new_on_owner_thread(
        core: Qwen3_5ExecutionCore<Qwen3_5Model, StreamingRequestExtension>,
        engine_settings: StreamingEngineSettings,
    ) -> Self {
        Self {
            core,
            decode_astro_expert_cache_policy: engine_settings.decode_astro_expert_cache_policy,
            prompt_span_to_fail_once_for_tests: None,
            startup_residency_facts: None,
        }
    }
}
