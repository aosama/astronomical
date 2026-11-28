//! How the engine builds itself on the MLX owner thread. The
//! shared constructors live in
//! `text_generation::generation_loop::engine_construction`.

use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;

use crate::qwen3_5_core::shared_execution::execution_core::Qwen3_5ExecutionCore;
use text_generation::generation_loop::engine_construction::TextEngineConstruction;

use super::super::model::Qwen3_5Model;
use super::Qwen3_5InferenceExecution;
use super::engine_request::Qwen3_5RequestExtension;

/// What only the engine is started with.
#[derive(Debug, Default)]
pub struct TextEngineSettings {
    /// The decode policy of the `astro_expert_cache`; only controlled
    /// benchmarks select anything but the baseline.
    pub decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    /// The `astro_neural_engine_program_cache` folder of this model revision.
    pub astro_neural_engine_program_cache_directory: Option<std::path::PathBuf>,
    /// The prompt chunk the model's startup settled on, and why; reported
    /// once the load settles.
    pub prompt_processing_chunk_report: Option<ipc_protocol::PromptProcessingChunkReport>,
}

impl TextEngineConstruction for Qwen3_5InferenceExecution {
    type EngineSettings = TextEngineSettings;

    fn new_on_owner_thread(
        core: Qwen3_5ExecutionCore<Qwen3_5Model, Qwen3_5RequestExtension>,
        engine_settings: TextEngineSettings,
    ) -> Self {
        Self {
            core,
            decode_astro_expert_cache_policy: engine_settings.decode_astro_expert_cache_policy,
            neural_engine_prefill: crate::qwen3_5_core::neural_engine::neural_engine_prefill_lifecycle::NeuralEnginePrefillLifecycle::new(
                engine_settings.astro_neural_engine_program_cache_directory,
            ),
            prompt_span_to_fail_once_for_tests: None,
            prompt_processing_chunk_report: engine_settings.prompt_processing_chunk_report,
        }
    }
}
