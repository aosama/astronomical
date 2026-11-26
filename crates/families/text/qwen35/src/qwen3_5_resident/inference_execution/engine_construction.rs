//! How the resident engine builds itself on the MLX owner thread. The
//! constructors it shares with the streaming engine live in
//! `text_generation::generation_loop::engine_construction`.

use std::path::PathBuf;

use crate::qwen3_5_core::shared_execution::execution_core::Qwen3_5ExecutionCore;
use text_generation::generation_loop::engine_construction::TextEngineConstruction;

use super::super::model::Qwen3_5ResidentModel;
use super::Qwen3_5InferenceExecution;
use super::engine_request::ResidentRequestExtension;

/// What only the resident engine is started with.
#[derive(Debug, Default)]
pub struct ResidentEngineSettings {
    /// Where this model revision's compiled Neural Engine programs are kept
    /// between loads; `None` compiles them again on every load.
    pub astro_neural_engine_program_cache_directory: Option<PathBuf>,
}

impl TextEngineConstruction for Qwen3_5InferenceExecution {
    type EngineSettings = ResidentEngineSettings;

    fn new_on_owner_thread(
        core: Qwen3_5ExecutionCore<Qwen3_5ResidentModel, ResidentRequestExtension>,
        engine_settings: ResidentEngineSettings,
    ) -> Self {
        Self {
            core,
            neural_engine_prefill: crate::qwen3_5_core::neural_engine::neural_engine_prefill_lifecycle::NeuralEnginePrefillLifecycle::new(
                engine_settings.astro_neural_engine_program_cache_directory,
            ),
            retained_live_prefix: None,
            speculative_draft_runtime: None,
            draft_model_status: ipc_protocol::DraftModelStatus::None,
        }
    }
}
