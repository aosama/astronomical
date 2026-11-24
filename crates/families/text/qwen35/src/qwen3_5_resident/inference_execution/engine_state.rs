//! The resident Qwen3.5 engine state owner and its state-local methods.

use crate::qwen3_5_core::shared_execution::execution_core::{
    Qwen3_5ExecutionCore, Qwen3_5ExecutionCoreOwner,
};

use super::super::model::Qwen3_5ResidentModel;
use super::astro_ssd_prompt_cache::retained_live_prefix::RetainedLiveDecoderPrefix;
use super::engine_request::ResidentRequestExtension;

/// Everything the resident engine keeps between engine steps: the state it
/// shares with the streaming engine, plus its Neural Engine programs, its
/// retained live prefix and its drafter.
pub struct Qwen3_5InferenceExecution {
    pub(crate) core: Qwen3_5ExecutionCore<Qwen3_5ResidentModel, ResidentRequestExtension>,
    /// Where this model revision's compiled Neural Engine programs are kept
    /// between loads; `None` compiles them again on every load.
    pub(in crate::qwen3_5_resident::inference_execution) astro_neural_engine_program_cache_directory:
        Option<std::path::PathBuf>,
    /// What the opt-in Neural Engine prefill is doing for the loaded model.
    pub(in crate::qwen3_5_resident::inference_execution) neural_engine_prefill_status:
        ipc_protocol::NeuralEnginePrefillStatus,
    /// The programs still on their way to the loaded model, which serves on the GPU
    /// until they arrive.
    pub(in crate::qwen3_5_resident::inference_execution) neural_engine_prefill_preparation: Option<
        crate::qwen3_5_resident::neural_engine_prefill_loading::NeuralEnginePrefillPreparation,
    >,
    /// The last completed generation's live decoder state, retained for
    /// in-memory prefix adoption (issue #1167). Dropped when the next request
    /// fails to adopt it or when the retention policy says so at completion.
    pub(in crate::qwen3_5_resident) retained_live_prefix: Option<RetainedLiveDecoderPrefix>,
    /// The loaded draft runtime; present only when loading found a paired
    /// draft artifact and the active limit had headroom for it.
    pub(in crate::qwen3_5_resident::inference_execution) speculative_draft_runtime: Option<
        super::speculative_decode_lifecycle::engine_draft_runtime::Qwen3_5SpeculativeDraftRuntime,
    >,
    /// What looking for a drafter found at load, for status.
    pub(in crate::qwen3_5_resident::inference_execution) draft_model_status:
        ipc_protocol::DraftModelStatus,
}

pub type Qwen3_5EngineState = Qwen3_5InferenceExecution;

impl Qwen3_5ExecutionCoreOwner for Qwen3_5InferenceExecution {
    type Model = Qwen3_5ResidentModel;
    type RequestExtension = ResidentRequestExtension;
    type Artifact = crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;

    fn execution_core(
        &self,
    ) -> &Qwen3_5ExecutionCore<Qwen3_5ResidentModel, ResidentRequestExtension> {
        &self.core
    }

    fn execution_core_mut(
        &mut self,
    ) -> &mut Qwen3_5ExecutionCore<Self::Model, Self::RequestExtension> {
        &mut self.core
    }
}
