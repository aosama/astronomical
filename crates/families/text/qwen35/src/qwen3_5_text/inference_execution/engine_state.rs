//! The engine's execution state, owned by the MLX owner thread.

use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;

use crate::qwen3_5_core::shared_execution::execution_core::Qwen3_5ExecutionCore;
use text_generation::generation_loop::execution_core::TextExecutionCoreOwner;

use super::super::model::Qwen3_5Model;
use super::engine_request::Qwen3_5RequestExtension;

/// Everything the engine keeps between engine steps: the loaded
/// model, the active request, the limits it was started with and its
/// `astro_ssd_prompt_cache` ownership.
pub struct Qwen3_5InferenceExecution {
    pub(crate) core: Qwen3_5ExecutionCore<Qwen3_5Model, Qwen3_5RequestExtension>,
    pub(in crate::qwen3_5_text) decode_astro_expert_cache_policy:
        WorkerDecodeAstroExpertCachePolicy,
    /// The opt-in Neural Engine prompt reading for the loaded model, which
    /// only a dense model with every layer kept in memory gets.
    pub(in crate::qwen3_5_text) neural_engine_prefill:
        crate::qwen3_5_core::neural_engine::neural_engine_prefill_lifecycle::NeuralEnginePrefillLifecycle,
    /// A span, as (layer, span), that the next layer-at-a-time prompt fails
    /// once on purpose; set only by a test.
    pub(in crate::qwen3_5_text) prompt_span_to_fail_once_for_tests: Option<(usize, usize)>,
    /// The prompt chunk startup settled on and what decided it, reported
    /// with the load result.
    pub(in crate::qwen3_5_text) prompt_processing_chunk_report:
        Option<ipc_protocol::PromptProcessingChunkReport>,
}

pub type Qwen3_5EngineState = Qwen3_5InferenceExecution;

impl TextExecutionCoreOwner for Qwen3_5InferenceExecution {
    /// The engine loads the Qwen 3.5 checkpoint as validated by the shared core.
    type Artifact = crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
    type Model = Qwen3_5Model;
    type RequestExtension = Qwen3_5RequestExtension;

    fn execution_core(&self) -> &Qwen3_5ExecutionCore<Qwen3_5Model, Qwen3_5RequestExtension> {
        &self.core
    }

    fn execution_core_mut(
        &mut self,
    ) -> &mut Qwen3_5ExecutionCore<Self::Model, Self::RequestExtension> {
        &mut self.core
    }
}
