//! Trait bridge from the engine-neutral execution contract onto the state
//! methods; the methods both Qwen 3.5 engines share are expanded from
//! `shared_execution::inference_execution_bridge`.

use crate::{MlxInferenceExecution, Qwen3_5InferenceRequest};

use super::engine_state::Qwen3_5EngineState;
use super::engine_state::Qwen3_5InferenceExecution;

impl MlxInferenceExecution for Qwen3_5InferenceExecution {
    type Request = Qwen3_5InferenceRequest;

    /// The resident engine builds Neural Engine programs beside the model.
    const PREPARES_NEURAL_ENGINE_PROGRAMS: bool = true;

    crate::qwen3_5_core::shared_execution::inference_execution_bridge::qwen3_5_shared_inference_execution_methods!();

    fn advance_neural_engine_preparation(
        &mut self,
    ) -> Option<ipc_protocol::NeuralEnginePrefillStatus> {
        Qwen3_5EngineState::advance_neural_engine_preparation(self)
    }
}
