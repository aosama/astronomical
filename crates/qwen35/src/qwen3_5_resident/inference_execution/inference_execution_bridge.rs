//! Trait bridge from the engine-neutral execution contract onto the state methods.

use ipc_protocol::{RequestId, WorkerEvent};

use crate::inference_engine::load_result::EngineLoadResult;
use crate::{
    EngineGenerationStart, GeneratedToken, GenerationFinalization, InferenceEngineError,
    MlxInferenceExecution, MlxMemoryLimitAdjustment, MlxMemoryTelemetry, Qwen3_5InferenceRequest,
};

use super::engine_state::Qwen3_5EngineState;
use super::engine_state::Qwen3_5InferenceExecution;

impl MlxInferenceExecution for Qwen3_5InferenceExecution {
    type Request = Qwen3_5InferenceRequest;

    fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Qwen3_5EngineState::load(self)
    }

    fn start_generation(
        &mut self,
        inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        Qwen3_5EngineState::start_generation(self, inference_request)
    }

    fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        Qwen3_5EngineState::advance_generation(self, request_id)
    }

    fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        Qwen3_5EngineState::inject_input_tokens(self, request_id, input_token_ids)
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        Qwen3_5EngineState::cancel_generation(self, request_id)
    }

    fn advance_neural_engine_preparation(
        &mut self,
    ) -> Option<ipc_protocol::NeuralEnginePrefillStatus> {
        Qwen3_5EngineState::advance_neural_engine_preparation(self)
    }

    fn collect_astro_ssd_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        Ok(self.core.astro_ssd_prompt_cache_stats_event())
    }

    fn clear_astro_ssd_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        self.core.clear_astro_ssd_prompt_cache(model_id.as_deref())
    }

    fn collect_mlx_memory_telemetry(
        &self,
    ) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
        Qwen3_5EngineState::collect_current_mlx_memory_telemetry(self)
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        Qwen3_5EngineState::update_mlx_memory_limit(self, requested_mlx_memory_ceiling_bytes)
    }
}
