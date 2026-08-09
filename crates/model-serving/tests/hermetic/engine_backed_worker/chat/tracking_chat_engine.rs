use super::*;

pub(super) struct TrackingChatEngine {
    is_active: bool,
}

impl TrackingChatEngine {
    pub(super) fn new() -> Self {
        Self { is_active: false }
    }
}

impl InferenceEngine for TrackingChatEngine {
    type Request = ScriptedInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Ok(EngineLoadResult::new())
    }

    async fn start_generation(
        &mut self,
        _generation_request: ScriptedInferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        if self.is_active {
            return Err(InferenceEngineError::EngineBusy);
        }
        self.is_active = true;
        Ok(EngineGenerationStart::new(0))
    }

    async fn decode_next_token(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        Ok(GeneratedToken::TokenId {
            token_id: 1,
            is_reasoning_token: false,
            expert_memory_mode: None,
            mlx_memory_telemetry: None,
            generation_finalization: None,
        })
    }

    async fn inject_input_tokens(
        &mut self,
        _request_id: RequestId,
        _input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }

    async fn cancel_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        self.is_active = false;
        Ok(GenerationFinalization::default())
    }
}
