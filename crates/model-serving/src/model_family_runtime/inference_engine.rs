use astronomical_ipc_protocol::{RequestId, WorkerEvent};

use crate::inference_engine::load_result::EngineLoadResult;
use crate::{
    GeneratedToken, GenerationFinalization, InferenceEngine, InferenceEngineError,
    PrismHadamardQwen35Engine, Qwen3_5ResidentEngine, Qwen3_5StreamingEngine,
};

use super::ModelFamilyInferenceRequest;

/// Family-tagged inference engine used by the generic worker.
pub enum ModelFamilyInferenceEngine {
    Qwen3_5(Qwen3_5ResidentEngine),
    Qwen3_5Streaming(Qwen3_5StreamingEngine),
    PrismHadamardQwen35(PrismHadamardQwen35Engine),
}

impl InferenceEngine for ModelFamilyInferenceEngine {
    type Request = ModelFamilyInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        match self {
            Self::Qwen3_5(engine) => engine.load().await,
            Self::Qwen3_5Streaming(engine) => engine.load().await,
            Self::PrismHadamardQwen35(engine) => engine.load().await,
        }
    }

    async fn start_generation(
        &mut self,
        inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        match (self, inference_request) {
            (Self::Qwen3_5(engine), ModelFamilyInferenceRequest::Qwen3_5(inference_request)) => {
                engine.start_generation(inference_request).await
            }
            (
                Self::Qwen3_5Streaming(engine),
                ModelFamilyInferenceRequest::Qwen3_5(inference_request),
            ) => engine.start_generation(inference_request).await,
            (
                Self::PrismHadamardQwen35(engine),
                ModelFamilyInferenceRequest::PrismHadamardQwen35(inference_request),
            ) => engine.start_generation(inference_request).await,
            _ => Err(family_mismatch_error()),
        }
    }

    async fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        match self {
            Self::Qwen3_5(engine) => engine.decode_next_token(request_id).await,
            Self::Qwen3_5Streaming(engine) => engine.decode_next_token(request_id).await,
            Self::PrismHadamardQwen35(engine) => engine.decode_next_token(request_id).await,
        }
    }

    async fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        match self {
            Self::Qwen3_5(engine) => {
                engine
                    .inject_input_tokens(request_id, input_token_ids)
                    .await
            }
            Self::Qwen3_5Streaming(engine) => {
                engine
                    .inject_input_tokens(request_id, input_token_ids)
                    .await
            }
            Self::PrismHadamardQwen35(engine) => {
                engine
                    .inject_input_tokens(request_id, input_token_ids)
                    .await
            }
        }
    }

    async fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        match self {
            Self::Qwen3_5(engine) => engine.cancel_generation(request_id).await,
            Self::Qwen3_5Streaming(engine) => engine.cancel_generation(request_id).await,
            Self::PrismHadamardQwen35(engine) => engine.cancel_generation(request_id).await,
        }
    }

    async fn collect_persistent_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        match self {
            Self::Qwen3_5(engine) => engine.collect_persistent_prompt_cache_stats().await,
            Self::Qwen3_5Streaming(engine) => engine.collect_persistent_prompt_cache_stats().await,
            Self::PrismHadamardQwen35(engine) => {
                engine.collect_persistent_prompt_cache_stats().await
            }
        }
    }

    async fn clear_persistent_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        match self {
            Self::Qwen3_5(engine) => engine.clear_persistent_prompt_cache(model_id).await,
            Self::Qwen3_5Streaming(engine) => engine.clear_persistent_prompt_cache(model_id).await,
            Self::PrismHadamardQwen35(engine) => {
                engine.clear_persistent_prompt_cache(model_id).await
            }
        }
    }

    async fn collect_mlx_memory_telemetry(
        &self,
    ) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
        match self {
            Self::Qwen3_5(engine) => engine.collect_mlx_memory_telemetry().await,
            Self::Qwen3_5Streaming(engine) => engine.collect_mlx_memory_telemetry().await,
            Self::PrismHadamardQwen35(engine) => engine.collect_mlx_memory_telemetry().await,
        }
    }

    async fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        match self {
            Self::Qwen3_5(engine) => {
                engine
                    .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
                    .await
            }
            Self::Qwen3_5Streaming(engine) => {
                engine
                    .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
                    .await
            }
            Self::PrismHadamardQwen35(engine) => {
                engine
                    .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
                    .await
            }
        }
    }
}

fn family_mismatch_error() -> InferenceEngineError {
    InferenceEngineError::InvalidRequest {
        reason: "model-family engine and inference request do not match".to_owned(),
    }
}

use crate::{EngineGenerationStart, MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
