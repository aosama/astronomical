//! The chat engine contract in a form a worker can hold behind a box
//! (issue #2473, Step 4).
//!
//! `InferenceEngine` returns `impl Future`, which keeps each engine's futures
//! unboxed but cannot be a trait object. `DynInferenceEngine` is the same
//! contract with boxed futures: every `InferenceEngine` implements it, and a
//! `Box<dyn DynInferenceEngine<Request>>` is itself an `InferenceEngine`, so
//! the worker holds whichever chat family its factory built without naming
//! each family in its type. Boxing costs one allocation per call, beside an
//! owner-thread round trip and a model forward that take milliseconds.

use std::future::Future;
use std::pin::Pin;

use ipc_protocol::{NeuralEnginePrefillStatus, RequestId, WorkerEvent};

use super::contract::{
    EngineGenerationStart, GeneratedToken, GenerationFinalization, InferenceEngine,
    PreparedInferenceRequest,
};
use super::{EngineLoadResult, InferenceEngineError};
use crate::memory::telemetry::{MlxMemoryLimitAdjustment, MlxMemoryTelemetry};

/// A boxed engine future borrowing the engine for `'engine`.
pub type EngineFuture<'engine, Output> =
    Pin<Box<dyn Future<Output = Result<Output, InferenceEngineError>> + Send + 'engine>>;

/// `InferenceEngine` with boxed futures, so it can be a trait object.
pub trait DynInferenceEngine<Request>: Send {
    fn load_boxed(&mut self) -> EngineFuture<'_, EngineLoadResult>;

    fn start_generation_boxed(
        &mut self,
        inference_request: Request,
    ) -> EngineFuture<'_, EngineGenerationStart>;

    fn decode_next_token_boxed(
        &mut self,
        request_id: RequestId,
    ) -> EngineFuture<'_, GeneratedToken>;

    fn inject_input_tokens_boxed(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> EngineFuture<'_, ()>;

    fn cancel_generation_boxed(
        &mut self,
        request_id: RequestId,
    ) -> EngineFuture<'_, GenerationFinalization>;

    fn advance_neural_engine_preparation_boxed(
        &mut self,
    ) -> EngineFuture<'_, Option<NeuralEnginePrefillStatus>>;

    fn collect_astro_ssd_prompt_cache_stats_boxed(&self) -> EngineFuture<'_, Option<WorkerEvent>>;

    fn clear_astro_ssd_prompt_cache_boxed(
        &mut self,
        model_id: Option<String>,
    ) -> EngineFuture<'_, Option<WorkerEvent>>;

    fn collect_mlx_memory_telemetry_boxed(&self) -> EngineFuture<'_, Option<MlxMemoryTelemetry>>;

    fn update_mlx_memory_limit_boxed(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> EngineFuture<'_, MlxMemoryLimitAdjustment>;
}

impl<Engine> DynInferenceEngine<Engine::Request> for Engine
where
    Engine: InferenceEngine + Send + Sync,
{
    fn load_boxed(&mut self) -> EngineFuture<'_, EngineLoadResult> {
        Box::pin(self.load())
    }

    fn start_generation_boxed(
        &mut self,
        inference_request: Engine::Request,
    ) -> EngineFuture<'_, EngineGenerationStart> {
        Box::pin(self.start_generation(inference_request))
    }

    fn decode_next_token_boxed(
        &mut self,
        request_id: RequestId,
    ) -> EngineFuture<'_, GeneratedToken> {
        Box::pin(self.decode_next_token(request_id))
    }

    fn inject_input_tokens_boxed(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> EngineFuture<'_, ()> {
        Box::pin(self.inject_input_tokens(request_id, input_token_ids))
    }

    fn cancel_generation_boxed(
        &mut self,
        request_id: RequestId,
    ) -> EngineFuture<'_, GenerationFinalization> {
        Box::pin(self.cancel_generation(request_id))
    }

    fn advance_neural_engine_preparation_boxed(
        &mut self,
    ) -> EngineFuture<'_, Option<NeuralEnginePrefillStatus>> {
        Box::pin(self.advance_neural_engine_preparation())
    }

    fn collect_astro_ssd_prompt_cache_stats_boxed(&self) -> EngineFuture<'_, Option<WorkerEvent>> {
        Box::pin(self.collect_astro_ssd_prompt_cache_stats())
    }

    fn clear_astro_ssd_prompt_cache_boxed(
        &mut self,
        model_id: Option<String>,
    ) -> EngineFuture<'_, Option<WorkerEvent>> {
        Box::pin(self.clear_astro_ssd_prompt_cache(model_id))
    }

    fn collect_mlx_memory_telemetry_boxed(&self) -> EngineFuture<'_, Option<MlxMemoryTelemetry>> {
        Box::pin(self.collect_mlx_memory_telemetry())
    }

    fn update_mlx_memory_limit_boxed(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> EngineFuture<'_, MlxMemoryLimitAdjustment> {
        Box::pin(self.update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes))
    }
}

/// A boxed engine is an engine.
impl<Request> InferenceEngine for Box<dyn DynInferenceEngine<Request> + Sync>
where
    Request: PreparedInferenceRequest + Send + 'static,
{
    type Request = Request;

    fn load(
        &mut self,
    ) -> impl Future<Output = Result<EngineLoadResult, InferenceEngineError>> + Send {
        self.as_mut().load_boxed()
    }

    fn start_generation(
        &mut self,
        inference_request: Request,
    ) -> impl Future<Output = Result<EngineGenerationStart, InferenceEngineError>> + Send {
        self.as_mut().start_generation_boxed(inference_request)
    }

    fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> impl Future<Output = Result<GeneratedToken, InferenceEngineError>> + Send {
        self.as_mut().decode_next_token_boxed(request_id)
    }

    fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> impl Future<Output = Result<(), InferenceEngineError>> + Send {
        self.as_mut()
            .inject_input_tokens_boxed(request_id, input_token_ids)
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> impl Future<Output = Result<GenerationFinalization, InferenceEngineError>> + Send {
        self.as_mut().cancel_generation_boxed(request_id)
    }

    fn advance_neural_engine_preparation(
        &mut self,
    ) -> impl Future<Output = Result<Option<NeuralEnginePrefillStatus>, InferenceEngineError>> + Send
    {
        self.as_mut().advance_neural_engine_preparation_boxed()
    }

    fn collect_astro_ssd_prompt_cache_stats(
        &self,
    ) -> impl Future<Output = Result<Option<WorkerEvent>, InferenceEngineError>> + Send {
        self.as_ref().collect_astro_ssd_prompt_cache_stats_boxed()
    }

    fn clear_astro_ssd_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> impl Future<Output = Result<Option<WorkerEvent>, InferenceEngineError>> + Send {
        self.as_mut().clear_astro_ssd_prompt_cache_boxed(model_id)
    }

    fn collect_mlx_memory_telemetry(
        &self,
    ) -> impl Future<Output = Result<Option<MlxMemoryTelemetry>, InferenceEngineError>> + Send {
        self.as_ref().collect_mlx_memory_telemetry_boxed()
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> impl Future<Output = Result<MlxMemoryLimitAdjustment, InferenceEngineError>> + Send {
        self.as_mut()
            .update_mlx_memory_limit_boxed(requested_mlx_memory_ceiling_bytes)
    }
}

#[cfg(test)]
#[path = "dyn_engine_tests.rs"]
mod dyn_engine_tests;
