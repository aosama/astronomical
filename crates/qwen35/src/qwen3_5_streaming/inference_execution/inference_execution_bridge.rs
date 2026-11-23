//! Trait bridge from the engine-neutral execution contract onto the state methods.

use ipc_protocol::{RequestId, WorkerEvent};

use crate::inference_engine::load_result::EngineLoadResult;
use crate::{
    EngineGenerationStart, GeneratedToken, GenerationFinalization, InferenceEngineError,
    MlxInferenceExecution, MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
    PerformanceAttributionOutcome, Qwen3_5InferenceRequest,
    build_astro_ssd_prompt_cache_stats_event,
};

use super::engine_state::{Qwen3_5EngineState, Qwen3_5InferenceExecution};
use super::fatal_engine_error;

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

    fn collect_astro_ssd_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        Ok(Qwen3_5EngineState::collect_astro_ssd_prompt_cache_stats(
            self,
        ))
    }

    fn clear_astro_ssd_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        let Some(astro_ssd_prompt_cache) = self.astro_ssd_prompt_cache.as_ref() else {
            return Ok(None);
        };
        let clear_outcome = astro_ssd_prompt_cache
            .clear_astro_ssd_prompt_cache(model_id.as_deref())
            .map_err(|clear_error| fatal_engine_error(clear_error.to_string()))?;
        Ok(Some(WorkerEvent::AstroSsdPromptCacheCleared {
            model_id: clear_outcome.model_id,
            blocks_removed: clear_outcome.blocks_removed,
            bytes_freed: clear_outcome.bytes_freed,
        }))
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

impl Qwen3_5EngineState {
    fn collect_astro_ssd_prompt_cache_stats(&self) -> Option<WorkerEvent> {
        let astro_ssd_prompt_cache = self.astro_ssd_prompt_cache.as_ref()?;
        let global_astro_ssd_prompt_cache_maximum_size_bytes = self
            .astro_ssd_prompt_cache_disk_store_config
            .as_ref()
            .map(|disk_store_config| {
                disk_store_config.global_astro_ssd_prompt_cache_maximum_size_bytes()
            })?;
        Some(build_astro_ssd_prompt_cache_stats_event(
            &self.astro_ssd_prompt_cache_counters,
            u64::try_from(
                astro_ssd_prompt_cache
                    .model_contract_ref()
                    .block_token_count(),
            )
            .unwrap_or(u64::MAX),
            u64::try_from(astro_ssd_prompt_cache.sequence_state_block_count()).unwrap_or(u64::MAX),
            u64::try_from(astro_ssd_prompt_cache.boundary_state_snapshot_count())
                .unwrap_or(u64::MAX),
            u64::try_from(astro_ssd_prompt_cache.visual_embedding_count()).unwrap_or(u64::MAX),
            astro_ssd_prompt_cache.total_size_bytes(),
            astro_ssd_prompt_cache.visual_embedding_total_size_bytes(),
            global_astro_ssd_prompt_cache_maximum_size_bytes,
        ))
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        let Some(active_request) = self.active_request.take() else {
            return Ok(GenerationFinalization::default());
        };
        if active_request.request_id != request_id {
            self.active_request = Some(active_request);
            Ok(GenerationFinalization::default())
        } else {
            Ok(self.finalize_generation_request(
                active_request,
                PerformanceAttributionOutcome::Cancelled,
                None,
            ))
        }
    }
}
