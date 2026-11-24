//! The `MlxInferenceExecution` engine implementation for the owner-thread
//! folded-ternary execution. Every entry point runs through the shared Qwen
//! 3.5 steps: load, start, decode, feedback injection, cancellation and the
//! live memory-limit change; the folded-ternary answers to their hooks live
//! in the sibling modules. As a child module it reaches the parent's private
//! fields directly.

use ipc_protocol::RequestId;

use super::PrismHadamardQwen35InferenceExecution;
use super::active_generation::{PrismRequestStart, prism_active_generation};
use crate::EngineGenerationStart;
use crate::GeneratedToken;
use crate::GenerationFinalization;
use crate::InferenceEngineError;
use crate::MlxInferenceExecution;
use crate::MlxMemoryLimitAdjustment;
use crate::MlxMemoryTelemetry;
use crate::inference_engine::load_result::EngineLoadResult;
use crate::qwen3_5_core::engine_errors::fatal_engine_error;
use crate::qwen3_5_core::shared_execution::decode_step::advance_generation;
use crate::qwen3_5_core::shared_execution::{
    generation_finalization_steps, generation_request_validation, generation_start_steps,
    input_injection, mlx_memory_limit_change, model_loading_lifecycle,
};
use ipc_protocol::WorkerEvent;

impl MlxInferenceExecution for PrismHadamardQwen35InferenceExecution {
    type Request = crate::Qwen3_5InferenceRequest;

    fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        model_loading_lifecycle::load_model(self)
    }

    fn start_generation(
        &mut self,
        mut inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        let taken_performance_attribution = inference_request.take_performance_attribution();
        let mut performance_attribution = generation_start_steps::ensure_no_active_generation(
            self,
            &inference_request,
            taken_performance_attribution,
        )?;
        // The returned reservation is what the request's context may occupy;
        // the folded-ternary engine keeps no memory governor, so nothing is
        // admitted against it, and a restore may not go past it.
        let reserved_context_tokens =
            generation_request_validation::validate_generation_request_and_resolve_reserved_context(
                &self.execution_core.generation_request_limits(),
                &inference_request,
            )?;
        let thinking_budget_state = generation_start_steps::build_thinking_budget_state(
            self.execution_core.think_end_token_id,
            &inference_request,
        )?;
        let random_state = generation_start_steps::resolve_sampling_random_state(
            self,
            inference_request.sampling_strategy(),
        )?;
        let prompt_token_ids = inference_request.input_token_ids().to_vec();
        let starting_decoder_state = self.starting_request_decoder_state(
            inference_request.request_id(),
            &prompt_token_ids,
            reserved_context_tokens,
            &mut performance_attribution,
        )?;
        let request_start = PrismRequestStart {
            prompt_token_ids,
            request_decoder_state: starting_decoder_state.request_decoder_state,
            restore_outcome: starting_decoder_state.restore_outcome,
            random_state,
            thinking_budget_state,
            performance_attribution,
            can_use_astro_ssd_prompt_cache: self.execution_core.astro_ssd_prompt_cache.is_some(),
        };
        let active_generation = prism_active_generation(&mut inference_request, request_start);
        let restored_prompt_prefix_token_count = u32::try_from(active_generation.prefill_cursor)
            .map_err(|_| fatal_engine_error("restored prompt prefix exceeds the u32 range"))?;
        let initial_prompt_processing_phase = (active_generation.prefill_cursor
            < active_generation.input_token_ids.len())
        .then_some(ipc_protocol::WorkerPromptProcessingPhase::Target);
        let astro_ssd_prompt_cache_diagnostics =
            active_generation.astro_ssd_prompt_cache_diagnostics.clone();
        self.execution_core.active_request = Some(active_generation);
        generation_start_steps::build_generation_start_contract(
            self,
            restored_prompt_prefix_token_count,
            initial_prompt_processing_phase,
            astro_ssd_prompt_cache_diagnostics,
        )
    }

    /// Runs one step of the active request through the shared Qwen 3.5
    /// decode step: a prompt chunk while any remains, then one reply token.
    fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        advance_generation(self, request_id)
    }

    fn collect_mlx_memory_telemetry(
        &self,
    ) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
        generation_finalization_steps::collect_current_mlx_memory_telemetry(self)
    }

    fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        input_injection::inject_input_tokens(self, request_id, input_token_ids)
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        Ok(generation_finalization_steps::cancel_generation(
            self, request_id,
        ))
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        mlx_memory_limit_change::update_mlx_memory_limit(self, requested_mlx_memory_ceiling_bytes)
    }

    fn collect_astro_ssd_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        Ok(self.execution_core.astro_ssd_prompt_cache_stats_event())
    }

    fn clear_astro_ssd_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        self.execution_core
            .clear_astro_ssd_prompt_cache(model_id.as_deref())
    }
}

/// Maps any boundary failure onto the engine error surface, keeping the
/// cause's own rendering as the fatal reason.
pub(super) fn gpu_error<E: std::fmt::Display>(error: E) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: error.to_string(),
    }
}

#[cfg(test)]
#[path = "engine_trait_impl_tests.rs"]
mod engine_trait_impl_tests;
