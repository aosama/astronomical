//! The `MlxInferenceExecution` engine implementation for the owner-thread
//! folded-ternary execution: load, start, decode through the shared Qwen 3.5
//! decode step, correction injection, cancellation, and live memory-limit
//! adjustment. Split from the execution state vocabulary in the parent so
//! each file stays inside the source-size budget; as a child module it
//! reaches the parent's private fields directly.

use ipc_protocol::{ExpertMemoryMode, RequestId};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::PrismHadamardQwen35InferenceExecution;
use super::active_generation::{PrismRequestStart, prism_active_generation};
use crate::EngineGenerationStart;
use crate::GeneratedToken;
use crate::GenerationFinalization;
use crate::InferenceEngineError;
use crate::MlxInferenceExecution;
use crate::MlxMemoryLimitAdjustment;
use crate::MlxMemoryTelemetry;
use crate::PerformanceAttributionOutcome;
use crate::PerformanceOperation;
use crate::RequestDecoderStateStack;
use crate::inference_engine::load_result::EngineLoadResult;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::qwen3_5_core::engine_errors::fatal_engine_error;
use crate::qwen3_5_core::shared_execution::decode_step::advance_generation;
use crate::qwen3_5_core::shared_execution::generation_finalization_steps::Qwen3_5FinalizationHooks;
use crate::qwen3_5_core::shared_execution::model_loading_steps::open_astro_ssd_prompt_cache_with_contract_resolver;
use crate::qwen3_5_core::shared_execution::{
    generation_request_validation, generation_start_steps,
};
use ipc_protocol::WorkerEvent;

impl MlxInferenceExecution for PrismHadamardQwen35InferenceExecution {
    type Request = crate::Qwen3_5InferenceRequest;

    fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        let pending_load = self
            .pending_load
            .take()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "folded-ternary startup was already consumed".to_owned(),
            })?;
        let mut performance_attribution = self
            .execution_core
            .take_model_loading_performance_attribution();
        let validated_pack = self
            .execution_core
            .validated_artifact
            .take()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "folded-ternary validated pack was already consumed".to_owned(),
            })?;
        let model_id = validated_pack.model_id().to_owned();
        let model_revision = validated_pack.revision().to_owned();
        let memory_limits = MlxMemoryLimits::new(
            pending_load.effective_mlx_memory_ceiling_bytes,
            pending_load.mlx_buffer_cache_limit_bytes,
        )
        .map_err(|error| InferenceEngineError::Fatal {
            reason: format!("folded-ternary MLX limits failed: {error}"),
        })?;
        let runtime = performance_attribution
            .measure_operation(PerformanceOperation::MlxRuntimeInitialization, |_| {
                MlxRuntime::initialize(memory_limits)
            })
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!("folded-ternary MLX runtime failed: {error}"),
            })?;
        let model = performance_attribution
            .measure_operation(PerformanceOperation::ModelSafetensorsMapping, |_| {
                PrismHadamardTextModel::new(runtime, validated_pack)
            })
            .map_err(gpu_error)?;
        // The dense resident family always owns its whole weight payload; the
        // safe idle minimum is the weights plus decode working set headroom.
        let weights_payload_bytes = model.pack().total_payload_bytes();
        let minimum_ceiling_bytes =
            crate::memory::dense_resident_safe_minimum_mlx_memory_ceiling_bytes(
                weights_payload_bytes,
            );
        self.minimum_ceiling_bytes = minimum_ceiling_bytes;
        self.execution_core.model_id = Some(model_id);
        self.execution_core.model_revision = Some(model_revision);
        self.execution_core.model = Some(model);
        self.open_astro_ssd_prompt_cache_store_after_load(&mut performance_attribution)?;
        self.record_model_loading_performance_attribution(performance_attribution);
        Ok(EngineLoadResult::new()
            .with_expert_memory_mode(Some(ExpertMemoryMode::Resident))
            .with_minimum_mlx_memory_ceiling_bytes(minimum_ceiling_bytes)
            .with_neural_engine_prefill_status(
                EngineLoadResult::neural_engine_prefill_unsupported_when_enabled(
                    pending_load.neural_engine_mlp_prefill_requested,
                ),
            ))
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
        Ok(self.collect_current_mlx_memory_telemetry())
    }

    fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        super::input_injection::inject_input_tokens(self, request_id, input_token_ids)
    }

    /// Ends a request the client stopped. A request that already ended on
    /// its own was finalized by the decode step and is no longer here.
    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        let Some(active_generation) = self.execution_core.active_request.take() else {
            return Ok(GenerationFinalization::new(
                Some(ExpertMemoryMode::Resident),
                self.collect_current_mlx_memory_telemetry(),
                None,
            ));
        };
        if active_generation.request_id != request_id {
            self.execution_core.active_request = Some(active_generation);
            return Err(InferenceEngineError::Fatal {
                reason: "folded-ternary cancellation request correlation mismatch".to_owned(),
            });
        }
        Ok(self.finalize_generation_request(
            active_generation,
            PerformanceAttributionOutcome::Cancelled,
            None,
        ))
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        if let Some(refusal) = super::memory_limit::ceiling_change_refusal(
            self.execution_core.active_request.is_some(),
            requested_mlx_memory_ceiling_bytes,
            self.minimum_ceiling_bytes,
        ) {
            return Err(refusal);
        }
        let mlx_buffer_cache_bytes = super::memory_limit::mlx_buffer_cache_bytes_for_ceiling(
            self.configured_mlx_buffer_cache_bytes,
            requested_mlx_memory_ceiling_bytes,
        );
        self.execution_core.memory_limits = super::memory_limit::memory_limits_for_ceiling(
            requested_mlx_memory_ceiling_bytes,
            mlx_buffer_cache_bytes,
        )?;
        self.apply_memory_limits_to_loaded_runtime()?;
        let mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry();
        Ok(MlxMemoryLimitAdjustment::new(
            requested_mlx_memory_ceiling_bytes,
            mlx_buffer_cache_bytes,
            self.minimum_ceiling_bytes,
            ExpertMemoryMode::Resident,
            mlx_memory_telemetry,
        ))
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

impl PrismHadamardQwen35InferenceExecution {
    /// Opens the `astro_ssd_prompt_cache` store for the model `load` just
    /// seated, when the worker configured one, through the shared open step
    /// with the quota-bounded contract the folded-ternary store has always
    /// been sized with.
    fn open_astro_ssd_prompt_cache_store_after_load(
        &mut self,
        performance_attribution: &mut crate::PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        let execution_core = &self.execution_core;
        let loaded_model = execution_core.loaded_model()?;
        let (astro_ssd_prompt_cache_model_contract, astro_ssd_prompt_cache) =
            open_astro_ssd_prompt_cache_with_contract_resolver(
                execution_core
                    .astro_ssd_prompt_cache_disk_store_config
                    .clone(),
                loaded_model,
                &execution_core.chunking,
                loaded_model.pack().model_id(),
                loaded_model.pack().revision(),
                None,
                super::astro_ssd_prompt_cache_hooks::prism_quota_bounded_contract,
                performance_attribution,
            )?;
        self.execution_core.astro_ssd_prompt_cache_model_contract =
            astro_ssd_prompt_cache_model_contract;
        self.execution_core.astro_ssd_prompt_cache = astro_ssd_prompt_cache;
        Ok(())
    }

    /// Applies the current ceiling and `mlx_buffer_cache` limit to the MLX
    /// runtime once the model is loaded; before that, `load` applies them.
    fn apply_memory_limits_to_loaded_runtime(&mut self) -> Result<(), InferenceEngineError> {
        let memory_limits = self.execution_core.memory_limits;
        let Some(model) = self.execution_core.model.as_mut() else {
            return Ok(());
        };
        model
            .runtime
            .update_memory_limits(memory_limits)
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!("folded-ternary could not apply the MLX memory ceiling: {error}"),
            })
    }

    /// Samples the current MLX memory footprint for the status surface,
    /// charging the active request's decoder state while it is in the core.
    pub(super) fn collect_current_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        self.mlx_memory_telemetry_with_live_decoder_state(
            self.execution_core
                .active_request
                .as_ref()
                .map(|active_generation| &active_generation.request_decoder_state),
        )
    }

    /// The MLX memory footprint with `live_request_decoder_state` charged to
    /// the context; the shared decode step passes the request it holds.
    pub(super) fn mlx_memory_telemetry_with_live_decoder_state(
        &self,
        live_request_decoder_state: Option<&RequestDecoderStateStack>,
    ) -> Option<MlxMemoryTelemetry> {
        let model = self.execution_core.model.as_ref()?;
        let memory_snapshot = match model.runtime.memory_snapshot() {
            Ok(memory_snapshot) => memory_snapshot,
            Err(memory_snapshot_error) => {
                tracing::warn!(
                    error = %memory_snapshot_error,
                    "folded-ternary execution could not sample current MLX memory"
                );
                return None;
            }
        };
        let active_memory_bytes = u64::try_from(memory_snapshot.active_memory_bytes()).ok()?;
        let mlx_buffer_cache_bytes =
            u64::try_from(memory_snapshot.mlx_buffer_cache_bytes()).ok()?;
        let peak_memory_bytes = u64::try_from(memory_snapshot.peak_memory_bytes()).ok()?;
        let active_memory_breakdown = super::active_memory_breakdown::prism_active_memory_breakdown(
            model,
            active_memory_bytes,
            live_request_decoder_state,
        );
        Some(MlxMemoryTelemetry::new(
            active_memory_bytes,
            mlx_buffer_cache_bytes,
            peak_memory_bytes,
            active_memory_breakdown,
        ))
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
