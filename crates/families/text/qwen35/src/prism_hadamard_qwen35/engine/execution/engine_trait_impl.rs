//! The `MlxInferenceExecution` engine implementation for the owner-thread
//! folded-ternary execution: load, start, decode through the shared Qwen 3.5
//! decode step, correction injection, cancellation, and live memory-limit
//! adjustment. Split from the execution state vocabulary in the parent so
//! each file stays inside the source-size budget; as a child module it
//! reaches the parent's private fields directly.

use ipc_protocol::{ExpertMemoryMode, RequestId};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::active_generation::{PrismRequestStart, prism_active_generation};
use super::{PrismHadamardQwen35InferenceExecution, PrismHadamardQwen35PendingLoad};
use crate::EngineGenerationStart;
use crate::GeneratedToken;
use crate::GenerationFinalization;
use crate::InferenceEngineError;
use crate::MlxInferenceExecution;
use crate::MlxMemoryLimitAdjustment;
use crate::MlxMemoryTelemetry;
use crate::PerformanceAttributionOutcome;
use crate::PerformanceOperation;
use crate::Qwen3_5ThinkingBudgetState;
use crate::RequestDecoderStateStack;
use crate::inference_engine::load_result::EngineLoadResult;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::qwen3_5_core::shared_execution::decode_step::advance_generation;
use crate::qwen3_5_core::shared_execution::generation_finalization_steps::Qwen3_5FinalizationHooks;
use crate::sampling_seed;
use crate::sampling_seed::current_time_millis_since_unix_epoch;
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
        self.open_astro_ssd_prompt_cache_store_after_load(
            &pending_load,
            &mut performance_attribution,
        )?;
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
        if self.execution_core.active_request.is_some() {
            return Err(InferenceEngineError::EngineBusy);
        }
        let maximum_context_tokens = self.maximum_context_tokens();
        let model =
            self.execution_core
                .model
                .as_ref()
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: "folded-ternary model is not loaded".to_owned(),
                })?;
        super::request_checks::check_serving_request(
            inference_request.input_token_ids(),
            inference_request.max_output_tokens(),
            self.execution_core.vocabulary_size,
            maximum_context_tokens,
        )?;
        let sampling_seed = sampling_seed::resolve_sampling_seed(
            match inference_request.sampling_strategy() {
                crate::Qwen3_5SamplingStrategy::HighestLogit => None,
                crate::Qwen3_5SamplingStrategy::TopKTopP { seed, .. } => seed,
            },
            current_time_millis_since_unix_epoch,
        );
        let random_state = model.runtime.random_key(sampling_seed).map_err(|error| {
            InferenceEngineError::Fatal {
                reason: format!("folded-ternary random key failed: {error}"),
            }
        })?;
        let thinking_budget_state = Qwen3_5ThinkingBudgetState::new(
            inference_request.generation_starts_inside_thinking_block(),
            inference_request.thinking_budget(),
            inference_request
                .forced_thinking_transition_token_ids()
                .to_vec(),
            inference_request.natural_reasoning_end_token_ids().to_vec(),
        )
        .map_err(|budget_error| InferenceEngineError::InvalidRequest {
            reason: format!("folded-ternary thinking budget is invalid: {budget_error}"),
        })?;
        let mut performance_attribution = inference_request.take_performance_attribution();
        let prompt_token_ids = inference_request.input_token_ids().to_vec();
        let (request_decoder_state, restore_outcome) =
            super::starting_decoder_state::starting_request_decoder_state(
                model,
                self.execution_core.astro_ssd_prompt_cache.as_deref(),
                &prompt_token_ids,
                &mut self.execution_core.astro_ssd_prompt_cache_counters,
                &mut performance_attribution,
            )?;
        let astro_ssd_prompt_cache_restored_token_count = restore_outcome.restored_token_count;
        let request_start = PrismRequestStart {
            prompt_token_ids,
            request_decoder_state,
            restored_token_count: astro_ssd_prompt_cache_restored_token_count,
            last_restored_block_key: restore_outcome.last_restored_astro_ssd_prompt_cache_block_key,
            random_state,
            thinking_budget_state,
            performance_attribution,
            can_use_astro_ssd_prompt_cache: self.execution_core.astro_ssd_prompt_cache.is_some(),
        };
        self.execution_core.active_request = Some(prism_active_generation(
            &mut inference_request,
            request_start,
        ));
        Ok(EngineGenerationStart::with_expert_memory_mode(
            astro_ssd_prompt_cache_restored_token_count,
            ExpertMemoryMode::Resident,
        ))
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
        Ok(self.astro_ssd_prompt_cache_stats_event())
    }

    fn clear_astro_ssd_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        let Some(astro_ssd_prompt_cache) = self.execution_core.astro_ssd_prompt_cache.as_ref()
        else {
            return Ok(None);
        };
        let clear_outcome = astro_ssd_prompt_cache
            .clear_astro_ssd_prompt_cache(model_id.as_deref())
            .map_err(|clear_error| InferenceEngineError::Fatal {
                reason: format!(
                    "folded-ternary astro SSD prompt cache could not be cleared: {clear_error}"
                ),
            })?;
        Ok(Some(WorkerEvent::AstroSsdPromptCacheCleared {
            model_id: clear_outcome.model_id,
            blocks_removed: clear_outcome.blocks_removed,
            bytes_freed: clear_outcome.bytes_freed,
        }))
    }
}

impl PrismHadamardQwen35InferenceExecution {
    /// Opens the `astro_ssd_prompt_cache` store for the model `load` just
    /// seated, when the worker configured one, against the startup ceiling
    /// the store's contract has always been sized with.
    fn open_astro_ssd_prompt_cache_store_after_load(
        &mut self,
        pending_load: &PrismHadamardQwen35PendingLoad,
        performance_attribution: &mut crate::PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        let execution_core = &mut self.execution_core;
        let Some(astro_ssd_prompt_cache_disk_store_config) = execution_core
            .astro_ssd_prompt_cache_disk_store_config
            .clone()
        else {
            return Ok(());
        };
        let loaded_model =
            execution_core
                .model
                .as_ref()
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: "folded-ternary model is not loaded".to_owned(),
                })?;
        let astro_ssd_prompt_cache =
            super::super::astro_ssd_prompt_cache::open_astro_ssd_prompt_cache_store(
                pending_load.effective_mlx_memory_ceiling_bytes,
                loaded_model,
                loaded_model.pack().model_id(),
                loaded_model.pack().revision(),
                pending_load.configured_astro_ssd_prompt_cache_block_token_count,
                pending_load.ssd_prompt_cache_common_prefix_stride_blocks,
                astro_ssd_prompt_cache_disk_store_config,
                performance_attribution,
            )?;
        execution_core.astro_ssd_prompt_cache = Some(astro_ssd_prompt_cache);
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
