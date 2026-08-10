use std::time::Instant;

use astronomical_ipc_protocol::RequestId;

use crate::{
    GeneratedToken, InferenceEngineError, PerformanceCounter, PerformanceOperation,
    persistent_prompt_cache_boundary_clamped_prefill_chunck_end,
};

use super::super::model::memory_admission::invalid_request_error;
use super::memory_admission::collect_completed_forward_memory_snapshot;
use super::prefill_execution_context::Qwen3_5PrefillExecutionContext;
use super::prompt_prefill_errors::PromptPrefillChunckAttemptError;
use super::{
    Qwen3_5EngineState, fatal_engine_error,
    qwen3_5_prefill_chunck_end_at_ordinary_target_control_span_boundary, qwen3_5_runtime_error,
    qwen3_5_speculative_prefill_sparse_target_is_active,
};
use crate::qwen3_5_moe::reclaim_retained_experts_for_request_memory_pressure;

impl Qwen3_5EngineState {
    pub(super) fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        let final_prompt_index = active_request.input_token_ids.len() - 1;
        if active_request.prefill_cursor >= final_prompt_index {
            return Ok(None);
        }
        if qwen3_5_speculative_prefill_sparse_target_is_active(
            active_request.should_use_speculative_prefill,
            active_request.prefill_cursor,
            active_request.ordinary_target_prefill_control_span_token_count,
        ) {
            let speculative_prefill_selection_preparation = match self
                .prepare_speculative_prefill_selection(
                    active_request,
                    active_request.prefill_cursor,
                    final_prompt_index,
                ) {
                Ok(speculative_prefill_selection_preparation) => {
                    speculative_prefill_selection_preparation
                }
                Err(speculative_prefill_failure) => {
                    self.log_speculative_prefill_drafter_failure_diagnostics(active_request);
                    return Err(speculative_prefill_failure);
                }
            };
            if matches!(
                speculative_prefill_selection_preparation,
                super::speculative_prefill_scoring::SpeculativePrefillSelectionPreparation::DrafterPhaseStarted
            ) {
                return Ok(Some(GeneratedToken::PromptProcessingPhaseStarted {
                    prompt_processing_phase:
                        astronomical_ipc_protocol::WorkerPromptProcessingPhase::Drafter,
                    total_token_count: u32::try_from(active_request.input_token_ids.len())
                        .unwrap_or(u32::MAX),
                }));
            }
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let prefill_start = active_request.prefill_cursor;
        let prefill_execution_context = Qwen3_5PrefillExecutionContext::new(
            active_request.visual_embeddings.is_some(),
            active_request.has_optional_prediction_session(),
            model.sparse_experts_are_paged(),
            self.persistent_prompt_cache.is_some()
                && active_request.can_use_persistent_prompt_cache
                && !active_request.has_optional_prediction_session(),
        )
        .with_target_only_prefix(active_request.has_optional_prediction_session())
        .with_speculative_prefill_sparse_target(
            qwen3_5_speculative_prefill_sparse_target_is_active(
                active_request.should_use_speculative_prefill,
                prefill_start,
                active_request.ordinary_target_prefill_control_span_token_count,
            ),
        );
        let candidate_prefill_chunck_end = self
            .prefill_chunck_sizer
            .next_prefill_chunck_end_for_execution_context(
                active_request.prefill_cursor,
                final_prompt_index,
                prefill_execution_context,
            );
        let requested_prefill_chunck_end =
            qwen3_5_prefill_chunck_end_at_ordinary_target_control_span_boundary(
                prefill_start,
                candidate_prefill_chunck_end,
                active_request.ordinary_target_prefill_control_span_token_count,
            )
            .ok_or_else(|| fatal_engine_error("prompt-processing chunk did not advance"))?;
        // Required publication is synchronous, so do not let one forward cross
        // multiple durable boundaries. A successful forward produces one exact
        // checkpoint, publishes it, and only then advances the request cursor.
        let requested_prefill_chunck_end = if self.persistent_prompt_cache.is_some()
            && active_request.can_use_persistent_prompt_cache
            && !active_request.has_optional_prediction_session()
            && !qwen3_5_speculative_prefill_sparse_target_is_active(
                active_request.should_use_speculative_prefill,
                prefill_start,
                active_request.ordinary_target_prefill_control_span_token_count,
            ) {
            let persistent_prompt_cache_block_token_count = self
                .persistent_prompt_cache
                .as_ref()
                .map(|persistent_prompt_cache| {
                    persistent_prompt_cache
                        .model_contract_ref()
                        .block_token_count()
                })
                .unwrap_or(0);
            persistent_prompt_cache_boundary_clamped_prefill_chunck_end(
                prefill_start,
                requested_prefill_chunck_end,
                persistent_prompt_cache_block_token_count,
            )
        } else {
            requested_prefill_chunck_end
        };
        let forward_chunk_started_at = Instant::now();
        let requested_prefill_chunck_token_count = requested_prefill_chunck_end - prefill_start;
        let mut attempted_prefill_chunck_token_count =
            active_request.clamped_prefill_chunck_token_count(requested_prefill_chunck_token_count);
        let mut has_retried_current_prefill_chunck_after_reclamation = false;
        let mut has_observed_prefill_capacity_constraint = false;
        let (prefill_end, prompt_prefill_chunck_outcome) = loop {
            let prefill_end = prefill_start
                .checked_add(attempted_prefill_chunck_token_count)
                .ok_or_else(|| fatal_engine_error("prefill chunk end overflowed"))?;
            match self.execute_prompt_prefill_chunck(
                request_id,
                active_request,
                prefill_start,
                prefill_end,
            ) {
                Ok(prompt_prefill_chunck_outcome) => {
                    break (prefill_end, prompt_prefill_chunck_outcome);
                }
                Err(PromptPrefillChunckAttemptError::Engine(generation_error)) => {
                    return Err(generation_error);
                }
                Err(PromptPrefillChunckAttemptError::AdaptiveMemoryLimitExceeded { reason }) => {
                    if attempted_prefill_chunck_token_count == 1 {
                        return Err(invalid_request_error(
                            "one prompt token cannot fit under the configured MLX ceiling",
                        ));
                    }
                    tracing::warn!(
                        request_id = request_id.value(),
                        attempted_prefill_chunck_token_count,
                        reason,
                        "adaptive MLX prefill admission reduced the prompt-processing chunk"
                    );
                    has_observed_prefill_capacity_constraint = true;
                    attempted_prefill_chunck_token_count /= 2;
                    has_retried_current_prefill_chunck_after_reclamation = false;
                    active_request
                        .performance_attribution
                        .record_counter(PerformanceCounter::PrefillCapacityRetryCount, 1);
                }
                Err(PromptPrefillChunckAttemptError::ActiveMemoryLimitExceeded {
                    active_memory_bytes,
                    attempted_allocation_bytes,
                    allowed_active_memory_bytes,
                    prefill_request_checkpoint,
                }) => {
                    active_request
                        .performance_attribution
                        .record_counter(PerformanceCounter::PrefillCapacityRejectionCount, 1);
                    active_request
                        .restore_prefill_request_checkpoint(prefill_request_checkpoint)
                        .map_err(qwen3_5_runtime_error)?;
                    active_request
                        .performance_attribution
                        .measure_operation(
                            PerformanceOperation::MlxAllocatorCacheCleanup,
                            |_performance_attribution| match model
                                .runtime()
                                .synchronize_gpu_stream()
                            {
                                Ok(()) => model.runtime().clear_allocator_cache(),
                                Err(mlx_runtime_error)
                                    if mlx_runtime_error
                                        .is_recoverable_graphics_processor_out_of_memory() =>
                                {
                                    model.runtime().clear_allocator_cache()
                                }
                                Err(mlx_runtime_error) => Err(mlx_runtime_error),
                            },
                        )
                        .map_err(qwen3_5_runtime_error)?;
                    let memory_snapshot_before_expert_reclamation = model
                        .runtime()
                        .memory_snapshot()
                        .map_err(qwen3_5_runtime_error)?;
                    let target_expert_payload_bytes_before_reclamation = model
                        .expert_weight_memory_cache_statistics()
                        .resident_payload_byte_count;
                    let native_capacity_deficit_bytes = active_memory_bytes
                        .saturating_add(attempted_allocation_bytes)
                        .saturating_sub(allowed_active_memory_bytes);
                    let memory_snapshot_after_expert_reclamation =
                        if native_capacity_deficit_bytes == 0 {
                            None
                        } else {
                            reclaim_retained_experts_for_request_memory_pressure(
                                model,
                                native_capacity_deficit_bytes,
                            )?
                        };
                    let active_memory_bytes_after_expert_reclamation =
                        memory_snapshot_after_expert_reclamation.as_ref().map_or(
                            memory_snapshot_before_expert_reclamation.active_memory_bytes(),
                            |memory_snapshot_after_expert_reclamation| {
                                memory_snapshot_after_expert_reclamation.active_memory_bytes()
                            },
                        );
                    if active_request.should_use_speculative_prefill {
                        let target_expert_payload_bytes_after_reclamation = model
                            .expert_weight_memory_cache_statistics()
                            .resident_payload_byte_count;
                        active_request.performance_attribution.record_counter(
                            PerformanceCounter::SpeculativePrefillContextTargetExpertReclaimedPayloadBytes,
                            target_expert_payload_bytes_before_reclamation.saturating_sub(
                                target_expert_payload_bytes_after_reclamation,
                            ),
                        );
                    }
                    let should_retry_same_prefill_chunck =
                        !has_retried_current_prefill_chunck_after_reclamation
                            && active_memory_bytes_after_expert_reclamation
                                < memory_snapshot_before_expert_reclamation.active_memory_bytes();
                    tracing::warn!(
                        request_id = request_id.value(),
                        attempted_prefill_chunck_token_count,
                        active_memory_bytes,
                        attempted_allocation_bytes,
                        allowed_active_memory_bytes,
                        native_capacity_deficit_bytes,
                        active_memory_bytes_before_expert_reclamation =
                            memory_snapshot_before_expert_reclamation.active_memory_bytes(),
                        active_memory_bytes_after_expert_reclamation,
                        should_retry_same_prefill_chunck,
                        "native MLX prefill allocation reached the active-memory ceiling"
                    );
                    has_observed_prefill_capacity_constraint = true;
                    if should_retry_same_prefill_chunck {
                        has_retried_current_prefill_chunck_after_reclamation = true;
                    } else {
                        if attempted_prefill_chunck_token_count == 1 {
                            return Err(invalid_request_error(
                                "one prompt token cannot fit under the configured MLX ceiling",
                            ));
                        }
                        attempted_prefill_chunck_token_count /= 2;
                        has_retried_current_prefill_chunck_after_reclamation = false;
                    }
                    active_request
                        .performance_attribution
                        .record_counter(PerformanceCounter::PrefillCapacityRetryCount, 1);
                }
                Err(PromptPrefillChunckAttemptError::GraphicsProcessorMemoryExhausted {
                    reason,
                    prefill_request_checkpoint,
                }) => {
                    active_request
                        .performance_attribution
                        .record_counter(PerformanceCounter::PrefillCapacityRejectionCount, 1);
                    active_request
                        .restore_prefill_request_checkpoint(prefill_request_checkpoint)
                        .map_err(qwen3_5_runtime_error)?;
                    active_request
                        .performance_attribution
                        .measure_operation(
                            PerformanceOperation::MlxAllocatorCacheCleanup,
                            |_performance_attribution| {
                                // The failed synchronization already waited for command-buffer
                                // completion; synchronizing its poisoned event chain repeats the error.
                                model.runtime().clear_allocator_cache()
                            },
                        )
                        .map_err(qwen3_5_runtime_error)?;
                    if attempted_prefill_chunck_token_count == 1 {
                        return Err(invalid_request_error(
                            "one prompt token exhausted available GPU memory",
                        ));
                    }
                    tracing::warn!(
                        request_id = request_id.value(),
                        attempted_prefill_chunck_token_count,
                        reason,
                        "Metal memory exhaustion reduced the prompt-processing chunk"
                    );
                    has_observed_prefill_capacity_constraint = true;
                    attempted_prefill_chunck_token_count /= 2;
                    has_retried_current_prefill_chunck_after_reclamation = false;
                    active_request
                        .performance_attribution
                        .record_counter(PerformanceCounter::PrefillCapacityRetryCount, 1);
                }
            }
        };
        let active_memory_bytes_before_growth =
            prompt_prefill_chunck_outcome.active_memory_bytes_before_growth;
        let forward_chunk_elapsed_millis =
            prompt_prefill_chunck_outcome.forward_chunk_elapsed_millis;
        let adaptive_ram_growth_context = prompt_prefill_chunck_outcome.adaptive_ram_growth_context;
        let exact_temporary_workspace_bytes =
            prompt_prefill_chunck_outcome.exact_temporary_workspace_bytes;
        let boundary_checkpoints = prompt_prefill_chunck_outcome.boundary_checkpoints;
        let speculative_prefill_chunck_mode =
            prompt_prefill_chunck_outcome.speculative_prefill_chunck_mode;
        let prefill_token_count = prefill_end - prefill_start;
        let should_retain_adaptive_ram_growth_observation = requested_prefill_chunck_token_count
            == self.prefill_chunck_sizer.active_prefill_chunck_tokens();
        if has_observed_prefill_capacity_constraint {
            active_request.record_successful_capacity_prefill_chunck(prefill_token_count);
        }
        // Publication deliberately precedes `advance_position` and cursor update.
        // If storage fails, the request stops at its last durable parent rather
        // than continuing with an in-memory chain that restart cannot reproduce.
        if let Some(persistent_prompt_cache) = self.persistent_prompt_cache.as_ref() {
            if active_request.can_use_persistent_prompt_cache && !boundary_checkpoints.is_empty() {
                self.capture_persistent_prompt_cache_blocks(
                    persistent_prompt_cache,
                    model,
                    active_request,
                    prefill_start,
                    prefill_end,
                    boundary_checkpoints,
                    super::persistent_prompt_cache_capture::PromptStatePersistenceOwner::for_active_request(active_request),
                )?;
            }
        }
        active_request.advance_position(prefill_token_count)?;
        active_request.prefill_cursor = prefill_end;
        // Keep the established end-of-chunk cleanup as well. A chunk without a cache boundary,
        // or the work created while serializing a boundary, must not retain temporary MLX
        // allocations until the next prompt chunk or request finalization.
        let memory_snapshot_before_end_of_prefill_chunck_cleanup =
            model.runtime().memory_snapshot().ok();
        let end_of_prefill_chunck_cleanup_started_at = Instant::now();
        active_request
            .performance_attribution
            .measure_operation(
                PerformanceOperation::MlxAllocatorCacheCleanup,
                |_performance_attribution| {
                    model
                        .runtime()
                        .synchronize_gpu_stream_and_clear_allocator_cache()
                },
            )
            .map_err(qwen3_5_runtime_error)?;
        let memory_snapshot_after_end_of_prefill_chunck_cleanup =
            model.runtime().memory_snapshot().ok();
        tracing::info!(
            request_id = request_id.value(),
            cleanup_stage = "end_of_prefill_chunck_cleanup",
            prefill_start,
            prefill_end,
            active_memory_bytes_before_cleanup =
                memory_snapshot_before_end_of_prefill_chunck_cleanup
                    .as_ref()
                    .map(|memory_snapshot| memory_snapshot.active_memory_bytes()),
            allocator_cache_memory_bytes_before_cleanup =
                memory_snapshot_before_end_of_prefill_chunck_cleanup
                    .as_ref()
                    .map(|memory_snapshot| memory_snapshot.allocator_cache_memory_bytes()),
            active_memory_bytes_after_cleanup = memory_snapshot_after_end_of_prefill_chunck_cleanup
                .as_ref()
                .map(|memory_snapshot| memory_snapshot.active_memory_bytes()),
            allocator_cache_memory_bytes_after_cleanup =
                memory_snapshot_after_end_of_prefill_chunck_cleanup
                    .as_ref()
                    .map(|memory_snapshot| memory_snapshot.allocator_cache_memory_bytes()),
            runtime_active_memory_limit_bytes =
                model.runtime().memory_limits().active_memory_limit_bytes(),
            cleanup_elapsed_millis = end_of_prefill_chunck_cleanup_started_at
                .elapsed()
                .as_millis(),
            "cleared MLX allocator-cache storage after prompt-processing chunk"
        );
        let mlx_memory_snapshot = collect_completed_forward_memory_snapshot(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context,
            should_retain_adaptive_ram_growth_observation,
            model,
            active_memory_bytes_before_growth,
            exact_temporary_workspace_bytes,
            &mut active_request.performance_attribution,
        )?;
        let prefill_chunck_elapsed_millis = forward_chunk_started_at.elapsed().as_millis() as u64;
        let next_prefill_execution_context = Qwen3_5PrefillExecutionContext::new(
            active_request.visual_embeddings.is_some(),
            active_request.has_optional_prediction_session(),
            model.sparse_experts_are_paged(),
            self.persistent_prompt_cache.is_some()
                && active_request.can_use_persistent_prompt_cache
                && !active_request.has_optional_prediction_session(),
        )
        .with_target_only_prefix(active_request.has_optional_prediction_session())
        .with_speculative_prefill_sparse_target(
            qwen3_5_speculative_prefill_sparse_target_is_active(
                active_request.should_use_speculative_prefill,
                prefill_end,
                active_request.ordinary_target_prefill_control_span_token_count,
            ),
        );
        if matches!(
            speculative_prefill_chunck_mode,
            super::Qwen3_5SpeculativePrefillChunckMode::TerminalAdditionalHistoryCapture
        ) {
            self.prefill_chunck_sizer
                .discard_pending_prefill_chunck_decision();
        } else {
            self.prefill_chunck_sizer.record_prefill_chunck_transition(
                prefill_token_count,
                prefill_chunck_elapsed_millis,
                has_observed_prefill_capacity_constraint,
                next_prefill_execution_context,
            );
        }
        let prefill_optimizer_insight = self
            .prefill_chunck_sizer
            .take_latest_prefill_optimizer_insight();
        tracing::trace!(
            request_id = request_id.value(),
            prefill_start_token = prefill_start,
            prefill_end_token = prefill_end,
            prefill_token_count,
            forward_chunk_elapsed_millis,
            prefill_chunck_elapsed_millis,
            "completed synchronous Qwen3.5 prompt-processing chunk"
        );
        Ok(Some(GeneratedToken::PrefillProgress {
            processed_token_count: prefill_token_count as u32,
            elapsed_millis: prefill_chunck_elapsed_millis,
            forward_prefill_chunck_elapsed_millis: forward_chunk_elapsed_millis,
            completed_prefill_chunck_tokens: u32::try_from(prefill_token_count).map_err(|_| {
                fatal_engine_error("completed_prefill_chunck_tokens exceeds the u32 range")
            })?,
            prefill_optimizer_insight,
            mlx_memory_telemetry: mlx_memory_snapshot
                .map(|mlx_memory_snapshot| {
                    let active_memory_bytes =
                        u64::try_from(mlx_memory_snapshot.active_memory_bytes()).map_err(|_| {
                            fatal_engine_error("MLX active memory bytes exceed the u64 range")
                        })?;
                    Ok::<crate::MlxMemoryTelemetry, InferenceEngineError>(
                        crate::MlxMemoryTelemetry::new(
                            active_memory_bytes,
                            u64::try_from(mlx_memory_snapshot.allocator_cache_memory_bytes())
                                .map_err(|_| {
                                    fatal_engine_error(
                                        "MLX allocator-cache memory bytes exceed the u64 range",
                                    )
                                })?,
                            u64::try_from(mlx_memory_snapshot.peak_memory_bytes()).map_err(
                                |_| {
                                    fatal_engine_error("MLX peak memory bytes exceed the u64 range")
                                },
                            )?,
                            model.active_memory_breakdown(
                                &active_request.request_decoder_state,
                                active_request.additional_context_state_payload_bytes(),
                                active_memory_bytes,
                                0,
                            ),
                        ),
                    )
                })
                .transpose()?,
            speculative_prefill_draft_memory_telemetry: active_request
                .speculative_prefill_draft_memory_telemetry
                .take(),
            expert_memory_mode: Some(model.expert_memory_mode()),
            prompt_work_reuse: active_request.prompt_work_reuse,
            persistent_prompt_cache_diagnostics: active_request
                .persistent_prompt_cache_diagnostics
                .clone(),
        }))
    }
}
