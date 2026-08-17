//! Advances one user-visible prompt-processing chunk with bounded recovery.
//!
//! The order in this file is a correctness contract:
//!
//! 1. Choose a deterministic fixed chunk from resolved configuration.
//! 2. Clamp it at semantic control-span and durable prompt-cache boundaries.
//! 3. Execute that unchanged chunk under adaptive memory admission.
//! 4. On a typed capacity failure, restore the request checkpoint before cleanup.
//! 5. Reclaim exact elastic expert bytes and retry the same chunk at most once.
//! 6. Publish required prompt-cache state before advancing the in-memory cursor.
//! 7. Synchronize the chunk tape and reclaim allocator cache only above threshold.
//! 8. Retain request-local capacity evidence; decode fills demand-selected pages after prefill.
//! 9. Emit progress only after all required state transitions succeeded.
//!
//! Recovery retries the unchanged chunk once after exact expert reclamation,
//! then halves executable work deterministically after another capacity failure.

use std::time::Instant;

use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES;

use crate::{
    ExpertResidencyPhase, GeneratedToken, InferenceEngineError, PerformanceCounter,
    PerformanceOperation, last_prefill_chunk_demand_weight,
    persistent_prompt_cache_boundary_clamped_prefill_chunck_end,
};

use super::super::model::memory_admission::invalid_request_error;
use super::completed_forward_memory::collect_completed_forward_memory_snapshot;
use super::prompt_prefill_errors::PromptPrefillChunckAttemptError;
use super::{
    Qwen3_5EngineState, Qwen3_5PromptProcessingChunkSizer, fatal_engine_error,
    qwen3_5_prefill_chunck_end_at_ordinary_target_control_span_boundary, qwen3_5_runtime_error,
    qwen3_5_speculative_prefill_sparse_target_is_active,
    speculative_prefill::SpeculativePrefillSelectionPreparation,
};

impl Qwen3_5EngineState {
    pub(super) fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        // The final prompt token is reserved as the generation seed and is
        // forwarded by generation startup. Prefill stops immediately before it.
        let final_prompt_index = active_request.input_token_ids.len() - 1;
        if active_request.prefill_cursor >= final_prompt_index {
            return Ok(None);
        }
        if qwen3_5_speculative_prefill_sparse_target_is_active(
            active_request.should_use_speculative_prefill,
            active_request.prefill_cursor,
            active_request.ordinary_target_prefill_control_span_token_count,
        ) {
            // Preparation first attempts cheap exact selection reuse. On a miss,
            // it becomes a two-call state machine: this call yields a stream-visible
            // Drafter phase event and the next performs request-scoped scoring.
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
                    // Capture complete target/drafter/cache evidence while request
                    // state still exists, then preserve the original error.
                    self.log_speculative_prefill_drafter_failure_diagnostics(active_request);
                    return Err(speculative_prefill_failure);
                }
            };
            if matches!(
                speculative_prefill_selection_preparation,
                SpeculativePrefillSelectionPreparation::DrafterPhaseStarted
            ) {
                return Ok(Some(GeneratedToken::PromptProcessingPhaseStarted {
                    prompt_processing_phase:
                        astronomical_ipc_protocol::WorkerPromptProcessingPhase::Drafter,
                    total_token_count: u32::try_from(active_request.input_token_ids.len())
                        .unwrap_or(u32::MAX),
                }));
            }
        }
        let prefill_start = active_request.prefill_cursor;
        let sparse_experts_are_paged = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .sparse_experts_are_paged();
        self.model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .refresh_phase_aware_expert_residency_plan(
                ExpertResidencyPhase::Prefill,
                u64::try_from(active_request.input_token_ids.len()).unwrap_or(u64::MAX),
                &mut active_request.performance_attribution,
            )
            .map_err(qwen3_5_runtime_error)?;
        let configured_prompt_processing_chunk_end = self
            .prompt_processing_chunk_sizer
            .next_prompt_processing_chunk_end_with_maximum_executable_capacity(
                active_request.prefill_cursor,
                final_prompt_index,
                sparse_experts_are_paged,
                active_request
                    .maximum_successful_prefill_chunck_tokens()
                    .unwrap_or(usize::MAX),
            );
        // First clamp: never combine the dense control prefix and sparse selected
        // conversation body in one call. They have different execution contracts.
        let requested_prefill_chunck_end =
            qwen3_5_prefill_chunck_end_at_ordinary_target_control_span_boundary(
                prefill_start,
                configured_prompt_processing_chunk_end,
                active_request.ordinary_target_prefill_control_span_token_count,
            )
            .ok_or_else(|| fatal_engine_error("prompt-processing chunk did not advance"))?;
        // The control-span clamp guarantees this chunk is wholly dense or wholly
        // sparse; execute_prompt_prefill_chunck never receives a mixed contract.
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
        // The token count becomes immutable for each attempt. Runtime capacity
        // recovery restores the checkpoint before retrying or halving this size.
        let forward_chunk_started_at = Instant::now();
        let requested_prefill_chunck_token_count = requested_prefill_chunck_end - prefill_start;
        let mut attempted_prefill_chunck_token_count =
            active_request.clamped_prefill_chunck_token_count(requested_prefill_chunck_token_count);
        let mut has_retried_current_prefill_chunck_after_reclamation = false;
        let mut has_observed_prefill_capacity_constraint = false;
        let (prefill_end, prompt_prefill_chunck_outcome) = loop {
            // Decode continues from the prompt tail, so the last successful
            // attempt records routed assignments with a density-matching weight.
            let last_chunk_ends_prompt = prefill_start
                .saturating_add(attempted_prefill_chunck_token_count)
                >= final_prompt_index;
            let demand_assignment_weight = if last_chunk_ends_prompt {
                last_prefill_chunk_demand_weight(
                    u64::try_from(prefill_start).unwrap_or(u64::MAX),
                    u64::try_from(attempted_prefill_chunck_token_count).unwrap_or(u64::MAX),
                )
            } else {
                1
            };
            if let Some(model) = self.model.as_ref() {
                model.set_expert_demand_assignment_weight(demand_assignment_weight);
            }
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
                    if let Some(smaller_executable_chunk_size_tokens) =
                        Qwen3_5PromptProcessingChunkSizer::next_smaller_executable_chunk_size_tokens(
                            attempted_prefill_chunck_token_count,
                        )
                    {
                        tracing::warn!(
                            request_id = request_id.value(),
                            attempted_prefill_chunck_token_count,
                            smaller_executable_chunk_size_tokens,
                            reason = %reason,
                            "prefill admission selected the next smaller executable chunk"
                        );
                        attempted_prefill_chunck_token_count = smaller_executable_chunk_size_tokens;
                        has_observed_prefill_capacity_constraint = true;
                        has_retried_current_prefill_chunck_after_reclamation = false;
                        continue;
                    }
                    tracing::warn!(
                        request_id = request_id.value(),
                        attempted_prefill_chunck_token_count,
                        reason = %reason,
                        "fixed prefill chunk cannot fit under the MLX ceiling after expert-memory admission"
                    );
                    return Err(invalid_request_error(format!(
                        "configured prefill chunk of {attempted_prefill_chunck_token_count} tokens cannot fit under the MLX ceiling after reclaiming elastic experts: {reason}"
                    )));
                }
                Err(PromptPrefillChunckAttemptError::ActiveMemoryLimitExceeded {
                    active_memory_bytes,
                    attempted_allocation_bytes,
                    allowed_active_memory_bytes,
                    prefill_request_checkpoint,
                }) => {
                    let should_retry_same_prefill_chunck = self
                        .recover_fixed_prefill_chunck_after_active_memory_limit(
                            request_id,
                            active_request,
                            attempted_prefill_chunck_token_count,
                            active_memory_bytes,
                            attempted_allocation_bytes,
                            allowed_active_memory_bytes,
                            prefill_request_checkpoint,
                            has_retried_current_prefill_chunck_after_reclamation,
                        )?;
                    has_observed_prefill_capacity_constraint = true;
                    if should_retry_same_prefill_chunck {
                        has_retried_current_prefill_chunck_after_reclamation = true;
                        active_request
                            .performance_attribution
                            .record_counter(PerformanceCounter::PrefillCapacityRetryCount, 1);
                    } else {
                        if let Some(smaller_executable_chunk_size_tokens) =
                            Qwen3_5PromptProcessingChunkSizer::next_smaller_executable_chunk_size_tokens(
                                attempted_prefill_chunck_token_count,
                            )
                        {
                            attempted_prefill_chunck_token_count =
                                smaller_executable_chunk_size_tokens;
                            has_retried_current_prefill_chunck_after_reclamation = false;
                            continue;
                        }
                        return Err(invalid_request_error(format!(
                            "configured prefill chunk of {attempted_prefill_chunck_token_count} tokens cannot fit under the MLX ceiling after reclaiming elastic experts"
                        )));
                    }
                }
                Err(PromptPrefillChunckAttemptError::GraphicsProcessorMemoryExhausted {
                    reason,
                    prefill_request_checkpoint,
                }) => {
                    let should_retry_same_prefill_chunck = self
                        .recover_fixed_prefill_chunck_after_graphics_processor_exhaustion(
                            request_id,
                            active_request,
                            attempted_prefill_chunck_token_count,
                            reason.as_str(),
                            prefill_request_checkpoint,
                            has_retried_current_prefill_chunck_after_reclamation,
                        )?;
                    has_observed_prefill_capacity_constraint = true;
                    if should_retry_same_prefill_chunck {
                        has_retried_current_prefill_chunck_after_reclamation = true;
                        active_request
                            .performance_attribution
                            .record_counter(PerformanceCounter::PrefillCapacityRetryCount, 1);
                    } else {
                        if let Some(smaller_executable_chunk_size_tokens) =
                            Qwen3_5PromptProcessingChunkSizer::next_smaller_executable_chunk_size_tokens(
                                attempted_prefill_chunck_token_count,
                            )
                        {
                            attempted_prefill_chunck_token_count =
                                smaller_executable_chunk_size_tokens;
                            has_retried_current_prefill_chunck_after_reclamation = false;
                            continue;
                        }
                        return Err(invalid_request_error(format!(
                            "configured prefill chunk of {attempted_prefill_chunck_token_count} tokens exhausted GPU memory after reclaiming elastic experts: {reason}"
                        )));
                    }
                }
            }
        };
        let active_memory_bytes_before_growth =
            prompt_prefill_chunck_outcome.active_memory_bytes_before_growth;
        let retained_expert_payload_bytes_before_growth =
            prompt_prefill_chunck_outcome.retained_expert_payload_bytes_before_growth;
        let forward_chunk_elapsed_millis =
            prompt_prefill_chunck_outcome.forward_chunk_elapsed_millis;
        let adaptive_ram_growth_context = prompt_prefill_chunck_outcome.adaptive_ram_growth_context;
        let exact_temporary_workspace_bytes =
            prompt_prefill_chunck_outcome.exact_temporary_workspace_bytes;
        let boundary_checkpoints = prompt_prefill_chunck_outcome.boundary_checkpoints;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let prefill_token_count = prefill_end - prefill_start;
        // Every completed forward is valid model-local transient evidence. Cache
        // boundaries and terminal tails deliberately produce smaller chunks; if
        // they are discarded, a long OpenCode request can complete real prefill
        // work without ever teaching subsequent admission about its workspace.
        let should_retain_adaptive_ram_growth_observation = true;
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
        // Only durable publication success may move these frontiers. If capture
        // failed above, the function returned with both position and cursor at
        // their previous parent, allowing restart to reproduce valid state.
        active_request.advance_position(prefill_token_count)?;
        active_request.prefill_cursor = prefill_end;
        // Retire the chunk tape. Reclaim the allocator cache only when it is
        // large enough to justify an IOGPU-visible clear.
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
                        .synchronize_gpu_stream_and_reclaim_allocator_cache_above_threshold(
                            ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES,
                        )
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
        collect_completed_forward_memory_snapshot(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context,
            should_retain_adaptive_ram_growth_observation,
            model,
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            exact_temporary_workspace_bytes,
            &mut active_request.performance_attribution,
        )?;
        // Demand-selected pages are materialized once after all prompt chunks.
        // Rebuilding them at every barrier would repeatedly read the same model
        // payload while the demand histogram is still changing.
        let mlx_memory_snapshot = model.runtime().memory_snapshot().ok();
        let prefill_chunck_elapsed_millis = forward_chunk_started_at.elapsed().as_millis() as u64;
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
            forward_prefill_chunk_elapsed_millis: forward_chunk_elapsed_millis,
            completed_prefill_chunk_tokens: u32::try_from(prefill_token_count).map_err(|_| {
                fatal_engine_error("completed_prefill_chunk_tokens exceeds the u32 range")
            })?,
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
            expert_residency_telemetry: Some(model.expert_residency_telemetry()),
            speculative_prefill_draft_memory_telemetry: active_request
                .speculative_prefill_draft_memory_telemetry
                .take(),
            // Drafter telemetry is emitted once on the next completed target
            // progress item, then removed so later chunks cannot duplicate it.
            expert_memory_mode: Some(model.expert_memory_mode()),
            prompt_work_reuse: active_request.prompt_work_reuse,
            persistent_prompt_cache_diagnostics: active_request
                .persistent_prompt_cache_diagnostics
                .clone(),
        }))
    }
}
