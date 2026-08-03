use std::time::Instant;

use astronomical_ipc_protocol::RequestId;

use crate::{
    GeneratedToken, InferenceEngineError, PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT,
    PerformanceCounter, PerformanceOperation, persistent_prompt_cache_aligned_prefill_end,
};

use super::super::model::memory_admission::{
    collect_completed_forward_memory_snapshot, invalid_request_error,
};
use super::super::reclaim_retained_experts_for_request_memory_pressure;
use super::prompt_prefill::PromptPrefillChunckAttemptError;
use super::{Qwen3_5MoEEngineState, fatal_engine_error, qwen3_5_moe_runtime_error};

impl Qwen3_5MoEEngineState {
    pub(super) fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5MoEEngineRequest,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        let final_prompt_index = active_request.input_token_ids.len() - 1;
        if active_request.prefill_cursor >= final_prompt_index {
            return Ok(None);
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5-MoE engine lost its loaded model"))?;
        let prefill_start = active_request.prefill_cursor;
        let candidate_prefill_chunck_end = self
            .prefill_chunck_sizer
            .next_prefill_chunck_end(active_request.prefill_cursor, final_prompt_index);
        let requested_prefill_chunck_end = if self.persistent_prompt_cache.is_some()
            && active_request.can_use_persistent_prompt_cache
        {
            persistent_prompt_cache_aligned_prefill_end(
                active_request.prefill_cursor,
                candidate_prefill_chunck_end,
                final_prompt_index,
            )
        } else {
            candidate_prefill_chunck_end
        };
        let forward_chunk_started_at = Instant::now();
        let requested_prefill_chunck_token_count = requested_prefill_chunck_end - prefill_start;
        let mut attempted_prefill_chunck_token_count =
            active_request.clamped_prefill_chunck_token_count(requested_prefill_chunck_token_count);
        let mut has_retried_current_prefill_chunck_after_reclamation = false;
        let mut has_observed_prefill_capacity_constraint = false;
        let (
            prefill_end,
            active_memory_bytes_before_growth,
            forward_chunk_elapsed_millis,
            adaptive_ram_growth_context,
        ) = loop {
            let prefill_end = prefill_start
                .checked_add(attempted_prefill_chunck_token_count)
                .ok_or_else(|| fatal_engine_error("prefill chunk end overflowed"))?;
            match self.execute_prompt_prefill_chunck(
                request_id,
                active_request,
                prefill_start,
                prefill_end,
            ) {
                Ok((
                    active_memory_bytes_before_growth,
                    forward_chunk_elapsed_millis,
                    adaptive_ram_growth_context,
                )) => {
                    break (
                        prefill_end,
                        active_memory_bytes_before_growth,
                        forward_chunk_elapsed_millis,
                        adaptive_ram_growth_context,
                    );
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
                        .map_err(qwen3_5_moe_runtime_error)?;
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
                        .map_err(qwen3_5_moe_runtime_error)?;
                    let memory_snapshot_before_expert_reclamation = model
                        .runtime()
                        .memory_snapshot()
                        .map_err(qwen3_5_moe_runtime_error)?;
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
            }
        };
        let prefill_token_count = prefill_end - prefill_start;
        let should_retain_adaptive_ram_growth_observation = requested_prefill_chunck_token_count
            == self.prefill_chunck_sizer.active_prefill_chunck_tokens();
        if has_observed_prefill_capacity_constraint {
            active_request.record_successful_capacity_prefill_chunck(prefill_token_count);
        }
        active_request.advance_position(prefill_token_count)?;
        active_request.prefill_cursor = prefill_end;
        if let (Some(persistent_prompt_cache), Some(persistent_prompt_cache_write_queue)) = (
            self.persistent_prompt_cache.as_ref(),
            self.persistent_prompt_cache_write_queue.as_ref(),
        ) {
            let position = active_request.next_position_tokens as usize;
            if active_request.can_use_persistent_prompt_cache
                && position > 0
                && position.is_multiple_of(PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT)
                && position < active_request.input_token_ids.len()
            {
                self.capture_persistent_prompt_cache_block(
                    persistent_prompt_cache,
                    persistent_prompt_cache_write_queue,
                    model,
                    active_request,
                );
            }
        }
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
            .map_err(qwen3_5_moe_runtime_error)?;
        let mlx_memory_snapshot = collect_completed_forward_memory_snapshot(
            &mut self.adaptive_ram_growth_guard,
            adaptive_ram_growth_context,
            should_retain_adaptive_ram_growth_observation,
            model,
            active_memory_bytes_before_growth,
            &mut active_request.performance_attribution,
        )?;
        let prefill_chunck_elapsed_millis = forward_chunk_started_at.elapsed().as_millis() as u64;
        self.prefill_chunck_sizer
            .record_prefill_chunck_elapsed_millis(
                prefill_token_count,
                prefill_chunck_elapsed_millis,
            );
        tracing::trace!(
            request_id = request_id.value(),
            prefill_start_token = prefill_start,
            prefill_end_token = prefill_end,
            prefill_token_count,
            forward_chunk_elapsed_millis,
            prefill_chunck_elapsed_millis,
            "completed synchronous Qwen3.5-MoE prompt-processing chunk"
        );
        Ok(Some(GeneratedToken::PrefillProgress {
            processed_token_count: prefill_token_count as u32,
            elapsed_millis: prefill_chunck_elapsed_millis,
            forward_prefill_chunck_elapsed_millis: forward_chunk_elapsed_millis,
            completed_prefill_chunck_tokens: u32::try_from(prefill_token_count).map_err(|_| {
                fatal_engine_error("completed_prefill_chunck_tokens exceeds the u32 range")
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
                                active_request.mtp_request_state.as_ref(),
                                active_memory_bytes,
                            ),
                        ),
                    )
                })
                .transpose()?,
            expert_memory_mode: Some(model.expert_memory_mode()),
        }))
    }
}
