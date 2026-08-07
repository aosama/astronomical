use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationOutput, WorkerEvent,
};
use std::{
    sync::{Arc, RwLock},
    time::Duration,
};
use tokio::time::Instant;

use crate::{
    ActiveRequestProgress, ChatGenerationStreamEvent, GenerationPerformanceLog,
    GenerationPerformanceRecord, WorkerActivity, WorkerControlError, WorkerHealthSnapshot,
    chat_generation_executor::try_send_stream_event,
    generation_performance_log::unix_epoch_millis,
    prefill_optimizer_observability::record_prefill_optimizer_insight,
    worker_health::{
        clear_active_request_progress, clear_latest_mlx_memory_snapshot,
        publish_active_request_progress, publish_activity, publish_expert_memory_mode,
        publish_health, publish_latest_mlx_memory_snapshot, publish_mlx_memory_limit_changed,
        publish_mlx_memory_limit_rejection, publish_persistent_prompt_cache_stats,
        record_serving_session,
    },
    worker_loop_types::ActiveGeneration,
};

pub(super) fn handle_worker_event(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_generation: &mut Option<ActiveGeneration>,
    performance_log: &mut GenerationPerformanceLog,
) -> Result<(), WorkerControlError> {
    match worker_event {
        WorkerEvent::ExpertMemoryModeChanged { expert_memory_mode } => {
            publish_expert_memory_mode(health_snapshot, expert_memory_mode);
        }
        WorkerEvent::Ready {
            model_id,
            capabilities,
            mtp_runtime_state,
            mtp_unavailable_reason,
        } => {
            if *is_ready || active_generation.is_some() {
                return Err(protocol_violation("duplicate worker readiness"));
            }
            *is_ready = true;
            tracing::info!(model = %model_id, "worker model is ready");
            *model_load_deadline = None;
            publish_health(
                health_snapshot,
                WorkerHealthSnapshot::ready_with_model(
                    model_id,
                    capabilities,
                    mtp_runtime_state,
                    mtp_unavailable_reason,
                ),
            );
        }
        WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes,
            effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
        } => {
            if *is_ready || active_generation.is_some() {
                return Err(protocol_violation("duplicate worker idle event"));
            }
            *is_ready = true;
            *model_load_deadline = None;
            publish_health(
                health_snapshot,
                WorkerHealthSnapshot::ready_without_model_with_memory_limits(
                    machine_mlx_memory_ceiling_bytes,
                    effective_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes,
                ),
            );
        }
        WorkerEvent::MlxMemorySample {
            mlx_memory_snapshot,
        } => {
            if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
                publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
            } else {
                clear_latest_mlx_memory_snapshot(health_snapshot);
            }
        }
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            expert_memory_mode,
            mlx_memory_snapshot,
        } => {
            publish_mlx_memory_limit_changed(
                health_snapshot,
                effective_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                expert_memory_mode,
                mlx_memory_snapshot,
            );
        }
        WorkerEvent::MlxMemoryLimitRejected {
            minimum_mlx_memory_ceiling_bytes,
            reason,
            ..
        } => {
            publish_mlx_memory_limit_rejection(
                health_snapshot,
                minimum_mlx_memory_ceiling_bytes,
                reason,
            );
        }
        WorkerEvent::ModelSwapped {
            model_id,
            capabilities,
            minimum_mlx_memory_ceiling_bytes,
            mtp_runtime_state,
            mtp_unavailable_reason,
        } => {
            if !*is_ready {
                return Err(protocol_violation("model swapped before initial readiness"));
            }
            tracing::info!(model = %model_id, "worker model swapped");
            let ready_worker_health_snapshot =
                if let Ok(previous_health_snapshot) = health_snapshot.read() {
                    WorkerHealthSnapshot::ready_with_replacement_model(
                        model_id,
                        capabilities,
                        minimum_mlx_memory_ceiling_bytes,
                        mtp_runtime_state,
                        mtp_unavailable_reason,
                        &previous_health_snapshot,
                    )
                } else {
                    let mut replacement_health_snapshot = WorkerHealthSnapshot::ready_with_model(
                        model_id,
                        capabilities,
                        mtp_runtime_state,
                        mtp_unavailable_reason,
                    );
                    replacement_health_snapshot.minimum_mlx_memory_ceiling_bytes =
                        minimum_mlx_memory_ceiling_bytes;
                    replacement_health_snapshot
                };
            publish_health(health_snapshot, ready_worker_health_snapshot);
        }
        WorkerEvent::ModelSwapFailed { .. } => {
            return Err(protocol_violation(
                "model swap failure outside model swap wait",
            ));
        }
        WorkerEvent::GenerationFinalized {
            request_id,
            expert_memory_mode,
            mlx_memory_snapshot,
        } => {
            let Some(active_request) = active_generation.as_mut() else {
                return Err(protocol_violation(
                    "finalized generation state without an active request",
                ));
            };
            if request_id != active_request.request_id {
                return Err(protocol_violation(
                    "finalized generation state request mismatch",
                ));
            }
            if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
                active_request.last_mlx_peak_memory_bytes =
                    Some(mlx_memory_snapshot.peak_memory_bytes);
                active_request.last_mlx_active_memory_bytes =
                    Some(mlx_memory_snapshot.active_memory_bytes);
                publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
            } else {
                clear_latest_mlx_memory_snapshot(health_snapshot);
            }
            if let Some(expert_memory_mode) = expert_memory_mode {
                publish_expert_memory_mode(health_snapshot, expert_memory_mode);
            }
        }
        WorkerEvent::Output {
            request_id,
            sequence_number,
            generated_token_count,
            outputs,
            mlx_memory_snapshot,
        } => {
            let Some(active_request) = active_generation.as_mut() else {
                return Err(protocol_violation("output without an active request"));
            };
            let latest_known_generated_token_count = active_request
                .generated_token_count
                .max(active_request.latest_generation_progress_token_count);
            if request_id != active_request.request_id
                || sequence_number != active_request.next_sequence_number
                || generated_token_count == 0
                || generated_token_count < latest_known_generated_token_count
                || generated_token_count > active_request.max_output_tokens
            {
                return Err(protocol_violation(
                    "output correlation or sequence mismatch",
                ));
            }
            if outputs.is_empty() {
                return Err(protocol_violation("output batch must not be empty"));
            }
            let output_count = u16::try_from(outputs.len())
                .map_err(|_| protocol_violation("output batch count exceeds the u16 range"))?;
            let next_sequence_number = active_request
                .next_sequence_number
                .checked_add(output_count)
                .ok_or_else(|| protocol_violation("output sequence overflow"))?;
            let mut next_tool_call_index = active_request.next_tool_call_index;
            for output in &outputs {
                if let ChatGenerationOutput::ToolCall {
                    tool_call_index, ..
                } = output
                {
                    if *tool_call_index != next_tool_call_index {
                        return Err(protocol_violation("tool-call index mismatch"));
                    }
                    next_tool_call_index = next_tool_call_index
                        .checked_add(1)
                        .ok_or_else(|| protocol_violation("tool-call index overflow"))?;
                }
            }
            for output in outputs {
                let stream_event = ChatGenerationStreamEvent::from_worker_output(output);
                try_send_stream_event(&active_request.stream_event_sender, stream_event)?;
            }
            if active_request.generation_started_at.is_none() {
                active_request.generation_started_at = Some(Instant::now());
                publish_activity(health_snapshot, WorkerActivity::Generating);
            }
            let elapsed_millis = active_request
                .generation_started_at
                .map_or(0, |generation_started_at| {
                    generation_started_at.elapsed().as_millis() as u64
                });
            publish_active_request_progress(
                health_snapshot,
                ActiveRequestProgress::Generation {
                    generated_token_count: u32::from(generated_token_count),
                    maximum_output_tokens: u32::from(active_request.max_output_tokens),
                    elapsed_millis,
                },
            );
            active_request.next_sequence_number = next_sequence_number;
            active_request.next_tool_call_index = next_tool_call_index;
            active_request.generated_token_count = generated_token_count;
            active_request.latest_generation_progress_token_count = generated_token_count;
            if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
                publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
            }
        }
        WorkerEvent::PrefillProgress {
            request_id,
            processed_tokens,
            total_tokens,
            elapsed_millis,
            forward_prefill_chunck_elapsed_millis,
            completed_prefill_chunck_tokens,
            prefill_optimizer_insight,
            mlx_memory_snapshot,
        } => {
            if let Some(active_request) = active_generation.as_mut()
                && request_id == active_request.request_id
            {
                // The worker's elapsed_millis is cumulative across all prefill
                // chunks (accumulated in EngineBackedWorker::advance_generation),
                // so we take the latest value rather than adding it again.
                active_request.prefill_elapsed_millis = elapsed_millis;
                if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
                    active_request.last_mlx_peak_memory_bytes =
                        Some(mlx_memory_snapshot.peak_memory_bytes);
                    active_request.last_mlx_active_memory_bytes =
                        Some(mlx_memory_snapshot.active_memory_bytes);
                    publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
                }
                try_send_stream_event(
                    &active_request.stream_event_sender,
                    ChatGenerationStreamEvent::PrefillProgress {
                        processed_tokens,
                        total_tokens,
                        elapsed_millis,
                        forward_prefill_chunck_elapsed_millis,
                        completed_prefill_chunck_tokens,
                        mlx_active_memory_bytes: mlx_memory_snapshot
                            .map(|snapshot| snapshot.active_memory_bytes),
                        mlx_allocator_cache_memory_bytes: mlx_memory_snapshot
                            .map(|snapshot| snapshot.allocator_cache_memory_bytes),
                        mlx_peak_memory_bytes: mlx_memory_snapshot
                            .map(|snapshot| snapshot.peak_memory_bytes),
                    },
                )?;
                publish_active_request_progress(
                    health_snapshot,
                    ActiveRequestProgress::Prefill {
                        processed_tokens,
                        total_tokens,
                        request_started_at: active_request.request_started_at,
                        elapsed_millis,
                        completed_prefill_chunck_tokens,
                    },
                );
                if let Some(prefill_optimizer_insight) = prefill_optimizer_insight {
                    record_prefill_optimizer_insight(health_snapshot, prefill_optimizer_insight);
                }
            }
        }
        WorkerEvent::GenerationProgress {
            request_id,
            generated_token_count,
            maximum_output_tokens,
            elapsed_millis,
            mlx_memory_snapshot,
        } => {
            let Some(active_request) = active_generation.as_mut() else {
                return Err(protocol_violation(
                    "generation progress without an active request",
                ));
            };
            let latest_known_generated_token_count = active_request
                .generated_token_count
                .max(active_request.latest_generation_progress_token_count);
            if request_id != active_request.request_id
                || generated_token_count == 0
                || generated_token_count < latest_known_generated_token_count
                || generated_token_count > active_request.max_output_tokens
                || maximum_output_tokens != active_request.max_output_tokens
            {
                return Err(protocol_violation(
                    "generation progress correlation or count mismatch",
                ));
            }
            if active_request.generation_started_at.is_none() {
                let elapsed_duration = Duration::from_millis(elapsed_millis);
                active_request.generation_started_at = Some(
                    Instant::now()
                        .checked_sub(elapsed_duration)
                        .unwrap_or_else(Instant::now),
                );
            }
            active_request.latest_generation_progress_token_count = generated_token_count;
            publish_activity(health_snapshot, WorkerActivity::Generating);
            publish_active_request_progress(
                health_snapshot,
                ActiveRequestProgress::Generation {
                    generated_token_count: u32::from(generated_token_count),
                    maximum_output_tokens: u32::from(maximum_output_tokens),
                    elapsed_millis,
                },
            );
            if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
                publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
            }
        }
        WorkerEvent::Completed {
            request_id,
            prompt_token_count,
            generated_token_count,
            reasoning_token_count,
            cached_token_count,
            reason,
        } => {
            let Some(active_request) = active_generation.as_ref() else {
                return Err(protocol_violation("completion without an active request"));
            };
            let latest_known_generated_token_count = active_request
                .generated_token_count
                .max(active_request.latest_generation_progress_token_count);
            let is_valid_reason = match reason {
                ChatGenerationCompletionReason::EndOfSequence => {
                    generated_token_count <= active_request.max_output_tokens
                }
                ChatGenerationCompletionReason::MaximumOutputTokens => {
                    generated_token_count == active_request.max_output_tokens
                }
                ChatGenerationCompletionReason::ToolCalls => {
                    active_request.next_tool_call_index > 0
                }
                ChatGenerationCompletionReason::Cancelled => false,
            };
            if request_id != active_request.request_id
                || generated_token_count < latest_known_generated_token_count
                || generated_token_count > active_request.max_output_tokens
                || !is_valid_reason
            {
                return Err(protocol_violation(
                    "completion correlation or count mismatch",
                ));
            }
            let Some(completed_request) = active_generation.take() else {
                return Ok(());
            };
            let total_elapsed_millis = completed_request
                .request_started_at
                .elapsed()
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX);
            let generation_elapsed_millis = completed_request
                .generation_started_at
                .map_or(0, |started_at| started_at.elapsed().as_millis() as u64);
            let prefill_elapsed_millis = completed_request.prefill_elapsed_millis;
            let (prefill_tok_per_second, generation_tok_per_second) =
                GenerationPerformanceRecord::compute_throughput(
                    prompt_token_count,
                    cached_token_count,
                    generated_token_count,
                    prefill_elapsed_millis,
                    generation_elapsed_millis,
                );
            let model_id = health_snapshot
                .read()
                .map(|snapshot| snapshot.ready_model_id.clone())
                .ok()
                .flatten()
                .unwrap_or_default();
            tracing::info!(
                request_id = request_id.value(),
                prompt_token_count,
                generated_token_count,
                cached_token_count,
                completion_reason = ?reason,
                prefill_elapsed_millis,
                generation_elapsed_millis,
                total_elapsed_millis,
                prefill_tok_per_second = ?prefill_tok_per_second,
                generation_tok_per_second = ?generation_tok_per_second,
                "worker generation completed"
            );
            performance_log.record(&GenerationPerformanceRecord {
                timestamp_millis: unix_epoch_millis(),
                request_id: request_id.value(),
                model_id,
                prompt_token_count,
                cached_token_count,
                generated_token_count,
                completion_reason: match reason {
                    ChatGenerationCompletionReason::EndOfSequence => "end_of_sequence",
                    ChatGenerationCompletionReason::MaximumOutputTokens => "maximum_output_tokens",
                    ChatGenerationCompletionReason::ToolCalls => "tool_calls",
                    ChatGenerationCompletionReason::Cancelled => "cancelled",
                }
                .to_owned(),
                prefill_elapsed_millis,
                generation_elapsed_millis,
                total_elapsed_millis,
                prefill_tok_per_second,
                generation_tok_per_second,
                mlx_peak_memory_bytes: completed_request.last_mlx_peak_memory_bytes,
                mlx_active_memory_bytes: completed_request.last_mlx_active_memory_bytes,
            });
            record_serving_session(
                health_snapshot,
                prompt_token_count,
                cached_token_count,
                prefill_tok_per_second,
                generation_tok_per_second,
            );
            publish_activity(health_snapshot, WorkerActivity::Idle);
            clear_active_request_progress(health_snapshot);
            let stream_event = ChatGenerationStreamEvent::Completed {
                prompt_token_count,
                generated_token_count,
                reasoning_token_count,
                cached_token_count,
                reason,
            };
            try_send_stream_event(&completed_request.stream_event_sender, stream_event)?;
        }
        WorkerEvent::Failed { request_id, reason } => {
            let Some(active_request) = active_generation.as_ref() else {
                return Err(protocol_violation("failure without an active request"));
            };
            if request_id != active_request.request_id {
                return Err(protocol_violation("failure request mismatch"));
            }
            let Some(failed_request) = active_generation.take() else {
                return Ok(());
            };
            tracing::warn!(request_id = request_id.value(), failure_reason = ?reason,
                "worker generation failed");
            publish_activity(health_snapshot, WorkerActivity::Idle);
            clear_active_request_progress(health_snapshot);
            try_send_stream_event(
                &failed_request.stream_event_sender,
                ChatGenerationStreamEvent::Failed { reason },
            )?;
        }
        WorkerEvent::PersistentPromptCacheStats { .. } => {
            publish_persistent_prompt_cache_stats(health_snapshot, worker_event);
        }
    }
    Ok(())
}

pub(super) fn protocol_violation(description: &'static str) -> WorkerControlError {
    WorkerControlError::WorkerProtocolViolation { description }
}
