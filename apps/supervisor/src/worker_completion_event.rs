//! Completion-event validation, local performance logging, and stream delivery.
//!
//! Worker completion has three audiences with different contracts: protocol
//! validation protects supervisor state, the local performance log receives
//! cache diagnostics, and the public stream receives only OpenAI-compatible
//! completion fields. Keeping the fan-out here makes that separation explicit.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, RequestId, WorkerPersistentPromptCacheRequestDiagnostics,
};

use crate::{
    ChatGenerationStreamEvent, GenerationPerformanceLog, GenerationPerformanceRecord,
    WorkerActivity, WorkerControlError, WorkerHealthSnapshot,
    chat_generation_executor::try_send_stream_event,
    generation_performance_log::unix_epoch_millis,
    worker_event_handler::protocol_violation,
    worker_health::{clear_active_request_progress, publish_activity, record_serving_session},
    worker_loop_types::ActiveGeneration,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_worker_completion_event(
    request_id: RequestId,
    prompt_token_count: u32,
    generated_token_count: u16,
    reasoning_token_count: u16,
    cached_token_count: u32,
    persistent_prompt_cache_diagnostics: Option<WorkerPersistentPromptCacheRequestDiagnostics>,
    reason: ChatGenerationCompletionReason,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_generation: &mut Option<ActiveGeneration>,
    performance_log: &mut GenerationPerformanceLog,
) -> Result<(), WorkerControlError> {
    // Cache diagnostics are request-scoped completion evidence. They go only to
    // the local performance row; the public generation stream keeps its stable
    // OpenAI-compatible completion shape unchanged.
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
        ChatGenerationCompletionReason::ToolCalls => active_request.next_tool_call_index > 0,
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
        time_to_first_output_millis = ?completed_request.time_to_first_output_millis,
        generation_preparation_elapsed_millis = ?completed_request
            .generation_preparation_elapsed_millis,
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
        time_to_first_output_millis: completed_request.time_to_first_output_millis,
        generation_preparation_elapsed_millis: completed_request
            .generation_preparation_elapsed_millis,
        first_decode_forward_elapsed_millis: completed_request.first_decode_forward_elapsed_millis,
        generation_preparation_expert_source_read_byte_count: 0,
        final_complete_expert_layer_count: completed_request.final_complete_expert_layer_count,
        final_complete_expert_payload_bytes: completed_request.final_complete_expert_payload_bytes,
        final_partial_expert_layer_count: completed_request.final_partial_expert_layer_count,
        final_partial_expert_payload_bytes: completed_request.final_partial_expert_payload_bytes,
        prefill_tok_per_second,
        generation_tok_per_second,
        mlx_peak_memory_bytes: completed_request.maximum_mlx_peak_memory_bytes,
        mlx_active_memory_bytes: completed_request.last_mlx_active_memory_bytes,
        persistent_prompt_cache_diagnostics,
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
    try_send_stream_event(
        &completed_request.stream_event_sender,
        ChatGenerationStreamEvent::Completed {
            prompt_token_count,
            generated_token_count,
            reasoning_token_count,
            cached_token_count,
            reason,
        },
    )
}
