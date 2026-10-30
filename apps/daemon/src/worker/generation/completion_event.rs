//! Completion-event validation, local performance logging, and stream delivery.
//!
//! Worker completion has four audiences with different contracts: protocol
//! validation protects daemon state, the local performance log receives
//! cache diagnostics, the completion attribution log receives emitted tool
//! calls when enabled, and the public stream receives only OpenAI-compatible
//! completion fields. Keeping the fan-out here makes that separation explicit.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, RequestId, WorkerPersistentPromptCacheRequestDiagnostics,
};

use crate::{
    ChatGenerationStreamEvent, CompletionAttributionLog, GenerationPerformanceLog,
    GenerationPerformanceRecord, WorkerActivity, WorkerControlError, WorkerHealthSnapshot,
    performance::completion_attribution_log::record_completion_at_now,
    performance::generation_log::unix_epoch_millis,
    serving::chat_generation_executor::try_send_stream_event,
    worker::event_handler::protocol_violation,
    worker::health::{clear_active_request_progress, publish_activity, record_serving_session},
    worker::loop_types::{ActiveGeneration, ActiveWorkerRequest},
};

/// Worker-reported completion scalars from one worker completion event,
/// grouped so the fan-out helpers receive one cohesive value instead of
/// parallel scalar parameters.
struct CompletionMeasurements {
    request_id: RequestId,
    prompt_token_count: u32,
    generated_token_count: u16,
    reasoning_token_count: u16,
    cached_token_count: u32,
    reason: ChatGenerationCompletionReason,
}

/// Derived wall-clock timings and throughput for one completed generation,
/// shared by the log line, the performance record, and the serving-session
/// record so all three audiences report identical numbers.
struct CompletionTimings {
    total_elapsed_millis: u64,
    generation_elapsed_millis: u64,
    prefill_elapsed_millis: u64,
    prefill_tok_per_second: Option<f64>,
    generation_tok_per_second: Option<f64>,
}

/// Everything the completion fan-out needs after the active request is
/// retired: the worker-reported measurements, the derived timings, the
/// serving model, and the retired per-request state.
struct CompletionObservation<'a> {
    completion_measurements: &'a CompletionMeasurements,
    completed_request: &'a ActiveGeneration,
    completion_timings: &'a CompletionTimings,
    model_id: String,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::worker) fn handle_worker_completion_event(
    request_id: RequestId,
    prompt_token_count: u32,
    generated_token_count: u16,
    reasoning_token_count: u16,
    cached_token_count: u32,
    persistent_prompt_cache_diagnostics: Option<WorkerPersistentPromptCacheRequestDiagnostics>,
    reason: ChatGenerationCompletionReason,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_worker_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) -> Result<(), WorkerControlError> {
    let completion_measurements = CompletionMeasurements {
        request_id,
        prompt_token_count,
        generated_token_count,
        reasoning_token_count,
        cached_token_count,
        reason,
    };
    // Cache diagnostics are request-scoped completion evidence. They go only to
    // the local performance row; the public generation stream keeps its stable
    // OpenAI-compatible completion shape unchanged.
    let Some(active_request) = active_worker_request
        .as_ref()
        .and_then(ActiveWorkerRequest::chat)
    else {
        return Err(protocol_violation("completion without an active request"));
    };
    validate_chat_completion_event(&completion_measurements, active_request)?;
    let Some(ActiveWorkerRequest::Chat(completed_request)) = active_worker_request.take() else {
        return Err(protocol_violation(
            "chat completion while image generation is active",
        ));
    };
    let completion_timings =
        measure_completion_timings(&completion_measurements, &completed_request);
    let completion_observation = CompletionObservation {
        completion_measurements: &completion_measurements,
        completed_request: &completed_request,
        completion_timings: &completion_timings,
        model_id: read_ready_model_id(health_snapshot),
    };
    log_completed_generation(&completion_observation);
    record_completion_performance_and_attribution(
        &completion_observation,
        persistent_prompt_cache_diagnostics,
        performance_log,
        completion_log,
    );
    publish_post_completion_activity(&completion_observation, health_snapshot);
    try_send_stream_event(
        &completed_request.stream_event_sender,
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: completion_measurements.prompt_token_count,
            generated_token_count: completion_measurements.generated_token_count,
            reasoning_token_count: completion_measurements.reasoning_token_count,
            cached_token_count: completion_measurements.cached_token_count,
            reason: completion_measurements.reason,
        },
    )
}

/// Validates the completion against the still-active chat request before any
/// daemon state advances: correlation, monotonic token counts, the output
/// budget, and the completion reason's preconditions all must hold.
fn validate_chat_completion_event(
    completion_measurements: &CompletionMeasurements,
    active_request: &ActiveGeneration,
) -> Result<(), WorkerControlError> {
    let latest_known_generated_token_count = active_request
        .generated_token_count
        .max(active_request.latest_generation_progress_token_count);
    let is_valid_reason = match completion_measurements.reason {
        ChatGenerationCompletionReason::EndOfSequence => {
            completion_measurements.generated_token_count <= active_request.max_output_tokens
        }
        ChatGenerationCompletionReason::MaximumOutputTokens => {
            completion_measurements.generated_token_count == active_request.max_output_tokens
        }
        ChatGenerationCompletionReason::ToolCalls => active_request.next_tool_call_index > 0,
        ChatGenerationCompletionReason::Cancelled => false,
    };
    if completion_measurements.request_id != active_request.request_id
        || completion_measurements.generated_token_count < latest_known_generated_token_count
        || completion_measurements.generated_token_count > active_request.max_output_tokens
        || !is_valid_reason
    {
        return Err(protocol_violation(
            "completion correlation or count mismatch",
        ));
    }
    Ok(())
}

/// Derives the wall-clock timings and throughput figures every completion
/// audience (log line, performance row, serving session) must agree on.
fn measure_completion_timings(
    completion_measurements: &CompletionMeasurements,
    completed_request: &ActiveGeneration,
) -> CompletionTimings {
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
            completion_measurements.prompt_token_count,
            completion_measurements.cached_token_count,
            completion_measurements.generated_token_count,
            prefill_elapsed_millis,
            generation_elapsed_millis,
        );
    CompletionTimings {
        total_elapsed_millis,
        generation_elapsed_millis,
        prefill_elapsed_millis,
        prefill_tok_per_second,
        generation_tok_per_second,
    }
}

/// Maps the wire reason to the stable snake_case label the performance and
/// attribution logs have always used.
fn completion_reason_label(reason: ChatGenerationCompletionReason) -> &'static str {
    match reason {
        ChatGenerationCompletionReason::EndOfSequence => "end_of_sequence",
        ChatGenerationCompletionReason::MaximumOutputTokens => "maximum_output_tokens",
        ChatGenerationCompletionReason::ToolCalls => "tool_calls",
        ChatGenerationCompletionReason::Cancelled => "cancelled",
    }
}

/// Reads the currently published ready model so completion rows identify the
/// producing model even across a mid-request swap.
fn read_ready_model_id(health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>) -> String {
    health_snapshot
        .read()
        .map(|snapshot| snapshot.ready_model_id.clone())
        .ok()
        .flatten()
        .unwrap_or_default()
}

/// Logs the daemon-visible completion summary.
fn log_completed_generation(completion_observation: &CompletionObservation<'_>) {
    tracing::info!(
        request_id = completion_observation.completion_measurements.request_id.value(),
        prompt_token_count = completion_observation.completion_measurements.prompt_token_count,
        generated_token_count =
            completion_observation.completion_measurements.generated_token_count,
        cached_token_count = completion_observation.completion_measurements.cached_token_count,
        completion_reason = ?completion_observation.completion_measurements.reason,
        prefill_elapsed_millis = completion_observation.completion_timings.prefill_elapsed_millis,
        generation_elapsed_millis =
            completion_observation.completion_timings.generation_elapsed_millis,
        total_elapsed_millis = completion_observation.completion_timings.total_elapsed_millis,
        time_to_first_output_millis = ?completion_observation.completed_request
            .time_to_first_output_millis,
        generation_preparation_elapsed_millis = ?completion_observation
            .completed_request
            .generation_preparation_elapsed_millis,
        prefill_tok_per_second = ?completion_observation.completion_timings.prefill_tok_per_second,
        generation_tok_per_second =
            ?completion_observation.completion_timings.generation_tok_per_second,
        "worker generation completed"
    );
}

/// Writes the completed request to the local performance row and, when the
/// operator enabled the toggle, to the completion attribution log.
fn record_completion_performance_and_attribution(
    completion_observation: &CompletionObservation<'_>,
    persistent_prompt_cache_diagnostics: Option<WorkerPersistentPromptCacheRequestDiagnostics>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) {
    let completion_reason =
        completion_reason_label(completion_observation.completion_measurements.reason);
    performance_log.record(&GenerationPerformanceRecord {
        timestamp_millis: unix_epoch_millis(),
        request_id: completion_observation
            .completion_measurements
            .request_id
            .value(),
        model_id: completion_observation.model_id.clone(),
        prompt_token_count: completion_observation
            .completion_measurements
            .prompt_token_count,
        cached_token_count: completion_observation
            .completion_measurements
            .cached_token_count,
        generated_token_count: completion_observation
            .completion_measurements
            .generated_token_count,
        completion_reason: completion_reason.to_owned(),
        prefill_elapsed_millis: completion_observation
            .completion_timings
            .prefill_elapsed_millis,
        generation_elapsed_millis: completion_observation
            .completion_timings
            .generation_elapsed_millis,
        total_elapsed_millis: completion_observation
            .completion_timings
            .total_elapsed_millis,
        time_to_first_output_millis: completion_observation
            .completed_request
            .time_to_first_output_millis,
        generation_preparation_elapsed_millis: completion_observation
            .completed_request
            .generation_preparation_elapsed_millis,
        first_decode_forward_elapsed_millis: completion_observation
            .completed_request
            .first_decode_forward_elapsed_millis,
        generation_preparation_expert_source_read_byte_count: 0,
        final_resident_expert_count: completion_observation
            .completed_request
            .final_resident_expert_count,
        final_resident_expert_payload_bytes: completion_observation
            .completed_request
            .final_resident_expert_payload_bytes,
        prefill_tok_per_second: completion_observation
            .completion_timings
            .prefill_tok_per_second,
        generation_tok_per_second: completion_observation
            .completion_timings
            .generation_tok_per_second,
        mlx_peak_memory_bytes: completion_observation
            .completed_request
            .maximum_mlx_peak_memory_bytes,
        mlx_active_memory_bytes: completion_observation
            .completed_request
            .last_mlx_active_memory_bytes,
        persistent_prompt_cache_diagnostics,
    });
    // Attribute what the model emitted — function names and the arguments JSON —
    // so argument pollution and foreign-dialect regressions are diagnosable
    // instead of invisible. No-op unless the operator enabled the toggle.
    record_completion_at_now(
        completion_log,
        completion_observation
            .completion_measurements
            .request_id
            .value(),
        &completion_observation.model_id,
        completion_reason,
        &completion_observation
            .completed_request
            .completed_tool_calls,
    );
}

/// Folds the completed request into the serving-session health record and
/// returns the worker to its idle published state.
fn publish_post_completion_activity(
    completion_observation: &CompletionObservation<'_>,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) {
    record_serving_session(
        health_snapshot,
        completion_observation
            .completion_measurements
            .prompt_token_count,
        completion_observation
            .completion_measurements
            .cached_token_count,
        completion_observation
            .completion_timings
            .prefill_tok_per_second,
        completion_observation
            .completion_timings
            .generation_tok_per_second,
    );
    publish_activity(health_snapshot, WorkerActivity::Idle);
    clear_active_request_progress(health_snapshot);
}

#[cfg(test)]
#[path = "completion_event_tests.rs"]
mod completion_event_tests;
