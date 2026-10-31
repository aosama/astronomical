//! Owns generation-scoped output and progress event handling.
//!
//! Correlation is validated before any stream or health mutation so malformed
//! worker events cannot leave externally visible state partially advanced.

use astronomical_ipc_protocol::WorkerEvent;
use std::sync::{Arc, RwLock};
use tokio::time::Instant;

use crate::{
    ActiveRequestProgress, ChatGenerationStreamEvent, CompletedToolCall, WorkerActivity,
    WorkerControlError, WorkerHealthSnapshot,
    serving::chat_generation_executor::try_send_stream_event,
    worker::event_handler::protocol_violation,
    worker::health::{
        publish_active_request_progress, publish_activity, publish_latest_mlx_memory_snapshot,
    },
    worker::loop_types::ActiveGeneration,
};

pub(in crate::worker) fn handle_worker_output(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut ActiveGeneration,
) -> Result<(), WorkerControlError> {
    let WorkerEvent::Output {
        request_id,
        sequence_number,
        generated_token_count,
        outputs,
        mlx_memory_snapshot,
        expert_residency,
    } = worker_event
    else {
        return Err(protocol_violation(
            "handle_worker_output received a non-Output event",
        ));
    };
    let ValidatedOutputBatch {
        next_sequence_number,
    } = validate_output_correlation_and_sequence(
        active_request,
        request_id,
        sequence_number,
        generated_token_count,
        &outputs,
    )?;
    let next_tool_call_index = accumulate_ordered_tool_calls(active_request, &outputs)?;
    for output in outputs {
        let stream_event = ChatGenerationStreamEvent::from_worker_output(output);
        try_send_stream_event(&active_request.stream_event_sender, stream_event)?;
    }
    record_first_output_timings(active_request, health_snapshot);
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
        active_request.fold_observed_mlx_peak_memory_bytes(mlx_memory_snapshot.peak_memory_bytes);
        active_request.last_mlx_active_memory_bytes = Some(mlx_memory_snapshot.active_memory_bytes);
        publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
    }
    if let Some(expert_residency) = expert_residency {
        crate::worker::health::publish_worker_expert_residency(health_snapshot, expert_residency);
    }
    Ok(())
}

/// The sequence number the next output batch must present, proven by the
/// correlation check before any output is consumed.
struct ValidatedOutputBatch {
    next_sequence_number: u16,
}

/// Validates one Output event against the request it claims to serve: the
/// request id and expected sequence number must match, the reported token
/// count must advance monotonically inside the request's ceiling, the batch
/// must be non-empty and u16-sized, and the sequence must not wrap. Any
/// violation is a protocol violation — the stream is desynchronized and the
/// request must fail rather than skip.
fn validate_output_correlation_and_sequence(
    active_request: &ActiveGeneration,
    request_id: astronomical_ipc_protocol::RequestId,
    sequence_number: u16,
    generated_token_count: u16,
    outputs: &[astronomical_ipc_protocol::ChatGenerationOutput],
) -> Result<ValidatedOutputBatch, WorkerControlError> {
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
    Ok(ValidatedOutputBatch {
        next_sequence_number,
    })
}

/// Accumulates the batch's tool calls in emission order, enforcing that each
/// tool-call index continues exactly from the previous one. The accumulated
/// calls serve completion attribution so the completion event can log
/// function names and arguments without re-reading the public stream; the
/// arguments are bounded at write time. Returns the next expected index.
fn accumulate_ordered_tool_calls(
    active_request: &mut ActiveGeneration,
    outputs: &[astronomical_ipc_protocol::ChatGenerationOutput],
) -> Result<u16, WorkerControlError> {
    let mut next_tool_call_index = active_request.next_tool_call_index;
    for output in outputs {
        if let astronomical_ipc_protocol::ChatGenerationOutput::ToolCall {
            tool_call_index,
            function_name,
            arguments_json,
        } = output
        {
            if *tool_call_index != next_tool_call_index {
                return Err(protocol_violation("tool-call index mismatch"));
            }
            active_request.completed_tool_calls.push(CompletedToolCall {
                tool_call_index: *tool_call_index,
                function_name: function_name.clone(),
                arguments_json: arguments_json.clone(),
            });
            next_tool_call_index = next_tool_call_index
                .checked_add(1)
                .ok_or_else(|| protocol_violation("tool-call index overflow"))?;
        }
    }
    Ok(next_tool_call_index)
}

/// Records the one-time timings of the first output batch: time to first
/// output from request start, the generation-preparation window when one was
/// measured, and — the first time generation itself begins — the generation
/// clock plus the activity flip to Generating.
fn record_first_output_timings(
    active_request: &mut ActiveGeneration,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) {
    if active_request.time_to_first_output_millis.is_none() {
        active_request.time_to_first_output_millis = Some(
            u64::try_from(active_request.request_started_at.elapsed().as_millis())
                .unwrap_or(u64::MAX),
        );
    }
    if active_request
        .generation_preparation_elapsed_millis
        .is_none()
    {
        active_request.generation_preparation_elapsed_millis = active_request
            .generation_preparation_started_at
            .map(|preparation_started_at| {
                u64::try_from(preparation_started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
            });
    }
    if active_request.generation_started_at.is_none() {
        active_request.generation_started_at = Some(Instant::now());
        publish_activity(health_snapshot, WorkerActivity::Generating);
    }
}

pub(in crate::worker) fn handle_worker_generation_progress(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut ActiveGeneration,
) -> Result<(), WorkerControlError> {
    let WorkerEvent::GenerationProgress {
        request_id,
        generated_token_count,
        maximum_output_tokens,
        elapsed_millis,
        mlx_memory_snapshot,
        expert_residency,
    } = worker_event
    else {
        return Err(protocol_violation(
            "handle_worker_generation_progress received a non-GenerationProgress event",
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
        let elapsed_duration = std::time::Duration::from_millis(elapsed_millis);
        active_request.generation_started_at = Some(
            Instant::now()
                .checked_sub(elapsed_duration)
                .unwrap_or_else(Instant::now),
        );
    }
    if active_request
        .generation_preparation_elapsed_millis
        .is_none()
    {
        active_request.generation_preparation_elapsed_millis = active_request
            .generation_preparation_started_at
            .map(|preparation_started_at| {
                u64::try_from(preparation_started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
            });
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
        active_request.fold_observed_mlx_peak_memory_bytes(mlx_memory_snapshot.peak_memory_bytes);
        active_request.last_mlx_active_memory_bytes = Some(mlx_memory_snapshot.active_memory_bytes);
        publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
    }
    if let Some(expert_residency) = expert_residency {
        crate::worker::health::publish_worker_expert_residency(health_snapshot, expert_residency);
    }
    Ok(())
}

pub(in crate::worker) fn handle_worker_first_decode_completed(
    worker_event: WorkerEvent,
    active_request: &mut ActiveGeneration,
) -> Result<(), WorkerControlError> {
    let WorkerEvent::FirstDecodeCompleted {
        request_id,
        elapsed_millis,
    } = worker_event
    else {
        return Err(protocol_violation(
            "handle_worker_first_decode_completed received a non-FirstDecodeCompleted event",
        ));
    };
    if request_id != active_request.request_id
        || active_request.first_decode_forward_elapsed_millis.is_some()
    {
        return Err(protocol_violation(
            "first decode completion correlation or duplication mismatch",
        ));
    }
    active_request.first_decode_forward_elapsed_millis = Some(elapsed_millis);
    if active_request
        .generation_preparation_elapsed_millis
        .is_none()
    {
        active_request.generation_preparation_elapsed_millis = active_request
            .generation_preparation_started_at
            .map(|preparation_started_at| {
                u64::try_from(preparation_started_at.elapsed().as_millis())
                    .unwrap_or(u64::MAX)
                    .saturating_sub(elapsed_millis)
            });
    }
    Ok(())
}

pub(in crate::worker) fn handle_worker_prompt_work_reuse(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &ActiveGeneration,
) -> Result<(), WorkerControlError> {
    let WorkerEvent::PromptWorkReuse {
        request_id,
        prompt_work_reuse,
    } = worker_event
    else {
        return Err(protocol_violation(
            "handle_worker_prompt_work_reuse received a non-PromptWorkReuse event",
        ));
    };
    if request_id != active_request.request_id {
        return Err(protocol_violation("prompt-work reuse request mismatch"));
    }
    crate::worker::health::record_prompt_work_reuse(health_snapshot, prompt_work_reuse);
    Ok(())
}
