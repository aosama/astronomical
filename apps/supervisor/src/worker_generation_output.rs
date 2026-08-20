//! Owns generation-scoped output and progress event handling.
//!
//! Correlation is validated before any stream or health mutation so malformed
//! worker events cannot leave externally visible state partially advanced.

use astronomical_ipc_protocol::WorkerEvent;
use std::sync::{Arc, RwLock};
use tokio::time::Instant;

use crate::{
    ActiveRequestProgress, ChatGenerationStreamEvent, WorkerActivity, WorkerControlError,
    WorkerHealthSnapshot,
    chat_generation_executor::try_send_stream_event,
    worker_event_handler::protocol_violation,
    worker_health::{
        publish_active_request_progress, publish_activity, publish_latest_mlx_memory_snapshot,
    },
    worker_loop_types::ActiveGeneration,
};

pub(super) fn handle_worker_output(
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
    } = worker_event
    else {
        return Err(protocol_violation(
            "handle_worker_output received a non-Output event",
        ));
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
        if let astronomical_ipc_protocol::ChatGenerationOutput::ToolCall {
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
    Ok(())
}

pub(super) fn handle_worker_generation_progress(
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
        publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
    }
    Ok(())
}

pub(super) fn handle_worker_first_decode_completed(
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

pub(super) fn handle_worker_prompt_work_reuse(
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
    crate::worker_health::record_prompt_work_reuse(health_snapshot, prompt_work_reuse);
    Ok(())
}
