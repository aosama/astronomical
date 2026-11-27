use std::sync::{Arc, RwLock};

use ipc_protocol::WorkerEvent;

use crate::{
    ActiveRequestProgress, ChatGenerationStreamEvent, WorkerControlError, WorkerHealthSnapshot,
    serving::chat_generation_executor::try_send_stream_event,
    worker::health::{publish_active_request_progress, publish_latest_mlx_memory_snapshot},
    worker::loop_types::ActiveGeneration,
};

pub(in crate::worker) fn handle_worker_prefill_progress(
    worker_prefill_progress_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut ActiveGeneration,
) -> Result<(), WorkerControlError> {
    let WorkerEvent::PrefillProgress {
        request_id,
        prompt_processing_phase,
        processed_tokens,
        total_tokens,
        elapsed_millis,
        forward_prefill_chunk_elapsed_millis,
        completed_prefill_chunk_tokens,
        mlx_memory_snapshot,
        expert_residency,
        prompt_layer_progress,
    } = worker_prefill_progress_event
    else {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "prefill progress handler received another event",
        });
    };
    if request_id != active_request.request_id {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "prefill progress request mismatch",
        });
    }

    // Worker elapsed_millis is cumulative, so retain the latest amount.
    active_request.prefill_work = crate::worker::loop_types::RequestPrefillWork {
        elapsed_millis,
        computed_token_count: processed_tokens,
    };
    crate::worker::health::record_running_prefill(
        health_snapshot,
        processed_tokens,
        running_prefill_elapsed_millis(elapsed_millis, active_request.request_started_at.elapsed()),
    );
    if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
        active_request.fold_observed_mlx_peak_memory_bytes(mlx_memory_snapshot.peak_memory_bytes);
        active_request.last_mlx_active_memory_bytes = Some(mlx_memory_snapshot.active_memory_bytes);
    }
    if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
        publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
    }
    if let Some(expert_residency) = expert_residency {
        crate::worker::health::publish_worker_expert_residency(health_snapshot, expert_residency);
    }
    try_send_stream_event(
        &active_request.stream_event_sender,
        ChatGenerationStreamEvent::PrefillProgress {
            processed_tokens,
            total_tokens,
            elapsed_millis,
            forward_prefill_chunk_elapsed_millis,
            completed_prefill_chunk_tokens,
            mlx_active_memory_bytes: mlx_memory_snapshot
                .map(|snapshot| snapshot.active_memory_bytes),
            mlx_buffer_cache_bytes: mlx_memory_snapshot
                .map(|snapshot| snapshot.mlx_buffer_cache_bytes),
            mlx_peak_memory_bytes: mlx_memory_snapshot.map(|snapshot| snapshot.peak_memory_bytes),
        },
    )?;
    publish_active_request_progress(
        health_snapshot,
        ActiveRequestProgress::Prefill {
            prompt_processing_phase,
            processed_tokens,
            total_tokens,
            request_started_at: active_request.request_started_at,
            elapsed_millis,
            completed_prefill_chunk_tokens,
            prompt_layer_progress,
        },
    );
    Ok(())
}

/// The prompt time the session's running reading speed uses. A prompt read
/// one layer at a time from the SSD reports its tokens as layers finish, but
/// its measured prompt time only once the whole prompt is read: until then
/// the measured time is 0, and tokens with no time would leave the reading
/// speed blank for the whole prompt. Until a measured time arrives, the
/// request's own clock stands in, the elapsed time the status shows.
pub(in crate::worker) fn running_prefill_elapsed_millis(
    measured_elapsed_millis: u64,
    request_elapsed: std::time::Duration,
) -> u64 {
    if measured_elapsed_millis > 0 {
        return measured_elapsed_millis;
    }
    u64::try_from(request_elapsed.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[path = "prefill_progress_tests.rs"]
mod prefill_progress_tests;
