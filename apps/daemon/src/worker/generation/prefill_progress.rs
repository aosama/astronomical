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
        running_prefill_token_count(processed_tokens, total_tokens, prompt_layer_progress),
        elapsed_millis,
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

/// The prompt tokens the session's running reading speed counts. A prompt
/// read one layer at a time from the SSD computes every token through each
/// layer in turn, so it reports layers done and no computed tokens until the
/// last layer: its work so far is the prompt's tokens times the share of the
/// layers done, the same share the progress bar shows.
pub(in crate::worker) fn running_prefill_token_count(
    processed_tokens: u32,
    total_tokens: u32,
    prompt_layer_progress: Option<ipc_protocol::WorkerPromptLayerProgress>,
) -> u32 {
    let Some(prompt_layer_progress) = prompt_layer_progress else {
        return processed_tokens;
    };
    if prompt_layer_progress.layer_count == 0 {
        return processed_tokens;
    }
    let layer_share_tokens = u64::from(total_tokens)
        * u64::from(
            prompt_layer_progress
                .completed_layer_count
                .min(prompt_layer_progress.layer_count),
        )
        / u64::from(prompt_layer_progress.layer_count);
    processed_tokens.max(u32::try_from(layer_share_tokens).unwrap_or(u32::MAX))
}

#[cfg(test)]
#[path = "prefill_progress_tests.rs"]
mod prefill_progress_tests;
